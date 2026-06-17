"""Run missing Local ordering seeds for HM12 and HM16."""
import torch, sys, time, gc, json
from pathlib import Path
ROOT = Path('/home/s110/code/ttpi_iclr')
sys.path.insert(0, str(ROOT))
from ttpi import TTPI
from dynamic_systems import HardMove

torch.set_default_dtype(torch.float64)
device = 'cuda'

def run_local(n_act, seed, n_iter=30):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()

    if n_act == 12: NS, NA = 30, 40
    else:           NS, NA = 25, 30

    L=1.0; vmax=0.25
    domain_acc = torch.linspace(-vmax, vmax, NA, device=device)
    domain_sw = torch.tensor([0.,1.], device=device)
    smin = torch.tensor([-L,-L,0.,0.], device=device)
    smax = torch.tensor([L,L,vmax,vmax], device=device)
    state_domain = [torch.linspace(smin[i], smax[i], NS, device=device) for i in range(4)]
    action_domain = [domain_acc, domain_sw] * n_act  # Local = identity order

    dyn = HardMove(dt=0.01, w_goal=1e3, w_action=1e4, n=n_act, device=device)
    ttpi = TTPI(domain_state=state_domain, domain_action=action_domain,
                reward=dyn.reward_state_action, normalize_reward=True,
                forward_model=dyn.forward_simulate,
                gamma=0.99, rmax_v=60, rmax_a=60,
                nswp_v=5, nswp_a=10, kickrank_v=10, kickrank_a=10,
                max_batch_v=5000, max_batch_a=20000,
                eps_cross_v=1e-3, eps_cross_a=1e-3, eps_round_v=1e-3, eps_round_a=1e-3,
                n_samples=50, verbose=False, device=device)
    ttpi.plt_training_stat = lambda: None; ttpi.save_model = lambda *a,**kw: None
    _orig = ttpi.PI_update
    def _new(n_iter_v=1): _orig(n_iter_v=n_iter_v); torch.cuda.empty_cache(); gc.collect()
    ttpi.PI_update = _new

    g = torch.Generator(device='cpu'); g.manual_seed(12345+seed)
    n_test = 50
    init_state = torch.empty((n_test,4), dtype=torch.float64)
    for i in range(4):
        r = torch.rand(n_test, generator=g, dtype=torch.float64).clip(0.25,0.75)
        init_state[:,i] = smin[i].cpu() + r*(smax[i].cpu()-smin[i].cpu())
    init_state[:,2:4]=0.0; init_state = init_state.to(device)

    eval_hist=[]; rank_hist=[]; last_cb=-1
    tr={'S':0,'mu':0,'to':0}; bj={'S':0,'mu':0,'to':0,'cb':-1}

    def cb(ttpi, state=init_state, callback_count=0):
        nonlocal last_cb; last_cb=callback_count
        s=state.clone(); Th=1000; ac=torch.ones(n_test,dtype=torch.bool,device=device)
        su=torch.zeros(n_test,dtype=torch.bool,device=device)
        sp=s[:,:2].clone(); pp=s[:,:2].clone(); pl=torch.zeros(n_test,device=device)
        for _ in range(Th):
            a=ttpi.policy(s); ns=dyn.forward_simulate(s,a); np=ns[:,:2]
            pl+=ac.float()*torch.linalg.norm(np-pp,dim=-1)
            d=torch.linalg.norm(np,dim=-1); nsu=ac&(d<=0.02)
            su|=nsu; ac&=~nsu; s=ns; pp=np
        sh=torch.linalg.norm(sp,dim=-1); mu_all=(sh/(pl+1e-12)).clamp(max=1.0)**2
        mu_s=mu_all[su].mean() if su.any() else torch.tensor(0.)
        S_=float(su.float().mean()); mv=float(mu_s); to=S_*mv
        tr['S']=max(tr['S'],S_); tr['mu']=max(tr['mu'],mv); tr['to']=max(tr['to'],to)
        if to>bj['to']: bj.update({'S':S_,'mu':mv,'to':to,'cb':callback_count})
        eval_hist.append({'S':S_,'mu':mv,'to':to,'cb':callback_count})
        pr=ttpi.policy_model.tt().ranks_tt.tolist(); vr=ttpi.v_model.ranks_tt.tolist()
        ar=ttpi.a_model.ranks_tt.tolist()
        rank_hist.append({'cb':callback_count,'Vr':max(vr),'Ar':max(ar),'Pr':max(pr)})
        al=torch.cuda.memory_allocated()/1e9; pk=torch.cuda.max_memory_allocated()/1e9
        print(f'  cb{callback_count:>3}: S={S_:.3f} mu={mv:.4f} Vr={max(vr)} Ar={max(ar)} Pr={max(pr)} p={pk:.2f}G',flush=True)
        torch.cuda.empty_cache(); gc.collect()
        return torch.tensor(0.),torch.tensor(0.)

    t0=time.time(); oom=False
    try:
        ttpi.train(resume=False,n_iter_max=n_iter,n_iter_v=1,callback=cb,
                   callback_freq=10,verbose=False,file_name=None)
    except torch.OutOfMemoryError:
        oom=True; print('OOM!',flush=True)
    except Exception as e:
        print(f'Error:{e}',flush=True)

    t=time.time()-t0
    pr=ttpi.policy_model.tt().ranks_tt.tolist(); vr=ttpi.v_model.ranks_tt.tolist()
    ar=ttpi.a_model.ranks_tt.tolist()
    peak=torch.cuda.max_memory_allocated()/1e9
    status="OOM" if oom else ("OK" if last_cb>=n_iter-1 else f"cb{last_cb}")

    return {"n_act":n_act,"label":"Local","seed":seed,"status":status,
            "S_joint":bj['S'],"mu_joint":bj['mu'],"to_joint":bj['to'],
            "Vr_max":max(vr),"Ar_max":max(ar),"Pr_max":max(pr),
            "peak_gb":peak,"time_s":t,
            "eval_history":eval_hist,"rank_history":rank_hist}

if __name__ == "__main__":
    results = {}
    for n_act, seeds in [(12, [1,2]), (16, [1,2])]:
        for seed in seeds:
            print(f"\n=== HM{n_act} Local seed={seed} ===", flush=True)
            r = run_local(n_act, seed, n_iter=30)
            key = f"n{n_act}_Local_s{seed}"
            results[key] = r
            print(f"  {key}: {r['status']} S×μ={r['to_joint']:.4f} Ar={r['Ar_max']} Pr={r['Pr_max']} peak={r['peak_gb']:.2f}G T={r['time_s']:.0f}s", flush=True)

    with open("repro/results/local_extra_seeds.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: repro/results/local_extra_seeds.json")
