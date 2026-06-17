"""PAM scaling: n_act=8/12/16, Local/BadSplit/Random, 1 seed each."""
import torch, sys, time, gc, json, statistics
from pathlib import Path
ROOT = Path('/home/s110/code/ttpi_iclr')
sys.path.insert(0, str(ROOT))
from ttpi import TTPI
from dynamic_systems import HardMove

torch.set_default_dtype(torch.float64)
device = 'cuda'

def get_configs(n_act):
    if n_act == 8:  return 40, 50
    if n_act == 12: return 30, 40
    if n_act == 16: return 25, 30

def build_orders(n_act, label):
    D_a = 2 * n_act
    if label == "Local": return list(range(D_a))
    if label == "BadSplit": return list(range(0, D_a, 2)) + list(range(1, D_a, 2))
    if label == "Random":
        g = torch.Generator().manual_seed(42)
        return torch.randperm(D_a, generator=g).tolist()
    return None

def run_one(n_act, a_order, seed, n_iter, label):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()

    NS, NA = get_configs(n_act)
    L=1.0; vmax=0.25
    domain_acc = torch.linspace(-vmax, vmax, NA, device=device)
    domain_sw = torch.tensor([0.,1.], device=device)
    smin = torch.tensor([-L,-L,0.,0.], device=device)
    smax = torch.tensor([L,L,vmax,vmax], device=device)
    state_domain = [torch.linspace(smin[i], smax[i], NS, device=device) for i in range(4)]
    action_domain_phys = [domain_acc, domain_sw] * n_act
    action_domain_tt = [action_domain_phys[i] for i in a_order]
    a_inv = torch.tensor([a_order.index(i) for i in range(2*n_act)])

    dyn = HardMove(dt=0.01, w_goal=1e3, w_action=1e4, n=n_act, device=device)

    ttpi = TTPI(domain_state=state_domain, domain_action=action_domain_tt,
                reward=lambda s,a: dyn.reward_state_action(s, a[..., a_inv.to(a.device)]),
                normalize_reward=True,
                forward_model=lambda s,a: dyn.forward_simulate(s, a[..., a_inv.to(a.device)]),
                gamma=0.99, rmax_v=60, rmax_a=60,
                nswp_v=5, nswp_a=10, kickrank_v=10, kickrank_a=10,
                max_batch_v=5000, max_batch_a=20000,
                eps_cross_v=1e-3, eps_cross_a=1e-3, eps_round_v=1e-3, eps_round_a=1e-3,
                n_samples=50, verbose=False, device=device)
    ttpi.plt_training_stat = lambda: None; ttpi.save_model = lambda *a,**kw: None
    _orig = ttpi.PI_update
    def _new(n_iter_v=1): _orig(n_iter_v=n_iter_v); torch.cuda.empty_cache(); gc.collect()
    ttpi.PI_update = _new

    g = torch.Generator(device='cpu'); g.manual_seed(12345)
    n_test = 50
    init_state = torch.empty((n_test,4), dtype=torch.float64)
    for i in range(4):
        r = torch.rand(n_test, generator=g, dtype=torch.float64).clip(0.25,0.75)
        init_state[:,i] = smin[i].cpu() + r*(smax[i].cpu()-smin[i].cpu())
    init_state[:,2:4]=0.0; init_state = init_state.to(device)

    eval_hist = []; rank_hist = []; last_cb = -1
    tracker = {'S':0,'mu':0,'to':0}
    best_joint = {'S':0,'mu':0,'to':0,'cb':-1}

    def cb(ttpi, state=init_state, callback_count=0):
        nonlocal last_cb; last_cb = callback_count
        s = state.clone()
        Th=int(10/0.01); ac=torch.ones(n_test,dtype=torch.bool,device=device)
        su=torch.zeros(n_test,dtype=torch.bool,device=device)
        sp=s[:,:2].clone(); pp=s[:,:2].clone(); pl=torch.zeros(n_test,device=device)
        for _ in range(Th):
            a_tt = ttpi.policy(s); a = a_tt[..., a_inv.to(a_tt.device)]
            ns = dyn.forward_simulate(s, a); np = ns[:,:2]
            pl += ac.float()*torch.linalg.norm(np-pp,dim=-1)
            d=torch.linalg.norm(np,dim=-1); nsu=ac&(d<=0.02)
            su|=nsu; ac&=~nsu; s=ns; pp=np
        sh=torch.linalg.norm(sp,dim=-1)
        mu_all=(sh/(pl+1e-12)).clamp(max=1.0)**2
        mu_s=mu_all[su].mean() if su.any() else torch.tensor(0.)
        S_=float(su.float().mean()); mv=float(mu_s); to=S_*mv
        tracker['S']=max(tracker['S'],S_); tracker['mu']=max(tracker['mu'],mv); tracker['to']=max(tracker['to'],to)
        if to>best_joint['to']: best_joint.update({'S':S_,'mu':mv,'to':to,'cb':callback_count})
        eval_hist.append({'S':S_,'mu':mv,'to':to,'cb':callback_count})
        p_r=ttpi.policy_model.tt().ranks_tt.tolist(); v_r=ttpi.v_model.ranks_tt.tolist()
        a_r=ttpi.a_model.ranks_tt.tolist()
        rank_hist.append({'cb':callback_count,'Vr':max(v_r),'Ar':max(a_r),'Pr':max(p_r)})
        al=torch.cuda.memory_allocated()/1e9; pk=torch.cuda.max_memory_allocated()/1e9
        print(f'  cb{callback_count:>3}: S={S_:.3f} mu={mv:.4f} Vr={max(v_r)} Ar={max(a_r)} Pr={max(p_r)} p={pk:.2f}G',flush=True)
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

    return {"n_act":n_act,"label":label,"status":status,"seed":seed,
            "S_joint":best_joint['S'],"mu_joint":best_joint['mu'],"to_joint":best_joint['to'],
            "Vr_max":max(vr),"Ar_max":max(ar),"Pr_max":max(pr),
            "peak_gb":peak,"time_s":t,"eval_history":eval_hist,"rank_history":rank_hist}

if __name__ == "__main__":
    scaling_results = {}
    for n_act in [8, 12, 16]:
        for label in ["Local", "BadSplit", "Random"]:
            a_ord = build_orders(n_act, label)
            print(f"\n=== n_act={n_act} {label} ===", flush=True)
            r = run_one(n_act, a_ord, seed=0, n_iter=30, label=label)
            key = f"n{n_act}_{label}"
            scaling_results[key] = r
            print(f"  {key}: {r['status']} S×μ={r['to_joint']:.4f} Vr={r['Vr_max']} Ar={r['Ar_max']} Pr={r['Pr_max']} peak={r['peak_gb']:.2f}G T={r['time_s']:.0f}s", flush=True)
        torch.cuda.empty_cache(); gc.collect()

    # Summary
    print(f"\n{'='*75}")
    print("SCALING SUMMARY")
    print(f"{'='*75}")
    print(f"{'n_act':<8} {'Order':<12} {'S×μ':<10} {'Ar':<6} {'Pr':<6} {'Peak(G)':<10} {'T(s)':<10} {'OOM':<6}")
    print("-"*65)
    for n_act in [8, 12, 16]:
        for label in ["Local", "BadSplit", "Random"]:
            r = scaling_results[f"n{n_act}_{label}"]
            print(f"{n_act:<8} {label:<12} {r['to_joint']:<10.4f} {r['Ar_max']:<6} {r['Pr_max']:<6} {r['peak_gb']:<10.2f} {r['time_s']:<10.0f} {'Y' if r['status']=='OOM' else 'N':<6}")

    with open("repro/results/pam_scaling.json", "w") as f:
        json.dump(scaling_results, f, indent=2)
    print("\nSaved: repro/results/pam_scaling.json")
