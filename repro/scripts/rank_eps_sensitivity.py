"""Rank/epsilon sensitivity under Local vs BadSplit ordering.

This is not parameter-augmented robust TTPI. It only varies TT-Cross rank
budget and approximation tolerance.
"""
import torch, gc, time, json, sys, itertools
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ttpi import TTPI
from dynamic_systems import HardMove

torch.set_default_dtype(torch.float64)
device = 'cuda'

def build_orders(n_act, label):
    D_a = 2 * n_act
    if label == "Local": return list(range(D_a))
    if label == "BadSplit": return list(range(0, D_a, 2)) + list(range(1, D_a, 2))
    return None

def run_one(n_act, a_order, seed, n_iter, rmax_val, eps_val, label):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()

    NS, NA = 40, 50
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
                gamma=0.99, rmax_v=rmax_val, rmax_a=rmax_val,
                nswp_v=5, nswp_a=10, kickrank_v=10, kickrank_a=10,
                max_batch_v=5000, max_batch_a=20000,
                eps_cross_v=eps_val, eps_cross_a=eps_val,
                eps_round_v=1e-3, eps_round_a=1e-3,
                n_samples=50, verbose=False, device=device)
    ttpi.plt_training_stat = lambda: None; ttpi.save_model = lambda *a,**kw: None
    _orig = ttpi.PI_update
    def _new(n_iter_v=1): _orig(n_iter_v=n_iter_v); torch.cuda.empty_cache(); gc.collect()
    ttpi.PI_update = _new

    g = torch.Generator(device='cpu'); g.manual_seed(12345)
    n_test = 30
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
            a_tt=ttpi.policy(s); a=a_tt[..., a_inv.to(a_tt.device)]
            ns=dyn.forward_simulate(s,a); np=ns[:,:2]
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
        torch.cuda.empty_cache(); gc.collect()
        return torch.tensor(0.),torch.tensor(0.)

    t0=time.time(); oom=False
    try:
        ttpi.train(resume=False,n_iter_max=n_iter,n_iter_v=1,callback=cb,
                   callback_freq=10,verbose=False,file_name=None)
    except torch.OutOfMemoryError:
        oom=True
    except Exception as e:
        pass

    t=time.time()-t0
    pr=ttpi.policy_model.tt().ranks_tt.tolist(); vr=ttpi.v_model.ranks_tt.tolist()
    ar=ttpi.a_model.ranks_tt.tolist()
    peak=torch.cuda.max_memory_allocated()/1e9
    status="OOM" if oom else ("OK" if last_cb>=n_iter-1 else f"cb{last_cb}")

    return {"label":label,"seed":seed,"rmax":rmax_val,"eps":eps_val,"status":status,
            "S_joint":bj['S'],"mu_joint":bj['mu'],"to_joint":bj['to'],
            "Vr_max":max(vr),"Ar_max":max(ar),"Pr_max":max(pr),
            "peak_gb":peak,"time_s":t,
            "eval_history":eval_hist,"rank_history":rank_hist}


if __name__ == "__main__":
    N_ACT = 8; N_ITER = 20
    # Parameter grid
    rmax_vals = [30, 60, 100]
    eps_vals = [1e-3, 5e-3]
    orders = ["Local", "BadSplit"]

    all_results = {}
    for rmax_val, eps_val, order in itertools.product(rmax_vals, eps_vals, orders):
        a_ord = build_orders(N_ACT, order)
        print(f"\n=== {order} rmax={rmax_val} eps={eps_val} ===", flush=True)
        r = run_one(N_ACT, a_ord, seed=0, n_iter=N_ITER,
                    rmax_val=rmax_val, eps_val=eps_val, label=order)
        key = f"{order}_r{rmax_val}_e{eps_val}"
        all_results[key] = r
        print(f"  {key}: {r['status']} S×μ={r['to_joint']:.4f} Ar={r['Ar_max']} Pr={r['Pr_max']} peak={r['peak_gb']:.2f}G T={r['time_s']:.0f}s", flush=True)
        torch.cuda.empty_cache(); gc.collect()

    # Summary
    print(f"\n{'='*70}")
    print("RANK/EPS SENSITIVITY: Local vs BadSplit")
    print(f"{'='*70}")
    print(f"{'Order':<10} {'rmax':<6} {'eps':<8} {'S×μ':<10} {'Ar':<6} {'Pr':<6} {'Peak':<10} {'Time':<10} {'OOM':<6}")
    print("-"*72)
    for rmax_val in rmax_vals:
        for eps_val in eps_vals:
            for order in orders:
                r = all_results[f"{order}_r{rmax_val}_e{eps_val}"]
                print(f"{order:<10} {rmax_val:<6} {eps_val:<8.0e} {r['to_joint']:<10.4f} "
                      f"{r['Ar_max']:<6} {r['Pr_max']:<6} {r['peak_gb']:<10.2f} "
                      f"{r['time_s']:<10.0f} {'Y' if r['status']=='OOM' else 'N':<6}")

    out_path = ROOT / "repro" / "archive" / "results" / "exploratory" / "rank_eps_sensitivity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")
