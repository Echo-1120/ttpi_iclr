"""MLA (Minimum Linear Arrangement) objective validation for PAM-TTPI.

Compares Local, BadSplit, Random, Auto(spectral) on HM8.
Reports MLA cost alongside empirical rank and performance.
"""
import torch, numpy as np, time, gc, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ttpi import TTPI
from dynamic_systems import HardMove

torch.set_default_dtype(torch.float64)
device = 'cuda'

# ============================================================
# MLA objective
# ============================================================
def compute_mla_cost(order, W):
    """sum_{i,j} w_{ij} * |pi(i) - pi(j)|"""
    D = len(order)
    pos = {order[i]: i for i in range(D)}
    cost = 0.0
    for i in range(D):
        for j in range(i+1, D):
            if W[i, j] > 0:
                cost += W[i, j] * abs(pos[i] - pos[j])
    return cost

def build_coupling_matrix(n_act):
    """Physics-based coupling: same actuator=2.0, opposing=1.5, adjacent=0.5"""
    D = 2 * n_act; half = n_act // 2
    W = np.zeros((D, D))
    for i in range(n_act):
        a_i, s_i = 2*i, 2*i+1
        W[a_i, s_i] = W[s_i, a_i] = 2.0
        opp = (i + half) % n_act
        a_opp = 2*opp
        W[a_i, a_opp] = W[a_opp, a_i] = 1.5
        adj = (i + 1) % n_act
        a_adj = 2*adj
        W[a_i, a_adj] = W[a_adj, a_i] = 0.5
    return W

def auto_spectral_order(W):
    """Spectral ordering via Fiedler vector."""
    D = W.shape[0]
    D_mat = np.diag(W.sum(axis=1))
    L = D_mat - W
    eigvals, eigvecs = np.linalg.eigh(L)
    fiedler = eigvecs[:, 1]
    return list(np.argsort(fiedler))

# ============================================================
# TTPI runner (same as PAM v4)
# ============================================================
def run_ttpi(n_act, a_order, seed, n_iter, label):
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
        al=torch.cuda.memory_allocated()/1e9; pk=torch.cuda.max_memory_allocated()/1e9
        print(f'  cb{callback_count:>3}: S={S_:.3f} mu={mv:.4f} Vr={max(vr)} Ar={max(ar)} Pr={max(pr)} p={pk:.2f}G',flush=True)
        torch.cuda.empty_cache(); gc.collect()
        return torch.tensor(0.),torch.tensor(0.)

    t0=time.time(); oom=False
    try:
        ttpi.train(resume=False,n_iter_max=n_iter,n_iter_v=1,callback=cb,
                   callback_freq=10,verbose=False,file_name=None)
    except torch.OutOfMemoryError:
        oom=True
    except Exception:
        pass

    t=time.time()-t0
    pr=ttpi.policy_model.tt().ranks_tt.tolist(); vr=ttpi.v_model.ranks_tt.tolist()
    ar=ttpi.a_model.ranks_tt.tolist()
    peak=torch.cuda.max_memory_allocated()/1e9
    status="OOM" if oom else ("OK" if last_cb>=n_iter-1 else f"cb{last_cb}")

    return {"label":label,"seed":seed,"status":status,
            "S_joint":bj['S'],"mu_joint":bj['mu'],"to_joint":bj['to'],
            "Ar_max":max(ar),"Pr_max":max(pr),
            "peak_gb":peak,"time_s":t,
            "eval_history":eval_hist,"rank_history":rank_hist}


if __name__ == "__main__":
    N_ACT = 8; N_ITER = 30; SEEDS = [0, 1, 2]

    # Build coupling matrix
    W = build_coupling_matrix(N_ACT)

    # Define orderings
    D_a = 2 * N_ACT
    half = N_ACT // 2
    orderings = {
        "Local": list(range(D_a)),
        "BadSplit": list(range(0, D_a, 2)) + list(range(1, D_a, 2)),
        "Random": [int(x) for x in torch.randperm(D_a, generator=torch.Generator().manual_seed(42)).tolist()],
        "Auto": auto_spectral_order(W),
    }

    # Compute MLA costs
    print("MLA Objective (lower = better coupling preservation):")
    print(f"{'Order':<12} {'MLA cost':<12} {'Order preview'}")
    print("-"*60)
    for label, order in orderings.items():
        cost = compute_mla_cost(order, W)
        preview = str(order[:8]) + "..."
        print(f"{label:<12} {cost:<12.1f} {preview}")
    print()

    # Run TTPI for each ordering × seed
    all_results = {}
    for label, a_order in orderings.items():
        print(f"\n{'='*50}\n  {label}: MLA={compute_mla_cost(a_order, W):.1f}\n{'='*50}")
        for seed in SEEDS:
            r = run_ttpi(N_ACT, a_order, seed=seed, n_iter=N_ITER, label=label)
            key = f"{label}_s{seed}"
            all_results[key] = r
            print(f"  === {label} s{seed}: {r['status']} "
                  f"S×μ={r['to_joint']:.4f} Ar={r['Ar_max']} Pr={r['Pr_max']} "
                  f"peak={r['peak_gb']:.2f}G T={r['time_s']:.0f}s ===", flush=True)
        torch.cuda.empty_cache(); gc.collect()

    # Summary with MLA
    print(f"\n{'='*75}")
    print("MLA + EMPIRICAL RESULTS")
    print(f"{'='*75}")
    print(f"{'Order':<12} {'MLA':<10} {'S×μ':<12} {'Ar':<8} {'Pr':<8} {'Peak(G)':<10} {'T(s)':<10} {'OOM':<6}")
    print("-"*75)

    import statistics
    for label in ["Local", "BadSplit", "Random", "Auto"]:
        keys = [f"{label}_s{s}" for s in SEEDS]
        rs = [all_results[k] for k in keys]
        mla = compute_mla_cost(orderings[label], W)
        to_mean = statistics.mean([r["to_joint"] for r in rs])
        to_std = statistics.stdev([r["to_joint"] for r in rs]) if len(rs)>1 else 0
        ar_mean = statistics.mean([r["Ar_max"] for r in rs])
        pr_mean = statistics.mean([r["Pr_max"] for r in rs])
        pk_mean = statistics.mean([r["peak_gb"] for r in rs])
        t_mean = statistics.mean([r["time_s"] for r in rs])
        ooms = sum(1 for r in rs if r["status"]=="OOM")
        print(f"{label:<12} {mla:<10.1f} {to_mean:.3f}±{to_std:.3f}   "
              f"{ar_mean:<8.1f} {pr_mean:<8.1f} {pk_mean:<10.2f} {t_mean:<10.0f} {ooms}/{len(SEEDS)}")

    # Save (convert numpy types)
    out = {"n_act": N_ACT, "n_iter": N_ITER, "n_seeds": len(SEEDS),
           "mla_costs": {k: float(compute_mla_cost(v, W)) for k, v in orderings.items()},
           "orderings": {k: [int(x) for x in v] for k, v in orderings.items()},
           "results": {}}
    for k, r in all_results.items():
        out["results"][k] = {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else
                                  [float(x) for x in vv] if isinstance(vv, (np.ndarray, list)) and len(vv)>0 and isinstance(vv[0], (np.floating, np.integer)) else vv)
                             for kk, vv in r.items() if kk not in ("eval_history","rank_history")}
    out_path = ROOT / "repro" / "archive" / "results" / "exploratory" / "mla_experiment.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    print(f"\nSaved: {out_path}")
