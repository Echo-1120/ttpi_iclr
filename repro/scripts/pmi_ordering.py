"""PMI (Path Mutual Information) ordering for TTPI.

Upgrades from MLA to PMI: measures adjacent mutual information along the TT chain.
PMI(pi) = sum_{k=1}^{D-1} I(z_{pi_k}; z_{pi_{k+1}})

Compares: Local, BadSplit, MLA-greedy, PMI-greedy on HM8.
"""
import torch, numpy as np, time, gc, json, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ttpi import TTPI
from dynamic_systems import HardMove

torch.set_default_dtype(torch.float64)
device = 'cuda'

# ============================================================
# PMI computation from trajectory samples
# ============================================================
def sample_trajectories(n_act, n_episodes=50, n_steps=200):
    """Collect state-action pairs from random exploration."""
    dyn = HardMove(dt=0.01, w_goal=1e3, w_action=1e4, n=n_act, device='cpu')
    d_action = 2 * n_act
    all_actions = []
    for _ in range(n_episodes):
        s = torch.rand(4) * 2 - 1
        s[2:] = 0
        for _ in range(n_steps):
            a = torch.zeros(d_action)
            for i in range(n_act):
                a[2*i] = (torch.rand(1).item() - 0.5) * 0.5
                a[2*i+1] = 1.0 if torch.rand(1).item() > 0.5 else 0.0
            all_actions.append(a.numpy())
            s = dyn.forward_simulate(s.unsqueeze(0), a.unsqueeze(0)).squeeze(0)
    return np.stack(all_actions)


def discretize(actions, n_bins=20):
    """StD: Statistics-then-Discretization. Discretize each dimension into bins."""
    D = actions.shape[1]
    disc = np.zeros_like(actions, dtype=np.int32)
    for d in range(D):
        vals = actions[:, d]
        if len(np.unique(vals)) <= 2:
            disc[:, d] = (vals > 0.5).astype(np.int32)
        else:
            bins = np.percentile(vals, np.linspace(0, 100, n_bins + 1))
            bins[0], bins[-1] = -np.inf, np.inf
            disc[:, d] = np.digitize(vals, bins) - 1
    return disc


def mutual_info(x, y):
    """Mutual information between two discrete variables."""
    n = len(x)
    # Joint histogram
    joint = defaultdict(int)
    for i in range(n):
        joint[(int(x[i]), int(y[i]))] += 1

    # Marginal
    mx = defaultdict(int); my = defaultdict(int)
    for (a, b), c in joint.items():
        mx[a] += c; my[b] += c

    # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
    mi = 0.0
    for (a, b), c_xy in joint.items():
        p_xy = c_xy / n
        p_x = mx[a] / n
        p_y = my[b] / n
        mi += p_xy * np.log(p_xy / (p_x * p_y))
    return max(mi, 0.0)


def build_mi_matrix(actions_disc):
    """Build pairwise mutual information matrix."""
    D = actions_disc.shape[1]
    MI = np.zeros((D, D))
    for i in range(D):
        for j in range(i+1, D):
            mi = mutual_info(actions_disc[:, i], actions_disc[:, j])
            MI[i, j] = MI[j, i] = mi
    return MI


def greedy_pmi_chain(MI):
    """Greedy chain construction maximizing PMI = sum I(adjacent)."""
    D = MI.shape[0]
    unplaced = set(range(D))
    # Start from pair with highest MI
    best_pair = max(((i, j) for i in range(D) for j in range(i+1, D)),
                    key=lambda p: MI[p[0], p[1]])
    chain = [best_pair[0], best_pair[1]]
    unplaced -= {best_pair[0], best_pair[1]}

    while unplaced:
        # Try prepending or appending each remaining node, pick max PMI gain
        best_gain = -1; best_node = -1; best_pos = 'append'
        for v in unplaced:
            gain_prepend = MI[v, chain[0]]
            gain_append = MI[v, chain[-1]]
            if gain_prepend > best_gain:
                best_gain = gain_prepend; best_node = v; best_pos = 'prepend'
            if gain_append > best_gain:
                best_gain = gain_append; best_node = v; best_pos = 'append'
        if best_pos == 'prepend':
            chain.insert(0, best_node)
        else:
            chain.append(best_node)
        unplaced.remove(best_node)
    return chain


# ============================================================
# Block-constrained PMI (within actuator blocks fixed)
# ============================================================
def pmi_block_ordering(MI, n_act):
    """PMI on blocks: each block = [acc_i, sw_i], blocks ordered by PMI."""
    # Build block-level MI by averaging within-block pairwise MI
    MI_block = np.zeros((n_act, n_act))
    for i in range(n_act):
        for j in range(i+1, n_act):
            # Average MI between all pairs of dimensions in blocks i and j
            dims_i = [2*i, 2*i+1]
            dims_j = [2*j, 2*j+1]
            mi_sum = 0.0
            for di in dims_i:
                for dj in dims_j:
                    mi_sum += MI[di, dj]
            MI_block[i, j] = MI_block[j, i] = mi_sum / 4.0

    # Greedy chain on blocks
    block_chain = greedy_pmi_chain(MI_block)

    # Expand to full ordering (keep (acc_i, sw_i) adjacent internally)
    a_order = []
    for b in block_chain:
        a_order.extend([2*b, 2*b+1])
    return a_order, block_chain


def compute_pmi_score(order, MI):
    """Compute PMI(pi) = sum_{k} I(z_{pi_k}; z_{pi_{k+1}})."""
    score = 0.0
    for k in range(len(order) - 1):
        score += MI[order[k], order[k+1]]
    return score


# ============================================================
# TTPI runner
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
        print(f'  cb{callback_count:>3}: S={S_:.3f} mu={mv:.4f} Ar={max(ar)} Pr={max(pr)}',flush=True)
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
            "Ar_max":int(max(ar)),"Pr_max":int(max(pr)),
            "peak_gb":float(peak),"time_s":float(t),
            "eval_history":eval_hist,"rank_history":rank_hist}


if __name__ == "__main__":
    N_ACT = 8; N_ITER = 30; SEEDS = [0, 1, 2]

    # Build MI matrix from trajectory samples
    print("Sampling trajectories for PMI estimation...")
    actions = sample_trajectories(N_ACT, n_episodes=50, n_steps=200)
    actions_disc = discretize(actions, n_bins=20)
    MI = build_mi_matrix(actions_disc)
    print(f"MI matrix shape: {MI.shape}")

    # Define orderings
    D_a = 2 * N_ACT
    half = N_ACT // 2

    # Block-level PMI
    pmi_aorder, pmi_blocks = pmi_block_ordering(MI, N_ACT)

    # Block-level MLA (spectral)
    W = np.zeros((N_ACT, N_ACT))
    for i in range(N_ACT):
        opp = (i + half) % N_ACT
        W[i, opp] = W[opp, i] = 2.0
        W[i, (i+1)%N_ACT] = W[(i+1)%N_ACT, i] = 1.0
    D_mat = np.diag(W.sum(axis=1)); L = D_mat - W
    _, eigvecs = np.linalg.eigh(L)
    mla_blocks = list(np.argsort(eigvecs[:, 1]))
    mla_aorder = []
    for b in mla_blocks:
        mla_aorder.extend([2*b, 2*b+1])

    badsplit = list(range(0, D_a, 2)) + list(range(1, D_a, 2))
    random_order = [int(x) for x in torch.randperm(D_a, generator=torch.Generator().manual_seed(42)).tolist()]

    order_configs = {
        "Local": list(range(D_a)),
        "BadSplit": badsplit,
        "Random": random_order,
        "MLA_Block": mla_aorder,
        "PMI_Block": pmi_aorder,
    }

    # Compute PMI scores. This is an exploratory adjacency score, not a
    # standalone predictor of TTPI trainability.
    print("\nPMI adjacency scores (exploratory; interpret with rank/OOM):")
    local_pmi = compute_pmi_score(order_configs["Local"], MI)
    for label, order in order_configs.items():
        pmi = compute_pmi_score(order, MI)
        print(f"  {label:<12} PMI={pmi:.4f}  (vs Local Δ={pmi-local_pmi:+.4f})")
    print()

    # Run TTPI
    all_results = {}
    for label, a_order in order_configs.items():
        print(f"\n{'='*50}\n  {label}  PMI={compute_pmi_score(a_order, MI):.4f}\n{'='*50}")
        for seed in SEEDS:
            r = run_ttpi(N_ACT, a_order, seed=seed, n_iter=N_ITER, label=label)
            key = f"{label}_s{seed}"
            all_results[key] = r
            print(f"  === {label} s{seed}: {r['status']} "
                  f"S×μ={r['to_joint']:.4f} Ar={r['Ar_max']} Pr={r['Pr_max']} "
                  f"peak={r['peak_gb']:.2f}G T={r['time_s']:.0f}s ===", flush=True)
        torch.cuda.empty_cache(); gc.collect()

    # Summary
    print(f"\n{'='*75}")
    print("PMI vs MLA vs BASELINE COMPARISON")
    print(f"{'='*75}")
    print(f"{'Order':<12} {'PMI':<10} {'S×μ':<14} {'Ar':<8} {'Pr':<8} {'Peak':<10} {'T(s)':<10} {'OOM':<6}")
    print("-"*78)

    import statistics
    for label in ["Local", "BadSplit", "Random", "MLA_Block", "PMI_Block"]:
        keys = [f"{label}_s{s}" for s in SEEDS]
        rs = [all_results[k] for k in keys]
        pmi = compute_pmi_score(order_configs[label], MI)
        ooms = sum(1 for r in rs if r["status"]=="OOM")
        complete = [r for r in rs if r["status"] == "OK"]
        stat_rs = complete if complete else rs
        to_mean = statistics.mean([r["to_joint"] for r in stat_rs])
        to_std = statistics.stdev([r["to_joint"] for r in stat_rs]) if len(stat_rs)>1 else 0
        ar = statistics.mean([r["Ar_max"] for r in stat_rs])
        pr = statistics.mean([r["Pr_max"] for r in stat_rs])
        pk = statistics.mean([r["peak_gb"] for r in stat_rs])
        tm = statistics.mean([r["time_s"] for r in stat_rs])
        stat_note = "OK-only" if complete and ooms else "all"
        print(f"{label:<12} {pmi:<10.4f} {to_mean:.3f}±{to_std:.3f}     "
              f"{ar:<8.1f} {pr:<8.1f} {pk:<10.2f} {tm:<10.0f} {ooms}/{len(SEEDS)} {stat_note}")

    # Save
    out = {"n_act": N_ACT, "n_iter": N_ITER, "n_seeds": len(SEEDS),
           "pmi_scores": {k: float(compute_pmi_score(v, MI)) for k, v in order_configs.items()},
           "pmi_note": "PMI is an exploratory adjacent-MI score. It must be interpreted with rank, memory, and OOM status.",
           "orderings": {k: [int(x) for x in v] for k, v in order_configs.items()},
           "summary": {}, "results": {}}
    for k, r in all_results.items():
        out["results"][k] = {kk: vv for kk, vv in r.items() if kk not in ("eval_history","rank_history")}
    for label in ["Local", "BadSplit", "Random", "MLA_Block", "PMI_Block"]:
        keys = [f"{label}_s{s}" for s in SEEDS]
        rs = [all_results[k] for k in keys]
        complete = [r for r in rs if r["status"] == "OK"]
        stat_rs = complete if complete else rs
        out["summary"][label] = {
            "pmi": float(compute_pmi_score(order_configs[label], MI)),
            "n_total": len(rs),
            "n_complete": len(complete),
            "n_oom": sum(1 for r in rs if r["status"] == "OOM"),
            "stats_scope": "OK-only" if complete and len(complete) < len(rs) else "all",
            "to_mean": float(statistics.mean([r["to_joint"] for r in stat_rs])),
            "Ar_mean": float(statistics.mean([r["Ar_max"] for r in stat_rs])),
            "Pr_mean": float(statistics.mean([r["Pr_max"] for r in stat_rs])),
            "peak_gb_mean": float(statistics.mean([r["peak_gb"] for r in stat_rs])),
        }
    out_path = ROOT / "repro" / "archive" / "results" / "exploratory" / "pmi_experiment.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")
