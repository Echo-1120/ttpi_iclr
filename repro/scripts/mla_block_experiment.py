"""Constrained MLA: block-level ordering. Blocks = [acc_i, sw_i], fixed internally.
Only permute blocks. Compares: Local, BadSplit, Random, Auto(block-spectral/greedy).
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
# Block-level MLA: blocks are fixed, only block order varies
# ============================================================
def blocks_to_order(block_order):
    """Expand block permutation to full action order."""
    a_order = []
    for b in block_order:
        a_order.extend([2*b, 2*b+1])  # [acc_b, sw_b]
    return a_order

def compute_mla_block(block_order, W_block):
    """MLA cost at block level."""
    n = len(block_order)
    pos = {block_order[i]: i for i in range(n)}
    cost = 0.0
    for i in range(n):
        for j in range(i+1, n):
            if W_block[i,j] > 0:
                cost += W_block[i,j] * abs(pos[i] - pos[j])
    return cost

def build_block_coupling(n_act):
    """Block-level coupling: opposing=2.0, adjacent=1.0."""
    half = n_act // 2
    W = np.zeros((n_act, n_act))
    for i in range(n_act):
        opp = (i + half) % n_act
        W[i, opp] = W[opp, i] = 2.0
        adj = (i + 1) % n_act
        W[i, adj] = W[adj, i] = 1.0
    return W

def spectral_blocks(W):
    """Spectral ordering at block level."""
    D_mat = np.diag(W.sum(axis=1))
    L = D_mat - W
    _, eigvecs = np.linalg.eigh(L)
    fiedler = eigvecs[:, 1]
    return list(np.argsort(fiedler))

def greedy_blocks(W):
    """Greedy at block level."""
    n = W.shape[0]
    unplaced = set(range(n))
    start = int(np.argmax(W.sum(axis=1)))
    order = [start]; unplaced.remove(start)
    while unplaced:
        last = order[-1]
        best = max(unplaced, key=lambda x: W[last, x])
        order.append(best); unplaced.remove(best)
    return order

def format_mla(value):
    return "N/A" if value is None else f"{value:.1f}"

def reported_block_mla(label, block_orderings, W_block):
    """Block MLA is defined only when actuator blocks stay intact."""
    if label in ("BadSplit", "Random"):
        return None
    if label == "AutoSpec":
        return compute_mla_block(block_orderings["AutoSpectral"], W_block)
    if label == "AutoGreedy":
        return compute_mla_block(block_orderings["AutoGreedy"], W_block)
    return compute_mla_block(block_orderings[label], W_block)

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
            "Ar_max":int(max(ar)),"Pr_max":int(max(pr)),
            "peak_gb":float(peak),"time_s":float(t),
            "eval_history":eval_hist,"rank_history":rank_hist}


if __name__ == "__main__":
    N_ACT = 8; N_ITER = 30; SEEDS = [0, 1, 2]
    W_block = build_block_coupling(N_ACT)

    # Build orderings at BLOCK level
    half = N_ACT // 2
    manual_blocks = []
    for i in range(half):
        manual_blocks.extend([i, (i+half) % N_ACT])
    # manual_blocks: [0,4,1,5,2,6,3,7] (opposing pairs adjacent)

    block_orderings = {
        "Local": list(range(N_ACT)),                    # [0,1,2,3,4,5,6,7]
        "AutoSpectral": spectral_blocks(W_block),
        "AutoGreedy": greedy_blocks(W_block),
    }

    # For BadSplit: keep blocks [acc_i,sw_i] but expand with ALL acc first, ALL sw after
    # This means we DON'T use blocks for BadSplit — it's the destructive ordering
    badsplit_aorder = list(range(0, 2*N_ACT, 2)) + list(range(1, 2*N_ACT, 2))
    random_aorder = [int(x) for x in torch.randperm(2*N_ACT, generator=torch.Generator().manual_seed(42)).tolist()]

    order_configs = {
        "Local": blocks_to_order(block_orderings["Local"]),
        "BadSplit": badsplit_aorder,
        "Random": random_aorder,
        "AutoSpec": blocks_to_order(block_orderings["AutoSpectral"]),
        "AutoGreedy": blocks_to_order(block_orderings["AutoGreedy"]),
    }

    # MLA costs
    print("Block-Level MLA Costs:")
    print(f"{'Order':<14} {'Block order':<30} {'MLA':<10}")
    print("-"*58)
    for label, blocks in block_orderings.items():
        cost = compute_mla_block(blocks, W_block)
        print(f"{label:<14} {str(blocks):<30} {cost:<10.1f}")
    print("BadSplit/Random are not block-preserving, so block MLA is reported as N/A.")

    # TTPI experiments
    all_results = {}
    for label, a_order in order_configs.items():
        print(f"\n{'='*50}\n  {label}\n{'='*50}")
        for seed in SEEDS:
            r = run_ttpi(N_ACT, a_order, seed=seed, n_iter=N_ITER, label=label)
            key = f"{label}_s{seed}"
            all_results[key] = r
            print(f"  === {label} s{seed}: {r['status']} "
                  f"S×μ={r['to_joint']:.4f} Ar={r['Ar_max']} Pr={r['Pr_max']} "
                  f"peak={r['peak_gb']:.2f}G T={r['time_s']:.0f}s ===", flush=True)
        torch.cuda.empty_cache(); gc.collect()

    # Summary
    print(f"\n{'='*80}")
    print("BLOCK-CONSTRAINED MLA + EMPIRICAL RESULTS")
    print(f"{'='*80}")
    print(f"{'Order':<14} {'MLA':<8} {'S×μ':<14} {'Ar':<8} {'Pr':<8} {'Peak':<10} {'T(s)':<10} {'OOM':<6}")
    print("-"*78)

    import statistics
    mla_costs = {}
    for label in ["Local", "BadSplit", "Random", "AutoSpec", "AutoGreedy"]:
        keys = [f"{label}_s{s}" for s in SEEDS]
        rs = [all_results[k] for k in keys]

        mla = reported_block_mla(label, block_orderings, W_block)
        mla_costs[label] = None if mla is None else float(mla)

        to_mean = statistics.mean([r["to_joint"] for r in rs])
        to_std = statistics.stdev([r["to_joint"] for r in rs]) if len(rs)>1 else 0
        ar = statistics.mean([r["Ar_max"] for r in rs])
        pr = statistics.mean([r["Pr_max"] for r in rs])
        pk = statistics.mean([r["peak_gb"] for r in rs])
        tm = statistics.mean([r["time_s"] for r in rs])
        ooms = sum(1 for r in rs if r["status"]=="OOM")
        print(f"{label:<14} {format_mla(mla):<8} {to_mean:.3f}±{to_std:.3f}     "
              f"{ar:<8.1f} {pr:<8.1f} {pk:<10.2f} {tm:<10.0f} {ooms}/{len(SEEDS)}")

    # Save
    out = {"n_act": N_ACT, "n_iter": N_ITER, "n_seeds": len(SEEDS),
           "mla_costs": mla_costs,
           "block_orderings": {k: [int(x) for x in v] for k, v in block_orderings.items()},
           "orderings": {k: [int(x) for x in v] for k, v in order_configs.items()},
           "results": {}}
    for k, r in all_results.items():
        out["results"][k] = {kk: vv for kk, vv in r.items() if kk not in ("eval_history","rank_history")}
    out_path = ROOT / "repro" / "archive" / "results" / "exploratory" / "mla_block_experiment.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")
