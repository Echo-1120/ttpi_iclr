"""Auto Physics-Aware Mode Ordering for TTPI.

Builds a coupling graph from sampled trajectories, generates an ordering via
greedy nearest-neighbor or spectral ordering, and runs TTPI.
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


def to_jsonable(x):
    """Convert numpy/torch scalar containers to plain JSON types."""
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.item()
        return to_jsonable(x.detach().cpu().tolist())
    return x

# ============================================================
# Step 1: Coupling estimation from trajectory samples
# ============================================================

def sample_trajectories(n_act, n_episodes=50, n_steps=200):
    """Collect state-action pairs from random exploration."""
    dyn = HardMove(dt=0.01, w_goal=1e3, w_action=1e4, n=n_act, device='cpu')
    d_action = 2 * n_act

    all_actions = []
    for ep in range(n_episodes):
        # Random initial state
        s = torch.rand(4) * 2 - 1  # [-1, 1]
        s[2:] = 0  # zero velocity
        for t in range(n_steps):
            # Random action
            a = torch.zeros(d_action)
            for i in range(n_act):
                a[2*i] = (torch.rand(1).item() - 0.5) * 0.5  # acc in [-0.25, 0.25]
                a[2*i+1] = 1.0 if torch.rand(1).item() > 0.5 else 0.0  # switch
            all_actions.append(a.numpy())
            s = dyn.forward_simulate(s.unsqueeze(0), a.unsqueeze(0)).squeeze(0)
    return np.stack(all_actions)  # [N, d_action]


def build_coupling_matrix(actions, n_act):
    """Build coupling weight matrix from pairwise absolute correlation."""
    d_action = 2 * n_act
    W = np.zeros((d_action, d_action))

    # Method 1: Correlation-based
    corr = np.abs(np.corrcoef(actions.T))
    # Method 2: Mutual information (simplified: use correlation as proxy)
    W = corr

    # Inject known physics: same-actuator (acc, sw) pairs get boosted weight
    for i in range(n_act):
        a_i, s_i = 2*i, 2*i+1
        W[a_i, s_i] = W[s_i, a_i] = max(W[a_i, s_i], 0.8)

    np.fill_diagonal(W, 0)
    return W


def generate_ordering(W, method="greedy"):
    """Generate 1D ordering from coupling matrix."""
    D = W.shape[0]
    if method == "greedy":
        # Greedy nearest-neighbor
        unplaced = set(range(D))
        start = int(np.argmax(W.sum(axis=1)))
        order = [start]; unplaced.remove(start)
        while unplaced:
            last = order[-1]
            best = max(unplaced, key=lambda x: W[last, x])
            order.append(best); unplaced.remove(best)

    elif method == "spectral":
        # Spectral ordering using Fiedler vector
        D_mat = np.diag(W.sum(axis=1))
        L = D_mat - W
        eigvals, eigvecs = np.linalg.eigh(L)
        fiedler = eigvecs[:, 1]  # second smallest eigenvector
        order = list(np.argsort(fiedler))

    elif method == "rcm":
        # Reverse Cuthill-McKee (simplified BFS)
        from collections import deque
        D = W.shape[0]
        start = int(np.argmax(W.sum(axis=1)))
        visited = set(); queue = deque([start]); order = []
        while queue:
            v = queue.popleft()
            if v in visited: continue
            visited.add(v); order.append(v)
            neighbors = sorted(range(D), key=lambda x: -W[v, x])
            for u in neighbors:
                if u not in visited and W[v, u] > 0.01:
                    queue.append(u)
        # Add unvisited
        for v in range(D):
            if v not in visited:
                order.append(v)
        order.reverse()  # RCM reverses

    return [int(i) for i in order]


def weighted_arrangement_cost(order, W):
    """Lower is better: sum_ij w_ij * |pos(i)-pos(j)|."""
    pos = {mode: idx for idx, mode in enumerate(order)}
    cost = 0.0
    for i in range(W.shape[0]):
        for j in range(i + 1, W.shape[1]):
            cost += W[i, j] * abs(pos[i] - pos[j])
    return float(cost)


def adjacency_score(order, W):
    """Higher means stronger couplings are adjacent in this chain."""
    return float(sum(W[order[i], order[i + 1]] for i in range(len(order) - 1)))


# ============================================================
# Step 2: Run TTPI with generated ordering
# ============================================================

def run_ttpi_with_order(n_act, a_order, seed=0, n_iter=30, label="Auto"):
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

    t0=time.time(); oom=False; error_msg=None
    try:
        ttpi.train(resume=False,n_iter_max=n_iter,n_iter_v=1,callback=cb,
                   callback_freq=10,verbose=False,file_name=None)
    except torch.OutOfMemoryError:
        oom=True; print('OOM!',flush=True)
    except Exception as e:
        error_msg=repr(e)
        print(f'Error:{error_msg}',flush=True)

    t=time.time()-t0
    pr=ttpi.policy_model.tt().ranks_tt.tolist(); vr=ttpi.v_model.ranks_tt.tolist()
    ar=ttpi.a_model.ranks_tt.tolist()
    peak=torch.cuda.max_memory_allocated()/1e9
    status="OOM" if oom else ("ERROR" if error_msg else ("OK" if last_cb>=n_iter-1 else f"cb{last_cb}"))

    return {"label":label,"seed":seed,"status":status,"a_order":a_order,
            "S_joint":bj['S'],"mu_joint":bj['mu'],"to_joint":bj['to'],
            "Vr_max":max(vr),"Ar_max":max(ar),"Pr_max":max(pr),
            "peak_gb":peak,"time_s":t,
            "error":error_msg,
            "eval_history":eval_hist,"rank_history":rank_hist}


if __name__ == "__main__":
    N_ACT = 8
    print(f"Auto physics-aware ordering for HM{N_ACT}")
    print("="*50)

    # Step 1: Sample trajectories and build coupling
    print("Sampling trajectories...")
    actions = sample_trajectories(N_ACT, n_episodes=30, n_steps=200)
    W = build_coupling_matrix(actions, N_ACT)
    print(f"Coupling matrix shape: {W.shape}")

    # Step 2: Generate orderings via different methods
    results = {}
    ordering_metrics = {}
    for method in ["greedy", "spectral", "rcm"]:
        order = generate_ordering(W, method=method)
        ordering_metrics[method] = {
            "weighted_arrangement_cost": weighted_arrangement_cost(order, W),
            "adjacency_score": adjacency_score(order, W),
        }
        # Check if ordering differs from Local (identity)
        is_identity = (order == list(range(2*N_ACT)))
        is_badsplit = (order == list(range(0,2*N_ACT,2)) + list(range(1,2*N_ACT,2)))
        print(f"\n  {method}: {order}")
        print(f"    identity={is_identity} badsplit={is_badsplit}")

        print(f"  Running TTPI with {method} ordering...")
        r = run_ttpi_with_order(N_ACT, order, seed=0, n_iter=30, label=f"Auto_{method}")
        results[method] = r
        print(f"  === Auto_{method}: {r['status']} S×μ={r['to_joint']:.4f} Ar={r['Ar_max']} Pr={r['Pr_max']} peak={r['peak_gb']:.2f}G T={r['time_s']:.0f}s ===", flush=True)
        torch.cuda.empty_cache(); gc.collect()

    # Compare with known baselines
    print(f"\n{'='*60}")
    print("COMPARISON: Auto-generated vs Manual Orderings")
    print(f"{'='*60}")
    print(f"{'Method':<14} {'S×μ':<10} {'Ar':<6} {'Pr':<6} {'Peak':<10} {'Time':<10}")
    print("-"*56)
    for method, r in results.items():
        print(f"Auto_{method:<8} {r['to_joint']:<10.4f} {r['Ar_max']:<6} {r['Pr_max']:<6} {r['peak_gb']:<10.2f} {r['time_s']:<10.0f}")
    print("(Compare: Local ~0.62, BadSplit ~0.60, Random ~0.54)")

    out = {
        "n_act": N_ACT,
        "n_iter": 30,
        "methods": ["greedy", "spectral", "rcm"],
        "ordering_metrics": ordering_metrics,
        "results": results,
        "note": "Auto ordering is exploratory; random-action correlation plus manual pair boosts is not final evidence.",
    }
    out_path = ROOT / "repro" / "archive" / "results" / "exploratory" / "auto_ordering.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(to_jsonable(out), f, indent=2)
    print(f"\nSaved: {out_path}")
