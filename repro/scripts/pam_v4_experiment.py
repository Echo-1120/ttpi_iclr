"""PAM-TTPI v4: Structure-aware ordering necessity proof.
4 orderings × 5 seeds, n_iter=30, n_test=50.
Saves full eval/rank history per seed for curve plotting.
"""
import torch, sys, time, gc, json, statistics
from pathlib import Path
ROOT = Path('/home/s110/code/ttpi_iclr')
sys.path.insert(0, str(ROOT))
from ttpi import TTPI
from dynamic_systems import HardMove

torch.set_default_dtype(torch.float64)
device = 'cuda'

def build_orderings(n_act):
    D_a = 2 * n_act; half = n_act // 2
    orders = {}

    # 1. Local: natural (acc_i, sw_i) interleaved
    orders["Local"] = [list(range(D_a))]  # single order, 5 seeds use same order

    # 2. BadSplit: all acc first, then all sw
    bad = list(range(0, D_a, 2)) + list(range(1, D_a, 2))
    orders["BadSplit"] = [bad]

    # 3. Random: k different random shuffles
    random_orders = []
    for k in range(5):
        g = torch.Generator().manual_seed(100 + k)
        random_orders.append(torch.randperm(D_a, generator=g).tolist())
    orders["Random"] = random_orders

    # 4. OppositePair: (acc_i,sw_i, acc_{i+half},sw_{i+half})
    opp = []
    for i in range(half):
        opp.extend([2*i, 2*i+1, 2*(i+half), 2*(i+half)+1])
    orders["OppositePair"] = [opp]

    return orders

def run_one(n_act, a_order, seed, n_iter, label):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()

    L=1.0; vmax=0.25; NS=40; NA=50
    domain_acc = torch.linspace(-vmax, vmax, NA, device=device)
    domain_sw = torch.tensor([0.,1.], device=device)
    smin = torch.tensor([-L,-L,0.,0.], device=device)
    smax = torch.tensor([L,L,vmax,vmax], device=device)

    state_domain = [torch.linspace(smin[i], smax[i], NS, device=device) for i in range(4)]
    action_domain_phys = [domain_acc, domain_sw] * n_act
    action_domain_tt = [action_domain_phys[i] for i in a_order]

    a_inv = torch.tensor([a_order.index(i) for i in range(2*n_act)])

    dyn = HardMove(dt=0.01, w_goal=1e3, w_action=1e4, n=n_act, device=device)

    def reward_tt(s, a):
        return dyn.reward_state_action(s, a[..., a_inv.to(a.device)])
    def forward_tt(s, a):
        ns = dyn.forward_simulate(s, a[..., a_inv.to(a.device)])
        return ns

    ttpi = TTPI(domain_state=state_domain, domain_action=action_domain_tt,
                reward=reward_tt, normalize_reward=True, forward_model=forward_tt,
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
    n_test = 50  # increased
    init_state = torch.empty((n_test,4), dtype=torch.float64)
    for i in range(4):
        r = torch.rand(n_test, generator=g, dtype=torch.float64).clip(0.25,0.75)
        init_state[:,i] = smin[i].cpu() + r*(smax[i].cpu()-smin[i].cpu())
    init_state[:,2:4]=0.0; init_state = init_state.to(device)

    tracker = {'best_S':0,'best_mu':0,'best_to':0}
    best_joint = {'S':0,'mu':0,'to':0,'cb':-1}
    eval_hist = []; rank_hist = []; last_cb = -1
    cross_warnings = 0

    def cb(ttpi, state=init_state, callback_count=0):
        nonlocal last_cb; last_cb = callback_count
        s = state.clone()
        Th=int(10/0.01); ac=torch.ones(n_test,dtype=torch.bool,device=device)
        su=torch.zeros(n_test,dtype=torch.bool,device=device)
        sp=s[:,:2].clone(); pp=s[:,:2].clone(); pl=torch.zeros(n_test,device=device)
        for _ in range(Th):
            a_tt = ttpi.policy(s)
            a = a_tt[..., a_inv.to(a_tt.device)]
            ns = dyn.forward_simulate(s, a); np = ns[:,:2]
            pl += ac.float()*torch.linalg.norm(np-pp,dim=-1)
            d=torch.linalg.norm(np,dim=-1); nsu=ac&(d<=0.02)
            su|=nsu; ac&=~nsu; s=ns; pp=np
        sh=torch.linalg.norm(sp,dim=-1)
        mu_all=(sh/(pl+1e-12)).clamp(max=1.0)**2
        mu_s=mu_all[su].mean() if su.any() else torch.tensor(0.)
        S_=float(su.float().mean()); mv=float(mu_s); to=S_*mv
        tracker['best_S']=max(tracker['best_S'],S_)
        tracker['best_mu']=max(tracker['best_mu'],mv)
        tracker['best_to']=max(tracker['best_to'],to)
        if to > best_joint['to']: best_joint.update({'S':S_,'mu':mv,'to':to,'cb':callback_count})
        eval_hist.append({'S':S_,'mu':mv,'to':to,'cb':callback_count})
        p_r=ttpi.policy_model.tt().ranks_tt.tolist(); v_r=ttpi.v_model.ranks_tt.tolist()
        a_r=ttpi.a_model.ranks_tt.tolist()
        rank_hist.append({'cb':callback_count,'Vr':max(v_r),'Ar':max(a_r),'Pr':max(p_r),
                          'Vr_all':v_r,'Ar_all':a_r,'Pr_all':p_r})
        al=torch.cuda.memory_allocated()/1e9; pk=torch.cuda.max_memory_allocated()/1e9
        print(f'  cb{callback_count:>3}: S={S_:.3f} mu={mv:.4f} to={to:.4f} Vr={max(v_r)} Ar={max(a_r)} Pr={max(p_r)} a={al:.3f}G p={pk:.2f}G',flush=True)
        torch.cuda.empty_cache(); gc.collect()
        return torch.tensor(0.),torch.tensor(0.)

    # Monkey-patch log_data to count cross warnings
    _orig_log = ttpi.log_data
    def _new_log():
        nonlocal cross_warnings
        try: _orig_log()
        except: cross_warnings += 1
    ttpi.log_data = _new_log

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

    return {
        "label":label,"seed":seed,"status":status,"a_order":a_order,
        "S_joint":best_joint['S'],"mu_joint":best_joint['mu'],
        "to_joint":best_joint['to'],"cb_joint":best_joint['cb'],
        "S_best":tracker['best_S'],"mu_best":tracker['best_mu'],"to_best":tracker['best_to'],
        "Vr_max":max(vr),"Ar_max":max(ar),"Pr_max":max(pr),
        "Vr_all":vr,"Ar_all":ar,"Pr_all":pr,
        "peak_gb":peak,"time_s":t,"cross_warnings":cross_warnings,
        "n_test":n_test,"n_iter":n_iter,
        "eval_history":eval_hist,"rank_history":rank_hist,
    }

if __name__ == "__main__":
    N_ACT=8; N_ITER=30; N_SEEDS=5
    all_orders = build_orderings(N_ACT)

    print("Ordering summary:")
    for label, orders in all_orders.items():
        print(f"  {label}: {orders[0][:8]}... ({len(orders)} variants)")

    all_results = {}
    for label, order_variants in all_orders.items():
        print(f"\n{'='*50}\n  {label}\n{'='*50}")
        for seed in range(N_SEEDS):
            a_ord = order_variants[seed % len(order_variants)]
            r = run_one(N_ACT, a_ord, seed=seed, n_iter=N_ITER, label=label)
            key = f"{label}_s{seed}"
            all_results[key] = r
            print(f"  === {label} s{seed}: {r['status']} "
                  f"joint(S={r['S_joint']:.3f} mu={r['mu_joint']:.4f} to={r['to_joint']:.4f}@cb{r['cb_joint']}) "
                  f"Vr={r['Vr_max']} Ar={r['Ar_max']} Pr={r['Pr_max']} "
                  f"peak={r['peak_gb']:.2f}G T={r['time_s']:.0f}s ===", flush=True)
        torch.cuda.empty_cache(); gc.collect()

    # Summary table
    print(f"\n{'='*80}")
    print("PAM-TTPI v4: Structure-Aware Ordering Necessity")
    print(f"{'='*80}")
    hdr = f"{'Order':<14} {'S×μ':<10} {'Vr':<6} {'Ar':<6} {'Pr':<6} {'Pr_std':<8} {'Peak(G)':<10} {'T(s)':<10} {'OOM':<6}"
    print(hdr); print("-"*80)
    for label in ["Local","BadSplit","Random","OppositePair"]:
        keys = [f"{label}_s{s}" for s in range(N_SEEDS)]
        tos = [all_results[k]['to_joint'] for k in keys]
        Vrs = [all_results[k]['Vr_max'] for k in keys]
        Ars = [all_results[k]['Ar_max'] for k in keys]
        Prs = [all_results[k]['Pr_max'] for k in keys]
        Pr_std = statistics.stdev(Prs) if len(Prs)>1 else 0
        pks = [all_results[k]['peak_gb'] for k in keys]
        Ts = [all_results[k]['time_s'] for k in keys]
        ooms = sum(1 for k in keys if all_results[k]['status']=="OOM")
        print(f"{label:<14} {statistics.mean(tos):<10.4f} {max(Vrs):<6} {max(Ars):<6} "
              f"{statistics.mean(Prs):<6.1f} {Pr_std:<8.1f} {max(pks):<10.2f} "
              f"{statistics.mean(Ts):<10.0f} {ooms}/{N_SEEDS}")

    # Save full data including eval/rank history
    out = {"n_act": N_ACT, "n_iter": N_ITER, "n_test": 50,
           "orderings": {k: v[0] for k, v in all_orders.items()},
           "results": all_results}
    with open("repro/results/pam_v4_full.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: repro/results/pam_v4_full.json ({len(json.dumps(out))//1024} KB)")
