"""Low-batch HM experiment. Usage: python run_hm_lowbatch.py <n_act> <seed>"""
import torch, sys, time, gc, json
from pathlib import Path
ROOT = Path('/home/s110/code/ttpi_iclr')
sys.path.insert(0, str(ROOT))
from ttpi import TTPI
from dynamic_systems import HardMove

N_ACT = int(sys.argv[1]); SEED = int(sys.argv[2])
MBV = 5000   # reduced from 10000
MBA = 20000  # reduced from 100000

torch.set_default_dtype(torch.float64)
device = 'cuda'
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()

L = 1.0; vmax = 0.25; NS = 50
domain_acc = torch.linspace(-vmax, vmax, 100, device=device)
domain_sw = torch.tensor([0., 1.], device=device)
smin = torch.tensor([-L, -L, 0., 0.], device=device)
smax = torch.tensor([L, L, vmax, vmax], device=device)
domain_state = [torch.linspace(smin[i], smax[i], NS, device=device) for i in range(4)]
domain_action = [domain_acc, domain_sw] * N_ACT

dyn = HardMove(dt=0.01, w_goal=1e3, w_action=1e4, n=N_ACT, device=device)

ttpi = TTPI(domain_state=domain_state, domain_action=domain_action,
            reward=lambda s, a: dyn.reward_state_action(s, a),
            normalize_reward=True,
            forward_model=lambda s, a: dyn.forward_simulate(s, a),
            gamma=0.99, rmax_v=100, rmax_a=100,
            nswp_v=5, nswp_a=10, kickrank_v=10, kickrank_a=10,
            max_batch_v=MBV, max_batch_a=MBA,
            eps_cross_v=1e-3, eps_cross_a=1e-3, eps_round_v=1e-3, eps_round_a=1e-3,
            n_samples=50, verbose=False, device=device)
ttpi.plt_training_stat = lambda: None
ttpi.save_model = lambda *a, **kw: None

_orig = ttpi.PI_update
def _new(n_iter_v=1):
    _orig(n_iter_v=n_iter_v)
    torch.cuda.empty_cache(); gc.collect()
ttpi.PI_update = _new

g = torch.Generator(device='cpu'); g.manual_seed(12345 + SEED)
n_test = 20
init_state = torch.empty((n_test, 4), dtype=torch.float64)
for i in range(4):
    r = torch.rand(n_test, generator=g, dtype=torch.float64).clip(0.25, 0.75)
    init_state[:, i] = smin[i].cpu() + r * (smax[i].cpu() - smin[i].cpu())
init_state[:, 2:4] = 0.0; init_state = init_state.to(device)

tracker = {'best_S': 0, 'best_mu': 0, 'best_to': 0, 'joint_best': 0}
eval_history = []; last_cb = -1

def cb(ttpi, state=init_state, callback_count=0):
    global last_cb; last_cb = callback_count
    s = state.clone()
    T = int(10 / 0.01); active = torch.ones(n_test, dtype=torch.bool, device=device)
    success = torch.zeros(n_test, dtype=torch.bool, device=device)
    sp = s[:, :2].clone(); pp = s[:, :2].clone(); pl = torch.zeros(n_test, device=device)
    for _ in range(T):
        a = ttpi.policy(s); ns = dyn.forward_simulate(s, a); np = ns[:, :2]
        pl += active.float() * torch.linalg.norm(np - pp, dim=-1)
        d = torch.linalg.norm(np, dim=-1); ns_ = active & (d <= 0.02)
        success |= ns_; active &= ~ns_; s = ns; pp = np
    sh = torch.linalg.norm(sp, dim=-1)
    mu_all = (sh / (pl + 1e-12)).clamp(max=1.0) ** 2
    mu_s = mu_all[success].mean() if success.any() else torch.tensor(0.)
    S_ = float(success.float().mean()); mu_s_val = float(mu_s)
    tracker['best_S'] = max(tracker['best_S'], S_)
    tracker['best_mu'] = max(tracker['best_mu'], mu_s_val)
    tracker['best_to'] = max(tracker['best_to'], S_ * mu_s_val)
    if S_ >= 0.95: tracker['joint_best'] = max(tracker['joint_best'], mu_s_val)
    eval_history.append({'S': S_, 'mu': mu_s_val, 'tradeoff': S_*mu_s_val, 'cb': callback_count})
    alloc = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f'  cb{callback_count:>3}: S={S_:.3f} mu={mu_s_val:.4f} to={S_*mu_s_val:.4f} a={alloc:.3f}G p={peak:.2f}G', flush=True)
    torch.cuda.empty_cache(); gc.collect()
    return torch.tensor(0.), torch.tensor(0.)

t0 = time.time(); oom = False
try:
    ttpi.train(resume=False, n_iter_max=100, n_iter_v=1, callback=cb,
               callback_freq=10, verbose=False, file_name=None)
except torch.OutOfMemoryError:
    oom = True; print(f'OOM at cb{last_cb}!', flush=True)
except Exception as e:
    print(f'Error: {e}', flush=True)

t = time.time() - t0
p_rank = ttpi.policy_model.tt().ranks_tt.tolist()
v_rank = ttpi.v_model.ranks_tt.tolist()
status = "OOM" if oom else ("OK" if last_cb >= 99 else f"INCOMPLETE(cb{last_cb})")
print(f'=== HM({N_ACT}) s={SEED}: {status} S={tracker["best_S"]:.3f} mu={tracker["best_mu"]:.4f} to={tracker["best_to"]:.4f} T={t:.0f}s p={torch.cuda.max_memory_allocated()/1e9:.2f}G cb={last_cb} ===', flush=True)

result = {"seed": SEED, "n_actuator": N_ACT, "n_state": 50, "n_iter": 100, "n_test": 20,
    "max_batch_v": MBV, "max_batch_a": MBA, "gc_collect": True, "status": status,
    "best_S": tracker["best_S"], "best_mu": tracker["best_mu"], "best_tradeoff": tracker["best_to"],
    "joint_mu_Sge95": tracker["joint_best"], "train_time_sec": t,
    "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9, "last_cb": last_cb,
    "eval_history": eval_history}
fn = f"repro/results/HM{N_ACT}_ns50_lowbatch_seed{SEED}.json"
with open(fn, "w") as f: json.dump(result, f, indent=2)
print(f"Saved: {fn}", flush=True)
