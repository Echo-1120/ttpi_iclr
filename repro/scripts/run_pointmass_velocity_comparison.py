"""PointMassVelocity comparison experiment.

This script turns PointMassVelocity.ipynb into a reproducible experiment and
adds lightweight deterministic baselines for comparison. The baselines are not
the paper's HyAR baselines; the ICLR paper only showcases this PointMass task in
the appendix/video. They are included to quantify whether the TTPI policy is
doing more than simple hand-coded navigation.

Example smoke run:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/s110/miniconda3/envs/tt_5080/bin/python \
    repro/scripts/run_pointmass_velocity_comparison.py --preset smoke

Notebook-like run:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/s110/miniconda3/envs/tt_5080/bin/python \
    repro/scripts/run_pointmass_velocity_comparison.py --preset notebook
"""
import argparse
import csv
import gc
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dynamic_systems import PointMass
from ttpi import TTPI


torch.set_default_dtype(torch.float64)

RESULTS = ROOT / "repro" / "results"
FIGURES = ROOT / "repro" / "figures" / "pointmass_velocity"


def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def notebook_eval_states(device, dim=2):
    state = torch.tensor(
        [
            [0.0, 0.0],
            [-0.3, -0.0],
            [-0.3, 0.3],
            [-0.3, -0.3],
            [0.3, -0.0],
            [0.3, 0.3],
            [0.3, -0.3],
            [0.0, 0.3],
            [-0.75, -0.75],
            [-0.75, 0.75],
            [0.75, -0.75],
            [-0.6, -0.0],
            [-0.6, 0.2],
            [-0.5, -0.6],
            [0.6, -0.2],
            [0.6, 0.5],
            [0.6, -0.7],
            [0.1, 0.6],
            [-0.8, -0.0],
            [-0.8, 0.7],
            [-0.8, -0.5],
            [0.8, -0.0],
            [0.0, -0.9],
            [0.8, 0.4],
            [0.8, -0.6],
            [0.9, 0.0],
            [0.1, 0.8],
            [-0.1, -0.8],
            [0.1, -0.8],
            [0.2, -0.8],
        ],
        device=device,
        dtype=torch.float64,
    )
    if dim > 2:
        state = torch.cat((state, torch.zeros(state.shape[0], dim - 2, device=device)), dim=-1)
    return state


def build_task(args, device):
    dim = 2
    order = 1
    length = 1.0
    position_max = torch.tensor([length] * dim, device=device)
    position_min = -position_max
    velocity_max = position_max / 4.0
    velocity_min = -velocity_max

    state_min = position_min
    state_max = position_max
    action_min = velocity_min
    action_max = velocity_max

    domain_state = [
        torch.linspace(state_min[i], state_max[i], args.n_state, device=device)
        for i in range(len(state_max))
    ]
    domain_action = [
        torch.linspace(action_min[i], action_max[i], args.n_action, device=device)
        for i in range(len(action_max))
    ]

    x_obst = [torch.tensor([0.0, -0.4], device=device)]
    r_obst = [0.2]
    dyn = PointMass(
        order=order,
        dt=args.dt,
        dim=dim,
        x_obst=x_obst,
        r_obst=r_obst,
        w_obst=1e2,
        w_action=5e2,
        w_goal=1e2,
        w_scale=1,
        device=device,
    )
    return {
        "dim": dim,
        "length": length,
        "state_min": state_min,
        "state_max": state_max,
        "action_min": action_min,
        "action_max": action_max,
        "domain_state": domain_state,
        "domain_action": domain_action,
        "x_obst": x_obst,
        "r_obst": r_obst,
        "dyn": dyn,
    }


def reward_wrapper(dyn):
    def reward(state, action):
        rewards, _ = dyn.reward_state_action(state, action)
        return rewards

    return reward


class PolicyAdapter:
    def __init__(self, name, fcn):
        self.name = name
        self.fcn = fcn

    def policy(self, state):
        return self.fcn(state)


def clip_action_norm(action, vmax):
    norm = torch.linalg.norm(action, dim=-1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(vmax / norm, max=1.0)
    return action * scale


def make_baseline(name, task, seed):
    device = task["state_min"].device
    vmax = float(task["action_max"][0].detach().cpu())
    obstacle = task["x_obst"][0]
    influence = task["r_obst"][0] + 0.35
    g = torch.Generator(device="cpu")
    g.manual_seed(7000 + seed)

    if name == "straight":
        def policy(state):
            vec = -state[:, :2]
            return clip_action_norm(vec, vmax)

    elif name == "potential":
        def policy(state):
            pos = state[:, :2]
            attract = -pos
            away = pos - obstacle.view(1, 2)
            dist = torch.linalg.norm(away, dim=-1, keepdim=True).clamp_min(1e-6)
            strength = torch.clamp((influence - dist) / influence, min=0.0) ** 2
            repel = 1.8 * strength * away / dist
            return clip_action_norm(attract + repel, vmax)

    elif name == "random":
        def policy(state):
            lo = task["action_min"].detach().cpu()
            hi = task["action_max"].detach().cpu()
            u = torch.rand((state.shape[0], 2), generator=g, dtype=torch.float64)
            return (lo + u * (hi - lo)).to(device)

    elif name == "zero":
        def policy(state):
            return torch.zeros((state.shape[0], 2), device=device, dtype=torch.float64)

    else:
        raise ValueError(f"Unknown baseline: {name}")

    return PolicyAdapter(name, policy)


@torch.no_grad()
def evaluate_policy(policy_obj, dyn, init_state, args, task):
    state = init_state.clone()
    n = state.shape[0]
    active = torch.ones(n, dtype=torch.bool, device=state.device)
    success = torch.zeros(n, dtype=torch.bool, device=state.device)
    collided = torch.zeros(n, dtype=torch.bool, device=state.device)
    start_pos = state[:, :2].clone()
    prev_pos = state[:, :2].clone()
    path_len = torch.zeros(n, dtype=torch.float64, device=state.device)
    cum_reward = torch.zeros(n, dtype=torch.float64, device=state.device)
    final_reward = None

    traj = [state[:, :2].detach().cpu()]
    horizon = int(args.horizon / args.dt)
    obstacle = task["x_obst"][0].view(1, 2)
    obstacle_radius = float(task["r_obst"][0])

    for _ in range(horizon):
        action = policy_obj.policy(state)
        reward, _ = dyn.reward_state_action(state, action)
        final_reward = reward
        cum_reward += reward
        next_state = dyn.forward_simulate(state, action)
        next_pos = next_state[:, :2]
        step_len = torch.linalg.norm(next_pos - prev_pos, dim=-1)
        path_len += active.to(torch.float64) * step_len

        dist_goal = torch.linalg.norm(next_pos, dim=-1)
        newly_success = active & (dist_goal <= args.target_radius)
        success |= newly_success
        active &= ~newly_success

        dist_obst = torch.linalg.norm(next_pos - obstacle, dim=-1)
        collided |= dist_obst <= obstacle_radius

        state = next_state
        prev_pos = next_pos
        traj.append(state[:, :2].detach().cpu())

    shortest = torch.linalg.norm(start_pos, dim=-1)
    mu_all = (shortest / (path_len + 1e-12)).clamp(max=1.0) ** 2
    mu_success = mu_all[success].mean() if success.any() else torch.tensor(0.0, device=state.device)
    success_rate = float(success.to(torch.float64).mean().cpu())
    mu = float(mu_success.cpu())
    metrics = {
        "success_rate": success_rate,
        "mu_success": mu,
        "tradeoff_score": success_rate * mu,
        "collision_rate": float(collided.to(torch.float64).mean().cpu()),
        "final_dist_mean": float(torch.linalg.norm(state[:, :2], dim=-1).mean().cpu()),
        "path_len_mean": float(path_len.mean().cpu()),
        "cum_reward_mean": float(cum_reward.mean().cpu()),
        "final_reward_mean": float(final_reward.mean().cpu()) if final_reward is not None else None,
    }
    return metrics, torch.stack(traj, dim=0)


def save_trajectory_figure(trajs, task, path):
    FIGURES.mkdir(parents=True, exist_ok=True)
    methods = list(trajs.keys())
    n_cols = min(3, len(methods))
    n_rows = math.ceil(len(methods) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)

    for ax, method in zip(axes.flat, methods):
        traj = trajs[method].numpy()
        for i in range(traj.shape[1]):
            ax.plot(traj[:, i, 0], traj[:, i, 1], linewidth=0.9, alpha=0.75)
            ax.scatter(traj[0, i, 0], traj[0, i, 1], s=8, alpha=0.8)
        obst = task["x_obst"][0].detach().cpu().numpy()
        circle = plt.Circle(obst, task["r_obst"][0], color="#ef4444", alpha=0.25)
        ax.add_patch(circle)
        ax.scatter([0.0], [0.0], marker="x", s=80, color="black")
        ax.set_title(method)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)

    for ax in axes.flat[len(methods):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def select_best_eval(eval_history):
    return max(
        eval_history,
        key=lambda m: (
            m.get("tradeoff_score", 0.0),
            m.get("success_rate", 0.0),
            -m.get("collision_rate", 1.0),
        ),
    )


def train_ttpi(args, task, init_state):
    device = init_state.device
    dyn = task["dyn"]
    ttpi = TTPI(
        domain_state=task["domain_state"],
        domain_action=task["domain_action"],
        reward=reward_wrapper(dyn),
        forward_model=dyn.forward_simulate,
        gamma=args.gamma,
        rmax_v=args.rmax_v,
        rmax_a=args.rmax_a,
        nswp_v=args.nswp_v,
        nswp_a=args.nswp_a,
        kickrank_v=args.kickrank_v,
        kickrank_a=args.kickrank_a,
        max_batch_v=args.max_batch_v,
        max_batch_a=args.max_batch_a,
        eps_cross_v=args.eps_cross_v,
        eps_cross_a=args.eps_cross_a,
        eps_round_v=args.eps_round_v,
        eps_round_a=args.eps_round_a,
        n_samples=args.n_samples,
        normalize_reward=False,
        verbose=False,
        device=device,
    )
    ttpi.plt_training_stat = lambda: None
    ttpi.save_model = lambda *a, **kw: None

    eval_history = []
    rank_history = []

    def callback(model, callback_count=0):
        metrics, _ = evaluate_policy(model, dyn, init_state, args, task)
        metrics["cb"] = int(callback_count)
        eval_history.append(metrics)
        ar = [int(x) for x in model.a_model.ranks_tt.detach().cpu().tolist()]
        vr = [int(x) for x in model.v_model.ranks_tt.detach().cpu().tolist()]
        pr = [int(x) for x in model.policy_model.ranks_tt.detach().cpu().tolist()]
        rank_history.append(
            {
                "cb": int(callback_count),
                "Vr_max": max(vr),
                "Ar_max": max(ar),
                "Pr_max": max(pr),
                "Vr": vr,
                "Ar": ar,
                "Pr": pr,
            }
        )
        peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
        print(
            f"  cb{callback_count:>3}: S={metrics['success_rate']:.3f} "
            f"mu={metrics['mu_success']:.3f} Sxmu={metrics['tradeoff_score']:.3f} "
            f"coll={metrics['collision_rate']:.3f} Ar={max(ar)} Pr={max(pr)} peak={peak:.2f}G",
            flush=True,
        )
        torch.cuda.empty_cache()
        gc.collect()
        return torch.tensor(0.0, device=device), torch.tensor(metrics["cum_reward_mean"], device=device)

    t0 = time.time()
    oom = False
    try:
        ttpi.train(
            n_iter_max=args.n_iter,
            n_iter_v=1,
            resume=False,
            callback=callback,
            callback_freq=args.callback_freq,
            verbose=False,
            file_name=None,
        )
    except torch.OutOfMemoryError:
        oom = True
        print("OOM", flush=True)

    elapsed = time.time() - t0
    best = select_best_eval(eval_history) if eval_history else None
    final_metrics, traj = evaluate_policy(ttpi, dyn, init_state, args, task)
    return {
        "policy": ttpi,
        "result": {
            "method": "ttpi",
            "status": "OOM" if oom else "OK",
            "time_s": float(elapsed),
            "peak_gb": float(torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0),
            "best_eval": best,
            "final_eval": final_metrics,
            "eval_history": eval_history,
            "rank_history": rank_history,
        },
        "trajectory": traj,
    }


def apply_preset(args):
    if args.preset == "smoke":
        args.n_state = 20
        args.n_action = 20
        args.n_iter = 3
        args.callback_freq = 1
        args.horizon = 2.0
        args.n_samples = 20
        args.nswp_v = 3
        args.nswp_a = 3
        args.rmax_v = 30
        args.rmax_a = 30
        args.max_batch_v = 3000
        args.max_batch_a = 6000
    elif args.preset == "notebook":
        args.n_state = 50
        args.n_action = 50
        args.n_iter = 200
        args.callback_freq = 20
        args.horizon = 10.0
    return args


def write_outputs(args, payload, rows, trajs, task):
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem or (
        f"pointmass_velocity_{args.preset}_state{args.n_state}_action{args.n_action}_"
        f"iter{args.n_iter}_seed{args.seed}"
    )
    json_path = RESULTS / f"{stem}.json"
    csv_path = RESULTS / f"{stem}.summary.csv"
    fig_path = FIGURES / f"{stem}_trajectories.png"

    payload["trajectory_figure"] = save_trajectory_figure(trajs, task, fig_path)
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)

    fields = [
        "method",
        "status",
        "success_rate",
        "mu_success",
        "tradeoff_score",
        "collision_rate",
        "final_dist_mean",
        "path_len_mean",
        "cum_reward_mean",
        "time_s",
        "peak_gb",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    print(f"Saved results: {json_path}")
    print(f"Saved summary: {csv_path}")
    print(f"Saved figure: {fig_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["smoke", "notebook", "custom"], default="smoke")
    parser.add_argument("--methods", default="ttpi,straight,potential,random,zero")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-stem", default=None)

    parser.add_argument("--n-state", type=int, default=50)
    parser.add_argument("--n-action", type=int, default=50)
    parser.add_argument("--n-iter", type=int, default=200)
    parser.add_argument("--callback-freq", type=int, default=20)
    parser.add_argument("--horizon", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--target-radius", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-batch-v", type=int, default=10000)
    parser.add_argument("--max-batch-a", type=int, default=100000)
    parser.add_argument("--nswp-v", type=int, default=10)
    parser.add_argument("--nswp-a", type=int, default=10)
    parser.add_argument("--rmax-v", type=int, default=100)
    parser.add_argument("--rmax-a", type=int, default=100)
    parser.add_argument("--kickrank-v", type=int, default=5)
    parser.add_argument("--kickrank-a", type=int, default=10)
    parser.add_argument("--eps-cross-v", type=float, default=1e-3)
    parser.add_argument("--eps-cross-a", type=float, default=1e-3)
    parser.add_argument("--eps-round-v", type=float, default=1e-3)
    parser.add_argument("--eps-round-a", type=float, default=1e-3)
    parser.add_argument("--n-samples", type=int, default=50)
    return apply_preset(parser.parse_args())


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    device = torch.device(args.device)
    seed_everything(args.seed)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    task = build_task(args, device)
    init_state = notebook_eval_states(device, dim=task["dim"])
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    payload = {
        "source": "PointMassVelocity.ipynb",
        "note": (
            "Baselines are deterministic controls added for this reproduction; "
            "the ICLR paper does not report a formal PointMassVelocity baseline table."
        ),
        "config": vars(args),
        "obstacle": {
            "center": [float(x) for x in task["x_obst"][0].detach().cpu().tolist()],
            "radius": float(task["r_obst"][0]),
        },
        "results": {},
    }
    rows = []
    trajs = {}

    for method in methods:
        print(f"\n=== {method} ===", flush=True)
        if method == "ttpi":
            out = train_ttpi(args, task, init_state)
            result = out["result"]
            metrics = result["best_eval"] or result["final_eval"]
            trajs[method] = out["trajectory"]
            payload["results"][method] = result
            row = {"method": method, "status": result["status"], "time_s": result["time_s"], "peak_gb": result["peak_gb"]}
            row.update(metrics)
            rows.append(row)
        else:
            policy = make_baseline(method, task, args.seed)
            t0 = time.time()
            metrics, traj = evaluate_policy(policy, task["dyn"], init_state, args, task)
            elapsed = time.time() - t0
            trajs[method] = traj
            payload["results"][method] = {"method": method, "status": "OK", "eval": metrics, "time_s": elapsed}
            row = {"method": method, "status": "OK", "time_s": elapsed, "peak_gb": ""}
            row.update(metrics)
            rows.append(row)
            print(
                f"  S={metrics['success_rate']:.3f} mu={metrics['mu_success']:.3f} "
                f"Sxmu={metrics['tradeoff_score']:.3f} coll={metrics['collision_rate']:.3f}",
                flush=True,
            )

    write_outputs(args, payload, rows, trajs, task)


if __name__ == "__main__":
    main()
