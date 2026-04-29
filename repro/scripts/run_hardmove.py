import os
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import json
import time
import argparse
from pathlib import Path

import torch

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
plt.ioff()


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ttpi import TTPI
from dynamic_systems import HardMove


torch.set_default_dtype(torch.float64)


def seed_everything(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def make_init_states(n_test, state_min, state_max, device, seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 12345)

    dim_state = len(state_min)
    init_state = torch.empty((n_test, dim_state), dtype=torch.float64)

    # 保持作者 notebook 的初始化风格：rand().clip(0.25, 0.75)
    for i in range(dim_state):
        r = torch.rand(n_test, generator=g, dtype=torch.float64).clip(0.25, 0.75)
        init_state[:, i] = state_min[i].cpu() + r * (state_max[i].cpu() - state_min[i].cpu())

    init_state[:, 2:4] = 0.0
    return init_state.to(device)


@torch.no_grad()
def evaluate_policy(ttpi, dyn_system, init_state, dt, horizon_sec=10.0, target_radius=0.02):
    state = init_state.clone()
    n = state.shape[0]
    T = int(horizon_sec / dt)

    active = torch.ones(n, dtype=torch.bool, device=state.device)
    success = torch.zeros(n, dtype=torch.bool, device=state.device)

    start_pos = state[:, :2].clone()
    prev_pos = state[:, :2].clone()
    path_len = torch.zeros(n, dtype=torch.float64, device=state.device)
    cum_reward = torch.zeros(n, dtype=torch.float64, device=state.device)

    final_reward = None

    for _ in range(T):
        action = ttpi.policy(state)
        r = dyn_system.reward_state_action(state, action)
        final_reward = r
        cum_reward += r

        next_state = dyn_system.forward_simulate(state, action)
        next_pos = next_state[:, :2]

        step_len = torch.linalg.norm(next_pos - prev_pos, dim=-1)
        path_len += active.to(torch.float64) * step_len

        dist = torch.linalg.norm(next_pos, dim=-1)
        newly_success = active & (dist <= target_radius)
        success |= newly_success
        active &= ~newly_success

        state = next_state
        prev_pos = next_pos

    shortest = torch.linalg.norm(start_pos, dim=-1)
    mu_all = (shortest / (path_len + 1e-12)).clamp(max=1.0) ** 2

    if success.any():
        mu_success = mu_all[success].mean()
    else:
        mu_success = torch.tensor(0.0, dtype=torch.float64, device=state.device)

    metrics = {
        "success_rate": float(success.to(torch.float64).mean().cpu().item()),
        "mu_success": float(mu_success.cpu().item()),
        "mu_all": float(mu_all.mean().cpu().item()),
        "final_dist_mean": float(torch.linalg.norm(state[:, :2], dim=-1).mean().cpu().item()),
        "path_len_mean": float(path_len.mean().cpu().item()),
        "cum_reward_mean": float(cum_reward.mean().cpu().item()),
        "final_reward_mean": float(final_reward.mean().cpu().item()) if final_reward is not None else None,
    }
    return metrics

@torch.no_grad()
def save_trajectory_figure(ttpi, dyn_system, init_state, dt, fig_path, horizon_sec=10.0, max_traj=20):
    state = init_state.clone()
    T = int(horizon_sec / dt)
    n_plot = min(max_traj, state.shape[0])

    traj = [state[:n_plot, :2].detach().cpu()]

    for _ in range(T):
        action = ttpi.policy(state)
        state = dyn_system.forward_simulate(state, action)
        traj.append(state[:n_plot, :2].detach().cpu())

    traj = torch.stack(traj, dim=0).numpy()  # [T+1, n_plot, 2]

    fig, ax = plt.subplots(figsize=(6, 6))

    for i in range(n_plot):
        ax.plot(traj[:, i, 0], traj[:, i, 1], linewidth=1.0, alpha=0.8)
        ax.scatter(traj[0, i, 0], traj[0, i, 1], s=10)

    ax.scatter([0.0], [0.0], marker="x", s=80, label="goal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("HardMove trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    return str(fig_path)

def select_best_eval(eval_history):
    if not eval_history:
        return None

    return max(
        eval_history,
        key=lambda m: (
            m.get("success_rate", 0.0),
            m.get("mu_success", 0.0),
            -m.get("final_dist_mean", 1e9),
        ),
    )

def select_best_eval(eval_history):
    if not eval_history:
        return None
    return max(
        eval_history,
        key=lambda m: (
            m.get("success_rate", 0.0),
            m.get("mu_success", 0.0),
            -m.get("final_dist_mean", 1e9),
        ),
    )

class EarlyStopTraining(Exception):
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-actuator", type=int, required=True)
    parser.add_argument("--n-state", type=int, default=50)
    parser.add_argument("--n-action", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--n-iter-v", type=int, default=1)
    parser.add_argument("--callback-freq", type=int, default=10)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target-radius", type=float, default=0.02)

    parser.add_argument("--max-batch-v", type=int, default=10000)
    parser.add_argument("--max-batch-a", type=int, default=100000)
    parser.add_argument("--nswp-v", type=int, default=5)
    parser.add_argument("--nswp-a", type=int, default=10)
    parser.add_argument("--rmax-v", type=int, default=100)
    parser.add_argument("--rmax-a", type=int, default=100)
    parser.add_argument("--kickrank-v", type=int, default=10)
    parser.add_argument("--kickrank-a", type=int, default=10)
    parser.add_argument("--eps-cross-v", type=float, default=1e-3)
    parser.add_argument("--eps-cross-a", type=float, default=1e-3)
    parser.add_argument("--eps-round-v", type=float, default=1e-3)
    parser.add_argument("--eps-round-a", type=float, default=1e-3)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--early-stop-success", type=float, default=None)
    parser.add_argument("--early-stop-mu", type=float, default=0.0)
    parser.add_argument("--early-stop-after-callback", type=int, default=0)


    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but --device cuda was requested.")

    device = torch.device(args.device)
    seed_everything(args.seed)

    out_dir = ROOT / "repro" / "results"
    model_dir = ROOT / "repro" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    task_name = (
    f"HM{args.n_actuator}"
    f"_state{args.n_state}"
    f"_action{args.n_action}"
    f"_iter{args.n_iter}"
    f"_seed{args.seed}"
)

    L = 1.0
    position_max = L
    position_min = -L

    velocity_max = 0.25 * position_max
    velocity_min = 0.0 * velocity_max
    acc_max = 1.0 * velocity_max
    acc_min = -1.0 * acc_max

    domain_acc0 = torch.linspace(acc_min, acc_max, args.n_action, device=device)
    domain_switch0 = torch.arange(2, device=device, dtype=torch.float64)

    state_min = torch.tensor(
        [position_min, position_min, velocity_min, velocity_min],
        device=device,
        dtype=torch.float64,
    )
    state_max = torch.tensor(
        [position_max, position_max, velocity_max, velocity_max],
        device=device,
        dtype=torch.float64,
    )

    domain_state = [
        torch.linspace(state_min[i], state_max[i], args.n_state, device=device)
        for i in range(len(state_max))
    ]

    domain_action = [domain_acc0, domain_switch0] * args.n_actuator

    dyn_system = HardMove(
        dt=args.dt,
        w_goal=1e3,
        w_action=1e4,
        n=args.n_actuator,
        device=device,
    )

    def forward_model(state, action):
        return dyn_system.forward_simulate(state, action)

    def reward(state, action):
        return dyn_system.reward_state_action(state, action)

    init_state = make_init_states(
        n_test=args.n_test,
        state_min=state_min,
        state_max=state_max,
        device=device,
        seed=args.seed,
    )

    eval_history = []

    best_eval = {"metrics": None}

    def is_better_metric(new_m, old_m):
        if old_m is None:
            return True
        return (
            new_m.get("success_rate", 0.0),
            new_m.get("mu_success", 0.0),
            -new_m.get("final_dist_mean", 1e9),
        ) > (
            old_m.get("success_rate", 0.0),
            old_m.get("mu_success", 0.0),
            -old_m.get("final_dist_mean", 1e9),
        )

    def callback(ttpi, state=init_state, callback_count=0):
        metrics = evaluate_policy(
            ttpi=ttpi,
            dyn_system=dyn_system,
            init_state=state,
            dt=args.dt,
            horizon_sec=10.0,
            target_radius=args.target_radius,
        )
        metrics["callback_count"] = int(callback_count)
        eval_history.append(metrics)

        if is_better_metric(metrics, best_eval["metrics"]):
            best_eval["metrics"] = dict(metrics)

        print("[EVAL]", json.dumps(metrics, ensure_ascii=False))

        if (
            args.early_stop_success is not None
            and callback_count >= args.early_stop_after_callback
            and metrics["success_rate"] >= args.early_stop_success
            and metrics["mu_success"] >= args.early_stop_mu
        ):
            print(
                f"[EARLY_STOP] success_rate={metrics['success_rate']:.4f}, "
                f"mu_success={metrics['mu_success']:.4f}, "
                f"callback_count={callback_count}"
            )
            raise EarlyStopTraining()

        return (
            torch.tensor(metrics["final_reward_mean"]),
            torch.tensor(metrics["cum_reward_mean"]),
        )

    ttpi = TTPI(
        domain_state=domain_state,
        domain_action=domain_action,
        reward=reward,
        normalize_reward=True,
        forward_model=forward_model,
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
        verbose=True,
        device=device,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()

    stopped_early = False

    try:
        ttpi.train(
            resume=False,
            n_iter_max=args.n_iter,
            n_iter_v=args.n_iter_v,
            callback=callback,
            callback_freq=args.callback_freq,
            verbose=False,
            file_name=str(model_dir / task_name),
        )
    except EarlyStopTraining:
        stopped_early = True
        print("[INFO] training stopped early by evaluation criterion")

    train_time_sec = time.time() - t0

    final_metrics = evaluate_policy(
        ttpi=ttpi,
        dyn_system=dyn_system,
        init_state=init_state,
        dt=args.dt,
        horizon_sec=10.0,
        target_radius=args.target_radius,
    )
    best_metrics = select_best_eval(eval_history)

    fig_dir = ROOT / "repro" / "figures" / "hardmove"
    fig_path = fig_dir / f"{task_name}_traj.png"


    traj_fig = save_trajectory_figure(
        ttpi=ttpi,
        dyn_system=dyn_system,
        init_state=init_state,
        dt=args.dt,
        fig_path=fig_path,
        horizon_sec=10.0,
        max_traj=min(args.n_test, 20),
    )

    
    
    result = {
        "task": f"HM({args.n_actuator})",
        "seed": args.seed,
        "n_state": args.n_state,
        "n_action": args.n_action,
        "n_actuator": args.n_actuator,
        "action_dim": 2 * args.n_actuator,
        "dt": args.dt,
        "gamma": args.gamma,
        "n_iter": args.n_iter,
        "n_iter_v": args.n_iter_v,
        "n_test": args.n_test,
        "target_radius": args.target_radius,
        "max_batch_v": args.max_batch_v,
        "max_batch_a": args.max_batch_a,
        "nswp_v": args.nswp_v,
        "nswp_a": args.nswp_a,
        "rmax_v": args.rmax_v,
        "rmax_a": args.rmax_a,
        "eps_cross_v": args.eps_cross_v,
        "eps_cross_a": args.eps_cross_a,
        "eps_round_v": args.eps_round_v,
        "eps_round_a": args.eps_round_a,
        "n_samples": args.n_samples,
        "train_time_sec": train_time_sec,
        "final_metrics": final_metrics,
        "eval_history": eval_history,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "max_cuda_mem_gb": (
            torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
        ),
        "trajectory_figure": traj_fig,
        "best_metrics": best_metrics,
        "stopped_early": stopped_early,
        "early_stop_success": args.early_stop_success,
        "early_stop_mu": args.early_stop_mu,
        "early_stop_after_callback": args.early_stop_after_callback,
        "best_metrics": best_metrics,
    }

    out_file = out_dir / f"{task_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[DONE] saved result to {out_file}")
    print(json.dumps(result["final_metrics"], indent=2, ensure_ascii=False))





if __name__ == "__main__":
    main()
