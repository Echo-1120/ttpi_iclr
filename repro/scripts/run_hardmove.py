import os
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import json
import time
import argparse
import warnings
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
from repro.hardmove_variants import (
    canonical_env_variant,
    make_env_permutation,
    transform_hardmove_action,
    variant_display_name,
)
from repro.pam_ordering import ORDERING_NAMES, build_hardmove_orders
from repro.diagnostics import (
    append_csv,
    cross_process_rows,
    cross_query_rows,
    rank_profile_rows,
    round_event_rows,
    standard_run_log,
    summary_row,
    tensor_safe,
    write_csv,
)


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
def evaluate_policy(
    ttpi,
    dyn_system,
    init_state,
    dt,
    horizon_sec=10.0,
    target_radius=0.02,
    action_inverse=None,
    action_transform=None,
):
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
        if action_inverse is not None:
            action = action[..., action_inverse]
        if action_transform is not None:
            action = action_transform(action)
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

    S_val = float(success.to(torch.float64).mean().cpu().item())
    mu_success_val = float(mu_success.cpu().item())
    metrics = {
        "success_rate": S_val,
        "mu_success": mu_success_val,
        "tradeoff_score": S_val * mu_success_val,
        "mu_all": float(mu_all.mean().cpu().item()),
        "final_dist_mean": float(torch.linalg.norm(state[:, :2], dim=-1).mean().cpu().item()),
        "path_len_mean": float(path_len.mean().cpu().item()),
        "cum_reward_mean": float(cum_reward.mean().cpu().item()),
        "final_reward_mean": float(final_reward.mean().cpu().item()) if final_reward is not None else None,
    }
    return metrics

@torch.no_grad()
def save_trajectory_figure(
    ttpi,
    dyn_system,
    init_state,
    dt,
    fig_path,
    horizon_sec=10.0,
    max_traj=20,
    action_inverse=None,
    action_transform=None,
):
    state = init_state.clone()
    T = int(horizon_sec / dt)
    n_plot = min(max_traj, state.shape[0])

    traj = [state[:n_plot, :2].detach().cpu()]

    for _ in range(T):
        action = ttpi.policy(state)
        if action_inverse is not None:
            action = action[..., action_inverse]
        if action_transform is not None:
            action = action_transform(action)
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


def select_best_tradeoff(eval_history):
    """Select checkpoint with best S×μ tradeoff (Pareto-aware)."""
    if not eval_history:
        return None
    return max(
        eval_history,
        key=lambda m: m.get("tradeoff_score", m.get("success_rate", 0) * m.get("mu_success", 0)),
    )


class EarlyStopTraining(Exception):
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-actuator", type=int, required=True)
    parser.add_argument("--n-state", type=int, default=50)
    parser.add_argument("--n-action", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--training-seed", type=int, default=None)
    parser.add_argument("--environment-seed", type=int, default=None)
    parser.add_argument("--permutation-seed", type=int, default=None)
    parser.add_argument("--state-sampling-seed", type=int, default=None)
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--n-iter-v", type=int, default=1)
    parser.add_argument("--callback-freq", type=int, default=10)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target-radius", type=float, default=0.02)
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--include-permutation-seed-in-run-id", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--diagnostics-root", type=Path, default=None)
    parser.add_argument("--figure-root", type=Path, default=None)
    parser.add_argument("--model-root", type=Path, default=None)

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
    parser.add_argument("--vel-symmetric", action="store_true",
                        help="Use symmetric velocity domain [-vmax, vmax] instead of [0, vmax]")
    parser.add_argument(
        "--action-order",
        choices=ORDERING_NAMES,
        default="local",
        help="TT action-mode order. 'local' preserves the original TTPI HardMove layout.",
    )
    parser.add_argument(
        "--env-variant",
        choices=["standard", "index_permuted", "actuator_relabelled", "cross_coupled"],
        default="standard",
        help="HardMove stress-test variant.",
    )
    parser.add_argument("--env-permutation-seed", type=int, default=2026)
    parser.add_argument("--cross-coupling-strength", type=float, default=0.25)
    parser.add_argument("--order-random-seed", type=int, default=42)
    parser.add_argument("--pam-pair-weight", type=float, default=10.0)
    parser.add_argument("--pam-neighbor-weight", type=float, default=1.0)
    parser.add_argument("--pam-opposite-weight", type=float, default=2.0)


    args = parser.parse_args()
    training_seed = args.training_seed if args.training_seed is not None else args.seed
    environment_seed = args.environment_seed if args.environment_seed is not None else args.env_permutation_seed
    permutation_seed = args.permutation_seed if args.permutation_seed is not None else args.order_random_seed
    state_sampling_seed = args.state_sampling_seed if args.state_sampling_seed is not None else training_seed
    args.seed = training_seed
    args.env_permutation_seed = environment_seed
    args.order_random_seed = permutation_seed
    env_variant_canonical = canonical_env_variant(args.env_variant)

    if args.action_order == "opposite_pair":
        warnings.warn(
            "'opposite_pair' is a legacy alias for opposite_interleave_legacy; "
            "use reverse_blocks or flip_within_block for the formal diagnostic controls.",
            DeprecationWarning,
            stacklevel=2,
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but --device cuda was requested.")

    device = torch.device(args.device)
    seed_everything(training_seed)

    out_dir = args.output_root if args.output_root is not None else ROOT / "repro" / "results"
    diagnostics_dir = (
        args.diagnostics_root if args.diagnostics_root is not None else ROOT / "repro" / "diagnostics"
    )
    fig_dir = args.figure_root if args.figure_root is not None else ROOT / "repro" / "figures" / "hardmove"
    model_dir = args.model_root if args.model_root is not None else ROOT / "repro" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    env_name = f"HM{args.n_actuator}"
    if args.env_variant != "standard":
        env_name = f"{env_name}_{args.env_variant}"
    perm_seed_suffix = f"_permseed{permutation_seed}" if args.include_permutation_seed_in_run_id else ""
    run_tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
    task_name = (
        f"{env_name}"
        f"_state{args.n_state}"
        f"_action{args.n_action}"
        f"_iter{args.n_iter}"
        f"_order{args.action_order}"
        f"_seed{args.seed}"
        f"{perm_seed_suffix}"
        f"{run_tag_suffix}"
    )

    L = 1.0
    position_max = L
    position_min = -L

    velocity_max = 0.25 * position_max
    velocity_min = -1.0 * velocity_max if args.vel_symmetric else 0.0 * velocity_max
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

    domain_action_phys = [domain_acc0, domain_switch0] * args.n_actuator
    ordering_t0 = time.time()
    coupling, order_results = build_hardmove_orders(
        n_actuator=args.n_actuator,
        random_seed=permutation_seed,
        pair_weight=args.pam_pair_weight,
        neighbor_weight=args.pam_neighbor_weight,
        opposite_weight=args.pam_opposite_weight,
    )
    order_info = order_results[args.action_order]
    action_order = order_info.order
    action_inverse = torch.tensor(
        [action_order.index(i) for i in range(2 * args.n_actuator)],
        device=device,
        dtype=torch.long,
    )
    domain_action = [domain_action_phys[i] for i in action_order]
    ordering_search_time_sec = time.time() - ordering_t0

    dyn_system = HardMove(
        dt=args.dt,
        w_goal=1e3,
        w_action=1e4,
        n=args.n_actuator,
        device=device,
    )

    env_permutation = list(range(args.n_actuator))
    if env_variant_canonical == "actuator_relabelled":
        env_permutation = make_env_permutation(args.n_actuator, args.env_permutation_seed)

    def transform_physical_action(action_phys):
        return transform_hardmove_action(
            action_phys=action_phys,
            n_actuator=args.n_actuator,
            variant=args.env_variant,
            permutation=env_permutation,
            cross_coupling_strength=args.cross_coupling_strength,
        )

    def forward_model(state, action):
        return dyn_system.forward_simulate(state, transform_physical_action(action[..., action_inverse]))

    def reward(state, action):
        return dyn_system.reward_state_action(state, transform_physical_action(action[..., action_inverse]))

    init_state = make_init_states(
        n_test=args.n_test,
        state_min=state_min,
        state_max=state_max,
        device=device,
        seed=state_sampling_seed,
    )

    eval_history = []

    best_eval = {"metrics": None}
    best_tradeoff = {"metrics": None, "S_times_mu": -1.0}

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
            action_inverse=action_inverse,
            action_transform=transform_physical_action,
        )
        metrics["callback_count"] = int(callback_count)
        metrics["S_times_mu"] = metrics["success_rate"] * metrics["mu_success"]
        eval_history.append(metrics)

        if is_better_metric(metrics, best_eval["metrics"]):
            best_eval["metrics"] = dict(metrics)

        tradeoff = metrics["S_times_mu"]
        if tradeoff > best_tradeoff["S_times_mu"]:
            best_tradeoff["metrics"] = dict(metrics)
            best_tradeoff["S_times_mu"] = tradeoff

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
    ttpi.plt_training_stat = lambda: None  # fix: avoid matplotlib crash

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    training_t0 = time.time()
    stopped_early = False
    status = "ok"
    oom = False
    error_type = ""
    error_message = ""
    caught_exception = None

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
    except torch.OutOfMemoryError as exc:
        status = "oom"
        oom = True
        error_type = type(exc).__name__
        error_message = str(exc)
        print(f"[OOM] {error_message}", flush=True)
    except Exception as exc:
        status = "error"
        error_type = type(exc).__name__
        error_message = str(exc)
        caught_exception = exc
        print(f"[ERROR] {error_type}: {error_message}", flush=True)

    ttpi_training_time_sec = time.time() - training_t0
    train_time_sec = ttpi_training_time_sec
    total_time_sec = ordering_search_time_sec + ttpi_training_time_sec
    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    )

    if status == "ok":
        final_metrics = evaluate_policy(
            ttpi=ttpi,
            dyn_system=dyn_system,
            init_state=init_state,
            dt=args.dt,
            horizon_sec=10.0,
            target_radius=args.target_radius,
            action_inverse=action_inverse,
            action_transform=transform_physical_action,
        )
        best_metrics = select_best_eval(eval_history)
    else:
        final_metrics = {}
        best_metrics = None

    # Paper-aligned metrics: four reporting conventions from eval_history
    paper_metrics = {}
    if status == "ok" and eval_history:
        # 1. Best S (ignore mu)
        best_S_entry = max(eval_history, key=lambda m: m.get("success_rate", 0))
        paper_metrics["best_S"] = {
            "S": best_S_entry.get("success_rate", 0),
            "mu": best_S_entry.get("mu_success", 0),
            "S_times_mu": best_S_entry.get("success_rate", 0) * best_S_entry.get("mu_success", 0),
            "callback": best_S_entry.get("callback_count", -1),
        }
        # 2. Best S, then best mu among those with S == best_S
        max_S = paper_metrics["best_S"]["S"]
        best_S_then_mu = max(
            [e for e in eval_history if abs(e.get("success_rate", 0) - max_S) < 1e-6],
            key=lambda m: m.get("mu_success", 0),
            default=best_S_entry,
        )
        paper_metrics["best_S_then_mu"] = {
            "S": best_S_then_mu.get("success_rate", 0),
            "mu": best_S_then_mu.get("mu_success", 0),
            "S_times_mu": best_S_then_mu.get("success_rate", 0) * best_S_then_mu.get("mu_success", 0),
            "callback": best_S_then_mu.get("callback_count", -1),
        }
        # 3. Best S × mu (tradeoff)
        best_to_entry = max(
            eval_history,
            key=lambda m: m.get("tradeoff_score", m.get("success_rate", 0) * m.get("mu_success", 0)),
        )
        paper_metrics["best_S_times_mu"] = {
            "S": best_to_entry.get("success_rate", 0),
            "mu": best_to_entry.get("mu_success", 0),
            "S_times_mu": best_to_entry.get("success_rate", 0) * best_to_entry.get("mu_success", 0),
            "callback": best_to_entry.get("callback_count", -1),
        }

    fig_path = fig_dir / f"{task_name}_traj.png"

    traj_fig = None
    if status == "ok":
        traj_fig = save_trajectory_figure(
            ttpi=ttpi,
            dyn_system=dyn_system,
            init_state=init_state,
            dt=args.dt,
            fig_path=fig_path,
            horizon_sec=10.0,
            max_traj=min(args.n_test, 20),
            action_inverse=action_inverse,
            action_transform=transform_physical_action,
        )

    result = {
        "run_id": task_name,
        "task": f"HM({args.n_actuator})",
        "env_name": env_name,
        "env_variant": args.env_variant,
        "env_variant_canonical": env_variant_canonical,
        "env_variant_display": variant_display_name(args.env_variant),
        "env_permutation": env_permutation,
        "cross_coupling_strength": args.cross_coupling_strength,
        "seed": args.seed,
        "training_seed": training_seed,
        "environment_seed": environment_seed,
        "permutation_seed": permutation_seed,
        "state_sampling_seed": state_sampling_seed,
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
        "vel_symmetric": args.vel_symmetric,
        "state_min": [float(x) for x in state_min.detach().cpu().tolist()],
        "state_max": [float(x) for x in state_max.detach().cpu().tolist()],
        "velocity_min": float(velocity_min),
        "velocity_max": float(velocity_max),
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
        "action_order_name": args.action_order,
        "canonical_ordering": order_info.metadata.get("canonical_name", args.action_order),
        "display_ordering": order_info.metadata.get("display_name", args.action_order),
        "baseline_category": order_info.metadata.get("baseline_category"),
        "deprecated_ordering_alias": order_info.metadata.get("deprecated_alias", False),
        "action_order": action_order,
        "action_inverse": [int(x) for x in action_inverse.detach().cpu().tolist()],
        "pam_order_objective": order_info.objective,
        "pam_order_adjacency_score": order_info.adjacency_score,
        "pam_order_peak_cut_objective": order_info.metadata.get("peak_cut_objective"),
        "pam_order_rankaware_proxy_objective": order_info.metadata.get("rankaware_proxy_objective"),
        "pam_order_block_preserving": order_info.metadata.get("block_preserving"),
        "pam_order_metadata": order_info.metadata,
        "pam_order_weights": {
            "pair_weight": args.pam_pair_weight,
            "neighbor_weight": args.pam_neighbor_weight,
            "opposite_weight": args.pam_opposite_weight,
        },
        "pam_all_order_metrics": {
            name: {
                "order": result.order,
                "objective": result.objective,
                "adjacency_score": result.adjacency_score,
                "peak_cut_objective": result.metadata.get("peak_cut_objective"),
                "rankaware_proxy_objective": result.metadata.get("rankaware_proxy_objective"),
                "block_preserving": result.metadata.get("block_preserving"),
                "canonical_name": result.metadata.get("canonical_name"),
                "display_name": result.metadata.get("display_name"),
                "baseline_category": result.metadata.get("baseline_category"),
                "deprecated_alias": result.metadata.get("deprecated_alias"),
            }
            for name, result in order_results.items()
        },
        "train_time_sec": train_time_sec,
        "ordering_search_time_sec": ordering_search_time_sec,
        "ttpi_training_time_sec": ttpi_training_time_sec,
        "total_time_sec": total_time_sec,
        "final_metrics": final_metrics,
        "eval_history": eval_history,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "max_cuda_mem_gb": (
            torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
        ),
        "peak_memory_mb": peak_memory_mb,
        "trajectory_figure": traj_fig,
        "best_metrics": best_metrics,
        "best_tradeoff_metrics": best_tradeoff["metrics"],
        "paper_metrics": paper_metrics,
        "status": status,
        "oom": oom,
        "error_type": error_type,
        "error_message": error_message,
        "stopped_early": stopped_early,
        "early_stop_success": args.early_stop_success,
        "early_stop_mu": args.early_stop_mu,
        "early_stop_after_callback": args.early_stop_after_callback,
        "train_data": tensor_safe(ttpi.train_data),
        "diagnostics": tensor_safe(ttpi.diagnostics),
    }

    out_file = out_dir / f"{task_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(tensor_safe(result), f, indent=2, ensure_ascii=False)

    rank_rows = rank_profile_rows(
        task_name=task_name,
        seed=args.seed,
        env_name=env_name,
        ordering_name=args.action_order,
        train_data=ttpi.train_data,
    )
    cross_rows = cross_query_rows(
        task_name=task_name,
        seed=args.seed,
        env_name=env_name,
        ordering_name=args.action_order,
        diagnostics=ttpi.diagnostics,
    )
    cross_process = cross_process_rows(
        task_name=task_name,
        seed=args.seed,
        env_name=env_name,
        ordering_name=args.action_order,
        diagnostics=ttpi.diagnostics,
    )
    round_rows = round_event_rows(
        task_name=task_name,
        seed=args.seed,
        env_name=env_name,
        ordering_name=args.action_order,
        diagnostics=ttpi.diagnostics,
    )
    rank_csv = diagnostics_dir / "rank_profiles" / f"{task_name}.rank_profile.csv"
    cross_csv = diagnostics_dir / "cross_queries" / f"{task_name}.cross_queries.csv"
    cross_process_csv = diagnostics_dir / "cross_process" / f"{task_name}.cross_process.csv"
    round_csv = diagnostics_dir / "round_events" / f"{task_name}.round_events.csv"
    summary_csv = diagnostics_dir / "pam_ablation_summary.csv"
    write_csv(rank_csv, rank_rows)
    write_csv(cross_csv, cross_rows)
    write_csv(cross_process_csv, cross_process)
    write_csv(round_csv, round_rows)
    append_csv(
        summary_csv,
        summary_row(
            result=result,
            train_data=ttpi.train_data,
            diagnostics=ttpi.diagnostics,
            peak_memory_mb=peak_memory_mb,
        ),
    )
    standard_log = standard_run_log(
        run_id=task_name,
        result=result,
        train_data=ttpi.train_data,
        diagnostics=ttpi.diagnostics,
        peak_memory_mb=peak_memory_mb,
        status=status,
    )
    standard_file = diagnostics_dir / "run_json" / f"{task_name}.standard.json"
    standard_file.parent.mkdir(parents=True, exist_ok=True)
    standard_file.write_text(
        json.dumps(tensor_safe(standard_log), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[DONE] saved result to {out_file}")
    print(f"[DONE] saved rank diagnostics to {rank_csv}")
    print(f"[DONE] saved TT-Cross diagnostics to {cross_csv}")
    print(f"[DONE] saved TT-Cross process diagnostics to {cross_process_csv}")
    print(f"[DONE] saved TT-Round diagnostics to {round_csv}")
    print(f"[DONE] saved standard JSON log to {standard_file}")
    print(f"[DONE] appended summary to {summary_csv}")
    print(json.dumps(result["final_metrics"], indent=2, ensure_ascii=False))
    if caught_exception is not None:
        raise caught_exception

if __name__ == "__main__":
    main()
