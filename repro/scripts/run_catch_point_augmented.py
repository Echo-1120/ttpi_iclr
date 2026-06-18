"""Parameter-augmented Catch-Point experiment.

This script keeps the original CP state semantics but uses a hybrid action more
aligned with the benchmark wording:

  state_aug = [x, y, vx, vy, wind]
  action    = [footstep_id, force]

The wind parameter is persistent and affects the x acceleration. This gives a
7-mode TTPI domain and serves as a fast low-dimensional PAM/robustness sandbox.

Example smoke run:
  /home/s110/miniconda3/envs/tt_5080/bin/python \
    repro/scripts/run_catch_point_augmented.py --preset smoke --device cpu
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ttpi import TTPI


torch.set_default_dtype(torch.float64)

RESULTS = ROOT / "repro" / "results"
FIGURES = ROOT / "repro" / "figures" / "catchpoint_augmented"

STATE_ORDERS = {
    "baseline": ["x", "y", "vx", "vy", "wind"],
    "pam_local": ["x", "vx", "wind", "y", "vy"],
    "bad_locality": ["wind", "y", "vy", "x", "vx"],
}
ACTION_ORDER = ["footstep", "force"]


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_arg(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_domains(args, state_order, device):
    domains = {
        "x": torch.linspace(-args.pos_max, args.pos_max, args.n_state, device=device),
        "y": torch.linspace(-args.pos_max, args.pos_max, args.n_state, device=device),
        "vx": torch.linspace(-args.vel_max, args.vel_max, args.n_velocity, device=device),
        "vy": torch.linspace(-args.vel_max, args.vel_max, args.n_velocity, device=device),
        "wind": torch.linspace(-args.wind_max, args.wind_max, args.n_param, device=device),
        "footstep": torch.arange(args.n_footsteps, dtype=torch.float64, device=device),
        "force": torch.linspace(0.0, args.force_max, args.n_force, device=device),
    }
    return domains, [domains[m] for m in state_order], [domains[m] for m in ACTION_ORDER]


def decode(values, order):
    return {name: values[:, i] for i, name in enumerate(order)}


def encode(v, order):
    return torch.stack([v[name] for name in order], dim=-1)


def footstep_angle(footstep, n_footsteps):
    return 2.0 * math.pi * footstep.round().clamp(0, n_footsteps - 1) / float(n_footsteps)


def make_dynamics(args, state_order):
    def forward_model(state, action):
        s = decode(state, state_order)
        a = decode(action, ACTION_ORDER)
        theta = footstep_angle(a["footstep"], args.n_footsteps)
        force = a["force"]

        ax = force * torch.cos(theta) + s["wind"] - args.damping * s["vx"]
        ay = force * torch.sin(theta) - args.damping * s["vy"]

        vx_new = torch.clamp(s["vx"] + ax * args.dt, -args.vel_max, args.vel_max)
        vy_new = torch.clamp(s["vy"] + ay * args.dt, -args.vel_max, args.vel_max)
        x_new = torch.clamp(s["x"] + vx_new * args.dt, -args.pos_max, args.pos_max)
        y_new = torch.clamp(s["y"] + vy_new * args.dt, -args.pos_max, args.pos_max)
        next_s = {
            "x": x_new,
            "y": y_new,
            "vx": vx_new,
            "vy": vy_new,
            "wind": s["wind"],
        }
        return encode(next_s, state_order)

    def reward(state, action):
        s = decode(state, state_order)
        next_state = forward_model(state, action)
        sn = decode(next_state, state_order)
        pos = torch.linalg.norm(torch.stack((s["x"], s["y"]), dim=-1), dim=-1)
        next_pos = torch.linalg.norm(torch.stack((sn["x"], sn["y"]), dim=-1), dim=-1)
        vel = torch.linalg.norm(torch.stack((sn["vx"], sn["vy"]), dim=-1), dim=-1)
        force = action[:, 1]
        progress_bonus = args.w_progress * (pos - next_pos)
        cost = (
            args.w_goal * next_pos**2
            + args.w_velocity * vel**2
            + args.w_force * force**2
            - progress_bonus
        )
        return -cost

    return forward_model, reward


@torch.no_grad()
def make_init_states(args, state_order, domains, device, seed, offgrid=False):
    g = torch.Generator(device="cpu")
    g.manual_seed(21000 + seed + (1000 if offgrid else 0))
    values = {}
    for name in state_order:
        if name in {"vx", "vy"}:
            values[name] = torch.zeros(args.n_test, dtype=torch.float64)
        elif name == "wind":
            if offgrid:
                values[name] = (
                    torch.rand(args.n_test, generator=g, dtype=torch.float64) * 2.0 - 1.0
                ) * args.wind_test_max
            else:
                d = domains[name].detach().cpu()
                idx = torch.randint(0, len(d), (args.n_test,), generator=g)
                values[name] = d[idx]
        else:
            d = domains[name].detach().cpu()
            r = torch.rand(args.n_test, generator=g, dtype=torch.float64).clamp(0.25, 0.75)
            values[name] = d[0] + r * (d[-1] - d[0])
    return encode(values, state_order).to(device)


@torch.no_grad()
def evaluate_policy(model, init_state, state_order, forward_model, reward, args):
    state = init_state.clone()
    n = state.shape[0]
    active = torch.ones(n, dtype=torch.bool, device=state.device)
    success = torch.zeros(n, dtype=torch.bool, device=state.device)
    path_len = torch.zeros(n, dtype=torch.float64, device=state.device)
    cum_reward = torch.zeros(n, dtype=torch.float64, device=state.device)
    prev_pos = torch.stack((decode(state, state_order)["x"], decode(state, state_order)["y"]), dim=-1)
    final_reward = None

    for _ in range(int(args.horizon / args.dt)):
        action = model.policy(state)
        final_reward = reward(state, action)
        cum_reward += final_reward
        state = forward_model(state, action)
        s = decode(state, state_order)
        pos = torch.stack((s["x"], s["y"]), dim=-1)
        step_len = torch.linalg.norm(pos - prev_pos, dim=-1)
        path_len += active.to(torch.float64) * step_len
        dist = torch.linalg.norm(pos, dim=-1)
        newly_success = active & (dist <= args.target_radius)
        success |= newly_success
        active &= ~newly_success
        prev_pos = pos

    s = decode(state, state_order)
    final_dist = torch.linalg.norm(torch.stack((s["x"], s["y"]), dim=-1), dim=-1)
    mu_success = torch.where(
        success,
        1.0 / (1.0 + path_len),
        torch.zeros_like(path_len),
    )
    return {
        "success_rate": float(success.to(torch.float64).mean().cpu()),
        "mu_success": float(mu_success[success].mean().cpu()) if success.any() else 0.0,
        "tradeoff_score": float((success.to(torch.float64).mean() * (mu_success[success].mean() if success.any() else 0.0)).cpu()),
        "final_dist_mean": float(final_dist.mean().cpu()),
        "path_len_mean": float(path_len.mean().cpu()),
        "cum_reward_mean": float(cum_reward.mean().cpu()),
        "final_reward_mean": float(final_reward.mean().cpu()) if final_reward is not None else None,
    }


@torch.no_grad()
def save_trajectory_figure(model, init_state, state_order, forward_model, args, path):
    state = init_state.clone()
    n_plot = min(args.max_plot_traj, state.shape[0])
    traj = []
    for _ in range(int(args.horizon / args.dt) + 1):
        s = decode(state, state_order)
        traj.append(torch.stack((s["x"][:n_plot], s["y"][:n_plot]), dim=-1).detach().cpu())
        if len(traj) <= int(args.horizon / args.dt):
            state = forward_model(state, model.policy(state))
    traj = torch.stack(traj, dim=0).numpy()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for i in range(n_plot):
        ax.plot(traj[:, i, 0], traj[:, i, 1], linewidth=1.0, alpha=0.8)
        ax.scatter(traj[0, i, 0], traj[0, i, 1], s=10)
    ax.scatter([0.0], [0.0], marker="x", s=80)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-args.pos_max, args.pos_max)
    ax.set_ylim(-args.pos_max, args.pos_max)
    ax.grid(alpha=0.25)
    ax.set_title("Augmented Catch-Point trajectories")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def run_case(case, args, device):
    state_order = STATE_ORDERS[case]
    domains, domain_state, domain_action = build_domains(args, state_order, device)
    forward_model, reward = make_dynamics(args, state_order)
    init_state = make_init_states(args, state_order, domains, device, args.seed, offgrid=False)
    test_state = make_init_states(args, state_order, domains, device, args.seed, offgrid=True)

    model = TTPI(
        domain_state=domain_state,
        domain_action=domain_action,
        reward=reward,
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
        normalize_reward=True,
        verbose=False,
        device=device,
    )
    model.plt_training_stat = lambda: None
    model.save_model = lambda *a, **kw: None

    eval_history = []
    rank_history = []

    def callback(ttpi, callback_count=0):
        nominal = evaluate_policy(ttpi, init_state, state_order, forward_model, reward, args)
        offgrid = evaluate_policy(ttpi, test_state, state_order, forward_model, reward, args)
        nominal["cb"] = int(callback_count)
        offgrid["cb"] = int(callback_count)
        ar = [int(x) for x in ttpi.a_model.ranks_tt.detach().cpu().tolist()]
        vr = [int(x) for x in ttpi.v_model.ranks_tt.detach().cpu().tolist()]
        pr = [int(x) for x in ttpi.policy_model.ranks_tt.detach().cpu().tolist()]
        rank_history.append(
            {
                "cb": int(callback_count),
                "Ar": ar,
                "Vr": vr,
                "Pr": pr,
                "Ar_max": max(ar),
                "Vr_max": max(vr),
                "Pr_max": max(pr),
            }
        )
        eval_history.append({"nominal": nominal, "offgrid": offgrid})
        print(
            f"  {case} cb{callback_count}: nominal S={nominal['success_rate']:.3f} "
            f"offgrid S={offgrid['success_rate']:.3f} Ar={max(ar)} Pr={max(pr)}",
            flush=True,
        )
        if (
            args.early_stop_success is not None
            and callback_count >= args.early_stop_after_callback
            and nominal["success_rate"] >= args.early_stop_success
            and offgrid["success_rate"] >= args.early_stop_success
        ):
            raise StopIteration
        return torch.tensor(nominal["final_reward_mean"], device=device), torch.tensor(
            nominal["cum_reward_mean"], device=device
        )

    t0 = time.time()
    status = "OK"
    try:
        model.train(
            n_iter_max=args.n_iter,
            n_iter_v=args.n_iter_v,
            resume=False,
            callback=callback,
            callback_freq=args.callback_freq,
            verbose=False,
            file_name=None,
        )
    except StopIteration:
        status = "EARLY_STOP"
    except torch.OutOfMemoryError:
        status = "OOM"
    elapsed = time.time() - t0

    final_nominal = evaluate_policy(model, init_state, state_order, forward_model, reward, args)
    final_offgrid = evaluate_policy(model, test_state, state_order, forward_model, reward, args)
    fig_path = FIGURES / f"catch_point_augmented_{args.preset}_{case}_seed{args.seed}.png"
    traj_fig = save_trajectory_figure(model, test_state, state_order, forward_model, args, fig_path)
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
    return {
        "case": case,
        "status": status,
        "state_order": state_order,
        "action_order": ACTION_ORDER,
        "time_s": elapsed,
        "peak_gb": peak_gb,
        "final_nominal": final_nominal,
        "final_offgrid": final_offgrid,
        "eval_history": eval_history,
        "rank_history": rank_history,
        "trajectory_figure": traj_fig,
    }


def write_summary(path, records):
    fields = [
        "case",
        "status",
        "time_s",
        "peak_gb",
        "nominal_success",
        "offgrid_success",
        "nominal_tradeoff",
        "offgrid_tradeoff",
        "final_dist_offgrid",
        "Ar_max",
        "Pr_max",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            last_rank = r["rank_history"][-1] if r["rank_history"] else {}
            writer.writerow(
                {
                    "case": r["case"],
                    "status": r["status"],
                    "time_s": r["time_s"],
                    "peak_gb": r["peak_gb"],
                    "nominal_success": r["final_nominal"]["success_rate"],
                    "offgrid_success": r["final_offgrid"]["success_rate"],
                    "nominal_tradeoff": r["final_nominal"]["tradeoff_score"],
                    "offgrid_tradeoff": r["final_offgrid"]["tradeoff_score"],
                    "final_dist_offgrid": r["final_offgrid"]["final_dist_mean"],
                    "Ar_max": last_rank.get("Ar_max"),
                    "Pr_max": last_rank.get("Pr_max"),
                }
            )


def apply_preset(args):
    if args.preset == "smoke":
        args.n_state = args.n_state or 15
        args.n_velocity = args.n_velocity or 9
        args.n_param = args.n_param or 5
        args.n_footsteps = args.n_footsteps or 8
        args.n_force = args.n_force or 15
        args.n_iter = args.n_iter or 2
        args.callback_freq = args.callback_freq or 1
        args.n_test = args.n_test or 12
        args.horizon = args.horizon or 1.0
        args.nswp_v = args.nswp_v or 3
        args.nswp_a = args.nswp_a or 3
        args.rmax_v = args.rmax_v or 30
        args.rmax_a = args.rmax_a or 30
        args.max_batch_v = args.max_batch_v or 5000
        args.max_batch_a = args.max_batch_a or 12000
        args.n_samples = args.n_samples or 16
    elif args.preset == "pilot":
        args.n_state = args.n_state or 35
        args.n_velocity = args.n_velocity or 21
        args.n_param = args.n_param or 7
        args.n_footsteps = args.n_footsteps or 16
        args.n_force = args.n_force or 35
        args.n_iter = args.n_iter or 20
        args.callback_freq = args.callback_freq or 2
        args.n_test = args.n_test or 50
        args.horizon = args.horizon or 6.0
        args.nswp_v = args.nswp_v or 5
        args.nswp_a = args.nswp_a or 5
        args.rmax_v = args.rmax_v or 60
        args.rmax_a = args.rmax_a or 60
        args.max_batch_v = args.max_batch_v or 10000
        args.max_batch_a = args.max_batch_a or 30000
        args.n_samples = args.n_samples or 32
    else:
        raise ValueError(args.preset)
    return args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["smoke", "pilot"], default="smoke")
    parser.add_argument("--cases", default="baseline,pam_local,bad_locality")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-stem", default=None)

    parser.add_argument("--n-state", type=int, default=None)
    parser.add_argument("--n-velocity", type=int, default=None)
    parser.add_argument("--n-param", type=int, default=None)
    parser.add_argument("--n-footsteps", type=int, default=None)
    parser.add_argument("--n-force", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--pos-max", type=float, default=1.0)
    parser.add_argument("--vel-max", type=float, default=0.45)
    parser.add_argument("--force-max", type=float, default=1.2)
    parser.add_argument("--wind-max", type=float, default=0.25)
    parser.add_argument("--wind-test-max", type=float, default=0.35)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--horizon", type=float, default=None)
    parser.add_argument("--target-radius", type=float, default=0.05)
    parser.add_argument("--damping", type=float, default=0.15)
    parser.add_argument("--w-goal", type=float, default=25.0)
    parser.add_argument("--w-velocity", type=float, default=0.5)
    parser.add_argument("--w-force", type=float, default=0.02)
    parser.add_argument("--w-progress", type=float, default=8.0)
    parser.add_argument("--max-plot-traj", type=int, default=16)

    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--n-iter", type=int, default=None)
    parser.add_argument("--n-iter-v", type=int, default=1)
    parser.add_argument("--callback-freq", type=int, default=None)
    parser.add_argument("--max-batch-v", type=int, default=None)
    parser.add_argument("--max-batch-a", type=int, default=None)
    parser.add_argument("--nswp-v", type=int, default=None)
    parser.add_argument("--nswp-a", type=int, default=None)
    parser.add_argument("--rmax-v", type=int, default=None)
    parser.add_argument("--rmax-a", type=int, default=None)
    parser.add_argument("--kickrank-v", type=int, default=4)
    parser.add_argument("--kickrank-a", type=int, default=4)
    parser.add_argument("--eps-cross-v", type=float, default=1e-3)
    parser.add_argument("--eps-cross-a", type=float, default=1e-3)
    parser.add_argument("--eps-round-v", type=float, default=1e-3)
    parser.add_argument("--eps-round-a", type=float, default=1e-3)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--early-stop-success", type=float, default=None)
    parser.add_argument("--early-stop-after-callback", type=int, default=0)
    return apply_preset(parser.parse_args())


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = device_from_arg(args.device)
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    requested = [c.strip() for c in args.cases.split(",") if c.strip()]
    unknown = [c for c in requested if c not in STATE_ORDERS]
    if unknown:
        raise ValueError(f"Unknown cases: {unknown}. Known: {sorted(STATE_ORDERS)}")

    records = []
    for case in requested:
        print(f"[catch-point-augmented] Running {case}", flush=True)
        records.append(run_case(case, args, device))

    stem = args.output_stem or f"catch_point_augmented_{args.preset}_seed{args.seed}"
    json_path = RESULTS / f"{stem}.json"
    csv_path = RESULTS / f"{stem}.summary.csv"
    payload = {
        "script": str(Path(__file__).resolve()),
        "preset": args.preset,
        "seed": args.seed,
        "device": str(device),
        "config": vars(args),
        "notes": [
            "Parameter-augmented CP with persistent wind.",
            "Action modes are [discrete footstep_id, continuous force].",
            "This is a low-dimensional robustness/PAM diagnostic, not the old heading+move_flag CP baseline.",
        ],
        "records": records,
        "summary_csv": str(csv_path),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary(csv_path, records)
    print(f"[catch-point-augmented] Wrote {json_path}")
    print(f"[catch-point-augmented] Wrote {csv_path}")


if __name__ == "__main__":
    main()
