"""Parameter-augmented planar pushing TTPI smoke experiment.

This script is intentionally conservative: it keeps TTPI's native
``domain_state + domain_action`` layout and only reorders modes within the state
block. Full cross-block PAM interleaving is covered by
``run_planar_pushing_pam_proxy.py`` until TTPI supports arbitrary conditional
mode sets in policy optimization.

Example:
  /home/s110/miniconda3/envs/tt_5080/bin/python \
    repro/scripts/run_planar_pushing_augmented_ttpi.py --preset smoke --device cpu
"""
import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ttpi import TTPI
from repro.scripts.run_planar_pushing_pam_proxy import (
    ACTION_MODES,
    CONTACT_TOL,
    PUSHER_X,
    augmented_pushing_next,
)


torch.set_default_dtype(torch.float64)

RESULTS = ROOT / "repro" / "results"

STATE_ORDERS = {
    "baseline": [
        "slider_x",
        "slider_y",
        "theta",
        "pusher_x",
        "pusher_y",
        "current_face",
        "mass",
        "friction",
    ],
    "contact_state": [
        "slider_x",
        "slider_y",
        "current_face",
        "theta",
        "pusher_x",
        "pusher_y",
        "friction",
        "mass",
    ],
}


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_arg(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def mode_domains(args, device):
    w = args.workspace
    return {
        "slider_x": torch.linspace(-w, w, args.n_state, device=device),
        "slider_y": torch.linspace(-w, w, args.n_state, device=device),
        "theta": torch.linspace(-torch.pi, torch.pi, args.n_theta, device=device),
        "pusher_x": torch.linspace(-0.10, -0.065, args.n_pusher, device=device),
        "pusher_y": torch.linspace(-0.06, 0.06, args.n_pusher, device=device),
        "current_face": torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64, device=device),
        "mass": torch.linspace(0.2, 2.0, args.n_param, device=device),
        "friction": torch.linspace(0.1, 0.8, args.n_param, device=device),
        "next_face": torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64, device=device),
        "vx": torch.linspace(0.0, 0.05, args.n_action, device=device),
        "vy": torch.linspace(-0.05, 0.05, args.n_action, device=device),
    }


def decode(values, order):
    return {name: values[:, i] for i, name in enumerate(order)}


def encode_state(v, order):
    return torch.stack([v[name] for name in order], dim=-1)


def state_dict_from_next(next_state, prev_v):
    return {
        "slider_x": next_state[:, 0],
        "slider_y": next_state[:, 1],
        "theta": next_state[:, 2],
        "pusher_x": next_state[:, 3],
        "pusher_y": next_state[:, 4],
        "current_face": next_state[:, 5],
        "mass": prev_v["mass"],
        "friction": prev_v["friction"],
    }


def one_step_cost(v, action, args):
    dyn_v = dict(v)
    dyn_v.update(decode(action, ACTION_MODES))
    dyn_v["dt"] = args.dt
    next_state = augmented_pushing_next(dyn_v)

    current_pos = torch.stack((v["slider_x"], v["slider_y"]), dim=-1)
    next_pos = next_state[:, :2]
    pos_error = 0.5 * (
        torch.linalg.norm(current_pos, dim=-1) + torch.linalg.norm(next_pos, dim=-1)
    ) / (args.position_tol * args.position_scale)
    theta_error = 0.5 * (v["theta"].abs() + next_state[:, 2].abs()) / (
        args.theta_tol * torch.pi
    )
    face_switch = (v["current_face"].round() != action[:, 0].round()).to(torch.float64)
    speed = torch.linalg.norm(action[:, 1:], dim=-1)
    effort = v["mass"] * speed / 0.05
    contact_error = torch.clamp((torch.abs(v["pusher_x"] - PUSHER_X) - CONTACT_TOL) / CONTACT_TOL, min=0.0)
    return (
        0.45 * pos_error
        + 0.35 * theta_error
        + args.face_switch_weight * face_switch
        + args.effort_weight * effort
        + args.contact_weight * contact_error
    )


def build_task(args, state_order, device):
    domains = mode_domains(args, device)
    domain_state = [domains[m] for m in state_order]
    domain_action = [domains[m] for m in ACTION_MODES]

    def forward_model(state, action):
        v = decode(state, state_order)
        dyn_v = dict(v)
        dyn_v.update(decode(action, ACTION_MODES))
        dyn_v["dt"] = args.dt
        next_base = augmented_pushing_next(dyn_v)
        next_v = state_dict_from_next(next_base, v)
        return encode_state(next_v, state_order)

    def reward(state, action):
        v = decode(state, state_order)
        return -one_step_cost(v, action, args)

    return domains, domain_state, domain_action, forward_model, reward


@torch.no_grad()
def make_init_states(args, state_order, domains, device):
    g = torch.Generator(device="cpu")
    g.manual_seed(12000 + args.seed)
    values = {}
    for name in state_order:
        domain = domains[name].detach().cpu()
        if name in {"current_face"}:
            idx = torch.randint(0, len(domain), (args.n_test,), generator=g)
            values[name] = domain[idx]
        else:
            lo = domain[0]
            hi = domain[-1]
            r = torch.rand(args.n_test, generator=g, dtype=torch.float64).clamp(0.25, 0.75)
            values[name] = lo + r * (hi - lo)
    return encode_state(values, state_order).to(device)


@torch.no_grad()
def evaluate_policy(model, init_state, state_order, forward_model, reward, args):
    state = init_state.clone()
    n = state.shape[0]
    success = torch.zeros(n, dtype=torch.bool, device=state.device)
    cum_reward = torch.zeros(n, dtype=torch.float64, device=state.device)
    horizon = int(args.horizon / args.dt)
    final_reward = None

    for _ in range(horizon):
        action = model.policy(state)
        final_reward = reward(state, action)
        cum_reward += final_reward
        state = forward_model(state, action)
        v = decode(state, state_order)
        close = (
            (v["slider_x"].abs() <= args.success_xy)
            & (v["slider_y"].abs() <= args.success_xy)
            & (v["theta"].abs() <= args.success_theta)
        )
        success |= close

    v = decode(state, state_order)
    final_pos = torch.linalg.norm(torch.stack((v["slider_x"], v["slider_y"]), dim=-1), dim=-1)
    return {
        "success_rate": float(success.to(torch.float64).mean().cpu()),
        "final_pos_mean": float(final_pos.mean().cpu()),
        "final_theta_abs_mean": float(v["theta"].abs().mean().cpu()),
        "cum_reward_mean": float(cum_reward.mean().cpu()),
        "final_reward_mean": float(final_reward.mean().cpu()) if final_reward is not None else None,
    }


def run_case(case, args, device):
    state_order = STATE_ORDERS[case]
    domains, domain_state, domain_action, forward_model, reward = build_task(args, state_order, device)
    init_state = make_init_states(args, state_order, domains, device)

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
        normalize_reward=False,
        verbose=False,
        device=device,
    )
    model.plt_training_stat = lambda: None
    model.save_model = lambda *a, **kw: None

    eval_history = []
    rank_history = []

    def callback(ttpi, callback_count=0):
        metrics = evaluate_policy(ttpi, init_state, state_order, forward_model, reward, args)
        metrics["cb"] = int(callback_count)
        eval_history.append(metrics)
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
        print(
            f"  {case} cb{callback_count}: S={metrics['success_rate']:.3f} "
            f"pos={metrics['final_pos_mean']:.3f} Ar={max(ar)} Pr={max(pr)}",
            flush=True,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return torch.tensor(metrics["final_reward_mean"], device=device), torch.tensor(
            metrics["cum_reward_mean"], device=device
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
    except torch.OutOfMemoryError:
        status = "OOM"
    elapsed = time.time() - t0

    final_eval = evaluate_policy(model, init_state, state_order, forward_model, reward, args)
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
    return {
        "case": case,
        "status": status,
        "state_order": state_order,
        "action_order": ACTION_MODES,
        "time_s": elapsed,
        "peak_gb": peak_gb,
        "eval_history": eval_history,
        "rank_history": rank_history,
        "final_eval": final_eval,
    }


def apply_preset(args):
    if args.preset == "smoke":
        # TTPI's default get_tt_max uses top-k=100 during normalization.
        # Keep the first two continuous state modes large enough that the
        # internal deterministic top-k path has at least 100 candidates.
        args.n_state = args.n_state or 12
        args.n_theta = args.n_theta or 9
        args.n_pusher = args.n_pusher or 7
        args.n_action = args.n_action or 7
        args.n_param = args.n_param or 5
        args.n_iter = args.n_iter or 1
        args.callback_freq = args.callback_freq or 1
        args.n_test = args.n_test or 8
        args.horizon = args.horizon or 0.25
        args.nswp_v = args.nswp_v or 2
        args.nswp_a = args.nswp_a or 2
        args.rmax_v = args.rmax_v or 20
        args.rmax_a = args.rmax_a or 20
        args.max_batch_v = args.max_batch_v or 3000
        args.max_batch_a = args.max_batch_a or 6000
        args.n_samples = args.n_samples or 8
    elif args.preset == "pilot":
        args.n_state = args.n_state or 9
        args.n_theta = args.n_theta or 11
        args.n_pusher = args.n_pusher or 9
        args.n_action = args.n_action or 9
        args.n_param = args.n_param or 5
        args.n_iter = args.n_iter or 5
        args.callback_freq = args.callback_freq or 1
        args.n_test = args.n_test or 20
        args.horizon = args.horizon or 0.5
        args.nswp_v = args.nswp_v or 3
        args.nswp_a = args.nswp_a or 3
        args.rmax_v = args.rmax_v or 30
        args.rmax_a = args.rmax_a or 30
        args.max_batch_v = args.max_batch_v or 5000
        args.max_batch_a = args.max_batch_a or 10000
        args.n_samples = args.n_samples or 10
    else:
        raise ValueError(args.preset)
    return args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["smoke", "pilot"], default="smoke")
    parser.add_argument("--cases", default="baseline,contact_state")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-stem", default=None)

    parser.add_argument("--n-state", type=int, default=None)
    parser.add_argument("--n-theta", type=int, default=None)
    parser.add_argument("--n-pusher", type=int, default=None)
    parser.add_argument("--n-action", type=int, default=None)
    parser.add_argument("--n-param", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--workspace", type=float, default=0.2)
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--horizon", type=float, default=None)

    parser.add_argument("--position-tol", type=float, default=0.01)
    parser.add_argument("--position-scale", type=float, default=0.3)
    parser.add_argument("--theta-tol", type=float, default=0.01)
    parser.add_argument("--face-switch-weight", type=float, default=0.15)
    parser.add_argument("--effort-weight", type=float, default=0.04)
    parser.add_argument("--contact-weight", type=float, default=0.03)
    parser.add_argument("--success-xy", type=float, default=0.03)
    parser.add_argument("--success-theta", type=float, default=15.0 / 180.0 * torch.pi)

    parser.add_argument("--gamma", type=float, default=0.99)
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
    return apply_preset(parser.parse_args())


def main():
    args = parse_args()
    device = device_from_arg(args.device)
    seed_everything(args.seed)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    requested = [c.strip() for c in args.cases.split(",") if c.strip()]
    unknown = [c for c in requested if c not in STATE_ORDERS]
    if unknown:
        raise ValueError(f"Unknown cases: {unknown}. Known: {sorted(STATE_ORDERS)}")

    records = []
    for case in requested:
        print(f"[planar-pushing-aug-ttpi] Running {case}", flush=True)
        records.append(run_case(case, args, device))

    stem = args.output_stem or f"planar_pushing_augmented_ttpi_{args.preset}_seed{args.seed}"
    out_path = RESULTS / f"{stem}.json"
    payload = {
        "script": str(Path(__file__).resolve()),
        "preset": args.preset,
        "seed": args.seed,
        "device": str(device),
        "config": vars(args),
        "notes": [
            "TTPI controller smoke for parameter-augmented state.",
            "This keeps state modes before action modes; it is not full cross-block PAM.",
            "mass is persistent and enters the effort proxy; friction enters contact dynamics.",
        ],
        "records": records,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[planar-pushing-aug-ttpi] Wrote {out_path}")


if __name__ == "__main__":
    main()
