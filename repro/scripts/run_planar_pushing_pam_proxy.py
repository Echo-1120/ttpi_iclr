"""Planar pushing PAM ordering proxy experiment.

This is the first reproducible step for the parameter-augmented planar pushing
benchmark. It does not train a full TTPI controller yet. Instead, it compares
TT-Cross ranks for the same augmented one-step pushing objective under different
mode orderings.

The original TTPI implementation keeps modes as ``domain_state + domain_action``.
That is correct for controller training, but it makes arbitrary state/parameter/
action interleaving a larger algorithmic change. This proxy isolates the
ordering question by giving TT-Cross the same scalar function with different
domain orders.

Example:
  /home/s110/miniconda3/envs/tt_5080/bin/python \
    repro/scripts/run_planar_pushing_pam_proxy.py --preset smoke
"""
import argparse
import csv
import gc
import json
import math
import random
import resource
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tt_utils import cross_approximate


torch.set_default_dtype(torch.float64)

RESULTS = ROOT / "repro" / "results"
FIGURES = ROOT / "repro" / "figures" / "planar_pushing"

SLIDER_R = 0.12 / 2
PUSHER_R = 0.01 / 2
PUSHER_X = -SLIDER_R - PUSHER_R
CONTACT_TOL = 5e-3

CANONICAL_MODES = [
    "slider_x",
    "slider_y",
    "theta",
    "pusher_x",
    "pusher_y",
    "current_face",
    "mass",
    "friction",
    "next_face",
    "vx",
    "vy",
]

STATE_MODES = ["slider_x", "slider_y", "theta", "pusher_x", "pusher_y", "current_face"]
PARAM_MODES = ["mass", "friction"]
ACTION_MODES = ["next_face", "vx", "vy"]

ORDERINGS = {
    "baseline_spa": STATE_MODES + PARAM_MODES + ACTION_MODES,
    "pam_local": [
        "slider_x",
        "slider_y",
        "current_face",
        "next_face",
        "theta",
        "mass",
        "pusher_x",
        "pusher_y",
        "friction",
        "vx",
        "vy",
    ],
    "bad_locality_only": [
        "mass",
        "friction",
        "vx",
        "vy",
        "next_face",
        "theta",
        "slider_x",
        "slider_y",
        "pusher_x",
        "pusher_y",
        "current_face",
    ],
    "bad_split": [
        "theta",
        "slider_x",
        "vx",
        "mass",
        "pusher_y",
        "next_face",
        "slider_y",
        "friction",
        "current_face",
        "pusher_x",
        "vy",
    ],
    "face_local_only": [
        "slider_x",
        "slider_y",
        "theta",
        "friction",
        "mass",
        "vx",
        "vy",
        "pusher_x",
        "pusher_y",
        "current_face",
        "next_face",
    ],
    "pam_contact_local": [
        "slider_x",
        "slider_y",
        "mass",
        "current_face",
        "next_face",
        "theta",
        "pusher_x",
        "pusher_y",
        "friction",
        "vx",
        "vy",
    ],
    "pam_theta_contact": [
        "slider_x",
        "slider_y",
        "current_face",
        "next_face",
        "theta",
        "pusher_x",
        "pusher_y",
        "friction",
        "vx",
        "vy",
        "mass",
    ],
}


def validate_orderings():
    canonical = sorted(CANONICAL_MODES)
    for name, order in ORDERINGS.items():
        if len(order) != len(CANONICAL_MODES):
            raise ValueError(f"{name} has {len(order)} modes; expected {len(CANONICAL_MODES)}")
        if len(set(order)) != len(order):
            raise ValueError(f"{name} contains duplicated modes: {order}")
        if sorted(order) != canonical:
            missing = sorted(set(CANONICAL_MODES) - set(order))
            extra = sorted(set(order) - set(CANONICAL_MODES))
            raise ValueError(f"{name} is not a permutation; missing={missing}, extra={extra}")


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_arg(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def rss_gb():
    # Linux ru_maxrss is KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0**2)


def slider_c_constant(device):
    # Integral over [0, r]^2 of sqrt(x^2+y^2), divided by r^2.
    # Equivalent to r * integral over [0, 1]^2.
    value = SLIDER_R * (math.sqrt(2.0) + math.log(1.0 + math.sqrt(2.0))) / 3.0
    return torch.tensor(value, device=device, dtype=torch.float64)


def make_domains(args, device):
    w = args.workspace
    domains = {
        "slider_x": torch.linspace(-w, w, args.n_state, device=device),
        "slider_y": torch.linspace(-w, w, args.n_state, device=device),
        "theta": torch.linspace(-math.pi, math.pi, args.n_theta, device=device),
        "pusher_x": torch.linspace(-0.10, -0.065, args.n_pusher, device=device),
        "pusher_y": torch.linspace(-0.06, 0.06, args.n_pusher, device=device),
        "current_face": torch.tensor([0.0, 1.0, 2.0, 3.0], device=device),
        "mass": torch.linspace(0.2, 2.0, args.n_param, device=device),
        "friction": torch.linspace(0.1, 0.8, args.n_param, device=device),
        "next_face": torch.tensor([0.0, 1.0, 2.0, 3.0], device=device),
        "vx": torch.linspace(0.0, 0.05, args.n_action, device=device),
        "vy": torch.linspace(-0.05, 0.05, args.n_action, device=device),
    }
    return domains


def decode_ordered(x, order):
    by_name = {}
    for idx, name in enumerate(order):
        by_name[name] = x[:, idx]
    return by_name


def face_to_signed(face):
    # Match pushing_dyn_explicit_double.py:
    # faceid = (faceid - 2) * (faceid < 2) + (faceid - 1) ** (faceid > 1)
    # The boolean exponent contributes 1 for face 0/1, giving:
    # 0 -> -1, 1 -> 0, 2 -> 1, 3 -> 2.
    mapping = torch.tensor([-1.0, 0.0, 1.0, 2.0], device=face.device, dtype=face.dtype)
    return mapping[face.round().long().clamp(0, 3)]


def augmented_pushing_next(v):
    device = v["theta"].device
    bs = v["theta"].shape[0]
    c = slider_c_constant(device)
    px_contact = torch.tensor(PUSHER_X, device=device, dtype=torch.float64)
    dt = v["dt"]

    sx = v["slider_x"]
    sy = v["slider_y"]
    theta = v["theta"].view(-1, 1)
    p_x = v["pusher_x"]
    p_y = v["pusher_y"]
    mu = v["friction"].clamp_min(1e-6).view(-1, 1)
    vn = v["vx"].view(-1, 1)
    vt = v["vy"].view(-1, 1)
    u = torch.cat((vn, vt), dim=-1)

    signed_face = face_to_signed(v["next_face"]).view(-1, 1)
    face_beta = torch.abs(signed_face + 1.0) * math.pi / 2.0

    r_face = torch.empty(bs, 2, 2, device=device, dtype=torch.float64)
    r_face[:, 0, 0] = torch.cos(face_beta).view(-1)
    r_face[:, 1, 1] = torch.cos(face_beta).view(-1)
    r_face[:, 0, 1] = -torch.sin(face_beta).view(-1)
    r_face[:, 1, 0] = torch.sin(face_beta).view(-1)

    r_theta = torch.empty(bs, 2, 2, device=device, dtype=torch.float64)
    r_theta[:, 0, 0] = torch.cos(theta).view(-1)
    r_theta[:, 1, 1] = torch.cos(theta).view(-1)
    r_theta[:, 0, 1] = -torch.sin(theta).view(-1)
    r_theta[:, 1, 0] = torch.sin(theta).view(-1)

    q_mat = torch.empty(bs, 2, 2, device=device, dtype=torch.float64)
    q_mat[:, 0, 0] = c**2 + p_x**2
    q_mat[:, 1, 1] = c**2 + p_y**2
    q_mat[:, 0, 1] = p_x * p_y
    q_mat[:, 1, 0] = p_x * p_y
    q_mat = q_mat / (c**2 + p_x**2 + p_y**2).view(-1, 1, 1)

    gamma_t = (mu * c**2 - px_contact * p_y.view(-1, 1) + mu * px_contact**2) / (
        c**2 + p_y.view(-1, 1) ** 2 - mu * px_contact * p_y.view(-1, 1)
    )
    gamma_b = (-mu * c**2 - px_contact * p_y.view(-1, 1) - mu * px_contact**2) / (
        c**2 + p_y.view(-1, 1) ** 2 + mu * px_contact * p_y.view(-1, 1)
    )

    eye = torch.eye(2, device=device, dtype=torch.float64).view(1, 2, 2).expand(bs, -1, -1)
    zeros_12 = torch.zeros(bs, 1, 2, device=device, dtype=torch.float64)
    p1 = eye
    p2 = torch.cat(
        [eye[:, 0:1], torch.cat([gamma_t, torch.zeros_like(gamma_t)], dim=-1).view(bs, 1, 2)],
        dim=1,
    )
    p3 = torch.cat(
        [eye[:, 0:1], torch.cat([gamma_b, torch.zeros_like(gamma_b)], dim=-1).view(bs, 1, 2)],
        dim=1,
    )

    c1 = torch.zeros(bs, 1, 2, device=device, dtype=torch.float64)
    c2 = torch.cat([-gamma_t, torch.ones_like(gamma_t)], dim=-1).view(bs, 1, 2)
    c3 = torch.cat([-gamma_b, torch.ones_like(gamma_b)], dim=-1).view(bs, 1, 2)

    denom = c**2 + px_contact**2 + p_y**2
    b1 = torch.stack((-p_y / denom, px_contact.expand_as(p_y) / denom), dim=-1).view(bs, 1, 2)
    b2 = torch.stack(
        ((-p_y + gamma_t.view(-1) * px_contact) / denom, torch.zeros_like(p_y)), dim=-1
    ).view(bs, 1, 2)
    b3 = torch.stack(
        ((-p_y + gamma_b.view(-1) * px_contact) / denom, torch.zeros_like(p_y)), dim=-1
    ).view(bs, 1, 2)

    dyn1 = torch.cat([torch.einsum("bij,bjk,bkl->bil", r_theta, q_mat, p1), b1, zeros_12, c1], dim=1)
    dyn2 = torch.cat([torch.einsum("bij,bjk,bkl->bil", r_theta, q_mat, p2), b2, zeros_12, c2], dim=1)
    dyn3 = torch.cat([torch.einsum("bij,bjk,bkl->bil", r_theta, q_mat, p3), b3, zeros_12, c3], dim=1)
    dyn4 = torch.cat(
        [
            torch.zeros(bs, 3, 2, device=device, dtype=torch.float64),
            eye,
        ],
        dim=1,
    )

    cond_stick = ((vt >= gamma_b * vn) & (vt <= gamma_t * vn)).to(torch.float64)
    cond_up = (vt > gamma_t * vn).to(torch.float64)
    cond_down = (vt < gamma_b * vn).to(torch.float64)
    cond_touch = (torch.abs(p_x - px_contact) <= CONTACT_TOL).view(-1, 1).to(torch.float64)
    cond1 = cond_stick * cond_touch
    cond2 = cond_up * cond_touch
    cond3 = cond_down * cond_touch
    cond4 = 1.0 - cond1 - cond2 - cond3

    f = cond1 * torch.einsum("bij,bj->bi", dyn1, u)
    f = f + cond2 * torch.einsum("bij,bj->bi", dyn2, u)
    f = f + cond3 * torch.einsum("bij,bj->bi", dyn3, u)
    f = f + cond4 * torch.einsum("bij,bj->bi", dyn4, u)
    f_xy = torch.einsum("bij,bj->bi", r_face, f[:, :2])
    f = torch.cat((f_xy, f[:, 2:]), dim=-1)

    next_state = torch.empty(bs, 6, device=device, dtype=torch.float64)
    next_state[:, 0] = sx + f[:, 0] * dt
    next_state[:, 1] = sy + f[:, 1] * dt
    next_state[:, 2] = theta.view(-1) + f[:, 2] * dt
    next_state[:, 3] = p_x + f[:, 3] * dt
    next_state[:, 4] = p_y + f[:, 4] * dt
    next_state[:, 5] = v["next_face"]
    return next_state


def make_proxy_function(order, args):
    def proxy(ordered_x):
        v = decode_ordered(ordered_x, order)
        v["dt"] = args.dt
        next_state = augmented_pushing_next(v)

        current_pos = torch.stack((v["slider_x"], v["slider_y"]), dim=-1)
        next_pos = next_state[:, :2]
        current_theta = v["theta"]
        next_theta = next_state[:, 2]

        pos_error = 0.5 * (
            torch.linalg.norm(current_pos, dim=-1) + torch.linalg.norm(next_pos, dim=-1)
        ) / (args.position_tol * args.position_scale)
        theta_error = 0.5 * (current_theta.abs() + next_theta.abs()) / (args.theta_tol * math.pi)
        face_switch = (v["current_face"].round() != v["next_face"].round()).to(torch.float64)
        speed = torch.linalg.norm(torch.stack((v["vx"], v["vy"]), dim=-1), dim=-1)
        effort = v["mass"] * speed / 0.05
        contact_error = torch.clamp((torch.abs(v["pusher_x"] - PUSHER_X) - CONTACT_TOL) / CONTACT_TOL, min=0.0)

        cost = (
            0.45 * pos_error
            + 0.35 * theta_error
            + args.face_switch_weight * face_switch
            + args.effort_weight * effort
            + args.contact_weight * contact_error
        )
        if args.objective == "score":
            return torch.exp(-args.score_temperature * cost)
        if args.objective == "cost":
            return torch.log1p(cost)
        raise ValueError(f"Unknown objective: {args.objective}")

    return proxy


def rank_stats(tt_model):
    ranks = [int(x) for x in tt_model.ranks_tt.detach().cpu().tolist()]
    internal = ranks[1:-1] if len(ranks) > 2 else ranks
    return {
        "ranks": ranks,
        "max_rank": int(max(internal) if internal else max(ranks)),
        "mean_rank": float(sum(internal) / len(internal)) if internal else float(sum(ranks) / len(ranks)),
        "storage": int(tt_model.numcoef()),
    }


def run_case(name, order, domains_by_name, args, device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    start_rss = rss_gb()
    t0 = time.time()
    domain = [domains_by_name[m] for m in order]
    fcn = make_proxy_function(order, args)
    tt_model = cross_approximate(
        fcn=fcn,
        max_batch=args.max_batch,
        domain=domain,
        rmax=args.rmax,
        nswp=args.nswp,
        eps=args.eps,
        kickrank=args.kickrank,
        verbose=args.verbose,
        device=str(device),
    )
    elapsed = time.time() - t0
    stats = rank_stats(tt_model)
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    else:
        peak_gb = max(0.0, rss_gb() - start_rss)
    stats.update(
        {
            "case": name,
            "order": order,
            "time_s": elapsed,
            "peak_memory_gb": peak_gb,
        }
    )
    return stats


def write_csv(path, records):
    fields = [
        "case",
        "max_rank",
        "mean_rank",
        "storage",
        "peak_memory_gb",
        "time_s",
        "ranks",
        "order",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in records:
            out = dict(row)
            out["ranks"] = " ".join(str(x) for x in row["ranks"])
            out["order"] = " ".join(row["order"])
            writer.writerow({k: out[k] for k in fields})


def plot_records(path, records, title):
    labels = [r["case"] for r in records]
    max_rank = [r["max_rank"] for r in records]
    storage = [r["storage"] for r in records]
    memory = [r["peak_memory_gb"] for r in records]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
    axes[0].bar(labels, max_rank, color="#4477aa")
    axes[0].set_ylabel("Max TT rank")
    axes[1].bar(labels, storage, color="#66aa55")
    axes[1].set_ylabel("TT coefficients")
    axes[2].bar(labels, memory, color="#aa6644")
    axes[2].set_ylabel("Peak memory GB")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def apply_preset(args):
    if args.preset == "smoke":
        args.n_state = args.n_state or 7
        args.n_theta = args.n_theta or 9
        args.n_pusher = args.n_pusher or 7
        args.n_action = args.n_action or 7
        args.n_param = args.n_param or 5
        args.nswp = args.nswp or 4
        args.rmax = args.rmax or 32
        args.max_batch = args.max_batch or 8192
    elif args.preset == "proxy":
        args.n_state = args.n_state or 11
        args.n_theta = args.n_theta or 15
        args.n_pusher = args.n_pusher or 11
        args.n_action = args.n_action or 11
        args.n_param = args.n_param or 7
        args.nswp = args.nswp or 8
        args.rmax = args.rmax or 60
        args.max_batch = args.max_batch or 20000
    else:
        raise ValueError(f"Unknown preset: {args.preset}")
    return args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["smoke", "proxy"], default="smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, or auto")
    parser.add_argument(
        "--cases",
        default="baseline_spa,pam_local,pam_contact_local,pam_theta_contact,bad_locality_only,bad_split,face_local_only",
    )
    parser.add_argument("--output-stem", default=None)
    parser.add_argument("--objective", choices=["score", "cost"], default="score")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--n-state", type=int, default=None)
    parser.add_argument("--n-theta", type=int, default=None)
    parser.add_argument("--n-pusher", type=int, default=None)
    parser.add_argument("--n-action", type=int, default=None)
    parser.add_argument("--n-param", type=int, default=None)
    parser.add_argument("--workspace", type=float, default=0.2)

    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--position-tol", type=float, default=0.01)
    parser.add_argument("--position-scale", type=float, default=0.3)
    parser.add_argument("--theta-tol", type=float, default=0.01)
    parser.add_argument("--face-switch-weight", type=float, default=0.15)
    parser.add_argument("--effort-weight", type=float, default=0.04)
    parser.add_argument("--contact-weight", type=float, default=0.03)
    parser.add_argument("--score-temperature", type=float, default=0.12)

    parser.add_argument("--nswp", type=int, default=None)
    parser.add_argument("--rmax", type=int, default=None)
    parser.add_argument("--max-batch", type=int, default=None)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--kickrank", type=int, default=4)
    return apply_preset(parser.parse_args())


def main():
    args = parse_args()
    validate_orderings()
    seed_everything(args.seed)
    device = device_from_arg(args.device)
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    requested = [c.strip() for c in args.cases.split(",") if c.strip()]
    unknown = [c for c in requested if c not in ORDERINGS]
    if unknown:
        raise ValueError(f"Unknown cases: {unknown}. Known cases: {sorted(ORDERINGS)}")

    domains_by_name = make_domains(args, device)
    records = []
    for case in requested:
        print(f"[planar-pushing-proxy] Running {case}: {' '.join(ORDERINGS[case])}")
        records.append(run_case(case, ORDERINGS[case], domains_by_name, args, device))
        print(
            f"  max_rank={records[-1]['max_rank']} mean_rank={records[-1]['mean_rank']:.2f} "
            f"storage={records[-1]['storage']} peak_gb={records[-1]['peak_memory_gb']:.4f}"
        )

    stem = args.output_stem or f"planar_pushing_pam_proxy_{args.preset}_seed{args.seed}_{args.objective}"
    json_path = RESULTS / f"{stem}.json"
    csv_path = RESULTS / f"{stem}.summary.csv"
    fig_path = FIGURES / f"{stem}_rank_proxy.png"

    payload = {
        "script": str(Path(__file__).resolve()),
        "preset": args.preset,
        "seed": args.seed,
        "device": str(device),
        "objective": args.objective,
        "canonical_modes": CANONICAL_MODES,
        "state_modes": STATE_MODES,
        "param_modes": PARAM_MODES,
        "action_modes": ACTION_MODES,
        "domain_sizes": {k: int(len(v)) for k, v in domains_by_name.items()},
        "config": vars(args),
        "notes": [
            "This is a TT rank proxy, not a full TTPI controller training result.",
            "Face ids match pushing_dyn_explicit_double.py: 0->-1, 1->0, 2->1, 3->2.",
            "Friction enters the contact cone and one-step pushing dynamics.",
            "Mass enters the one-step effort term because the repository's explicit quasi-static pushing dynamics has no inertial mass term.",
            "Repository PushingTask action modes are [next_face, vx, vy], not a four-dimensional continuous action.",
            "bad_locality_only intentionally places parameters on the left and state/contact variables on the right.",
        ],
        "records": records,
        "summary_csv": str(csv_path),
        "figure": str(fig_path),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(csv_path, records)
    plot_records(fig_path, records, f"Planar pushing PAM proxy ({args.preset}, seed={args.seed})")

    print(f"[planar-pushing-proxy] Wrote {json_path}")
    print(f"[planar-pushing-proxy] Wrote {csv_path}")
    print(f"[planar-pushing-proxy] Wrote {fig_path}")


if __name__ == "__main__":
    main()
