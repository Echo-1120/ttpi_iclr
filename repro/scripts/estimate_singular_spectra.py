#!/usr/bin/env python3
"""Estimate conditional action-tensor singular spectra for HardMove PAM.

This script evaluates A_s(a) proxies at stratified representative states. For
small unfoldings it computes exact SVD; for large unfoldings it computes SVD on a
randomly sampled row/column sketch so HM16 diagnostics remain feasible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dynamic_systems import HardMove
from repro.pam_ordering import build_hardmove_orders

torch.set_default_dtype(torch.float64)


def parse_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def representative_states(k: int, device: torch.device, seed: int) -> list[tuple[str, torch.Tensor]]:
    """Return half high-return proxy states and half Latin-hypercube states."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 7919)
    n_proxy = max(1, k // 2)
    n_lhs = max(0, k - n_proxy)
    states: list[tuple[str, torch.Tensor]] = []

    # High-return/success proxy: near-goal states with small velocities. If
    # saved successful trajectories are unavailable, these are the closest
    # deterministic proxy for successful trajectory states in HardMove.
    radii = torch.linspace(0.0, 0.08, n_proxy + 1, dtype=torch.float64)[1:]
    for i, radius in enumerate(radii):
        theta = 2.0 * math.pi * i / max(1, n_proxy)
        state = torch.tensor(
            [[float(radius * math.cos(theta)), float(radius * math.sin(theta)), 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        states.append(("high_return_proxy", state))

    # Latin hypercube over [x, y, vx, vy], matching run_hardmove defaults.
    mins = torch.tensor([-1.0, -1.0, 0.0, 0.0], dtype=torch.float64)
    maxs = torch.tensor([1.0, 1.0, 0.25, 0.25], dtype=torch.float64)
    for i in range(n_lhs):
        coords = []
        for dim in range(4):
            perm = torch.randperm(max(1, n_lhs), generator=g)
            u = (perm[i].to(torch.float64) + torch.rand((), generator=g, dtype=torch.float64)) / max(1, n_lhs)
            coords.append(float(mins[dim] + u * (maxs[dim] - mins[dim])))
        state = torch.tensor([coords], dtype=torch.float64, device=device)
        states.append(("latin_hypercube", state))
    return states[:k]


def action_domains(n_actuator: int, n_action: int, cap: int, device: torch.device) -> list[torch.Tensor]:
    velocity_max = 0.25
    acc_max = velocity_max
    acc = torch.linspace(-acc_max, acc_max, n_action, device=device)
    if cap > 0 and n_action > cap:
        idx = torch.linspace(0, n_action - 1, cap, device=device).round().long()
        acc = acc[idx]
    sw = torch.arange(2, device=device, dtype=torch.float64)
    return [acc, sw] * n_actuator


def unravel_indices(flat: torch.Tensor, dims: list[int]) -> torch.Tensor:
    cols = []
    rem = flat.clone()
    for dim in reversed(dims):
        cols.append(rem % dim)
        rem = torch.div(rem, dim, rounding_mode="floor")
    return torch.stack(list(reversed(cols)), dim=1).long()


def multi_to_action(indices: torch.Tensor, domains: list[torch.Tensor]) -> torch.Tensor:
    values = []
    for site, domain in enumerate(domains):
        values.append(domain[indices[:, site]])
    return torch.stack(values, dim=1)


def transform_action(action_phys: torch.Tensor, n_actuator: int, env_variant: str, permutation: list[int], strength: float) -> torch.Tensor:
    action_phys = action_phys.clone()
    if env_variant == "index_permuted":
        perm = torch.tensor(permutation, device=action_phys.device, dtype=torch.long)
        action_phys = action_phys.view(-1, n_actuator, 2)[:, perm, :].reshape(action_phys.shape)
    elif env_variant == "cross_coupled":
        reshaped = action_phys.view(-1, n_actuator, 2).clone()
        acc = reshaped[:, :, 0]
        left = torch.roll(acc, shifts=1, dims=1)
        right = torch.roll(acc, shifts=-1, dims=1)
        reshaped[:, :, 0] = acc + strength * 0.5 * (left + right)
        action_phys = reshaped.reshape(action_phys.shape)
    return action_phys


@torch.no_grad()
def evaluate_matrix(
    *,
    dyn: HardMove,
    state: torch.Tensor,
    row_indices: torch.Tensor,
    col_indices: torch.Tensor,
    left_dims: list[int],
    right_dims: list[int],
    domains_ordered: list[torch.Tensor],
    action_order: list[int],
    n_actuator: int,
    env_variant: str,
    permutation: list[int],
    cross_strength: float,
) -> torch.Tensor:
    n_rows = row_indices.numel()
    n_cols = col_indices.numel()
    left_multi = unravel_indices(row_indices, left_dims) if left_dims else torch.zeros((n_rows, 0), dtype=torch.long, device=state.device)
    right_multi = unravel_indices(col_indices, right_dims) if right_dims else torch.zeros((n_cols, 0), dtype=torch.long, device=state.device)
    rows = []
    inverse = [action_order.index(i) for i in range(2 * n_actuator)]
    inverse_t = torch.tensor(inverse, dtype=torch.long, device=state.device)
    for i in range(n_rows):
        full_idx = torch.cat(
            [
                left_multi[i].view(1, -1).expand(n_cols, -1),
                right_multi,
            ],
            dim=1,
        )
        action_ordered = multi_to_action(full_idx, domains_ordered)
        action_phys = action_ordered[:, inverse_t]
        action_phys = transform_action(action_phys, n_actuator, env_variant, permutation, cross_strength)
        s = state.expand(n_cols, -1)
        values = dyn.reward_state_action(s, action_phys)
        rows.append(values.detach().cpu())
    return torch.stack(rows, dim=0)


def effective_rank(singular_values: list[float], rel_error: float) -> int:
    if not singular_values:
        return 0
    vals = torch.tensor(singular_values, dtype=torch.float64)
    energy = vals.square()
    total = float(energy.sum().item())
    if total <= 0:
        return 0
    tail = torch.flip(torch.cumsum(torch.flip(energy, dims=[0]), dim=0), dims=[0])
    for idx in range(len(vals)):
        err = math.sqrt(float(tail[idx].item()) / total)
        if err <= rel_error:
            return max(1, idx + 1)
    return len(vals)


def effective_rank_relative_sigma1(singular_values: list[float], threshold: float) -> int:
    if not singular_values:
        return 0
    sigma1 = abs(float(singular_values[0]))
    if sigma1 <= 0:
        return 0
    return int(sum(1 for value in singular_values if abs(float(value)) / sigma1 >= threshold))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-actuator", type=int, required=True)
    parser.add_argument("--env-variant", choices=["standard", "index_permuted", "cross_coupled"], default="standard")
    parser.add_argument("--env-permutation-seed", type=int, default=2026)
    parser.add_argument("--cross-coupling-strength", type=float, default=0.25)
    parser.add_argument("--n-action", type=int, default=50)
    parser.add_argument("--action-grid-cap", type=int, default=12)
    parser.add_argument("--orderings", type=str, default="local,block_pam,rankaware_proxy_pam,hybrid_pam")
    parser.add_argument("--k-states", type=int, default=8)
    parser.add_argument("--top-singular", type=int, default=32)
    parser.add_argument("--error-thresholds", type=str, default="0.1,0.01,0.001")
    parser.add_argument("--max-unfold-elements", type=int, default=200000)
    parser.add_argument("--sampled-rows", type=int, default=512)
    parser.add_argument("--sampled-cols", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "repro" / "diagnostics" / "spectra")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    env_name = f"HM{args.n_actuator}" + ("" if args.env_variant == "standard" else f"_{args.env_variant}")
    domains_phys = action_domains(args.n_actuator, args.n_action, args.action_grid_cap, device)
    coupling, orders = build_hardmove_orders(args.n_actuator, random_seed=args.seed)
    selected = parse_list(args.orderings)
    thresholds = [float(x) for x in parse_list(args.error_thresholds)]
    states = representative_states(args.k_states, device, args.seed)
    dyn = HardMove(dt=0.01, w_goal=1e3, w_action=1e4, n=args.n_actuator, device=device)

    g_perm = torch.Generator(device="cpu")
    g_perm.manual_seed(args.env_permutation_seed + args.n_actuator)
    permutation = torch.randperm(args.n_actuator, generator=g_perm).tolist()

    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for ordering in selected:
        if ordering not in orders:
            raise ValueError(f"Unknown ordering {ordering}; available={sorted(orders)}")
        order = orders[ordering].order
        domains_ordered = [domains_phys[i] for i in order]
        dims = [len(x) for x in domains_ordered]
        for state_idx, (state_type, state) in enumerate(states):
            for cut in range(1, len(dims)):
                left_dims = dims[:cut]
                right_dims = dims[cut:]
                n_rows = math.prod(left_dims) if left_dims else 1
                n_cols = math.prod(right_dims) if right_dims else 1
                exact = n_rows * n_cols <= args.max_unfold_elements
                if exact:
                    row_ids = torch.arange(n_rows, device=device)
                    col_ids = torch.arange(n_cols, device=device)
                else:
                    row_ids = torch.randint(n_rows, (min(args.sampled_rows, n_rows),), device=device)
                    col_ids = torch.randint(n_cols, (min(args.sampled_cols, n_cols),), device=device)

                matrix = evaluate_matrix(
                    dyn=dyn,
                    state=state,
                    row_indices=row_ids,
                    col_indices=col_ids,
                    left_dims=left_dims,
                    right_dims=right_dims,
                    domains_ordered=domains_ordered,
                    action_order=order,
                    n_actuator=args.n_actuator,
                    env_variant=args.env_variant,
                    permutation=permutation,
                    cross_strength=args.cross_coupling_strength,
                )
                singular_values = torch.linalg.svdvals(matrix).detach().cpu().tolist()[: args.top_singular]
                for sv_idx, value in enumerate(singular_values, start=1):
                    rows.append(
                        {
                            "env": env_name,
                            "ordering": ordering,
                            "state_index": state_idx,
                            "state_type": state_type,
                            "state_vector_json": json.dumps([float(x) for x in state.detach().cpu().view(-1).tolist()]),
                            "cut": cut,
                            "singular_index": sv_idx,
                            "singular_value": float(value),
                            "exact_unfolding": exact,
                            "matrix_rows": int(n_rows),
                            "matrix_cols": int(n_cols),
                            "used_rows": int(row_ids.numel()),
                            "used_cols": int(col_ids.numel()),
                        }
                    )
                summary = {
                    "env": env_name,
                    "ordering": ordering,
                    "state_index": state_idx,
                    "state_type": state_type,
                    "state_vector_json": json.dumps([float(x) for x in state.detach().cpu().view(-1).tolist()]),
                    "cut": cut,
                    "exact_unfolding": exact,
                    "approximation": "exact" if exact else "sampled_row_col_svd",
                    "matrix_rows": int(n_rows),
                    "matrix_cols": int(n_cols),
                    "top_singular_value": float(singular_values[0]) if singular_values else 0.0,
                }
                for threshold in thresholds:
                    summary[f"effective_rank_relerr_{threshold:g}"] = effective_rank(singular_values, threshold)
                    summary[f"effective_rank_sigma1_rel_{threshold:g}"] = effective_rank_relative_sigma1(
                        singular_values, threshold
                    )
                summary_rows.append(summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{env_name}_seed{args.seed}"
    detail_csv = args.out_dir / f"{prefix}.singular_values.csv"
    summary_csv = args.out_dir / f"{prefix}.effective_ranks.csv"
    payload_json = args.out_dir / f"{prefix}.spectra.json"

    for path, data in [(detail_csv, rows), (summary_csv, summary_rows)]:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()) if data else [])
            if data:
                writer.writeheader()
                writer.writerows(data)

    payload_json.write_text(
        json.dumps(
            {
                "env": env_name,
                "n_actuator": args.n_actuator,
                "env_variant": args.env_variant,
                "orderings": selected,
                "thresholds": thresholds,
                "state_sampling": "half_high_return_proxy_half_latin_hypercube",
                "detail_csv": str(detail_csv),
                "summary_csv": str(summary_csv),
                "singular_values": rows,
                "effective_ranks": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[DONE] wrote {detail_csv}")
    print(f"[DONE] wrote {summary_csv}")
    print(f"[DONE] wrote {payload_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
