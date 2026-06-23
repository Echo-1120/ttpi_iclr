#!/usr/bin/env python3
"""Build and inspect PAM HardMove action orderings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from repro.pam_ordering import build_hardmove_orders


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-actuator", type=int, required=True)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--pair-weight", type=float, default=10.0)
    parser.add_argument("--neighbor-weight", type=float, default=1.0)
    parser.add_argument("--opposite-weight", type=float, default=2.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path. Default: repro/results/pam_ordering_HM{n}.json",
    )
    args = parser.parse_args()

    coupling, orders = build_hardmove_orders(
        n_actuator=args.n_actuator,
        random_seed=args.random_seed,
        pair_weight=args.pair_weight,
        neighbor_weight=args.neighbor_weight,
        opposite_weight=args.opposite_weight,
    )

    payload = {
        "n_actuator": args.n_actuator,
        "n_modes": 2 * args.n_actuator,
        "mode_convention": "[acc_0, sw_0, acc_1, sw_1, ...]",
        "weights": {
            "pair_weight": args.pair_weight,
            "neighbor_weight": args.neighbor_weight,
            "opposite_weight": args.opposite_weight,
        },
        "coupling_matrix": coupling,
        "orders": {
            name: {
                "order": result.order,
                "objective": result.objective,
                "adjacency_score": result.adjacency_score,
                "metadata": result.metadata,
            }
            for name, result in orders.items()
        },
    }

    out = args.out or ROOT / "repro" / "results" / f"pam_ordering_HM{args.n_actuator}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved: {out}")
    print(f"{'name':<24} {'objective':>12} {'adjacency':>12} order")
    for name, result in sorted(orders.items(), key=lambda item: item[1].objective):
        print(f"{name:<24} {result.objective:12.2f} {result.adjacency_score:12.2f} {result.order}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
