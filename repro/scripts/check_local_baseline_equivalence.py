#!/usr/bin/env python3
"""Strict equivalence checks for original local TTPI vs ordering-wrapper local."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from repro.baseline_equivalence import (
    check_hardmove_local_action_equivalence,
    check_index_roundtrip,
    check_ttgo_local_equivalence,
)
from repro.pam_ordering import build_hardmove_orders


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-actuator", type=int, default=8)
    parser.add_argument("--n-state", type=int, default=20)
    parser.add_argument("--n-action", type=int, default=20)
    parser.add_argument("--n-index-samples", type=int, default=4096)
    parser.add_argument("--n-action-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--index-only", action="store_true", help="Only run pure index encode/decode checks.")
    parser.add_argument("--skip-ttgo", action="store_true")
    parser.add_argument("--ttgo-n-actuator", type=int, default=4)
    parser.add_argument("--ttgo-n-state", type=int, default=9)
    parser.add_argument("--ttgo-n-action", type=int, default=7)
    parser.add_argument("--ttgo-n-states", type=int, default=32)
    parser.add_argument("--ttgo-n-samples", type=int, default=8)
    parser.add_argument("--ttgo-rollout-steps", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    _, orders = build_hardmove_orders(n_actuator=args.n_actuator, random_seed=args.seed)
    results = []
    for name, ordering in orders.items():
        results.append(
            check_index_roundtrip(
                action_order=ordering.order,
                n_actuator=args.n_actuator,
                n_action=args.n_action,
                n_samples=args.n_index_samples,
                seed=args.seed + 1009,
            )
        )
        results[-1].details["ordering"] = name

    if not args.index_only:
        results.append(
            check_hardmove_local_action_equivalence(
                n_actuator=args.n_actuator,
                n_state=args.n_state,
                n_action=args.n_action,
                n_samples=args.n_action_samples,
                seed=args.seed,
                device=args.device,
            )
        )

    if not args.index_only and not args.skip_ttgo:
        results.append(
            check_ttgo_local_equivalence(
                n_actuator=args.ttgo_n_actuator,
                n_state=args.ttgo_n_state,
                n_action=args.ttgo_n_action,
                n_states=args.ttgo_n_states,
                n_ttgo_samples=args.ttgo_n_samples,
                rollout_steps=args.ttgo_rollout_steps,
                seed=args.seed,
                device=args.device,
            )
        )

    payload = {
        "passed": all(result.passed for result in results),
        "results": [
            {"name": result.name, "passed": result.passed, "details": result.details}
            for result in results
        ],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
