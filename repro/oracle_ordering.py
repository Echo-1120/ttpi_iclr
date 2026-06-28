"""Small-scale oracle diagnostics for HardMove action-mode orderings."""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Iterable

from repro.pam_ordering import (
    build_hardmove_orders,
    peak_cut_cost,
    rankaware_proxy_cost,
    validate_order,
    weighted_linear_arrangement,
)
from repro.pam_validation import validate_block_adjacency


ORACLE_MODES = ("full_modes", "block_with_flips")


def iter_full_mode_orders(n_actuator: int) -> Iterable[list[int]]:
    n_modes = 2 * n_actuator
    for order in itertools.permutations(range(n_modes)):
        yield list(order)


def iter_block_with_flip_orders(n_actuator: int) -> Iterable[list[int]]:
    blocks = [[2 * i, 2 * i + 1] for i in range(n_actuator)]
    for block_order in itertools.permutations(range(n_actuator)):
        for flips in itertools.product((False, True), repeat=n_actuator):
            order: list[int] = []
            for block_idx in block_order:
                block = blocks[block_idx]
                order.extend(reversed(block) if flips[block_idx] else block)
            yield order


def expected_oracle_count(n_actuator: int, mode: str) -> int:
    import math

    if mode == "full_modes":
        return math.factorial(2 * n_actuator)
    if mode == "block_with_flips":
        return math.factorial(n_actuator) * (2**n_actuator)
    raise ValueError(f"Unknown oracle mode {mode!r}; expected one of {ORACLE_MODES}")


def iter_oracle_orders(n_actuator: int, mode: str) -> Iterable[list[int]]:
    if mode == "full_modes":
        if n_actuator > 4:
            raise ValueError("full_modes oracle is intentionally limited to HM4 or smaller")
        return iter_full_mode_orders(n_actuator)
    if mode == "block_with_flips":
        return iter_block_with_flip_orders(n_actuator)
    raise ValueError(f"Unknown oracle mode {mode!r}; expected one of {ORACLE_MODES}")


def _regret(value: float, best: float, worst: float) -> float:
    denom = worst - best
    if denom <= 0:
        return 0.0
    return float((value - best) / denom)


def oracle_rows(n_actuator: int, mode: str, random_seed: int = 0) -> list[dict[str, object]]:
    coupling, named_orders = build_hardmove_orders(n_actuator=n_actuator, random_seed=random_seed)
    rows: list[dict[str, object]] = []
    named_by_order = {tuple(result.order): name for name, result in named_orders.items()}

    for idx, order in enumerate(iter_oracle_orders(n_actuator, mode)):
        order = validate_order(order, len(coupling))
        block_preserving = True
        try:
            validate_block_adjacency(order, n_actuator)
        except ValueError:
            block_preserving = False
        rows.append(
            {
                "candidate_index": idx,
                "n_actuator": n_actuator,
                "mode": mode,
                "ordering": named_by_order.get(tuple(order), f"oracle_{idx}"),
                "order_json": json.dumps(order),
                "la_objective": weighted_linear_arrangement(order, coupling),
                "peak_cut_objective": peak_cut_cost(order, coupling),
                "rankaware_proxy_objective": rankaware_proxy_cost(
                    order, coupling, lambda_sum=1.0, lambda_peak=2.0
                ),
                "block_preserving": block_preserving,
            }
        )

    for metric in ("la_objective", "peak_cut_objective", "rankaware_proxy_objective"):
        values = [float(row[metric]) for row in rows]
        best = min(values)
        worst = max(values)
        for row in rows:
            row[f"normalized_regret_{metric}"] = _regret(float(row[metric]), best, worst)
    return rows


def write_oracle_outputs(
    *,
    n_actuator: int,
    mode: str,
    out_dir: Path,
    random_seed: int = 0,
) -> tuple[Path, Path]:
    rows = oracle_rows(n_actuator=n_actuator, mode=mode, random_seed=random_seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"HM{n_actuator}_{mode}_seed{random_seed}"
    csv_path = out_dir / f"{prefix}.oracle.csv"
    json_path = out_dir / f"{prefix}.oracle.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "n_actuator": n_actuator,
        "mode": mode,
        "random_seed": random_seed,
        "expected_count": expected_oracle_count(n_actuator, mode),
        "actual_count": len(rows),
        "csv": str(csv_path),
        "best": {},
        "worst": {},
    }
    for metric in ("la_objective", "peak_cut_objective", "rankaware_proxy_objective"):
        summary["best"][metric] = min(rows, key=lambda row: float(row[metric])) if rows else None
        summary["worst"][metric] = max(rows, key=lambda row: float(row[metric])) if rows else None
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, json_path
