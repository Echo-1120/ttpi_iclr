#!/usr/bin/env python3
"""Aggregate PAM diagnostic summaries into final tables."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


METRICS = [
    "la_objective",
    "peak_cut_objective",
    "rankaware_proxy_objective",
    "adv_rank_max",
    "adv_rank_mean",
    "val_rank_max",
    "val_rank_mean",
    "peak_memory_mb",
    "runtime_sec",
    "success_rate",
    "avg_return",
    "mu",
    "best_s_times_mu",
    "tt_cross_calls",
    "tt_cross_function_evals",
    "queried_points_total",
]


def as_float(value: str) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=ROOT / "repro" / "diagnostics" / "pam_ablation_summary.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "repro" / "diagnostics" / "pam_ablation_final_table.csv")
    args = parser.parse_args()

    if not args.summary.exists():
        raise FileNotFoundError(args.summary)

    with args.summary.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    seeds: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        key = (row.get("env", "unknown"), row.get("ordering", "unknown"))
        seeds[key].add(row.get("seed", ""))
        for metric in METRICS:
            value = as_float(row.get(metric, ""))
            if value is not None:
                grouped[key][metric].append(value)

    out_rows = []
    for key in sorted(grouped):
        env, ordering = key
        out = {"env": env, "ordering": ordering, "n_seeds": len(seeds[key])}
        for metric in METRICS:
            values = grouped[key].get(metric, [])
            if values:
                out[f"{metric}_mean"] = sum(values) / len(values)
                out[f"{metric}_min"] = min(values)
                out[f"{metric}_max"] = max(values)
            else:
                out[f"{metric}_mean"] = ""
                out[f"{metric}_min"] = ""
                out[f"{metric}_max"] = ""
        out_rows.append(out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(out_rows[0].keys()) if out_rows else ["env", "ordering", "n_seeds"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[DONE] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
