"""Metric extraction for TTPI/PAM prestudy records."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from statistics import mean, median
from typing import Any

from repro.pam_ordering import (
    build_hardmove_orders,
    hardmove_action_coupling,
    peak_cut_cost,
    rankaware_proxy_cost,
    segmented_cut_costs,
    weighted_linear_arrangement,
)

from .result_loader import as_float, as_int


def _rank_profile_from_json(value: Any) -> list[int]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return [int(x) for x in value]


def max_numeric(rows: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
    values = [as_float(row.get(key)) for row in rows]
    values = [x for x in values if x is not None]
    return max(values) if values else default


def sum_numeric(rows: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
    values = [as_float(row.get(key)) for row in rows]
    values = [x for x in values if x is not None]
    return sum(values) if values else default


def rank_auc(rank_rows: list[dict[str, Any]], key: str = "adv_rank_max") -> float:
    values = [as_float(row.get(key)) for row in sorted(rank_rows, key=lambda r: as_int(r.get("iteration"), 0) or 0)]
    values = [x for x in values if x is not None]
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return float(sum((values[i] + values[i + 1]) * 0.5 for i in range(len(values) - 1)))


def final_rank(rank_rows: list[dict[str, Any]], key: str = "adv_rank_max") -> float:
    if not rank_rows:
        return 0.0
    rows = sorted(rank_rows, key=lambda r: as_int(r.get("iteration"), 0) or 0)
    return float(as_float(rows[-1].get(key), 0.0) or 0.0)


def last_completed_iteration(rank_rows: list[dict[str, Any]], round_rows: list[dict[str, Any]]) -> int:
    rank_iter = [as_int(row.get("iteration"), 0) or 0 for row in rank_rows]
    round_iter = [as_int(row.get("iteration"), 0) or 0 for row in round_rows]
    return max(rank_iter + round_iter + [0])


def per_cut_bottleneck(rank_rows: list[dict[str, Any]], prefix: str = "adv_rank_") -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rank_rows:
        for key, value in row.items():
            if key.startswith(prefix):
                out[key] = max(out.get(key, 0.0), float(as_float(value, 0.0) or 0.0))
    return out


def summarize_record(record: dict[str, Any]) -> dict[str, Any]:
    result = record.get("result", {})
    diagnostics = result.get("diagnostics", {})
    final_metrics = result.get("final_metrics") or {}
    rank_rows = record.get("rank_rows", [])
    cross_rows = record.get("cross_rows", [])
    cross_process = record.get("cross_process_rows", [])
    total_queries = result.get("diagnostics", {}).get("queried_points_total")
    if total_queries in (None, ""):
        total_queries = sum_numeric(cross_rows, "queried_points", 0.0)
    tt_cross_calls = diagnostics.get("tt_cross_calls")
    if tt_cross_calls in (None, ""):
        tt_cross_calls = len(cross_rows)
    completed_pi = last_completed_iteration(rank_rows, record.get("round_rows", []))
    total_queries = float(total_queries or 0.0)
    return {
        "run_id": record.get("run_id"),
        "env": record.get("env"),
        "ordering": record.get("ordering"),
        "seed": record.get("seed"),
        "status": record.get("status"),
        "oom": bool(record.get("oom")),
        "error_type": record.get("error_type", ""),
        "peak_memory_mb": as_float(result.get("peak_memory_mb"), 0.0) or 0.0,
        "ordering_search_time_sec": as_float(result.get("ordering_search_time_sec"), 0.0) or 0.0,
        "ttpi_training_time_sec": as_float(result.get("ttpi_training_time_sec", result.get("train_time_sec")), 0.0)
        or 0.0,
        "total_time_sec": as_float(result.get("total_time_sec"), 0.0) or 0.0,
        "success_rate": as_float(final_metrics.get("success_rate")),
        "avg_return": as_float(final_metrics.get("cum_reward_mean")),
        "mu": as_float(final_metrics.get("mu_success")),
        "tt_cross_calls": int(tt_cross_calls or 0),
        "tt_cross_function_evals": int(diagnostics.get("tt_cross_function_evals", total_queries) or 0),
        "queried_points_total": int(total_queries),
        "queries_per_completed_pi": total_queries / completed_pi if completed_pi else 0.0,
        "last_completed_pi": completed_pi,
        "adv_rank_auc": rank_auc(rank_rows, "adv_rank_max"),
        "val_rank_auc": rank_auc(rank_rows, "val_rank_max"),
        "adv_rank_final": final_rank(rank_rows, "adv_rank_max"),
        "val_rank_final": final_rank(rank_rows, "val_rank_max"),
        "adv_rank_max_over_run": max_numeric(rank_rows, "adv_rank_max", 0.0),
        "val_rank_max_over_run": max_numeric(rank_rows, "val_rank_max", 0.0),
        "cross_process_rank_max": max_numeric(cross_process, "rank_max", 0.0),
        "pam_order_objective": as_float(result.get("pam_order_objective")),
        "pam_order_peak_cut_objective": as_float(result.get("pam_order_peak_cut_objective")),
        "pam_order_rankaware_proxy_objective": as_float(result.get("pam_order_rankaware_proxy_objective")),
    }


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [summarize_record(record) for record in records]


def aggregate_rows(rows: list[dict[str, Any]], group_keys: tuple[str, ...] = ("env", "ordering")) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in group_keys)].append(row)
    out: list[dict[str, Any]] = []
    metrics = [
        "peak_memory_mb",
        "ttpi_training_time_sec",
        "success_rate",
        "avg_return",
        "mu",
        "queried_points_total",
        "queries_per_completed_pi",
        "adv_rank_auc",
        "adv_rank_max_over_run",
    ]
    for key, items in sorted(groups.items()):
        agg: dict[str, Any] = {group_keys[i]: key[i] for i in range(len(group_keys))}
        agg["runs"] = len(items)
        agg["ok_runs"] = sum(1 for item in items if item.get("status") == "ok")
        agg["oom_runs"] = sum(1 for item in items if item.get("status") == "oom" or item.get("oom"))
        agg["oom_rate"] = agg["oom_runs"] / len(items) if items else 0.0
        for metric in metrics:
            values = [item.get(metric) for item in items if isinstance(item.get(metric), (int, float))]
            if values:
                agg[f"{metric}_mean"] = mean(values)
                agg[f"{metric}_median"] = median(values)
            else:
                agg[f"{metric}_mean"] = ""
                agg[f"{metric}_median"] = ""
        out.append(agg)
    return out


def select_typical_seed(rows: list[dict[str, Any]], env: str, methods: list[str]) -> int | None:
    by_seed: dict[int, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row.get("env") == env and row.get("ordering") in methods and isinstance(row.get("seed"), int):
            by_seed[int(row["seed"])][row["ordering"]] = float(row.get("adv_rank_auc") or 0.0)
    candidates = {seed: vals for seed, vals in by_seed.items() if all(method in vals for method in methods)}
    if not candidates:
        return None
    scores = {seed: sum(vals.values()) / len(methods) for seed, vals in candidates.items()}
    med = median(scores.values())
    return min(scores, key=lambda seed: (abs(scores[seed] - med), seed))


def kendall_tau_distance(order: list[int], reference: list[int]) -> int:
    ref_pos = {value: idx for idx, value in enumerate(reference)}
    mapped = [ref_pos[value] for value in order]
    inversions = 0
    for i in range(len(mapped)):
        for j in range(i + 1, len(mapped)):
            inversions += int(mapped[i] > mapped[j])
    return inversions


def strongest_edge_adjacency_ratio(order: list[int], coupling: list[list[float]], top_k: int = 10) -> float:
    pos = {mode: idx for idx, mode in enumerate(order)}
    edges: list[tuple[float, int, int]] = []
    for i in range(len(coupling)):
        for j in range(i + 1, len(coupling)):
            if coupling[i][j] > 0:
                edges.append((float(coupling[i][j]), i, j))
    if not edges:
        return 0.0
    edges.sort(reverse=True)
    chosen = edges[: min(top_k, len(edges))]
    return sum(1 for _, i, j in chosen if abs(pos[i] - pos[j]) == 1) / len(chosen)


def block_displacement(order: list[int], n_actuator: int) -> float:
    pos = {mode: idx for idx, mode in enumerate(order)}
    displacements = []
    for actuator in range(n_actuator):
        center = (pos[2 * actuator] + pos[2 * actuator + 1]) * 0.5
        local_center = 2 * actuator + 0.5
        displacements.append(abs(center - local_center))
    return float(mean(displacements)) if displacements else 0.0


def permutation_metrics(n_actuator: int, ordering_names: list[str], random_seed: int = 0) -> list[dict[str, Any]]:
    coupling, orders = build_hardmove_orders(n_actuator=n_actuator, random_seed=random_seed)
    reference = list(range(2 * n_actuator))
    rows: list[dict[str, Any]] = []
    for name in ordering_names:
        if name not in orders:
            continue
        order = orders[name].order
        cuts = segmented_cut_costs(order, coupling)
        rows.append(
            {
                "env": f"HM{n_actuator}",
                "ordering": name,
                "weighted_edge_span": weighted_linear_arrangement(order, coupling),
                "peak_cut": peak_cut_cost(order, coupling),
                "mean_prefix_cut": mean(cuts) if cuts else 0.0,
                "max_prefix_cut": max(cuts) if cuts else 0.0,
                "rankaware_proxy": rankaware_proxy_cost(order, coupling, lambda_sum=1.0, lambda_peak=2.0),
                "strongest_edge_adjacency_ratio": strongest_edge_adjacency_ratio(order, coupling),
                "block_displacement_vs_local": block_displacement(order, n_actuator),
                "kendall_tau_vs_local": kendall_tau_distance(order, reference),
                "cut_side_weight_imbalance": max(cuts) / (mean(cuts) + 1e-12) if cuts else 0.0,
                "order_json": json.dumps(order),
            }
        )
    return rows


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    pairs = []
    for x, y in zip(xs, ys):
        try:
            fx = float(x)
            fy = float(y)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fx) and math.isfinite(fy):
            pairs.append((fx, fy))
    if len(pairs) < 2:
        return None
    xvals, yvals = zip(*pairs)
    mx, my = mean(xvals), mean(yvals)
    vx = sum((x - mx) ** 2 for x in xvals)
    vy = sum((y - my) ** 2 for y in yvals)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)
