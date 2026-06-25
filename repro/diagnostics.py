"""PAM/TTPI diagnostic helpers.

The helpers here keep experiment logging separate from training code: complete
rank profiles, TT-Cross query counts, and compact per-run summaries.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        return float(value)
    except Exception:
        return default


def _rank_list(value: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().view(-1).tolist()
    return [int(x) for x in value]


def rank_stats(profile: list[int]) -> dict[str, float]:
    values = [int(x) for x in profile]
    if not values:
        return {"max": 1.0, "mean": 1.0}
    return {"max": float(max(values)), "mean": float(sum(values) / len(values))}


def tensor_safe(obj: Any) -> Any:
    if hasattr(obj, "detach"):
        obj = obj.detach().cpu()
        if obj.numel() == 1:
            return obj.item()
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: tensor_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [tensor_safe(v) for v in obj]
    return obj


def rank_profile_rows(
    *,
    task_name: str,
    seed: int,
    env_name: str,
    ordering_name: str,
    train_data: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_iter = max(
        len(train_data.get("v_rank_profile", [])),
        len(train_data.get("a_rank_profile", [])),
        len(train_data.get("p_rank_profile", [])),
    )
    for iteration in range(n_iter):
        v_profile = _rank_list((train_data.get("v_rank_profile") or [])[iteration]) if iteration < len(train_data.get("v_rank_profile", [])) else []
        a_profile = _rank_list((train_data.get("a_rank_profile") or [])[iteration]) if iteration < len(train_data.get("a_rank_profile", [])) else []
        p_profile = _rank_list((train_data.get("p_rank_profile") or [])[iteration]) if iteration < len(train_data.get("p_rank_profile", [])) else []
        row: dict[str, Any] = {
            "task_name": task_name,
            "seed": int(seed),
            "env": env_name,
            "ordering": ordering_name,
            "iteration": int(iteration + 1),
            "adv_rank_max": rank_stats(a_profile)["max"],
            "adv_rank_mean": rank_stats(a_profile)["mean"],
            "val_rank_max": rank_stats(v_profile)["max"],
            "val_rank_mean": rank_stats(v_profile)["mean"],
            "policy_rank_max": rank_stats(p_profile)["max"],
            "policy_rank_mean": rank_stats(p_profile)["mean"],
            "adv_rank_profile_json": json.dumps(a_profile),
            "val_rank_profile_json": json.dumps(v_profile),
            "policy_rank_profile_json": json.dumps(p_profile),
        }
        for idx, rank in enumerate(a_profile, start=1):
            row[f"adv_rank_{idx}"] = int(rank)
        for idx, rank in enumerate(v_profile, start=1):
            row[f"val_rank_{idx}"] = int(rank)
        for idx, rank in enumerate(p_profile, start=1):
            row[f"policy_rank_{idx}"] = int(rank)
        rows.append(row)
    return rows


def cross_query_rows(
    *,
    task_name: str,
    seed: int,
    env_name: str,
    ordering_name: str,
    diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in diagnostics.get("cross_call_history", []):
        row = {
            "task_name": task_name,
            "seed": int(seed),
            "env": env_name,
            "ordering": ordering_name,
            "call_index": int(entry.get("call_index", 0)),
            "function_name": entry.get("function_name", ""),
            "domain_dim": int(entry.get("domain_dim", 0)),
            "domain_sizes_json": json.dumps(entry.get("domain_sizes", [])),
            "max_batch": int(entry.get("max_batch", 0)),
            "rmax": int(entry.get("rmax", 0)),
            "nswp": int(entry.get("nswp", 0)),
            "eps": _as_float(entry.get("eps")),
            "kickrank": int(entry.get("kickrank", 0)),
            "function_eval_requests": int(entry.get("function_eval_requests", 0)),
            "queried_points": int(entry.get("queried_points", 0)),
            "rank_profile_json": json.dumps(entry.get("rank_profile", [])),
            "rank_max": int(entry.get("rank_max", 1)),
        }
        rows.append(row)
    return rows


def latest_rank_profiles(train_data: dict[str, Any]) -> dict[str, list[int]]:
    def latest(key: str) -> list[int]:
        values = train_data.get(key) or []
        return _rank_list(values[-1]) if values else []

    return {
        "advantage": latest("a_rank_profile"),
        "value": latest("v_rank_profile"),
        "policy": latest("p_rank_profile"),
    }


def summary_row(
    *,
    result: dict[str, Any],
    train_data: dict[str, Any],
    diagnostics: dict[str, Any],
    peak_memory_mb: float | None,
) -> dict[str, Any]:
    profiles = latest_rank_profiles(train_data)
    adv_stats = rank_stats(profiles["advantage"])
    val_stats = rank_stats(profiles["value"])
    final_metrics = result.get("final_metrics") or {}
    best_tradeoff = result.get("best_tradeoff_metrics") or {}
    return {
        "seed": result.get("seed"),
        "env": result.get("env_name", result.get("task")),
        "ordering": result.get("action_order_name"),
        "la_objective": result.get("pam_order_objective"),
        "peak_cut_objective": result.get("pam_order_peak_cut_objective"),
        "rankaware_proxy_objective": result.get("pam_order_rankaware_proxy_objective"),
        "adv_rank_max": adv_stats["max"],
        "adv_rank_mean": adv_stats["mean"],
        "val_rank_max": val_stats["max"],
        "val_rank_mean": val_stats["mean"],
        "peak_memory_mb": peak_memory_mb,
        "runtime_sec": result.get("train_time_sec"),
        "success_rate": final_metrics.get("success_rate"),
        "avg_return": final_metrics.get("cum_reward_mean"),
        "mu": final_metrics.get("mu_success"),
        "best_success_rate": best_tradeoff.get("success_rate"),
        "best_mu": best_tradeoff.get("mu_success"),
        "best_s_times_mu": best_tradeoff.get("S_times_mu", best_tradeoff.get("tradeoff_score")),
        "tt_cross_calls": diagnostics.get("tt_cross_calls", 0),
        "tt_cross_function_evals": diagnostics.get("tt_cross_function_evals", 0),
        "queried_points_total": diagnostics.get("queried_points_total", 0),
        "adv_rank_profile_json": json.dumps(profiles["advantage"]),
        "val_rank_profile_json": json.dumps(profiles["value"]),
        "policy_rank_profile_json": json.dumps(profiles["policy"]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_header = reader.fieldnames
            existing_rows = list(reader)
        if existing_header:
            fieldnames = list(existing_header)
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
            if fieldnames != existing_header:
                with path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(existing_rows)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
