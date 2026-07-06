"""PAM/TTPI diagnostic helpers.

The helpers here keep experiment logging separate from training code: complete
rank profiles, TT-Cross query counts, and compact per-run summaries.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

REQUIRED_SUMMARY_FIELDS = [
    "seed",
    "training_seed",
    "environment_seed",
    "permutation_seed",
    "state_sampling_seed",
    "env",
    "ordering",
    "canonical_ordering",
    "display_ordering",
    "baseline_category",
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
    "tt_cross_calls",
    "tt_cross_function_evals",
    "queried_points_total",
    "ordering_search_time_sec",
    "ttpi_training_time_sec",
    "total_time_sec",
    "status",
    "oom",
    "error_type",
    "error_message",
]

ORDERING_GROUPS = {
    "local": "primary_baseline",
    "opposite_pair": "legacy_diagnostic",
    "badsplit": "diagnostic_control",
    "random": "diagnostic_control",
    "reverse_blocks": "diagnostic_control",
    "flip_within_block": "diagnostic_control",
    "pam_greedy": "surrogate_baseline",
    "pam_spectral": "surrogate_baseline",
    "pam_spectral_refined": "surrogate_baseline",
    "pam_greedy_refined": "surrogate_baseline",
    "block_pam": "ablation",
    "free_pam": "ablation",
    "peakcut_pam": "surrogate_baseline",
    "sensitivity_pam": "surrogate_baseline",
    "rankaware_proxy_pam": "main_method",
    "rankaware_spectral_pam": "ablation",
    "hybrid_pam": "main_method",
    "sensitivity_lite_fd5": "prestudy_candidate",
    "sensitivity_lite_first_order": "prestudy_candidate",
    "hybrid_block_only": "prestudy_ablation",
    "hybrid_block_plus_physics": "prestudy_ablation",
    "hybrid_block_plus_sensitivity": "prestudy_ablation",
    "hybrid_current": "prestudy_ablation",
}


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
            "elapsed_sec": _as_float(entry.get("elapsed_sec")),
            "memory_mb_start": entry.get("memory_mb_start"),
            "memory_mb_end": entry.get("memory_mb_end"),
            "function_eval_requests": int(entry.get("function_eval_requests", 0)),
            "queried_points": int(entry.get("queried_points", 0)),
            "rank_profile_json": json.dumps(entry.get("rank_profile", [])),
            "rank_max": int(entry.get("rank_max", 1)),
            "rank_mean": _as_float(entry.get("rank_mean"), 1.0),
        }
        rows.append(row)
    return rows


def cross_process_rows(
    *,
    task_name: str,
    seed: int,
    env_name: str,
    ordering_name: str,
    diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in diagnostics.get("cross_process_events", []):
        profile = _rank_list(entry.get("rank_profile", []))
        row: dict[str, Any] = {
            "task_name": task_name,
            "seed": int(seed),
            "env": env_name,
            "ordering": ordering_name,
            "call_index": int(entry.get("call_index", 0)),
            "function_name": entry.get("function_name", ""),
            "stage": entry.get("stage", ""),
            "elapsed_sec": _as_float(entry.get("elapsed_sec")),
            "memory_mb": entry.get("memory_mb"),
            "function_eval_requests_so_far": int(entry.get("function_eval_requests_so_far", 0)),
            "rank_profile_json": json.dumps(profile),
            "rank_max": int(entry.get("rank_max", max(profile, default=1))),
            "rank_mean": _as_float(entry.get("rank_mean"), rank_stats(profile)["mean"]),
        }
        for idx, rank in enumerate(profile, start=1):
            row[f"rank_{idx}"] = int(rank)
        rows.append(row)
    return rows


def round_event_rows(
    *,
    task_name: str,
    seed: int,
    env_name: str,
    ordering_name: str,
    diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, entry in enumerate(diagnostics.get("round_events", []), start=1):
        profile = _rank_list(entry.get("rank_profile", []))
        row: dict[str, Any] = {
            "task_name": task_name,
            "seed": int(seed),
            "env": env_name,
            "ordering": ordering_name,
            "event_index": idx,
            "model": entry.get("model", ""),
            "iteration": int(entry.get("iteration", 0)),
            "stage": entry.get("stage", ""),
            "memory_mb": entry.get("memory_mb"),
            "rank_profile_json": json.dumps(profile),
            "rank_max": int(entry.get("rank_max", max(profile, default=1))),
            "rank_mean": _as_float(entry.get("rank_mean"), rank_stats(profile)["mean"]),
        }
        for rank_idx, rank in enumerate(profile, start=1):
            row[f"rank_{rank_idx}"] = int(rank)
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
    training_seed = result.get("training_seed", result.get("seed"))
    environment_seed = result.get("environment_seed", result.get("env_permutation_seed", training_seed))
    permutation_seed = result.get("permutation_seed", result.get("order_random_seed", 0))
    state_sampling_seed = result.get("state_sampling_seed", training_seed)
    ordering_metadata = result.get("pam_order_metadata") or {}
    ordering_search_time_sec = _as_float(result.get("ordering_search_time_sec"), 0.0)
    ttpi_training_time_sec = _as_float(
        result.get("ttpi_training_time_sec", result.get("train_time_sec")),
        0.0,
    )
    total_time_sec = _as_float(
        result.get("total_time_sec"),
        ordering_search_time_sec + ttpi_training_time_sec,
    )
    status = result.get("status", "ok")
    row = {
        "seed": result.get("seed"),
        "training_seed": training_seed,
        "environment_seed": environment_seed,
        "permutation_seed": permutation_seed,
        "state_sampling_seed": state_sampling_seed,
        "env": result.get("env_name", result.get("task")),
        "ordering": result.get("action_order_name"),
        "canonical_ordering": result.get(
            "canonical_ordering",
            ordering_metadata.get("canonical_name", result.get("action_order_name")),
        ),
        "display_ordering": result.get(
            "display_ordering",
            ordering_metadata.get("display_name", result.get("action_order_name")),
        ),
        "baseline_category": result.get(
            "baseline_category",
            ordering_metadata.get("baseline_category", ordering_group(result.get("action_order_name"))),
        ),
        "ordering_group": ordering_group(result.get("action_order_name")),
        "la_objective": result.get("pam_order_objective"),
        "peak_cut_objective": result.get("pam_order_peak_cut_objective"),
        "rankaware_proxy_objective": result.get("pam_order_rankaware_proxy_objective"),
        "adv_rank_max": adv_stats["max"],
        "adv_rank_mean": adv_stats["mean"],
        "val_rank_max": val_stats["max"],
        "val_rank_mean": val_stats["mean"],
        "peak_memory_mb": peak_memory_mb if peak_memory_mb is not None else "",
        "runtime_sec": total_time_sec,
        "train_time_sec": result.get("train_time_sec", ttpi_training_time_sec),
        "ordering_search_time_sec": ordering_search_time_sec,
        "ttpi_training_time_sec": ttpi_training_time_sec,
        "total_time_sec": total_time_sec,
        "success_rate": final_metrics.get("success_rate", ""),
        "avg_return": final_metrics.get("cum_reward_mean", ""),
        "mu": final_metrics.get("mu_success", ""),
        "best_success_rate": best_tradeoff.get("success_rate"),
        "best_mu": best_tradeoff.get("mu_success"),
        "best_s_times_mu": best_tradeoff.get("S_times_mu", best_tradeoff.get("tradeoff_score")),
        "tt_cross_calls": diagnostics.get("tt_cross_calls", 0),
        "tt_cross_function_evals": diagnostics.get("tt_cross_function_evals", 0),
        "queried_points_total": diagnostics.get("queried_points_total", 0),
        "status": status,
        "oom": bool(result.get("oom", status == "oom")),
        "error_type": result.get("error_type", ""),
        "error_message": result.get("error_message", ""),
        "adv_rank_profile_json": json.dumps(profiles["advantage"]),
        "val_rank_profile_json": json.dumps(profiles["value"]),
        "policy_rank_profile_json": json.dumps(profiles["policy"]),
    }
    validate_required_summary_fields(row)
    return row


def ordering_group(ordering_name: Any) -> str:
    return ORDERING_GROUPS.get(str(ordering_name), "other")


def validate_required_summary_fields(row: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_SUMMARY_FIELDS if row.get(field) is None]
    if missing:
        raise ValueError(f"Missing required PAM diagnostic fields: {missing}")


def standard_run_log(
    *,
    run_id: str,
    result: dict[str, Any],
    train_data: dict[str, Any],
    diagnostics: dict[str, Any],
    peak_memory_mb: float | None,
    status: str = "ok",
) -> dict[str, Any]:
    profiles = latest_rank_profiles(train_data)
    adv_stats = rank_stats(profiles["advantage"])
    val_stats = rank_stats(profiles["value"])
    final_metrics = result.get("final_metrics") or {}
    training_seed = result.get("training_seed", result.get("seed"))
    environment_seed = result.get("environment_seed", result.get("env_permutation_seed", training_seed))
    permutation_seed = result.get("permutation_seed", result.get("order_random_seed", 0))
    state_sampling_seed = result.get("state_sampling_seed", training_seed)
    ordering_metadata = result.get("pam_order_metadata") or {}
    ordering_search_time_sec = _as_float(result.get("ordering_search_time_sec"), 0.0)
    ttpi_training_time_sec = _as_float(
        result.get("ttpi_training_time_sec", result.get("train_time_sec")),
        0.0,
    )
    total_time_sec = _as_float(
        result.get("total_time_sec"),
        ordering_search_time_sec + ttpi_training_time_sec,
    )
    status = result.get("status", status)
    return {
        "run_id": run_id,
        "seed": result.get("seed"),
        "training_seed": training_seed,
        "environment_seed": environment_seed,
        "permutation_seed": permutation_seed,
        "state_sampling_seed": state_sampling_seed,
        "env": result.get("env_name", result.get("task")),
        "ordering": result.get("action_order_name"),
        "canonical_ordering": result.get(
            "canonical_ordering",
            ordering_metadata.get("canonical_name", result.get("action_order_name")),
        ),
        "display_ordering": result.get(
            "display_ordering",
            ordering_metadata.get("display_name", result.get("action_order_name")),
        ),
        "baseline_category": result.get(
            "baseline_category",
            ordering_metadata.get("baseline_category", ordering_group(result.get("action_order_name"))),
        ),
        "ordering_group": ordering_group(result.get("action_order_name")),
        "objectives": {
            "la": result.get("pam_order_objective"),
            "peak_cut": result.get("pam_order_peak_cut_objective"),
            "rankaware_proxy": result.get("pam_order_rankaware_proxy_objective"),
        },
        "resources": {
            "peak_memory_mb": peak_memory_mb,
            "runtime_sec": total_time_sec,
            "train_time_sec": result.get("train_time_sec", ttpi_training_time_sec),
            "ordering_search_time_sec": ordering_search_time_sec,
            "ttpi_training_time_sec": ttpi_training_time_sec,
            "total_time_sec": total_time_sec,
        },
        "performance": {
            "success_rate": final_metrics.get("success_rate", ""),
            "avg_return": final_metrics.get("cum_reward_mean", ""),
            "mu": final_metrics.get("mu_success", ""),
        },
        "tt_cross": {
            "calls": diagnostics.get("tt_cross_calls", 0),
            "function_evals_total": diagnostics.get("tt_cross_function_evals", 0),
            "queried_points_total": diagnostics.get("queried_points_total", 0),
            "process_events": len(diagnostics.get("cross_process_events", [])),
        },
        "ranks": {
            "adv_rank_max": adv_stats["max"],
            "adv_rank_mean": adv_stats["mean"],
            "val_rank_max": val_stats["max"],
            "val_rank_mean": val_stats["mean"],
            "adv_rank_profile": profiles["advantage"],
            "val_rank_profile": profiles["value"],
            "policy_rank_profile": profiles["policy"],
        },
        "error": {
            "oom": bool(result.get("oom", status == "oom")),
            "error_type": result.get("error_type", ""),
            "error_message": result.get("error_message", ""),
        },
        "status": status,
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
