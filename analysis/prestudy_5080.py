"""RTX 5080 prestudy audit, analysis, and report entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from repro.pam_ordering import ORDERING_NAMES, build_hardmove_orders

from .metrics import (
    aggregate_rows,
    kendall_tau_distance,
    pearson_correlation,
    permutation_metrics,
    select_typical_seed,
    summarize_records,
)
from .plotting import bar_by_method, boxplot_by_method, cutwise_rank_heatmap, rank_trajectory, scatter
from .result_loader import ROOT, count_files, load_records, load_summary_rows, write_csv
from .statistics import oom_rate_table, paired_method_tests


REPORT_ROOT = ROOT / "reports" / "prestudy_5080"
MAIN_METHODS = ["local", "block_pam", "sensitivity_pam", "hybrid_pam", "rankaware_proxy_pam"]
CORE_METHODS = ["block_pam", "sensitivity_pam", "hybrid_pam"]
MAIN_ENVS = ["HM8", "HM12"]


def run_text(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep] + body)


def system_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "timestamp_utc": run_text(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]),
        "python": sys.version,
        "platform": platform.platform(),
        "os": os.name,
        "git_commit": run_text(["git", "rev-parse", "HEAD"]),
        "git_branch": run_text(["git", "branch", "--show-current"]),
        "git_dirty_short": run_text(["git", "status", "--short"]),
        "nvidia_smi": run_text(["nvidia-smi"]) if shutil.which("nvidia-smi") else "unavailable: nvidia-smi not found",
    }
    try:
        import torch

        manifest.update(
            {
                "torch_version": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "gpu_memory_total_mb": (
                    torch.cuda.get_device_properties(0).total_memory / 1e6 if torch.cuda.is_available() else None
                ),
            }
        )
    except Exception as exc:
        manifest.update(
            {
                "torch_version": None,
                "torch_cuda": None,
                "cuda_available": False,
                "gpu_available": False,
                "torch_error": f"{type(exc).__name__}: {exc}",
            }
        )
    return manifest


def write_system_manifest(report_root: Path) -> Path:
    path = report_root / "system_manifest.json"
    write_json(path, system_manifest())
    return path


def audit_status_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in summary_rows:
        env = str(row.get("env", ""))
        ordering = str(row.get("ordering", ""))
        if not env or not ordering:
            continue
        counts[(env, ordering)][str(row.get("status", "ok"))] += 1
    out = []
    for (env, ordering), counter in sorted(counts.items()):
        out.append(
            {
                "env": env,
                "ordering": ordering,
                "ok": counter.get("ok", 0),
                "oom": counter.get("oom", 0),
                "error": counter.get("error", 0),
                "unknown": sum(v for k, v in counter.items() if k not in {"ok", "oom", "error"}),
                "total": sum(counter.values()),
            }
        )
    return out


def ordering_definition_rows() -> list[dict[str, Any]]:
    _, orders = build_hardmove_orders(n_actuator=8, random_seed=0)
    rows = []
    for name in ORDERING_NAMES:
        meta = orders[name].metadata
        rows.append(
            {
                "ordering": name,
                "canonical": meta.get("canonical_name"),
                "category": meta.get("baseline_category"),
                "construction": meta.get("construction_method", ""),
                "hybrid_w": (
                    f"{meta.get('hybrid_physics_weight')}/{meta.get('hybrid_sensitivity_weight')}"
                    if "hybrid_physics_weight" in meta
                    else ""
                ),
                "deprecated": meta.get("deprecated_alias", False),
            }
        )
    return rows


def write_repository_audit(report_root: Path) -> Path:
    summary_rows = load_summary_rows()
    status_rows = audit_status_rows(summary_rows)
    ordering_rows = ordering_definition_rows()
    diagnostics_root = ROOT / "repro" / "diagnostics"
    text = [
        "# Repository Audit: RTX 5080 TTPI/PAM Prestudy",
        "",
        "## Scope",
        "This audit records the true current implementation before any supplementary 5080 prestudy reruns. It does not launch training.",
        "",
        "## True Entry Points",
        "- HardMove runner: `repro/scripts/run_hardmove.py`",
        "- Batch launcher: `repro/scripts/run_pam_ablation.py`",
        "- Ordering registry: `repro/pam_ordering.py`",
        "- Environment action wrapper: `repro/hardmove_variants.py`",
        "- Main prestudy entry: `python -m analysis.prestudy_5080`",
        "",
        "## Ordering Definitions",
        markdown_table(ordering_rows, ["ordering", "canonical", "category", "construction", "hybrid_w", "deprecated"]),
        "",
        "## Critical Implementation Facts",
        "- `sensitivity_pam` is a deterministic angular-distance coupling proxy, not a trajectory/gradient sensitivity estimator.",
        "- `hybrid_pam` is the block-preserving order induced by `0.5 * physical coupling + 0.5 * deterministic sensitivity coupling`.",
        "- `rankaware_proxy_pam` minimizes a static segmented cut-load proxy with `lambda_sum=1.0` and `lambda_peak=2.0`; it does not use real singular spectra.",
        "- New `sensitivity_lite_*` entries are prestudy ordering candidates only. They do not alter TTPI, dynamics, reward, grids, tolerances, or evaluation.",
        "",
        "## Existing Result Status",
        markdown_table(status_rows, ["env", "ordering", "ok", "oom", "error", "unknown", "total"]),
        "",
        "## Artifact Inventory",
        f"- Summary rows: {len(summary_rows)}",
        f"- Result JSON files: {count_files(ROOT / 'repro' / 'results', '*.json')}",
        f"- Rank profile CSV files: {count_files(diagnostics_root / 'rank_profiles', '*.csv')}",
        f"- TT-Cross query CSV files: {count_files(diagnostics_root / 'cross_queries', '*.csv')}",
        f"- TT-Cross process CSV files: {count_files(diagnostics_root / 'cross_process', '*.csv')}",
        f"- TT-Round event CSV files: {count_files(diagnostics_root / 'round_events', '*.csv')}",
        f"- Model/checkpoint files under `repro/models`: {count_files(ROOT / 'repro' / 'models', '*')}",
        f"- Singular spectrum files under `repro/diagnostics/spectra`: {count_files(diagnostics_root / 'spectra', '*')}",
        "",
        "## Field Mapping",
        "- `peak_memory_mb`: CUDA max allocated memory in MB when CUDA is available.",
        "- `runtime_sec` / `total_time_sec`: ordering construction plus TTPI training time.",
        "- `ordering_search_time_sec`: ordering construction only.",
        "- `ttpi_training_time_sec`: TTPI train call only.",
        "- `tt_cross_function_evals` and `queried_points_total`: wrapper-counted request totals; this is an upper-bound request count if tntorch caches internally.",
        "- `adv_rank_*` / `val_rank_*`: TT rank profiles recorded after PI/rank events; per-cut columns are present when the source rank vector is available.",
        "",
        "## Data Quality Notes",
        "- OOM rows are retained as first-class evidence.",
        "- Empty historical `status` values are normalized to `ok` only when no OOM/error flag is present.",
        "- Formal singular spectra are not present in the pulled data; existing spectra appear to be older diagnostics and are not used as formal evidence.",
    ]
    path = report_root / "repository_audit.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
    return path


def selected_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("env") in MAIN_ENVS and record.get("ordering") in MAIN_METHODS
    ]


def write_main_comparison(report_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = load_records(tag="formal5080")
    selected = selected_records(records)
    run_rows = summarize_records(selected)
    aggregate = aggregate_rows(run_rows)
    out_dir = report_root / "main_comparison"
    write_csv(out_dir / "run_level_summary.csv", run_rows)
    write_csv(out_dir / "summary_tables.csv", aggregate)

    tests: dict[str, Any] = {}
    for env in MAIN_ENVS:
        tests[env] = {
            metric: paired_method_tests(run_rows, env=env, methods=MAIN_METHODS, metric=metric, baseline="block_pam")
            for metric in ["peak_memory_mb", "adv_rank_auc", "queried_points_total", "success_rate"]
        }
        tests[env]["oom_rate"] = oom_rate_table(run_rows, env=env, methods=MAIN_METHODS)
    write_json(out_dir / "statistical_tests.json", tests)

    for env in MAIN_ENVS:
        env_rows = [row for row in run_rows if row.get("env") == env]
        env_records = [record for record in selected if record.get("env") == env]
        typical_seed = select_typical_seed(run_rows, env, CORE_METHODS)
        boxplot_by_method(env_rows, "peak_memory_mb", out_dir / f"{env}_peak_memory_boxplot.png", f"{env} peak memory", "MB")
        boxplot_by_method(
            env_rows,
            "queried_points_total",
            out_dir / f"{env}_total_queries_boxplot.png",
            f"{env} total TT-Cross queries",
            "queries",
        )
        boxplot_by_method(env_rows, "success_rate", out_dir / f"{env}_success_boxplot.png", f"{env} success", "success rate")
        bar_by_method(tests[env]["oom_rate"], "oom_rate", out_dir / f"{env}_oom_rate.png", f"{env} OOM rate", "OOM rate")
        rank_trajectory(env_records, out_dir / f"{env}_rank_trajectory_median.png", f"{env} rank trajectory", typical_seed)
        scatter(env_rows, "peak_memory_mb", "adv_rank_auc", out_dir / f"{env}_memory_rank_scatter.png", f"{env} memory vs rank-AUC")
        scatter(
            env_rows,
            "queried_points_total",
            "adv_rank_auc",
            out_dir / f"{env}_query_rank_auc_scatter.png",
            f"{env} queries vs rank-AUC",
        )

    summary_lines = [
        "# Main Comparison Summary",
        "",
        "Methods: `local`, `block_pam`, `sensitivity_pam`, `hybrid_pam`, `rankaware_proxy_pam`.",
        "Environments: `HM8`, `HM12` from existing `formal5080` artifacts.",
        "",
        "## Aggregate Table",
        markdown_table(
            aggregate,
            [
                "env",
                "ordering",
                "runs",
                "ok_runs",
                "oom_runs",
                "oom_rate",
                "peak_memory_mb_mean",
                "adv_rank_auc_mean",
                "queried_points_total_mean",
                "success_rate_mean",
            ],
        ),
        "",
        "## Typical Seeds",
    ]
    for env in MAIN_ENVS:
        summary_lines.append(f"- {env}: {select_typical_seed(run_rows, env, CORE_METHODS)}")
    summary_lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "- OOM rows are included in OOM-rate and partial-rank evidence.",
            "- Success-rate comparisons use completed run values; effective success with OOM-as-zero should be interpreted from `ok_runs` and `oom_rate` together.",
            "- Statistical tests are paired by seed and automatically downgrade to descriptive-only when paired seed count is too small.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return run_rows, selected


def write_rankaware_failure(report_root: Path, run_rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    out_dir = report_root / "rankaware_failure"
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = ["block_pam", "sensitivity_pam", "hybrid_pam", "rankaware_proxy_pam"]
    perm_rows = []
    for n_actuator in [8, 12]:
        perm_rows.extend(permutation_metrics(n_actuator, methods, random_seed=0))
    write_csv(out_dir / "permutation_metrics.csv", perm_rows)

    corr_rows = []
    for env in MAIN_ENVS:
        env_rows = [row for row in run_rows if row.get("env") == env]
        xs = [row.get("pam_order_rankaware_proxy_objective") for row in env_rows]
        for target in ["peak_memory_mb", "adv_rank_auc", "queried_points_total"]:
            corr_rows.append(
                {
                    "env": env,
                    "x": "pam_order_rankaware_proxy_objective",
                    "y": target,
                    "pearson": pearson_correlation(
                        [x for x in xs if isinstance(x, (int, float))],
                        [row.get(target) for row in env_rows if isinstance(row.get("pam_order_rankaware_proxy_objective"), (int, float))],
                    ),
                    "n": sum(1 for row in env_rows if isinstance(row.get(target), (int, float))),
                }
            )
    write_csv(out_dir / "proxy_actual_correlations.csv", corr_rows)

    rankaware_records = [
        record
        for record in records
        if record.get("env") in MAIN_ENVS and record.get("ordering") in methods
    ]
    rank_trajectory(rankaware_records, out_dir / "rankaware_vs_baselines_rank_trajectory.png", "Rankaware vs baselines")
    cutwise_rank_heatmap(
        [record for record in rankaware_records if record.get("ordering") == "rankaware_proxy_pam"],
        out_dir / "rankaware_cutwise_rank_heatmap.png",
        "Rankaware proxy cutwise rank bottlenecks",
    )
    scatter(
        [row for row in run_rows if row.get("ordering") in methods],
        "pam_order_rankaware_proxy_objective",
        "peak_memory_mb",
        out_dir / "static_proxy_vs_peak_memory.png",
        "Static proxy vs peak memory",
    )
    scatter(
        [row for row in run_rows if row.get("ordering") in methods],
        "pam_order_rankaware_proxy_objective",
        "adv_rank_auc",
        out_dir / "static_proxy_vs_actual_rank_auc.png",
        "Static proxy vs actual rank-AUC",
    )
    bar_by_method(
        [row for row in run_rows if row.get("ordering") in methods and row.get("env") == "HM12"],
        "last_completed_pi",
        out_dir / "HM12_oom_iteration_comparison.png",
        "HM12 last completed PI before stop/OOM",
        "last completed PI",
    )

    rankaware_hm12 = [
        row for row in run_rows if row.get("env") == "HM12" and row.get("ordering") == "rankaware_proxy_pam"
    ]
    hm12_rankaware_all_oom = bool(rankaware_hm12) and all(row.get("status") == "oom" or row.get("oom") for row in rankaware_hm12)
    text = [
        "# Rankaware Proxy Failure Analysis",
        "",
        "## Evidence Grades",
        "- directly proven: directly read from logs or deterministic ordering definitions.",
        "- strong evidence: repeated across seeds but not a formal causal intervention.",
        "- conjecture: plausible mechanism requiring checkpoint/spectrum evidence.",
        "- cannot verify: data missing in current artifacts.",
        "",
        "## Findings",
        f"- directly proven: `rankaware_proxy_pam` uses a static cut-load proxy, not singular spectra or dynamic TT-Cross ranks.",
        f"- directly proven: HM12 rankaware proxy all-OOM status is `{hm12_rankaware_all_oom}` in the loaded formal artifacts.",
        "- strong evidence: static proxy values should be interpreted against actual rank-AUC, peak memory, and query counts in `proxy_actual_correlations.csv`.",
        "- strong evidence: cut-level rank concentration is inspectable through `rankaware_cutwise_rank_heatmap.png` because per-cut rank columns exist.",
        "- conjecture: mismatch may come from optimizing average/log cut load while sacrificing dynamic TT-Cross bottleneck cuts.",
        "- cannot verify: formal singular-spectrum stable rank and entropy effective rank are not available in the pulled artifacts.",
        "",
        "## Output Files",
        "- `permutation_metrics.csv`",
        "- `proxy_actual_correlations.csv`",
        "- `rankaware_vs_baselines_rank_trajectory.png/.pdf`",
        "- `rankaware_cutwise_rank_heatmap.png/.pdf`",
    ]
    (out_dir / "rankaware_failure_analysis.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def write_prestudy_placeholders(report_root: Path) -> None:
    formal_records = load_records(tag="formal5080")
    lite_records = load_records(
        results_root=ROOT / "results" / "prestudy_5080" / "sensitivity_lite" / "results",
        diagnostics_root=ROOT / "results" / "prestudy_5080" / "sensitivity_lite" / "diagnostics",
        tag="prestudy5080lite",
    )
    hybrid_records = load_records(
        results_root=ROOT / "results" / "prestudy_5080" / "hybrid_ablation" / "results",
        diagnostics_root=ROOT / "results" / "prestudy_5080" / "hybrid_ablation" / "diagnostics",
        tag="prestudy5080hybrid",
    )
    sensitivity_dir = report_root / "sensitivity_lite"
    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    sensitivity_reference = [
        record
        for record in formal_records
        if record.get("env") == "HM8" and record.get("ordering") in {"block_pam", "sensitivity_pam"}
    ]
    sensitivity_rows = summarize_records(sensitivity_reference + lite_records)
    construction_rows = []
    for record in sensitivity_reference + lite_records:
        result = record.get("result", {})
        meta = result.get("pam_order_metadata") or {}
        if "construction_method" not in meta and record.get("ordering") == "sensitivity_pam":
            meta = {
                **meta,
                "construction_method": "deterministic_angular_distance_proxy",
                "trajectory_count": 0,
                "sensitivity_function_calls": 0,
                "construction_fallback": "metadata_backfilled_from_current_code_audit",
            }
        elif "construction_method" not in meta and record.get("ordering") == "block_pam":
            meta = {
                **meta,
                "construction_method": "physical_coupling_block_preserving_spectral_2opt",
                "trajectory_count": 0,
                "sensitivity_function_calls": 0,
                "construction_fallback": "metadata_backfilled_from_current_code_audit",
            }
        construction_rows.append(
            {
                "env": record.get("env"),
                "ordering": record.get("ordering"),
                "seed": record.get("seed"),
                "status": record.get("status"),
                "ordering_search_time_sec": result.get("ordering_search_time_sec", ""),
                "construction_method": meta.get("construction_method", ""),
                "trajectory_count": meta.get("trajectory_count", ""),
                "fd_eps": meta.get("fd_eps", ""),
                "sensitivity_function_calls": meta.get("sensitivity_function_calls", ""),
                "construction_fallback": meta.get("construction_fallback", ""),
            }
        )
    similarity_rows = []
    by_seed = {(record.get("env"), record.get("seed"), record.get("ordering")): record for record in sensitivity_reference + lite_records}
    for record in lite_records:
        ref = by_seed.get((record.get("env"), record.get("seed"), "sensitivity_pam"))
        if not ref:
            continue
        order = record.get("result", {}).get("action_order") or []
        ref_order = ref.get("result", {}).get("action_order") or []
        if order and ref_order and len(order) == len(ref_order):
            similarity_rows.append(
                {
                    "env": record.get("env"),
                    "seed": record.get("seed"),
                    "ordering": record.get("ordering"),
                    "reference": "sensitivity_pam",
                    "kendall_tau_distance": kendall_tau_distance(order, ref_order),
                    "exact_match": order == ref_order,
                }
            )
    write_csv(sensitivity_dir / "construction_cost.csv", construction_rows)
    write_csv(sensitivity_dir / "permutation_similarity.csv", similarity_rows)
    write_csv(sensitivity_dir / "run_level_summary.csv", sensitivity_rows)
    if lite_records:
        boxplot_by_method(
            sensitivity_rows,
            "peak_memory_mb",
            sensitivity_dir / "sensitivity_lite_peak_memory.png",
            "Sensitivity lite peak memory",
            "MB",
        )
        boxplot_by_method(
            sensitivity_rows,
            "adv_rank_auc",
            sensitivity_dir / "sensitivity_lite_rank_auc.png",
            "Sensitivity lite rank-AUC",
            "rank-AUC",
        )
    (sensitivity_dir / "sensitivity_lite_analysis.md").write_text(
        "\n".join(
            [
                "# Sensitivity Lite Analysis",
                "",
                "The current `sensitivity_pam` is already a deterministic, low-cost angular-distance proxy. Therefore `sensitivity_lite_fd5` and `sensitivity_lite_first_order` are new empirical/action-effect prestudy candidates, not reduced-budget versions of the current method.",
                "",
                "Run `bash scripts/run_prestudy_5080.sh sensitivity-lite` on the 5080 server to generate HM8 seed 0-2 results under `results/prestudy_5080/sensitivity_lite/`.",
                "",
                f"Loaded lite result rows: {len(lite_records)}.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    hybrid_dir = report_root / "hybrid_ablation"
    hybrid_dir.mkdir(parents=True, exist_ok=True)
    hybrid_rows = summarize_records(hybrid_records)
    write_csv(hybrid_dir / "run_level_summary.csv", hybrid_rows)
    write_csv(hybrid_dir / "summary_tables.csv", aggregate_rows(hybrid_rows) if hybrid_rows else [])
    objective_rows = []
    for record in hybrid_records:
        result = record.get("result", {})
        meta = result.get("pam_order_metadata") or {}
        objective_rows.append(
            {
                "env": record.get("env"),
                "ordering": record.get("ordering"),
                "seed": record.get("seed"),
                "status": record.get("status"),
                "hybrid_physics_weight": meta.get("hybrid_physics_weight", ""),
                "hybrid_sensitivity_weight": meta.get("hybrid_sensitivity_weight", ""),
                "construction_method": meta.get("construction_method", ""),
                "la_objective": result.get("pam_order_objective", ""),
                "peak_cut_objective": result.get("pam_order_peak_cut_objective", ""),
                "rankaware_proxy_objective": result.get("pam_order_rankaware_proxy_objective", ""),
            }
        )
    write_csv(hybrid_dir / "objective_components.csv", objective_rows)
    if hybrid_records:
        boxplot_by_method(hybrid_rows, "peak_memory_mb", hybrid_dir / "hybrid_peak_memory.png", "Hybrid ablation peak memory", "MB")
        boxplot_by_method(hybrid_rows, "adv_rank_auc", hybrid_dir / "hybrid_rank_auc.png", "Hybrid ablation rank-AUC", "rank-AUC")
    (hybrid_dir / "objective_definition.md").write_text(
        "\n".join(
            [
                "# Hybrid Ablation Objective Definition",
                "",
                "- `hybrid_block_only`: block legality only; zero objective; local block order tie-break.",
                "- `hybrid_block_plus_physics`: physical HardMove coupling objective.",
                "- `hybrid_block_plus_sensitivity`: deterministic angular-distance sensitivity coupling objective.",
                "- `hybrid_current`: `0.5 * physical + 0.5 * deterministic sensitivity`.",
                "",
                "All variants preserve the TTPI training configuration and change only the action-mode ordering.",
                "",
                f"Loaded hybrid ablation result rows: {len(hybrid_records)}.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def analyze_existing(report_root: Path) -> None:
    run_rows, records = write_main_comparison(report_root)
    write_rankaware_failure(report_root, run_rows, records)
    write_prestudy_placeholders(report_root)


def one_page_report(report_root: Path) -> Path:
    summary_path = report_root / "main_comparison" / "summary_tables.csv"
    rows = []
    if summary_path.exists():
        import csv

        with summary_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    hm16_rows = [row for row in load_summary_rows() if row.get("env") == "HM16"]
    hm16_oom = sum(1 for row in hm16_rows if row.get("status") == "oom" or row.get("oom"))
    hm16_total = len(hm16_rows)
    core_rows = [
        row
        for row in rows
        if row.get("env") in MAIN_ENVS and row.get("ordering") in ["block_pam", "sensitivity_pam", "hybrid_pam"]
    ]
    text = [
        "# TTPI 动作模式排序预实验与高显存服务器需求",
        "",
        "## Research Question",
        "本预实验检验动作 mode ordering 是否能稳定降低 TTPI 的 TT-rank、显存与 TT-Cross 查询负担，并确认 RTX 5080 是否足以支撑正式 HM12/HM16 消融。",
        "",
        "## RTX 5080 Findings",
        markdown_table(
            core_rows,
            [
                "env",
                "ordering",
                "runs",
                "ok_runs",
                "oom_runs",
                "peak_memory_mb_mean",
                "adv_rank_auc_mean",
                "queried_points_total_mean",
                "success_rate_mean",
            ],
        ),
        "",
        f"HM16 当前 formal 配置下 OOM/total = {hm16_oom}/{hm16_total}，因此 5080 不适合作为 HM16 全量正式实验平台。",
        "",
        "## Rankaware Proxy Mismatch",
        "`rankaware_proxy_pam` 目前只优化静态 cut-load proxy。已有日志允许比较 proxy、真实 rank-AUC、显存和 query；详细机制见 `rankaware_failure/rankaware_failure_analysis.md`。",
        "",
        "## Hardware Bottleneck",
        "瓶颈不是单纯训练时间，而是 TT-Cross 动态 rank burst 带来的显存峰值和 OOM。OOM 记录被保留为服务器申请证据。",
        "",
        "## Server Plan",
        "先在 5080 上只补 HM8 seed 0-2 的 `sensitivity_lite_*` 和 `hybrid_*` objective-toggle 预实验；HM12/HM16 全量实验等待更高显存服务器确认后再启动。",
        "",
        "## Resource Request",
        "服务器申请应基于当前 `peak_memory_mb`、OOM rate、rank-AUC 和 query 证据，不伪造服务器型号。建议申请显存显著高于 5080 的 GPU，并优先完成 HM12/HM16 多 seed 正式消融。",
    ]
    md_path = report_root / "prestudy_one_page.md"
    md_path.write_text("\n".join(text) + "\n", encoding="utf-8")
    html_path = report_root / "prestudy_one_page.html"
    html_path.write_text(
        "<html><head><meta charset='utf-8'><title>TTPI Prestudy 5080</title></head><body><pre>"
        + "\n".join(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        + "</pre></body></html>\n",
        encoding="utf-8",
    )
    pdf_note = report_root / "prestudy_one_page.pdf.missing.txt"
    pdf_note.write_text("PDF was not generated by default; Markdown and HTML reports are available.\n", encoding="utf-8")
    return md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["audit", "analyze-existing", "report", "all"], default="all")
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    args = parser.parse_args(argv)

    args.report_root.mkdir(parents=True, exist_ok=True)
    if args.stage in {"audit", "all"}:
        write_system_manifest(args.report_root)
        write_repository_audit(args.report_root)
    if args.stage in {"analyze-existing", "all"}:
        analyze_existing(args.report_root)
    if args.stage in {"report", "all"}:
        if not (args.report_root / "main_comparison" / "summary_tables.csv").exists():
            analyze_existing(args.report_root)
        one_page_report(args.report_root)
    print(f"[DONE] prestudy stage {args.stage} -> {args.report_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
