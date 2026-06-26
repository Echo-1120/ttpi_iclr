#!/usr/bin/env python3
"""Synchronize experiment artifacts into paper tables and notes.

This script is intentionally conservative: it reports exactly what is present in
the repository and avoids upgrading partially supported claims.
"""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
NOTES = ROOT / "notes"
DIAG = ROOT / "repro" / "diagnostics"


FIGURE_FILES = [
    "singular_value_decay_by_cut.png",
    "effective_rank_profile_by_ordering.png",
    "tt_actual_rank_profile_by_ordering.png",
    "spectral_proxy_vs_actual_tt_rank.png",
    "peak_memory_vs_ordering.png",
    "runtime_vs_ordering.png",
    "la_score_vs_peak_memory_scatter.png",
    "peakcut_vs_peak_memory_scatter.png",
    "rankaware_score_vs_peak_memory_scatter.png",
    "reordered_coupling_heatmaps.png",
    "performance_vs_memory_pareto.png",
]


def run_git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return ""


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value not in ("", None) else default
    except Exception:
        return default


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fmt(value: str | float | None, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "--"


def write_summary_table(rows: list[dict[str, str]]) -> None:
    out = PAPER / "tables" / "pam_diagnostics_summary.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text(
            "\\begin{table}[t]\n"
            "\\centering\n"
            "\\caption{PAM diagnostics summary. No diagnostic rows were found.}\n"
            "\\label{tab:pam-diagnostics-summary}\n"
            "\\begin{tabular}{ll}\n\\toprule\nStatus & Missing CSV \\\\\n\\bottomrule\n\\end{tabular}\n"
            "\\end{table}\n",
            encoding="utf-8",
        )
        return

    rows = sorted(rows, key=lambda r: (r.get("env", ""), f(r.get("peak_memory_mb"))))
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Repository-synchronized PAM diagnostics. Smoke results validate diagnostics but do not support performance claims.}",
        "\\label{tab:pam-diagnostics-summary}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.4pt}",
        "\\begin{tabular}{@{}lrrrrrrr@{}}",
        "\\toprule",
        "Ord. & LA & Cut & MB & s & Evals & Succ. & $\\mu$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{tex_escape(row.get('ordering', ''))} & "
            f"{fmt(row.get('la_objective'), 0)} & "
            f"{fmt(row.get('peak_cut_objective'), 0)} & "
            f"{fmt(row.get('peak_memory_mb'), 1)} & "
            f"{fmt(row.get('runtime_sec'), 2)} & "
            f"{fmt(row.get('tt_cross_function_evals'), 0)} & "
            f"{fmt(row.get('success_rate'), 2)} & "
            f"{fmt(row.get('mu'), 2)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\normalsize", "\\end{table}", ""]
    out.write_text("\n".join(lines), encoding="utf-8")


def write_main_ablation_table(rows: list[dict[str, str]]) -> None:
    out = PAPER / "tables" / "pam_ablation_main.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text(
            "\\begin{tabular}{ll}\n\\toprule\nStatus & Missing CSV \\\\\n\\bottomrule\n\\end{tabular}\n",
            encoding="utf-8",
        )
        return
    rows = sorted(rows, key=lambda r: (r.get("env", ""), f(r.get("peak_memory_mb"))))
    lines = [
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.4pt}",
        "\\begin{tabular}{@{}lrrrrrr@{}}",
        "\\toprule",
        "Ord. & MB & s & Succ. & $\\mu$ & A-rank & Evals \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{tex_escape(row.get('ordering', ''))} & "
            f"{fmt(row.get('peak_memory_mb'), 1)} & "
            f"{fmt(row.get('runtime_sec'), 2)} & "
            f"{fmt(row.get('success_rate'), 2)} & "
            f"{fmt(row.get('mu'), 2)} & "
            f"{fmt(row.get('adv_rank_max'), 1)} & "
            f"{fmt(row.get('tt_cross_function_evals'), 0)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\normalsize", ""]
    out.write_text("\n".join(lines), encoding="utf-8")


def write_final_table(rows: list[dict[str, str]]) -> None:
    out = PAPER / "tables" / "pam_final_table.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("% No final aggregate table available yet.\n", encoding="utf-8")
        return
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Aggregate PAM diagnostics by environment and ordering.}",
        "\\label{tab:pam-final-table}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Env & Ordering & Seeds & Mem. MB & Time s & Evals \\\\",
        "\\midrule",
    ]
    for row in sorted(rows, key=lambda r: (r.get("env", ""), r.get("ordering", ""))):
        lines.append(
            f"{tex_escape(row.get('env', ''))} & "
            f"{tex_escape(row.get('ordering', ''))} & "
            f"{tex_escape(row.get('n_seeds', ''))} & "
            f"{fmt(row.get('peak_memory_mb_mean'), 1)} & "
            f"{fmt(row.get('runtime_sec_mean'), 2)} & "
            f"{fmt(row.get('tt_cross_function_evals_mean'), 0)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    out.write_text("\n".join(lines), encoding="utf-8")


def scan_oom_logs() -> list[str]:
    hits: list[str] = []
    for path in sorted((ROOT / "repro" / "logs").glob("*.log")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        lowered = text.lower()
        if "out of memory" in lowered or "oom" in lowered or "cuda error" in lowered:
            hits.append(str(path.relative_to(ROOT)))
    return hits


def figure_inventory() -> list[tuple[str, bool]]:
    base = ROOT / "repro" / "figures" / "pam_diagnostics"
    return [(name, (base / name).exists()) for name in FIGURE_FILES]


def write_experiment_log(summary_rows: list[dict[str, str]], final_rows: list[dict[str, str]]) -> None:
    out = NOTES / "experiment_log.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "repro" / "experiments" / "pam_ablation_manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    envs = sorted({r.get("env", "") for r in summary_rows if r.get("env")})
    orderings = sorted({r.get("ordering", "") for r in summary_rows if r.get("ordering")})
    seeds = sorted({r.get("seed", "") for r in summary_rows if r.get("seed")})
    oom_hits = scan_oom_logs()
    git_head = run_git(["rev-parse", "--short", "HEAD"])
    git_status = run_git(["status", "--short"])
    changed_results = run_git(["diff", "--stat", "HEAD", "--", "repro/results", "repro/diagnostics", "repro/figures"])

    lines = [
        "# Experiment Log",
        "",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        f"Git HEAD: `{git_head}`",
        "",
        "## Current Artifact Summary",
        "",
        f"- Summary rows: {len(summary_rows)}",
        f"- Aggregate rows: {len(final_rows)}",
        f"- Environments: {', '.join(envs) if envs else 'none'}",
        f"- Orderings: {', '.join(orderings) if orderings else 'none'}",
        f"- Seeds: {', '.join(seeds) if seeds else 'none'}",
        "",
        "## Experiment Configuration",
        "",
        f"- Manifest: `repro/experiments/pam_ablation_manifest.json`",
        f"- Manifest environments: {len(manifest.get('environments', []))}",
        f"- Manifest orderings: {len(manifest.get('orderings', []))}",
        f"- Manifest seeds: {manifest.get('seeds', [])}",
        "",
        "## OOM / Abnormal Runs",
        "",
    ]
    if oom_hits:
        lines += [f"- `{hit}`" for hit in oom_hits]
    else:
        lines.append("- No OOM log hits found in `repro/logs/*.log`.")

    lines += [
        "",
        "## Figure Inventory",
        "",
    ]
    for name, exists in figure_inventory():
        status = "present" if exists else "missing"
        lines.append(f"- `{name}`: {status}")

    lines += [
        "",
        "## Changes Versus Current Commit",
        "",
        "Tracked/untracked working tree:",
        "",
        "```",
        git_status or "clean",
        "```",
        "",
        "Experiment artifact diff stat:",
        "",
        "```",
        changed_results or "no artifact diff",
        "```",
        "",
        "## Interpretation Guardrails",
        "",
        "- Do not claim control-performance improvement from smoke runs.",
        "- Promote claims only after `notes/claims.md` status is updated with multi-seed evidence.",
        "- Keep paper tables generated from CSV rather than hand-edited numbers.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    summary_rows = read_csv(DIAG / "pam_ablation_summary.csv")
    final_rows = read_csv(DIAG / "pam_ablation_final_table.csv")
    write_main_ablation_table(summary_rows)
    write_summary_table(summary_rows)
    write_final_table(final_rows)
    write_experiment_log(summary_rows, final_rows)
    print("[DONE] wrote paper/tables/pam_ablation_main.tex")
    print("[DONE] wrote paper/tables/pam_diagnostics_summary.tex")
    print("[DONE] wrote paper/tables/pam_final_table.tex")
    print("[DONE] wrote notes/experiment_log.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
