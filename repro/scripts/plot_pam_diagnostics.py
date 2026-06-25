#!/usr/bin/env python3
"""Generate PAM diagnostic figures from CSV logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from repro.pam_ordering import build_hardmove_orders


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_many(pattern: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(ROOT.glob(pattern)):
        rows.extend(read_csv(path))
    return rows


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value not in ("", None) else default
    except Exception:
        return default


def grouped_mean(rows: list[dict[str, str]], group_key: str, value_key: str) -> tuple[list[str], list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row.get(group_key, "unknown")].append(f(row, value_key))
    labels = sorted(grouped)
    values = [sum(grouped[k]) / max(1, len(grouped[k])) for k in labels]
    return labels, values


def save_placeholder(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.set_title(title)
    ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def bar_plot(rows: list[dict[str, str]], value_key: str, title: str, ylabel: str, path: Path) -> None:
    if not rows:
        save_placeholder(path, title, f"No summary rows found for {value_key}.")
        return
    labels, values = grouped_mean(rows, "ordering", value_key)
    fig, ax = plt.subplots(figsize=(max(8, 0.65 * len(labels)), 4.8))
    ax.bar(labels, values, color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def scatter_plot(rows: list[dict[str, str]], x_key: str, y_key: str, title: str, path: Path) -> None:
    if not rows:
        save_placeholder(path, title, f"No summary rows found for {x_key} vs {y_key}.")
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    labels = sorted({row.get("ordering", "unknown") for row in rows})
    colors = plt.cm.tab20([i / max(1, len(labels) - 1) for i in range(len(labels))])
    cmap = dict(zip(labels, colors))
    for row in rows:
        label = row.get("ordering", "unknown")
        ax.scatter(f(row, x_key), f(row, y_key), color=cmap[label], label=label, alpha=0.82, s=42)
    handles, handle_labels = ax.get_legend_handles_labels()
    unique = dict(zip(handle_labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=8, ncol=2)
    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.grid(alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_rank_profile(rank_rows: list[dict[str, str]], path: Path) -> None:
    title = "TT actual rank profile by cut"
    if not rank_rows:
        save_placeholder(path, title, "No rank profile CSV files found.")
        return
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rank_rows:
        ordering = row.get("ordering", "unknown")
        for key, value in row.items():
            if key.startswith("adv_rank_") and key[len("adv_rank_") :].isdigit():
                grouped[ordering][int(key.split("_")[-1])].append(float(value))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for ordering, cut_values in sorted(grouped.items()):
        cuts = sorted(cut_values)
        vals = [sum(cut_values[c]) / len(cut_values[c]) for c in cuts]
        ax.plot(cuts, vals, marker="o", linewidth=1.7, label=ordering)
    ax.set_title(title)
    ax.set_xlabel("TT cut")
    ax.set_ylabel("advantage TT rank")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_rank_by_iteration(rank_rows: list[dict[str, str]], path: Path) -> None:
    title = "Rank profile by cut and iteration"
    if not rank_rows:
        save_placeholder(path, title, "No rank profile CSV files found.")
        return
    grouped: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rank_rows:
        iteration = int(float(row.get("iteration", 0) or 0))
        for key, value in row.items():
            if key.startswith("adv_rank_") and key[len("adv_rank_") :].isdigit():
                grouped[iteration][int(key.split("_")[-1])].append(float(value))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for iteration in sorted(grouped):
        cuts = sorted(grouped[iteration])
        vals = [sum(grouped[iteration][c]) / len(grouped[iteration][c]) for c in cuts]
        ax.plot(cuts, vals, marker=".", alpha=0.65, label=f"iter {iteration}")
    ax.set_title(title)
    ax.set_xlabel("TT cut")
    ax.set_ylabel("mean advantage TT rank")
    ax.grid(alpha=0.25)
    if len(grouped) <= 12:
        ax.legend(fontsize=7, ncol=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_singular_decay(sv_rows: list[dict[str, str]], path: Path) -> None:
    title = "Singular value decay by cut"
    if not sv_rows:
        save_placeholder(path, title, "No singular spectrum CSV files found.")
        return
    grouped: dict[tuple[str, int], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in sv_rows:
        key = (row.get("ordering", "unknown"), int(float(row.get("cut", 0) or 0)))
        grouped[key][int(float(row.get("singular_index", 0) or 0))].append(f(row, "singular_value"))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for (ordering, cut), vals_by_idx in sorted(grouped.items())[:24]:
        idxs = sorted(vals_by_idx)
        vals = [sum(vals_by_idx[i]) / len(vals_by_idx[i]) for i in idxs]
        if vals and vals[0] != 0:
            vals = [v / abs(vals[0]) for v in vals]
        ax.semilogy(idxs, vals, linewidth=1.4, alpha=0.75, label=f"{ordering} c{cut}")
    ax.set_title(title)
    ax.set_xlabel("singular value index")
    ax.set_ylabel("normalized singular value")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_effective_rank(eff_rows: list[dict[str, str]], path: Path) -> None:
    title = "Effective rank profile by ordering"
    if not eff_rows:
        save_placeholder(path, title, "No effective-rank CSV files found.")
        return
    rank_key = next((k for k in eff_rows[0] if k.startswith("effective_rank_relerr_")), None)
    if rank_key is None:
        save_placeholder(path, title, "Effective-rank columns are missing.")
        return
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in eff_rows:
        grouped[row.get("ordering", "unknown")][int(float(row.get("cut", 0) or 0))].append(f(row, rank_key))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for ordering, cut_values in sorted(grouped.items()):
        cuts = sorted(cut_values)
        vals = [sum(cut_values[c]) / len(cut_values[c]) for c in cuts]
        ax.plot(cuts, vals, marker="o", linewidth=1.7, label=ordering)
    ax.set_title(title)
    ax.set_xlabel("TT cut")
    ax.set_ylabel(rank_key)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_proxy_vs_actual(eff_rows: list[dict[str, str]], rank_rows: list[dict[str, str]], path: Path) -> None:
    title = "Spectral proxy vs actual TT rank"
    if not eff_rows or not rank_rows:
        save_placeholder(path, title, "Need both singular spectra and TT rank profiles.")
        return
    rank_key = next((k for k in eff_rows[0] if k.startswith("effective_rank_relerr_")), None)
    actual: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rank_rows:
        ordering = row.get("ordering", "unknown")
        for key, value in row.items():
            if key.startswith("adv_rank_") and key[len("adv_rank_") :].isdigit():
                actual[(ordering, int(key.split("_")[-1]))].append(float(value))
    proxy: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in eff_rows:
        proxy[(row.get("ordering", "unknown"), int(float(row.get("cut", 0) or 0)))].append(f(row, rank_key))
    xs, ys = [], []
    for key in sorted(set(actual) & set(proxy)):
        xs.append(sum(proxy[key]) / len(proxy[key]))
        ys.append(sum(actual[key]) / len(actual[key]))
    if not xs:
        save_placeholder(path, title, "No matching ordering/cut pairs found.")
        return
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(xs, ys, color="#F58518", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("spectral effective rank")
    ax.set_ylabel("actual advantage TT rank")
    ax.grid(alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(path: Path, n_actuator: int) -> None:
    coupling, orders = build_hardmove_orders(n_actuator)
    names = ["local", "block_pam", "rankaware_proxy_pam", "hybrid_pam"]
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.8))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        order = orders[name].order
        matrix = [[coupling[i][j] for j in order] for i in order]
        ax.imshow(matrix, cmap="viridis")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Reordered coupling heatmaps")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-dir", type=Path, default=ROOT / "repro" / "diagnostics")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "repro" / "figures" / "pam_diagnostics")
    parser.add_argument("--n-actuator", type=int, default=8)
    args = parser.parse_args()

    summary_rows = read_csv(args.diagnostics_dir / "pam_ablation_summary.csv")
    rank_rows = read_many("repro/diagnostics/rank_profiles/*.csv")
    sv_rows = read_many("repro/diagnostics/spectra/*.singular_values.csv")
    eff_rows = read_many("repro/diagnostics/spectra/*.effective_ranks.csv")
    out = args.out_dir

    plot_singular_decay(sv_rows, out / "singular_value_decay_by_cut.png")
    plot_effective_rank(eff_rows, out / "effective_rank_profile_by_ordering.png")
    plot_rank_profile(rank_rows, out / "tt_actual_rank_profile_by_ordering.png")
    plot_proxy_vs_actual(eff_rows, rank_rows, out / "spectral_proxy_vs_actual_tt_rank.png")
    plot_rank_by_iteration(rank_rows, out / "rank_profile_by_cut_and_iteration.png")
    bar_plot(summary_rows, "peak_memory_mb", "Peak memory by ordering", "MB", out / "peak_memory_vs_ordering.png")
    bar_plot(summary_rows, "runtime_sec", "Runtime by ordering", "seconds", out / "runtime_vs_ordering.png")
    scatter_plot(summary_rows, "la_objective", "peak_memory_mb", "LA score vs peak memory", out / "la_score_vs_peak_memory_scatter.png")
    scatter_plot(summary_rows, "peak_cut_objective", "peak_memory_mb", "Peak-cut cost vs peak memory", out / "peakcut_vs_peak_memory_scatter.png")
    scatter_plot(summary_rows, "rankaware_proxy_objective", "peak_memory_mb", "Rank-aware score vs peak memory", out / "rankaware_score_vs_peak_memory_scatter.png")
    plot_heatmaps(out / "reordered_coupling_heatmaps.png", args.n_actuator)
    scatter_plot(summary_rows, "peak_memory_mb", "best_s_times_mu", "Performance-memory Pareto", out / "performance_vs_memory_pareto.png")
    print(f"[DONE] wrote figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
