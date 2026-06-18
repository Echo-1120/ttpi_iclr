"""Aggregate and plot planar pushing PAM proxy runs.

Inputs:
  repro/results/planar_pushing_pam_proxy_proxy_seed{0,1,2}_score.summary.csv

Outputs:
  repro/results/planar_pushing_pam_proxy_3seed_summary.csv
  repro/figures/planar_pushing/planar_pushing_pam_proxy_3seed_max_rank.png
  repro/figures/planar_pushing/planar_pushing_pam_proxy_3seed_mean_rank.png
  repro/figures/planar_pushing/planar_pushing_pam_proxy_3seed_storage.png
"""
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "repro" / "results"
FIGURES = ROOT / "repro" / "figures" / "planar_pushing"
SUMMARY = RESULTS / "planar_pushing_pam_proxy_3seed_summary.csv"

CASE_ORDER = [
    "baseline_spa",
    "pam_local",
    "pam_contact_local",
    "pam_theta_contact",
    "bad_locality_only",
    "bad_split",
    "face_local_only",
]


def mean_std(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def load_rows(seeds):
    rows = []
    for seed in seeds:
        path = RESULTS / f"planar_pushing_pam_proxy_proxy_seed{seed}_score.summary.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["seed"] = seed
                row["max_rank"] = float(row["max_rank"])
                row["mean_rank"] = float(row["mean_rank"])
                row["storage"] = float(row["storage"])
                row["peak_memory_gb"] = float(row["peak_memory_gb"])
                row["time_s"] = float(row["time_s"])
                rows.append(row)
    expected = {(case, seed) for case in CASE_ORDER for seed in seeds}
    present = {(r["case"], r["seed"]) for r in rows}
    missing = sorted(expected - present)
    if missing:
        raise RuntimeError(f"Missing rows: {missing}")
    return rows


def aggregate(rows):
    by_case = {case: [r for r in rows if r["case"] == case] for case in CASE_ORDER}
    baseline_storage = [r["storage"] for r in by_case["baseline_spa"]]
    baseline_rank = [r["max_rank"] for r in by_case["baseline_spa"]]

    summary = []
    for case in CASE_ORDER:
        group = by_case[case]
        max_mean, max_std = mean_std([r["max_rank"] for r in group])
        mean_rank_mean, mean_rank_std = mean_std([r["mean_rank"] for r in group])
        storage_mean, storage_std = mean_std([r["storage"] for r in group])
        mem_mean, mem_std = mean_std([r["peak_memory_gb"] for r in group])
        time_mean, time_std = mean_std([r["time_s"] for r in group])
        storage_ratio = statistics.mean(
            r["storage"] / b for r, b in zip(group, baseline_storage)
        )
        rank_ratio = statistics.mean(
            r["max_rank"] / b for r, b in zip(group, baseline_rank)
        )
        summary.append(
            {
                "case": case,
                "n_seeds": len(group),
                "max_rank_mean": max_mean,
                "max_rank_std": max_std,
                "mean_rank_mean": mean_rank_mean,
                "mean_rank_std": mean_rank_std,
                "storage_mean": storage_mean,
                "storage_std": storage_std,
                "peak_memory_gb_mean": mem_mean,
                "peak_memory_gb_std": mem_std,
                "time_s_mean": time_mean,
                "time_s_std": time_std,
                "max_rank_vs_baseline": rank_ratio,
                "storage_vs_baseline": storage_ratio,
            }
        )
    return summary


def save_summary(summary):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def plot_bar(summary, mean_key, std_key, ylabel, title, output_name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    labels = [r["case"] for r in summary]
    means = [r[mean_key] for r in summary]
    stds = [r[std_key] for r in summary]

    fig, ax = plt.subplots(figsize=(8.6, 4.7))
    colors = [
        "#6b7280",
        "#60a5fa",
        "#22c55e",
        "#16a34a",
        "#dc2626",
        "#ef4444",
        "#f59e0b",
    ]
    ax.bar(labels, means, yerr=stds, capsize=4, color=colors, edgecolor="#1f2937", linewidth=0.4)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.tick_params(axis="x", rotation=22)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / output_name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = load_rows(seeds=(0, 1, 2))
    summary = aggregate(rows)
    save_summary(summary)
    plot_bar(
        summary,
        "max_rank_mean",
        "max_rank_std",
        "Max TT rank",
        "Planar pushing PAM proxy: max rank",
        "planar_pushing_pam_proxy_3seed_max_rank.png",
    )
    plot_bar(
        summary,
        "mean_rank_mean",
        "mean_rank_std",
        "Mean internal TT rank",
        "Planar pushing PAM proxy: mean rank",
        "planar_pushing_pam_proxy_3seed_mean_rank.png",
    )
    plot_bar(
        summary,
        "storage_mean",
        "storage_std",
        "TT coefficients",
        "Planar pushing PAM proxy: storage",
        "planar_pushing_pam_proxy_3seed_storage.png",
    )
    print(f"Saved summary: {SUMMARY}")
    print(f"Saved figures in: {FIGURES}")


if __name__ == "__main__":
    main()
