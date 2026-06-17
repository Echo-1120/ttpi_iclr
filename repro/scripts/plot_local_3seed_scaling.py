"""Plot Local 3-seed scaling curves for HM8/HM12/HM16.

Inputs:
  repro/results/pam_full_results.csv

Outputs:
  repro/results/local_3seed_scaling_summary.csv
  repro/figures/pam/local_3seed_peak_memory.png
  repro/figures/pam/local_3seed_max_rank.png
  repro/figures/pam/local_3seed_success_tradeoff.png
"""
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "repro" / "results"
FIGURES = ROOT / "repro" / "figures" / "pam"
SOURCE = RESULTS / "pam_full_results.csv"
SUMMARY = RESULTS / "local_3seed_scaling_summary.csv"


def mean_std(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def load_local_rows():
    rows = []
    with SOURCE.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["order"] != "Local":
                continue
            n_act = int(row["n_act"])
            seed = int(row["seed"])
            if n_act in {8, 12, 16} and seed in {0, 1, 2}:
                rows.append(
                    {
                        "n_act": n_act,
                        "seed": seed,
                        "S_mu": float(row["S_mu"]),
                        "Ar": float(row["Ar"]),
                        "Pr": float(row["Pr"]),
                        "MaxRank": max(float(row["Ar"]), float(row["Pr"])),
                        "Peak_GB": float(row["Peak_GB"]),
                        "Time_s": float(row["Time_s"]),
                    }
                )
    expected = {(n, s) for n in (8, 12, 16) for s in (0, 1, 2)}
    present = {(r["n_act"], r["seed"]) for r in rows}
    missing = sorted(expected - present)
    if missing:
        raise RuntimeError(f"Missing Local 3-seed rows in {SOURCE}: {missing}")
    return sorted(rows, key=lambda r: (r["n_act"], r["seed"]))


def aggregate(rows):
    summary = []
    for n_act in (8, 12, 16):
        group = [r for r in rows if r["n_act"] == n_act]
        s_mu_mean, s_mu_std = mean_std([r["S_mu"] for r in group])
        ar_mean, ar_std = mean_std([r["Ar"] for r in group])
        pr_mean, pr_std = mean_std([r["Pr"] for r in group])
        max_rank_mean, max_rank_std = mean_std([r["MaxRank"] for r in group])
        peak_mean, peak_std = mean_std([r["Peak_GB"] for r in group])
        time_mean, time_std = mean_std([r["Time_s"] for r in group])
        summary.append(
            {
                "n_act": n_act,
                "n_seeds": len(group),
                "S_mu_mean": s_mu_mean,
                "S_mu_std": s_mu_std,
                "Ar_mean": ar_mean,
                "Ar_std": ar_std,
                "Pr_mean": pr_mean,
                "Pr_std": pr_std,
                "MaxRank_mean": max_rank_mean,
                "MaxRank_std": max_rank_std,
                "Peak_GB_mean": peak_mean,
                "Peak_GB_std": peak_std,
                "Time_s_mean": time_mean,
                "Time_s_std": time_std,
            }
        )
    return summary


def save_summary(summary):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def plot_curve(summary, mean_key, std_key, ylabel, title, output_name, ylim=None):
    FIGURES.mkdir(parents=True, exist_ok=True)
    xs = [r["n_act"] for r in summary]
    ys = [r[mean_key] for r in summary]
    stds = [r[std_key] for r in summary]
    lo = [y - s for y, s in zip(ys, stds)]
    hi = [y + s for y, s in zip(ys, stds)]

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(xs, ys, marker="o", markersize=8, linewidth=2.2, color="#2563eb")
    ax.fill_between(xs, lo, hi, color="#93c5fd", alpha=0.35, linewidth=0)
    ax.set_xlabel(r"Number of actuators $n_{\mathrm{act}}$", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(xs)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.28)
    fig.tight_layout()
    fig.savefig(FIGURES / output_name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = load_local_rows()
    summary = aggregate(rows)
    save_summary(summary)

    plot_curve(
        summary,
        "Peak_GB_mean",
        "Peak_GB_std",
        "Peak GPU memory (GB)",
        "Local 3-seed peak memory scaling",
        "local_3seed_peak_memory.png",
        ylim=(0, max(r["Peak_GB_mean"] + r["Peak_GB_std"] for r in summary) + 1.0),
    )
    plot_curve(
        summary,
        "MaxRank_mean",
        "MaxRank_std",
        "Max TT rank",
        "Local 3-seed max-rank scaling",
        "local_3seed_max_rank.png",
        ylim=(0, max(r["MaxRank_mean"] + r["MaxRank_std"] for r in summary) + 3.0),
    )
    plot_curve(
        summary,
        "S_mu_mean",
        "S_mu_std",
        r"Best $S \times \mu$",
        r"Local 3-seed $S \times \mu$ scaling",
        "local_3seed_success_tradeoff.png",
        ylim=(0, 1.0),
    )

    print(f"Saved summary: {SUMMARY}")
    print(f"Saved figures: {FIGURES / 'local_3seed_peak_memory.png'}")
    print(f"Saved figures: {FIGURES / 'local_3seed_max_rank.png'}")
    print(f"Saved figures: {FIGURES / 'local_3seed_success_tradeoff.png'}")


if __name__ == "__main__":
    main()
