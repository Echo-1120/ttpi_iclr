#!/usr/bin/env python3
"""Plot S vs μ curves and Pareto front from HM8 eval_history.

Usage:
    python repro/scripts/plot_pareto.py [--results-dir repro/results] [--fig-dir repro/figures/pareto]
"""

import json, argparse, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_eval_histories(results_dir, seeds=(0, 1, 2)):
    """Load eval_history from JSON result files."""
    all_eh = {}
    for seed in seeds:
        # Try iter100 and iter200 patterns
        patterns = [
            f"HM8_state100_action100_iter100_seed{seed}.json",
            f"HM8_state100_action100_iter200_seed{seed}.json",
        ]
        found = False
        for pat in patterns:
            p = Path(results_dir) / pat
            if p.exists():
                d = json.loads(p.read_text())
                eh = d.get("eval_history", [])
                if eh:
                    all_eh[seed] = eh
                    found = True
                break
        if not found:
            print(f"Warning: no result file for seed={seed}")
    return all_eh


def find_pareto_front(points):
    """Find Pareto-optimal points (maximizing both S and μ).
    A point dominates another if it has >= S AND >= μ, with at least one >.
    """
    pts = sorted(points, key=lambda x: (-x[0], -x[1]))  # sort by S desc, then μ desc
    pareto = []
    best_mu_so_far = -1
    for s, mu, cb, seed in pts:
        if mu > best_mu_so_far:
            pareto.append((s, mu, cb, seed))
            best_mu_so_far = mu
    return pareto


def plot_pareto(all_eh, fig_dir):
    """Plot S vs μ scatter + Pareto front for each seed and combined."""
    # ---- Per-seed plots ----
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_eh), 3)))

    fig, axes = plt.subplots(1, len(all_eh), figsize=(5 * len(all_eh), 4.5),
                             squeeze=False)
    axes = axes[0]

    for ax_idx, (seed, eh) in enumerate(sorted(all_eh.items())):
        ax = axes[ax_idx]
        S_vals = [e["success_rate"] for e in eh]
        mu_vals = [e["mu_success"] for e in eh]
        cbs = [e.get("callback_count", i * 10) for i, e in enumerate(eh)]
        tradeoffs = [s * m for s, m in zip(S_vals, mu_vals)]

        # Scatter points
        sc = ax.scatter(S_vals, mu_vals, c=cbs, cmap="viridis", s=40,
                        edgecolors="black", linewidth=0.5, zorder=3)

        # Connect points chronologically
        idx_sorted = sorted(range(len(S_vals)), key=lambda i: cbs[i])
        S_sorted = [S_vals[i] for i in idx_sorted]
        mu_sorted = [mu_vals[i] for i in idx_sorted]
        ax.plot(S_sorted, mu_sorted, "gray", linewidth=0.8, alpha=0.5, zorder=1)

        # Mark start and end
        ax.scatter([S_sorted[0]], [mu_sorted[0]], marker="o", s=80,
                   facecolors="white", edgecolors="red", linewidth=1.5, zorder=4, label="start")
        ax.scatter([S_sorted[-1]], [mu_sorted[-1]], marker="s", s=80,
                   facecolors="white", edgecolors="red", linewidth=1.5, zorder=4, label="end")

        # Mark best tradeoff
        best_idx = tradeoffs.index(max(tradeoffs))
        ax.scatter([S_vals[best_idx]], [mu_vals[best_idx]], marker="*", s=200,
                   facecolors="gold", edgecolors="black", linewidth=1, zorder=5,
                   label=f"max S×μ={max(tradeoffs):.3f}")

        # Mark best S and best μ
        best_S_idx = S_vals.index(max(S_vals))
        best_mu_idx = mu_vals.index(max(mu_vals))
        ax.scatter([S_vals[best_S_idx]], [mu_vals[best_S_idx]], marker="D", s=60,
                   facecolors="cyan", edgecolors="black", linewidth=1, zorder=5,
                   label=f"best S={max(S_vals):.2f}")
        ax.scatter([S_vals[best_mu_idx]], [mu_vals[best_mu_idx]], marker="D", s=60,
                   facecolors="magenta", edgecolors="black", linewidth=1, zorder=5,
                   label=f"best μ={max(mu_vals):.3f}")

        # Per-seed Pareto front
        pts = [(S_vals[i], mu_vals[i], cbs[i], seed) for i in range(len(S_vals))]
        pareto = find_pareto_front(pts)
        if pareto:
            p_S = [p[0] for p in pareto]
            p_mu = [p[1] for p in pareto]
            p_S.append(p_S[0] if p_S else 0)
            p_mu.append(0)
            ax.plot(p_S, p_mu, "r--", linewidth=1.5, alpha=0.7, zorder=2, label="Pareto front")

        ax.set_xlabel("Success Rate S")
        ax.set_ylabel("μ (path efficiency)")
        ax.set_title(f"HM(8) seed={seed}")
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)

        # Colorbar
        plt.colorbar(sc, ax=ax, label="callback", shrink=0.8)

    fig.suptitle("HM(8) S vs μ Trade-off — Per Seed", fontsize=14, fontweight="bold")
    fig.tight_layout()

    fig_path = fig_dir / "HM8_S_vs_mu_per_seed.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # ---- Combined plot with Pareto front ----
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    all_points = []
    for seed, eh in sorted(all_eh.items()):
        S_vals = [e["success_rate"] for e in eh]
        mu_vals = [e["mu_success"] for e in eh]
        cbs = [e.get("callback_count", i * 10) for i, e in enumerate(eh)]
        color = colors[seed % len(colors)]

        ax2.scatter(S_vals, mu_vals, c=[color], s=50, edgecolors="black",
                    linewidth=0.5, zorder=3, alpha=0.8, label=f"seed={seed}")

        idx_sorted = sorted(range(len(S_vals)), key=lambda i: cbs[i])
        S_sorted = [S_vals[i] for i in idx_sorted]
        mu_sorted = [mu_vals[i] for i in idx_sorted]
        ax2.plot(S_sorted, mu_sorted, color=color, linewidth=1, alpha=0.4, zorder=1)

        for i in range(len(S_vals)):
            all_points.append((S_vals[i], mu_vals[i], cbs[i], seed))

        # Mark best tradeoff per seed
        tradeoffs = [s * m for s, m in zip(S_vals, mu_vals)]
        best_i = tradeoffs.index(max(tradeoffs))
        ax2.scatter([S_vals[best_i]], [mu_vals[best_i]], marker="*", s=250,
                    facecolors="gold", edgecolors="black", linewidth=1.5, zorder=5)

    # Combined Pareto front
    pareto = find_pareto_front(all_points)
    if pareto:
        p_S = [p[0] for p in pareto]
        p_mu = [p[1] for p in pareto]
        # sort by S for proper step plot
        paired = sorted(zip(p_S, p_mu), key=lambda x: x[0])
        p_S, p_mu = zip(*paired)
        # Add corner at (max_S, 0)
        p_S_full = [0] + list(p_S) + [1.05]
        p_mu_full = [max(p_mu)] + list(p_mu) + [0]
        ax2.step(p_S_full, p_mu_full, "r--", linewidth=2.5, alpha=0.8, zorder=2,
                 where="post", label="Pareto front")
        ax2.fill_between(p_S_full, 0, p_mu_full, step="post", alpha=0.08, color="red")

    # Reference point: paper
    ax2.scatter([1.0], [0.93], marker="X", s=200, c="red", edgecolors="black",
                linewidth=1.5, zorder=6, label="Paper (S=1.0, μ=0.93)")

    ax2.set_xlabel("Success Rate S", fontsize=12)
    ax2.set_ylabel("μ (path efficiency)", fontsize=12)
    ax2.set_title("HM(8) S vs μ — Combined Pareto Front", fontsize=14, fontweight="bold")
    ax2.set_xlim(-0.02, 1.05)
    ax2.set_ylim(-0.02, 1.05)
    ax2.legend(fontsize=9, loc="lower left")
    ax2.grid(True, alpha=0.3)

    fig_path2 = fig_dir / "HM8_S_vs_mu_pareto.png"
    fig2.savefig(fig_path2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {fig_path2}")

    # ---- Print Pareto summary ----
    print("\n=== PARETO FRONT SUMMARY ===")
    print(f"{'S':<8} {'μ':<10} {'S×μ':<10} {'callback':<10} {'seed':<6}")
    print("-" * 45)
    for s, mu, cb, seed in sorted(pareto, key=lambda x: -x[1]):
        print(f"{s:<8.3f} {mu:<10.4f} {s*mu:<10.4f} {cb:<10} {seed:<6}")

    if pareto:
        best_tradeoff_point = max(pareto, key=lambda x: x[0] * x[1])
        print(f"\nBest tradeoff (Pareto): S={best_tradeoff_point[0]:.3f}, "
              f"μ={best_tradeoff_point[1]:.4f}, S×μ={best_tradeoff_point[0]*best_tradeoff_point[1]:.4f}")

    # Print gap to paper
    paper_S, paper_mu = 1.0, 0.93
    print(f"\nPaper: S={paper_S}, μ={paper_mu}, S×μ={paper_S*paper_mu:.4f}")
    for s, mu, _, seed in pareto:
        if s >= 0.95:
            print(f"  Pareto S≥0.95 (seed={seed}): S={s:.3f}, μ={mu:.4f}, "
                  f"Δμ={(paper_mu-mu):.3f}")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="repro/results")
    parser.add_argument("--fig-dir", default="repro/figures/pareto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    fig_dir = Path(args.fig_dir)

    all_eh = load_eval_histories(results_dir, args.seeds)
    if not all_eh:
        print("No eval_history found. Falling back to log-based extraction...")
        # Fallback: try loading from aggregated_stats.json
        stats_path = results_dir / "aggregated_stats.json"
        if stats_path.exists():
            d = json.loads(stats_path.read_text())
            for task_key, task_data in d.get("tasks", {}).items():
                if "HM8" in task_key:
                    for s in task_data.get("per_seed", []):
                        if s["seed"] in args.seeds:
                            all_eh[s["seed"]] = [{"success_rate": pt[0], "mu_success": pt[1],
                                                   "callback_count": pt[2]}
                                                  for pt in s.get("eval_history", [])]
        if not all_eh:
            print("No data available. Run training first.")
            sys.exit(1)

    plot_pareto(all_eh, fig_dir)


if __name__ == "__main__":
    main()
