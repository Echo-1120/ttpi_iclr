#!/usr/bin/env python3
"""Summarize HardMove reproduction results from repro/results/ JSON files."""

import json
import sys
from pathlib import Path


def load_results(results_dir):
    results = []
    for jf in sorted(Path(results_dir).glob("HM*.json")):
        with open(jf) as f:
            data = json.load(f)
        results.append(data)
    return results


def print_table(results):
    header = [
        "Experiment",
        "n_act",
        "n_state",
        "n_iter",
        "Seed",
        "S (final)",
        "S (best)",
        "μ (best)",
        "Dist (best)",
        "Time (s)",
        "EarlyStop",
    ]
    col_widths = [max(len(h), 12) for h in header]

    def fmt(label, width):
        return f"{label:>{width}}"

    print(" ".join(fmt(h, col_widths[i]) for i, h in enumerate(header)))
    print(" ".join("-" * w for w in col_widths))

    for r in results:
        best = r.get("best_metrics") or {}
        final = r.get("final_metrics") or {}

        row = [
            f"HM({r.get('n_actuator', '?')})",
            str(r.get("n_actuator", "")),
            str(r.get("n_state", "")),
            str(r.get("n_iter", "")),
            str(r.get("seed", "")),
            f"{final.get('success_rate', 0):.3f}",
            f"{best.get('success_rate', 0):.3f}",
            f"{best.get('mu_success', 0):.3f}",
            f"{best.get('final_dist_mean', 0):.4f}",
            f"{r.get('train_time_sec', 0):.0f}",
            "Y" if r.get("stopped_early") else "N",
        ]
        print(" ".join(fmt(v, col_widths[i]) for i, v in enumerate(row)))

    print()
    print(f"Total: {len(results)} experiments")


def print_grouped(results):
    print()
    print("=== Paper-style Table 1 (separate best S / best μ) ===")
    print()

    from collections import defaultdict
    import statistics

    groups = defaultdict(list)
    for r in results:
        key = f"HM({r.get('n_actuator', '?')})"
        groups[key].append(r)

    print(f"{'Task':<10} {'d':<4} {'m':<5} {'T(s)':<10} {'S':<12} {'μ':<12}")
    print("-" * 55)

    for key, runs in sorted(groups.items()):
        n_act = runs[0].get("n_actuator", 0)
        d = 4
        m = 2 * n_act

        # Best S from eval_history (any checkpoint)
        best_S_vals = []
        best_mu_vals = []
        times = []
        for r in runs:
            eh = r.get("eval_history", [])
            if eh:
                best_S_entry = max(eh, key=lambda x: x.get("success_rate", 0))
                best_S_vals.append(best_S_entry["success_rate"])
                # Best μ among entries with S >= 0.5
                high_S = [e for e in eh if e.get("success_rate", 0) >= 0.5]
                if high_S:
                    best_mu_entry = max(high_S, key=lambda x: x.get("mu_success", 0))
                    best_mu_vals.append(best_mu_entry["mu_success"])
            times.append(r.get("train_time_sec", 0))

        S_mean = statistics.mean(best_S_vals) if best_S_vals else 0
        S_std = statistics.stdev(best_S_vals) if len(best_S_vals) > 1 else 0
        mu_mean = statistics.mean(best_mu_vals) if best_mu_vals else 0
        mu_std = statistics.stdev(best_mu_vals) if len(best_mu_vals) > 1 else 0
        T_mean = statistics.mean(times) if times else 0

        print(f"{key:<10} {d:<4} {m:<5} {T_mean:<10.0f} {S_mean:.2f}±{S_std:.2f}   {mu_mean:.2f}±{mu_std:.2f}")

    print()
    print("Paper Table 1 reference:")
    print("CP      4    2    30        1.00        1.00")
    print("HM(8)   4    16   850       1.00        0.93±0.01")
    print("HM(12)  4    24   946       1.00        0.92±0.01")
    print("HM(16)  4    32   1743      1.00        0.92±0.02")


def main():
    results_dir = Path(__file__).resolve().parents[1] / "results"
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    results = load_results(results_dir)
    if not results:
        print("No HM*.json results found.")
        sys.exit(0)

    print_table(results)
    print_grouped(results)


if __name__ == "__main__":
    main()
