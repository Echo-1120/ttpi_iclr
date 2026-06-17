"""Generate 3 scaling charts from combined PAM data."""
import json, statistics
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---- Load all data ----
def load_json(p):
    with open(p) as f: return json.load(f)

# PAM v4: HM8 5 seeds for all orderings
v4 = load_json("repro/results/pam_v4_full.json")
# Scaling: HM8/12/16 1 seed
scaling = load_json("repro/results/pam_scaling.json")
# Local extra: HM12/16 seeds 1,2
extra = load_json("repro/results/local_extra_seeds.json")

# ---- Merge into per-(n_act, ordering) agg ----
# Collect per-seed data
data = {}  # (n_act, label) -> list of {S×μ, Ar, Pr, peak, time, oom}

# From PAM v4: HM8, 5 seeds
for label in ["Local","BadSplit","Random"]:
    key = (8, label)
    data[key] = []
    for s in range(5):
        r = v4["results"][f"{label}_s{s}"]
        data[key].append({"to": r["to_joint"], "Ar": r["Ar_max"], "Pr": r["Pr_max"],
                          "peak": r["peak_gb"], "time": r["time_s"], "oom": r["status"]=="OOM"})

# From scaling: HM8/12/16, 1 seed each
for k, r in scaling.items():
    n = r["n_act"]; label = r["label"]
    key = (n, label)
    # Avoid duplicating HM8 data from v4
    if n == 8: continue
    data[key] = data.get(key, [])
    data[key].append({"to": r["to_joint"], "Ar": r["Ar_max"], "Pr": r["Pr_max"],
                      "peak": r["peak_gb"], "time": r["time_s"], "oom": r["status"]=="OOM"})

# From extra: HM12/16 Local seeds 1,2
for k, r in extra.items():
    # key format: n12_Local_s1
    parts = k.split("_")
    n = int(parts[0][1:])
    label = parts[1]
    key = (n, label)
    data[key] = data.get(key, [])
    data[key].append({"to": r["to_joint"], "Ar": r["Ar_max"], "Pr": r["Pr_max"],
                      "peak": r["peak_gb"], "time": r["time_s"], "oom": r["status"]=="OOM"})

# Also add HM8 Local seeds from v4 are already in data[(8,"Local")]

# ---- Aggregate ----
n_acts = [8, 12, 16]
orderings = ["Local", "BadSplit", "Random"]
colors = {"Local": "#2ca02c", "BadSplit": "#d62728", "Random": "#ff7f0e"}
markers = {"Local": "o", "BadSplit": "s", "Random": "^"}

agg = {}
for n in n_acts:
    for label in orderings:
        key = (n, label)
        pts = data.get(key, [])
        if not pts: continue
        tos = [p["to"] for p in pts]; Ars = [p["Ar"] for p in pts]; Prs = [p["Pr"] for p in pts]
        pks = [p["peak"] for p in pts]; Tms = [p["time"] for p in pts]
        ooms = sum(1 for p in pts if p["oom"])
        agg[key] = {
            "n_seeds": len(pts), "oom_count": ooms,
            "to_mean": statistics.mean(tos), "to_std": statistics.stdev(tos) if len(tos)>1 else 0,
            "Ar_mean": statistics.mean(Ars), "Ar_std": statistics.stdev(Ars) if len(Ars)>1 else 0,
            "Pr_mean": statistics.mean(Prs), "Pr_std": statistics.stdev(Prs) if len(Prs)>1 else 0,
            "peak_mean": statistics.mean(pks), "peak_std": statistics.stdev(pks) if len(pks)>1 else 0,
            "T_mean": statistics.mean(Tms), "T_std": statistics.stdev(Tms) if len(Tms)>1 else 0,
            "all_to": tos, "all_Ar": Ars, "all_Pr": Prs, "all_peak": pks,
        }
        print(f"n={n} {label}: {agg[key]}")

# ---- Figure settings ----
FIG = Path("repro/figures/pam")
FIG.mkdir(parents=True, exist_ok=True)
GPU_LIMIT = 15.5  # RTX 5080
RMAX = 60

# ---- Fig 1: Peak Memory Scaling ----
fig, ax = plt.subplots(figsize=(8, 5.5))
for label in orderings:
    xs, ys, yerrs, oom_xs = [], [], [], []
    for n in n_acts:
        k = (n, label)
        if k not in agg: continue
        a = agg[k]
        xs.append(n); ys.append(a["peak_mean"]); yerrs.append(a["peak_std"])
        if a["oom_count"] > 0: oom_xs.append(n)
    ax.errorbar(xs, ys, yerr=yerrs, color=colors[label], marker=markers[label],
                markersize=10, linewidth=2, capsize=5, label=label)
    if oom_xs:
        ax.scatter(oom_xs, [GPU_LIMIT]*len(oom_xs), marker='x', s=150,
                   color=colors[label], linewidth=3, zorder=5)
ax.axhline(y=GPU_LIMIT, color='red', linestyle='--', linewidth=2, alpha=0.7, label='RTX 5080 limit (16 GB)')
ax.fill_between([7, 17], GPU_LIMIT, GPU_LIMIT+2, alpha=0.08, color='red')
ax.set_xlabel("Number of Actuators $n_{act}$", fontsize=13)
ax.set_ylabel("Peak GPU Memory (GB)", fontsize=13)
ax.set_title("Peak Memory Scaling (lower is better)", fontsize=14, fontweight='bold')
ax.set_xticks(n_acts); ax.legend(fontsize=11); ax.grid(alpha=0.3); ax.set_ylim(0, GPU_LIMIT+3)
fig.tight_layout(); fig.savefig(FIG/"scaling_peak_memory.png", dpi=200, bbox_inches='tight'); plt.close(fig)
print("Saved: scaling_peak_memory.png")

# ---- Fig 2: Max A-rank Scaling ----
fig, ax = plt.subplots(figsize=(8, 5.5))
for label in orderings:
    xs, ys, yerrs = [], [], []
    for n in n_acts:
        k = (n, label)
        if k not in agg: continue
        a = agg[k]
        xs.append(n); ys.append(a["Ar_mean"]); yerrs.append(a["Ar_std"])
    ax.errorbar(xs, ys, yerr=yerrs, color=colors[label], marker=markers[label],
                markersize=10, linewidth=2, capsize=5, label=label)
ax.axhline(y=RMAX, color='gray', linestyle=':', linewidth=2, alpha=0.6, label=f'$r_{{max}}$ = {RMAX}')
ax.set_xlabel("Number of Actuators $n_{act}$", fontsize=13)
ax.set_ylabel("Max Advantage Rank $A_r$", fontsize=13)
ax.set_title("TT-Rank Scaling (lower is better)", fontsize=14, fontweight='bold')
ax.set_xticks(n_acts); ax.legend(fontsize=11); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG/"scaling_rank.png", dpi=200, bbox_inches='tight'); plt.close(fig)
print("Saved: scaling_rank.png")

# ---- Fig 3: Performance (S×μ) Scaling ----
fig, ax = plt.subplots(figsize=(8, 5.5))
for label in orderings:
    xs, ys, yerrs = [], [], []
    for n in n_acts:
        k = (n, label)
        if k not in agg: continue
        a = agg[k]
        xs.append(n); ys.append(a["to_mean"]); yerrs.append(a["to_std"])
    ax.errorbar(xs, ys, yerr=yerrs, color=colors[label], marker=markers[label],
                markersize=10, linewidth=2, capsize=5, label=label)
ax.set_xlabel("Number of Actuators $n_{act}$", fontsize=13)
ax.set_ylabel("Best Joint $S \\times \\mu$", fontsize=13)
ax.set_title("Control Performance Scaling (higher is better)", fontsize=14, fontweight='bold')
ax.set_xticks(n_acts); ax.legend(fontsize=11); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG/"scaling_performance.png", dpi=200, bbox_inches='tight'); plt.close(fig)
print("Saved: scaling_performance.png")

print("\nAll 3 scaling charts saved.")
