"""Small-sample statistics for TTPI/PAM prestudy tables."""

from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from statistics import mean, median
from typing import Any


def bootstrap_ci(values: list[float], *, n_boot: int = 2000, seed: int = 0) -> tuple[float | None, float | None]:
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return None, None
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in values]
        stats.append(mean(sample))
    stats.sort()
    lo = stats[int(0.025 * (len(stats) - 1))]
    hi = stats[int(0.975 * (len(stats) - 1))]
    return lo, hi


def paired_permutation_pvalue(diffs: list[float]) -> float | None:
    diffs = [float(d) for d in diffs if math.isfinite(float(d))]
    if not diffs:
        return None
    observed = abs(mean(diffs))
    if len(diffs) > 18:
        rng = random.Random(0)
        samples = 20000
        count = 0
        for _ in range(samples):
            val = abs(mean([d if rng.random() < 0.5 else -d for d in diffs]))
            count += int(val >= observed - 1e-12)
        return (count + 1) / (samples + 1)
    count = 0
    total = 0
    for signs in itertools.product([-1, 1], repeat=len(diffs)):
        val = abs(mean([sign * diff for sign, diff in zip(signs, diffs)]))
        count += int(val >= observed - 1e-12)
        total += 1
    return count / total


def holm_correction(pvalues: dict[str, float | None]) -> dict[str, float | None]:
    valid = sorted((key, p) for key, p in pvalues.items() if p is not None)
    m = len(valid)
    corrected: dict[str, float | None] = {key: None for key in pvalues}
    running = 0.0
    for idx, (key, p) in enumerate(valid):
        value = min(1.0, (m - idx) * p)
        running = max(running, value)
        corrected[key] = running
    return corrected


def paired_method_tests(
    rows: list[dict[str, Any]],
    *,
    env: str,
    methods: list[str],
    metric: str,
    baseline: str,
) -> dict[str, Any]:
    by_seed: dict[int, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row.get("env") != env or row.get("ordering") not in methods:
            continue
        value = row.get(metric)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            by_seed[int(row["seed"])][str(row["ordering"])] = float(value)
    paired_seeds = [seed for seed, vals in by_seed.items() if baseline in vals]
    tests: dict[str, Any] = {
        "env": env,
        "metric": metric,
        "baseline": baseline,
        "paired_seed_count": len(paired_seeds),
        "comparisons": {},
    }
    raw_p: dict[str, float | None] = {}
    for method in methods:
        if method == baseline:
            continue
        seeds = [seed for seed in paired_seeds if method in by_seed[seed]]
        diffs = [by_seed[seed][method] - by_seed[seed][baseline] for seed in seeds]
        key = f"{method}_minus_{baseline}"
        lo, hi = bootstrap_ci(diffs) if diffs else (None, None)
        p = paired_permutation_pvalue(diffs) if len(diffs) >= 3 else None
        raw_p[key] = p
        tests["comparisons"][key] = {
            "seed_count": len(seeds),
            "mean_diff": mean(diffs) if diffs else None,
            "median_diff": median(diffs) if diffs else None,
            "bootstrap95_mean_diff": [lo, hi],
            "paired_permutation_p": p,
            "mode": "paired_permutation" if p is not None else "descriptive_only",
        }
    corrected = holm_correction(raw_p)
    for key, value in corrected.items():
        tests["comparisons"][key]["holm_p"] = value
    return tests


def oom_rate_table(rows: list[dict[str, Any]], *, env: str, methods: list[str]) -> list[dict[str, Any]]:
    out = []
    for method in methods:
        items = [row for row in rows if row.get("env") == env and row.get("ordering") == method]
        if not items:
            continue
        oom = sum(1 for item in items if item.get("status") == "oom" or item.get("oom"))
        out.append({"env": env, "ordering": method, "runs": len(items), "oom_runs": oom, "oom_rate": oom / len(items)})
    return out
