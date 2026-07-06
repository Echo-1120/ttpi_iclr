# Rankaware Proxy Failure Analysis

## Evidence Grades
- directly proven: directly read from logs or deterministic ordering definitions.
- strong evidence: repeated across seeds but not a formal causal intervention.
- conjecture: plausible mechanism requiring checkpoint/spectrum evidence.
- cannot verify: data missing in current artifacts.

## Findings
- directly proven: `rankaware_proxy_pam` uses a static cut-load proxy, not singular spectra or dynamic TT-Cross ranks.
- directly proven: HM12 rankaware proxy all-OOM status is `True` in the loaded formal artifacts.
- strong evidence: static proxy values should be interpreted against actual rank-AUC, peak memory, and query counts in `proxy_actual_correlations.csv`.
- strong evidence: cut-level rank concentration is inspectable through `rankaware_cutwise_rank_heatmap.png` because per-cut rank columns exist.
- conjecture: mismatch may come from optimizing average/log cut load while sacrificing dynamic TT-Cross bottleneck cuts.
- cannot verify: formal singular-spectrum stable rank and entropy effective rank are not available in the pulled artifacts.

## Output Files
- `permutation_metrics.csv`
- `proxy_actual_correlations.csv`
- `rankaware_vs_baselines_rank_trajectory.png/.pdf`
- `rankaware_cutwise_rank_heatmap.png/.pdf`
