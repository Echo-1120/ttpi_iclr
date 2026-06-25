# PAM Diagnostics Implementation

This implementation upgrades the HardMove PAM reproduction from ordering-only
comparisons to TT-Cross cost diagnostics.

## Main Entry Points

Dry-run the manifest:

```bash
python repro/scripts/run_pam_ablation.py \
  --manifest repro/experiments/pam_ablation_manifest.json
```

Run the full ablation and generate plots:

```bash
python repro/scripts/run_pam_ablation.py \
  --manifest repro/experiments/pam_ablation_manifest.json \
  --execute \
  --make-plots
```

For a small server smoke test:

```bash
python repro/scripts/run_pam_ablation.py \
  --manifest repro/experiments/pam_ablation_manifest.json \
  --smoke \
  --execute \
  --make-plots
```

Add singular spectrum diagnostics:

```bash
python repro/scripts/run_pam_ablation.py \
  --manifest repro/experiments/pam_ablation_manifest.json \
  --run-spectra \
  --execute \
  --make-plots
```

## Logged Evidence

Each HardMove run now saves:

- result JSON in `repro/results/`
- per-iteration rank profile CSV in `repro/diagnostics/rank_profiles/`
- TT-Cross query CSV in `repro/diagnostics/cross_queries/`
- appended summary row in `repro/diagnostics/pam_ablation_summary.csv`

The summary includes `tt_cross_calls`, `tt_cross_function_evals`,
`queried_points_total`, full rank-profile JSON fields, peak memory, runtime,
success rate, average return, and `mu`.

## Ordering Variants

The manifest covers:

`local`, `random`, `badsplit`, `opposite_pair`, `pam_greedy`,
`pam_spectral`, `block_pam`, `free_pam`, `sensitivity_pam`,
`rankaware_proxy_pam`, `rankaware_spectral_pam`, and `hybrid_pam`.

`block_pam` preserves `[acc_i, sw_i]` adjacency. Rank-aware variants currently
use a segment-load proxy unless a future spectral cache is wired in.

## Figures

`repro/scripts/plot_pam_diagnostics.py` generates the required figures under
`repro/figures/pam_diagnostics/`, including memory/runtime comparisons,
rank-profile plots, singular-spectrum plots, score-memory scatters, coupling
heatmaps, and performance-memory Pareto plots.
