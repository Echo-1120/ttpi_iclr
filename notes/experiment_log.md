# Experiment Log

Updated: 2026-06-25T22:30:19
Git HEAD: `beb0eec`

## Current Artifact Summary

- Summary rows: 4
- Aggregate rows: 4
- Environments: HM8
- Orderings: badsplit, local, opposite_pair, random
- Seeds: 0

## Experiment Configuration

- Manifest: `repro/experiments/pam_ablation_manifest.json`
- Manifest environments: 7
- Manifest orderings: 12
- Manifest seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

## OOM / Abnormal Runs

- No OOM log hits found in `repro/logs/*.log`.

## Figure Inventory

- `singular_value_decay_by_cut.png`: present
- `effective_rank_profile_by_ordering.png`: present
- `tt_actual_rank_profile_by_ordering.png`: present
- `spectral_proxy_vs_actual_tt_rank.png`: present
- `peak_memory_vs_ordering.png`: present
- `runtime_vs_ordering.png`: present
- `la_score_vs_peak_memory_scatter.png`: present
- `peakcut_vs_peak_memory_scatter.png`: present
- `rankaware_score_vs_peak_memory_scatter.png`: present
- `reordered_coupling_heatmaps.png`: present
- `performance_vs_memory_pareto.png`: present

## Changes Versus Current Commit

Tracked/untracked working tree:

```
M .DS_Store
?? notes/
?? paper/
?? repro/scripts/sync_paper_artifacts.py
```

Experiment artifact diff stat:

```
no artifact diff
```

## Interpretation Guardrails

- Do not claim control-performance improvement from smoke runs.
- Promote claims only after `notes/claims.md` status is updated with multi-seed evidence.
- Keep paper tables generated from CSV rather than hand-edited numbers.
