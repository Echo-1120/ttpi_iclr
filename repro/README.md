# Reproduction Artifact Layout

This directory is split into active paper evidence and archived exploration.

## Active Evidence

Keep these files in the active tree:

- `scripts/`: reproducible experiment and plotting scripts.
- `results/pam_v4_full.json`: HM8 multi-seed ordering results.
- `results/pam_scaling.json`: HM8/HM12/HM16 scaling stress results.
- `results/local_extra_seeds.json`: HM12/HM16 Local extra seeds.
- `results/pam_full_results.csv`: compact table backing the paper section.
- `results/main_table.tex`: LaTeX table generated from the active evidence.
- `figures/pam/`: paper-facing PAM figures.
- `EXPERIMENT_SECTION.md`: experiment-section draft.
- `WORKLOAD_AND_REVIEW.md`: workload, review, and theory boundary notes.
- `BASELINE.md`: paper-baseline reproduction notes.

## Archive

`archive/` stores logs, checkpoints, caches, early runs, and exploratory outputs.
Those files are useful for traceability but are not part of the main paper
evidence chain. The repository `.gitignore` ignores `repro/archive/` by default.

Current archive subdirectories:

- `archive/logs/`
- `archive/models/`
- `archive/results/early/`
- `archive/results/exploratory/`
- `archive/results/legacy/`
- `archive/figures/`
- `archive/cache/`
