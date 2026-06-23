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

## PAM v1

PAM v1 is implemented as a physics-aware action-mode ordering layer for
HardMove. It does not modify the TTPI core. It builds a weighted physical
coupling graph over action modes and reports the weighted linear arrangement
objective:

```text
sum_ij W_ij |pi(i) - pi(j)|
```

Main files:

- `repro/pam_ordering.py`: coupling graph, order generators, objective.
- `repro/scripts/build_pam_ordering.py`: inspect generated orders.
- `repro/scripts/run_hardmove.py`: supports `--action-order`.
- `repro/scripts/run_pam_ordering_smoke.sh`: GPU smoke-test entry point.

Inspect orders locally:

```bash
python repro/scripts/build_pam_ordering.py --n-actuator 8
python repro/scripts/build_pam_ordering.py --n-actuator 12
python repro/scripts/build_pam_ordering.py --n-actuator 16
```

Run GPU smoke tests on the server:

```bash
bash repro/scripts/run_pam_ordering_smoke.sh
```

Useful environment overrides:

```bash
N_ACTUATOR=12 SEED=0 N_ITER=30 bash repro/scripts/run_pam_ordering_smoke.sh
```

Do not rent a larger GPU until generated PAM orders pass HM8/HM12 smoke tests
on the RTX 5080. Larger GPUs should only be used after the method definition is
fixed and the remaining work is higher-resolution or multi-seed data generation.
