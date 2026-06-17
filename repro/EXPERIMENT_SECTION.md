# Experiment: Structure-Aware Mode Ordering for TTPI

## Setup

All experiments run on a single NVIDIA RTX 5080 (16 GB), PyTorch 2.11, CUDA 12.8.
TTPI parameters: $\gamma=0.99$, $n_{\text{iter\_{}v}}=1$, $n_{\text{samples}}=50$, $\epsilon_{\text{cross}}=10^{-3}$.
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` enables virtual memory paging.
`gc.collect()` follows every policy iteration to reduce fragmentation.

## Orderings

Four mode orderings are compared on HM8, HM12, HM16:

- **Local** (original TTPI): `[acc_0, sw_0, acc_1, sw_1, ..., acc_N, sw_N]`. Each actuator's continuous command and discrete switch are adjacent.
- **BadSplit**: `[acc_0, ..., acc_N, sw_0, ..., sw_N]`. All acceleration modes precede all switch modes, breaking every `(acc_i, sw_i)` pair across the chain midpoint.
- **Random**: Five independent random permutations of the action dimensions.
- **OppositePair**: `[acc_0, sw_0, acc_{N/2}, sw_{N/2}, acc_1, sw_1, acc_{N/2+1}, sw_{N/2+1}, ...]`. Preserves local `(acc, sw)` pairs while grouping opposing actuators.

## HM8 Multi-Seed (5 seeds, n_iter=30, n_test=50)

| Order | $S\times\mu$ | $A_r$ | $P_r$ | Peak (GB) | Time (s) |
|-------|-------------|-------|-------|-----------|----------|
| Local | 0.62 ± 0.05 | 7.0 ± 0.0 | 9.0 ± 0.0 | 2.8 ± 0.9 | 170 ± 9 |
| BadSplit | 0.60 ± 0.12 | 49.0 ± 3.2 | 51.0 ± 3.2 | 9.4 ± 0.4 | 673 ± 19 |
| Random | 0.54 ± 0.11 | 50.0 ± 4.2 | 52.0 ± 4.2 | 9.6 ± 1.5 | 673 ± 46 |
| OppositePair | 0.57 ± 0.10 | 7.0 ± 0.0 | 9.0 ± 0.0 | 2.8 ± 0.6 | 159 ± 7 |

BadSplit and Random inflate the policy rank by 5-6× over Local, consuming 3× more peak GPU memory and requiring 4× more training time. OppositePair achieves comparable rank and memory to Local, confirming that the key property is preserving local actuator coupling, not a specific fixed permutation.

## Scaling: n_act ∈ {8, 12, 16} (3 seeds Local, 1 seed BadSplit/Random)

| n_act | Order | Seeds | $S\times\mu$ | $A_r$ | $P_r$ | Peak (GB) | Time (s) | OOM |
|-------|-------|-------|-------------|-------|-------|-----------|----------|-----|
| 8 | Local | 5 | 0.62 ± 0.05 | 7.0 | 9.0 | 2.8 | 170 | N |
| 8 | BadSplit | 5 | 0.60 ± 0.12 | 49.0 | 51.0 | 9.4 | 673 | N |
| 8 | Random | 5 | 0.54 ± 0.11 | 50.0 | 52.0 | 9.6 | 673 | N |
| 12 | Local | 3 | 0.52 ± 0.04 | 12.0 | 14.0 | 5.9 | 404 | N |
| 12 | BadSplit | 1 | 0.19 | 54.0 | 56.0 | 8.9 | 372 | Y |
| 12 | Random | 1 | 0.24 | 60.0 | 62.0 | 10.3 | 1248 | Y |
| 16 | Local | 3 | 0.57 ± 0.04 | 15.0 | 17.0 | 5.9 | 837 | N |
| 16 | BadSplit | 1 | 0.26 | 60.0 | 62.0 | 7.8 | 2220 | N |
| 16 | Random | 1 | 0.20 | 60.0 | 62.0 | 7.8 | 2160 | N |

**Key findings:**

1. BadSplit and Random complete at n_act=8 but already show severe rank, memory, and time inflation.
2. BadSplit and Random trigger OOM at n_act=12 under the same 16 GB hardware budget.
3. Local completes all tested scales without OOM. Policy rank grows modestly (9→14→17).
4. BadSplit and Random hit or approach the rank ceiling (rmax=60) at n_act=12 and 16.
5. Performance ($S\times\mu$) for Local is stable (0.52-0.62), while BadSplit and Random degrade sharply (0.19-0.26 at n_act=16).

## Local 3-Seed Scaling Curves

The Local-only 3-seed scaling curves use exactly seeds 0, 1, and 2 for each
$n_{\text{act}}\in\{8,12,16\}$. The exported summary is
`repro/results/local_3seed_scaling_summary.csv`.

| n_act | $S\times\mu$ | Max rank | Peak (GB) |
|-------|-------------|----------|-----------|
| 8 | 0.620 ± 0.066 | 9.0 ± 0.0 | 2.86 ± 1.33 |
| 12 | 0.517 ± 0.036 | 14.0 ± 1.0 | 5.86 ± 0.00 |
| 16 | 0.567 ± 0.041 | 17.0 ± 3.6 | 5.87 ± 0.00 |

Figures:

- `repro/figures/pam/local_3seed_peak_memory.png`
- `repro/figures/pam/local_3seed_max_rank.png`
- `repro/figures/pam/local_3seed_success_tradeoff.png`

## DPRP and LaX Pilot

`repro/scripts/dprp_lax_ablation.py` now provides a separate experiment entry
for dynamic rank pruning (DPRP) and local action expansion (LaX), without
changing the original TTPI implementation.

Current verified runs:

- Smoke test: `repro/results/dprp_lax_smoke_seed0.json`
- LaX paper-budget single seed: `repro/results/dprp_lax_paper_local_lax_v2_seed0.json`

For LaX on HM8 Local seed 0 at low grid (`NS=30`, `NA=25`, `n_iter=30`):

| Metric | Value |
|--------|-------|
| Best No-LaX $S\times\mu$ | 0.6297 |
| Best LaX $S\times\mu$ | 0.6155 |
| Best paired LaX delta at same callback | +0.1355 at cb19 |
| LaX extra memory | ~0.000007 GB |

Interpretation: the current LaX implementation shows very low extra memory and
a positive same-model correction at cb19, but it does not yet improve the
best-over-training score on this single seed. This is pilot evidence, not a
paper-ready multi-seed claim.

DPRP is implemented and smoke-tested, but the smoke setting is too small to
support the intended Local-vs-Random rank-pruning claim. The paper claim still
requires paper-budget Local/Random DPRP runs.

## PointMassVelocity Comparison

The original `PointMassVelocity.ipynb` is an appendix-style continuous-control
demo: a 2D velocity-controlled point mass moves to the origin while avoiding a
circular obstacle at `(0, -0.4)` with radius `0.2`. The paper does not report a
formal PointMassVelocity baseline table, so the reproduction script adds simple
deterministic controls as sanity-check baselines rather than claiming a HyAR
comparison.

Script:

- `repro/scripts/run_pointmass_velocity_comparison.py`

Notebook-like run:

- `repro/results/pointmass_velocity_notebook_seed0.json`
- `repro/results/pointmass_velocity_notebook_seed0.summary.csv`
- `repro/figures/pointmass_velocity/pointmass_velocity_notebook_seed0_trajectories.png`

Configuration: `n_state=50`, `n_action=50`, `n_iter=200`,
`callback_freq=20`, horizon `10s`, seed `0`.

| Method | Success | $\mu$ | $S\times\mu$ | Collision | Time |
|--------|---------|-------|--------------|-----------|------|
| TTPI | 0.900 | 0.960 | 0.864 | 0.000 | 175.0s |
| Straight-to-goal | 1.000 | 0.967 | 0.967 | 0.133 | 0.27s |
| Potential field | 0.033 | 0.000 | 0.000 | 0.000 | 0.32s |
| Random | 0.033 | 0.000 | 0.000 | 0.033 | 0.27s |
| Zero action | 0.033 | 0.000 | 0.000 | 0.000 | 0.25s |

Interpretation: the straight-line controller reaches the target fastest but
collides with the obstacle on 13.3% of the notebook initial states. TTPI reaches
90% of states with zero collisions, which better matches the obstacle-aware
objective in the notebook. The potential-field baseline is collision-free but
fails to reach the target under this tuning.

## Conclusion

Structure-aware mode ordering is a key practical condition for scaling TTPI to high-dimensional hybrid action spaces under a limited GPU budget. Breaking local actuator coupling causes TT rank inflation, peak memory growth, and eventual OOM. The Local ordering, which preserves `(acc_i, sw_i)` adjacency, is therefore the correct natural baseline for HardMove and the main positive example of the locality-preserving principle.
