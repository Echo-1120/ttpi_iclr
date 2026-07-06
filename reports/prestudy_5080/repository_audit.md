# Repository Audit: RTX 5080 TTPI/PAM Prestudy

## Scope
This audit records the true current implementation before any supplementary 5080 prestudy reruns. It does not launch training.

## True Entry Points
- HardMove runner: `repro/scripts/run_hardmove.py`
- Batch launcher: `repro/scripts/run_pam_ablation.py`
- Ordering registry: `repro/pam_ordering.py`
- Environment action wrapper: `repro/hardmove_variants.py`
- Main prestudy entry: `python -m analysis.prestudy_5080`

## Ordering Definitions
| ordering | canonical | category | construction | hybrid_w | deprecated |
| --- | --- | --- | --- | --- | --- |
| local | local | primary_baseline |  |  | False |
| badsplit | badsplit | diagnostic_control |  |  | False |
| random | random | diagnostic_control |  |  | False |
| reverse_blocks | reverse_blocks | diagnostic_control |  |  | False |
| flip_within_block | flip_within_block | diagnostic_control |  |  | False |
| opposite_pair | opposite_interleave_legacy | legacy_diagnostic |  |  | True |
| pam_spectral | pam_spectral | surrogate_baseline |  |  | False |
| pam_greedy | pam_greedy | surrogate_baseline |  |  | False |
| pam_spectral_refined | pam_spectral_refined | surrogate_baseline |  |  | False |
| pam_greedy_refined | pam_greedy_refined | surrogate_baseline |  |  | False |
| block_pam | block_pam | ablation |  |  | False |
| free_pam | free_pam | ablation |  |  | False |
| peakcut_pam | peakcut_pam | surrogate_baseline |  |  | False |
| sensitivity_pam | sensitivity_pam | surrogate_baseline | deterministic_angular_distance_proxy |  | False |
| rankaware_proxy_pam | rankaware_proxy_pam | main_method |  |  | False |
| rankaware_spectral_pam | rankaware_spectral_pam | ablation |  |  | False |
| hybrid_pam | hybrid_pam | main_method | hybrid_physics_plus_deterministic_sensitivity | 0.5/0.5 | False |
| sensitivity_lite_fd5 | sensitivity_lite_fd5 | prestudy_candidate | lite_fd5_hardmove_action_effect_proxy |  | False |
| sensitivity_lite_first_order | sensitivity_lite_first_order | prestudy_candidate | analytic_first_order_hardmove_action_effect_proxy |  | False |
| hybrid_block_only | hybrid_block_only | prestudy_ablation | block_constraint_only_zero_objective_local_tiebreak | 0.0/0.0 | False |
| hybrid_block_plus_physics | hybrid_block_plus_physics | prestudy_ablation | hybrid_ablation_physics_only | 1.0/0.0 | False |
| hybrid_block_plus_sensitivity | hybrid_block_plus_sensitivity | prestudy_ablation | hybrid_ablation_sensitivity_only | 0.0/1.0 | False |
| hybrid_current | hybrid_current | prestudy_ablation | hybrid_ablation_current_0p5_physics_0p5_sensitivity | 0.5/0.5 | False |

## Critical Implementation Facts
- `sensitivity_pam` is a deterministic angular-distance coupling proxy, not a trajectory/gradient sensitivity estimator.
- `hybrid_pam` is the block-preserving order induced by `0.5 * physical coupling + 0.5 * deterministic sensitivity coupling`.
- `rankaware_proxy_pam` minimizes a static segmented cut-load proxy with `lambda_sum=1.0` and `lambda_peak=2.0`; it does not use real singular spectra.
- New `sensitivity_lite_*` entries are prestudy ordering candidates only. They do not alter TTPI, dynamics, reward, grids, tolerances, or evaluation.

## Existing Result Status
| env | ordering | ok | oom | error | unknown | total |
| --- | --- | --- | --- | --- | --- | --- |
| HM12 | badsplit | 0 | 10 | 0 | 0 | 10 |
| HM12 | block_pam | 6 | 4 | 0 | 0 | 10 |
| HM12 | flip_within_block | 0 | 10 | 0 | 0 | 10 |
| HM12 | free_pam | 0 | 10 | 0 | 0 | 10 |
| HM12 | hybrid_pam | 3 | 7 | 0 | 0 | 10 |
| HM12 | local | 7 | 3 | 0 | 0 | 10 |
| HM12 | pam_greedy | 0 | 10 | 0 | 0 | 10 |
| HM12 | pam_spectral | 0 | 10 | 0 | 0 | 10 |
| HM12 | peakcut_pam | 0 | 10 | 0 | 0 | 10 |
| HM12 | random | 0 | 30 | 0 | 0 | 30 |
| HM12 | rankaware_proxy_pam | 0 | 10 | 0 | 0 | 10 |
| HM12 | reverse_blocks | 2 | 8 | 0 | 0 | 10 |
| HM12 | sensitivity_pam | 9 | 1 | 0 | 0 | 10 |
| HM12_actuator_relabelled | badsplit | 0 | 10 | 0 | 0 | 10 |
| HM12_actuator_relabelled | block_pam | 5 | 5 | 0 | 0 | 10 |
| HM12_actuator_relabelled | flip_within_block | 0 | 10 | 0 | 0 | 10 |
| HM12_actuator_relabelled | free_pam | 0 | 10 | 0 | 0 | 10 |
| HM12_actuator_relabelled | hybrid_pam | 2 | 8 | 0 | 0 | 10 |
| HM12_actuator_relabelled | local | 4 | 6 | 0 | 0 | 10 |
| HM12_actuator_relabelled | pam_greedy | 0 | 10 | 0 | 0 | 10 |
| HM12_actuator_relabelled | pam_spectral | 0 | 10 | 0 | 0 | 10 |
| HM12_actuator_relabelled | peakcut_pam | 0 | 10 | 0 | 0 | 10 |
| HM12_actuator_relabelled | random | 0 | 30 | 0 | 0 | 30 |
| HM12_actuator_relabelled | rankaware_proxy_pam | 0 | 10 | 0 | 0 | 10 |
| HM12_actuator_relabelled | reverse_blocks | 2 | 8 | 0 | 0 | 10 |
| HM12_actuator_relabelled | sensitivity_pam | 1 | 9 | 0 | 0 | 10 |
| HM12_cross_coupled | badsplit | 0 | 10 | 0 | 0 | 10 |
| HM12_cross_coupled | block_pam | 0 | 10 | 0 | 0 | 10 |
| HM12_cross_coupled | flip_within_block | 0 | 10 | 0 | 0 | 10 |
| HM12_cross_coupled | free_pam | 0 | 10 | 0 | 0 | 10 |
| HM12_cross_coupled | hybrid_pam | 0 | 10 | 0 | 0 | 10 |
| HM12_cross_coupled | local | 0 | 10 | 0 | 0 | 10 |
| HM12_cross_coupled | pam_greedy | 0 | 10 | 0 | 0 | 10 |
| HM12_cross_coupled | pam_spectral | 0 | 10 | 0 | 0 | 10 |
| HM12_cross_coupled | peakcut_pam | 0 | 10 | 0 | 0 | 10 |
| HM12_cross_coupled | random | 0 | 30 | 0 | 0 | 30 |
| HM12_cross_coupled | rankaware_proxy_pam | 0 | 10 | 0 | 0 | 10 |
| HM12_cross_coupled | reverse_blocks | 0 | 10 | 0 | 0 | 10 |
| HM12_cross_coupled | sensitivity_pam | 0 | 10 | 0 | 0 | 10 |
| HM16 | badsplit | 0 | 10 | 0 | 0 | 10 |
| HM16 | block_pam | 0 | 10 | 0 | 0 | 10 |
| HM16 | flip_within_block | 0 | 10 | 0 | 0 | 10 |
| HM16 | free_pam | 0 | 10 | 0 | 0 | 10 |
| HM16 | hybrid_pam | 0 | 10 | 0 | 0 | 10 |
| HM16 | local | 0 | 10 | 0 | 0 | 10 |
| HM16 | pam_greedy | 0 | 10 | 0 | 0 | 10 |
| HM16 | pam_spectral | 0 | 10 | 0 | 0 | 10 |
| HM16 | peakcut_pam | 0 | 10 | 0 | 0 | 10 |
| HM16 | random | 0 | 30 | 0 | 0 | 30 |
| HM16 | rankaware_proxy_pam | 0 | 10 | 0 | 0 | 10 |
| HM16 | reverse_blocks | 0 | 10 | 0 | 0 | 10 |
| HM16 | sensitivity_pam | 0 | 10 | 0 | 0 | 10 |
| HM8 | badsplit | 2 | 10 | 0 | 0 | 12 |
| HM8 | block_pam | 10 | 0 | 0 | 0 | 10 |
| HM8 | flip_within_block | 3 | 7 | 0 | 0 | 10 |
| HM8 | free_pam | 10 | 0 | 0 | 0 | 10 |
| HM8 | hybrid_pam | 10 | 0 | 0 | 0 | 10 |
| HM8 | local | 13 | 0 | 0 | 0 | 13 |
| HM8 | opposite_pair | 2 | 0 | 0 | 0 | 2 |
| HM8 | pam_greedy | 10 | 0 | 0 | 0 | 10 |
| HM8 | pam_spectral | 1 | 9 | 0 | 0 | 10 |
| HM8 | peakcut_pam | 10 | 0 | 0 | 0 | 10 |
| HM8 | random | 2 | 30 | 0 | 0 | 32 |
| HM8 | rankaware_proxy_pam | 10 | 0 | 0 | 0 | 10 |
| HM8 | reverse_blocks | 10 | 0 | 0 | 0 | 10 |
| HM8 | sensitivity_pam | 10 | 0 | 0 | 0 | 10 |
| HM8_actuator_relabelled | badsplit | 0 | 10 | 0 | 0 | 10 |
| HM8_actuator_relabelled | block_pam | 10 | 0 | 0 | 0 | 10 |
| HM8_actuator_relabelled | flip_within_block | 0 | 10 | 0 | 0 | 10 |
| HM8_actuator_relabelled | free_pam | 10 | 0 | 0 | 0 | 10 |
| HM8_actuator_relabelled | hybrid_pam | 10 | 0 | 0 | 0 | 10 |
| HM8_actuator_relabelled | local | 10 | 0 | 0 | 0 | 10 |
| HM8_actuator_relabelled | pam_greedy | 2 | 8 | 0 | 0 | 10 |
| HM8_actuator_relabelled | pam_spectral | 0 | 10 | 0 | 0 | 10 |
| HM8_actuator_relabelled | peakcut_pam | 2 | 8 | 0 | 0 | 10 |
| HM8_actuator_relabelled | random | 0 | 30 | 0 | 0 | 30 |
| HM8_actuator_relabelled | rankaware_proxy_pam | 4 | 6 | 0 | 0 | 10 |
| HM8_actuator_relabelled | reverse_blocks | 10 | 0 | 0 | 0 | 10 |
| HM8_actuator_relabelled | sensitivity_pam | 10 | 0 | 0 | 0 | 10 |
| HM8_cross_coupled | badsplit | 0 | 10 | 0 | 0 | 10 |
| HM8_cross_coupled | block_pam | 0 | 10 | 0 | 0 | 10 |
| HM8_cross_coupled | flip_within_block | 0 | 10 | 0 | 0 | 10 |
| HM8_cross_coupled | free_pam | 0 | 10 | 0 | 0 | 10 |
| HM8_cross_coupled | hybrid_pam | 0 | 10 | 0 | 0 | 10 |
| HM8_cross_coupled | local | 0 | 10 | 0 | 0 | 10 |
| HM8_cross_coupled | pam_greedy | 0 | 10 | 0 | 0 | 10 |
| HM8_cross_coupled | pam_spectral | 0 | 10 | 0 | 0 | 10 |
| HM8_cross_coupled | peakcut_pam | 0 | 10 | 0 | 0 | 10 |
| HM8_cross_coupled | random | 0 | 30 | 0 | 0 | 30 |
| HM8_cross_coupled | rankaware_proxy_pam | 0 | 10 | 0 | 0 | 10 |
| HM8_cross_coupled | reverse_blocks | 0 | 10 | 0 | 0 | 10 |
| HM8_cross_coupled | sensitivity_pam | 0 | 10 | 0 | 0 | 10 |

## Artifact Inventory
- Summary rows: 1059
- Result JSON files: 1083
- Rank profile CSV files: 1055
- TT-Cross query CSV files: 1055
- TT-Cross process CSV files: 1050
- TT-Round event CSV files: 1050
- Model/checkpoint files under `repro/models`: 881
- Singular spectrum files under `repro/diagnostics/spectra`: 3

## Field Mapping
- `peak_memory_mb`: CUDA max allocated memory in MB when CUDA is available.
- `runtime_sec` / `total_time_sec`: ordering construction plus TTPI training time.
- `ordering_search_time_sec`: ordering construction only.
- `ttpi_training_time_sec`: TTPI train call only.
- `tt_cross_function_evals` and `queried_points_total`: wrapper-counted request totals; this is an upper-bound request count if tntorch caches internally.
- `adv_rank_*` / `val_rank_*`: TT rank profiles recorded after PI/rank events; per-cut columns are present when the source rank vector is available.

## Data Quality Notes
- OOM rows are retained as first-class evidence.
- Empty historical `status` values are normalized to `ok` only when no OOM/error flag is present.
- Formal singular spectra are not present in the pulled data; existing spectra appear to be older diagnostics and are not used as formal evidence.
