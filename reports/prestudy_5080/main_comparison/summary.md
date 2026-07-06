# Main Comparison Summary

Methods: `local`, `block_pam`, `sensitivity_pam`, `hybrid_pam`, `rankaware_proxy_pam`.
Environments: `HM8`, `HM12` from existing `formal5080` artifacts.

## Aggregate Table
| env | ordering | runs | ok_runs | oom_runs | oom_rate | peak_memory_mb_mean | adv_rank_auc_mean | queried_points_total_mean | success_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HM12 | block_pam | 10 | 6 | 4 | 0.4 | 11286.3303168 | 184.95 | 185532768.2 | 0.5683333333333334 |
| HM12 | hybrid_pam | 10 | 3 | 7 | 0.7 | 11223.2946688 | 178.75 | 190565554.6 | 0.5566666666666666 |
| HM12 | local | 10 | 7 | 3 | 0.3 | 10669.5346688 | 190.05 | 190556156 | 0.4714285714285714 |
| HM12 | rankaware_proxy_pam | 10 | 0 | 10 | 1.0 | 10914.0975616 | 73.35 | 77568916.6 |  |
| HM12 | sensitivity_pam | 10 | 9 | 1 | 0.1 | 9476.5677056 | 183.05 | 148803853.6 | 0.4811111111111111 |
| HM8 | block_pam | 10 | 10 | 0 | 0.0 | 2539.7644288 | 136.2 | 27254724.4 | 0.664 |
| HM8 | hybrid_pam | 10 | 10 | 0 | 0.0 | 2753.7698304 | 135.05 | 27814156.8 | 0.652 |
| HM8 | local | 10 | 10 | 0 | 0.0 | 2806.1961216 | 137.75 | 29521724.4 | 0.557 |
| HM8 | rankaware_proxy_pam | 10 | 10 | 0 | 0.0 | 5012.1925632 | 190.35 | 45155906 | 0.534 |
| HM8 | sensitivity_pam | 10 | 10 | 0 | 0.0 | 2597.1117056 | 135.75 | 28189154 | 0.717 |

## Typical Seeds
- HM8: 1
- HM12: 9

## Interpretation Guardrails
- OOM rows are included in OOM-rate and partial-rank evidence.
- Success-rate comparisons use completed run values; effective success with OOM-as-zero should be interpreted from `ok_runs` and `oom_rate` together.
- Statistical tests are paired by seed and automatically downgrade to descriptive-only when paired seed count is too small.
