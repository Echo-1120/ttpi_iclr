# Claims Ledger

Chinese synchronized version: `notes/claims_zh.md`.

## Paper Framing

Working title:
Mode Ordering Matters in Tensor-Train Policy Iteration: Block-Preserving
Coupling-Aware Reordering.

Core position:
Mode ordering is a first-class algorithmic variable in TTPI because TT ranks are
defined by ordering-induced sequential unfoldings. The current evidence no
longer supports presenting the static rank-aware proxy as the main method.
Instead, the strongest present story is block-preserving coupling-aware
ordering: a deterministic structural coupling prior reduces TTPI resource
burden more reliably than naive rank-aware cut proxies or simple
physics--sensitivity mixtures. The implementation name `sensitivity_pam` should
be treated as a code name; the current score is not an empirical trajectory
sensitivity estimator.

Safe main claim:
At fixed TTPI approximation budgets, action-mode ordering changes intermediate
matricization ranks, TT-Cross request burden, peak GPU memory, OOM rate, and
wall-clock time. In the current 5080 prestudy, deterministic CouplingAwarePAM
(implemented as `sensitivity_pam`) is the most stable implemented ordering
candidate.

Do not claim as a headline:
- Physics-aware ordering is always better.
- A lower physics linear-arrangement objective directly implies higher return.
- The current static RankAwareProxy objective is a solved rank-aware method.
- HybridPAM is the main method before redesigning the mixture rule.
- Completed-run success alone is sufficient when OOM rates differ.

## Claim C1

Mode ordering affects TTPI resource consumption under fixed approximation
budgets.

Status: Supported by the current HM8/HM12 RTX 5080 prestudy.

Evidence:
- HM8 formal runs complete for all main orderings. Peak memory differs
  substantially: BlockPAM 2539.8 MB, CouplingAwarePAM 2597.1 MB, Local
  2806.2 MB, and RankAwareProxy 5012.2 MB.
- HM12 exposes OOM differences: Local has 3/10 OOM, BlockPAM 4/10,
  CouplingAwarePAM 1/10, HybridPAM 7/10, and RankAwareProxy 10/10.
- TT-Cross request burden also changes: HM12 CouplingAwarePAM averages 148.80M
  request counts versus Local 190.56M, BlockPAM 185.53M, and HybridPAM
  190.57M.
- HM16 at the current formal configuration has 150/150 OOM on the RTX 5080,
  so it should be reserved for a higher-memory server.

Required experiments:
- Stress-test environments should be summarized separately after the existing
  actuator-relabelled and cross-coupled formal runs are curated.
- For final submission, report confidence intervals or paired seed tests for
  the main resource metrics.

Do not claim:
- Ordering always improves control performance.
- OOM-free behavior on HM8 implies scalability to HM16 on a single RTX 5080.

## Claim C2

Coupling-aware block-preserving ordering is currently the strongest main method
candidate.

Status: Supported as the current implementation direction; still requires
broader stress-test confirmation.

Evidence:
- On HM8, CouplingAwarePAM (`sensitivity_pam` in code) achieves the highest
  mean success among the main completed runs (0.717) while keeping memory close
  to BlockPAM.
- On HM12, CouplingAwarePAM is the most stable resource option: 1/10 OOM,
  9476.6 MB mean peak memory, and 148.80M mean TT-Cross requests.
- HM12 completion-aware performance favors CouplingAwarePAM: completion
  probability 0.90, conditional completed-run success 0.481, and
  OOM-as-failure operational success 0.433. Local and BlockPAM have lower
  operational success (0.330 and 0.341), while HybridPAM drops to 0.167 because
  it completes only 3/10 seeds.
- Hybrid objective-toggle prestudy on HM8 seeds 0--2 shows the
  sensitivity-only objective has the lowest memory (2505.7 MB), shortest time
  (156.2 s), lowest query count (25.88M), and highest success (0.747) among
  block-only, physics-only, sensitivity-only, and current-hybrid variants.

Required experiments:
- Confirm on actuator-relabelled and cross-coupled HardMove stress tests.
- Compare against Local, BlockPAM, BadSplit, and Random with identical seeds
  and evaluation settings.

Do not claim:
- The deterministic sensitivity proxy is an empirical trajectory sensitivity
  estimator. It is currently a deterministic angular-distance coupling proxy.
- The method's advantage is purely a control-performance advantage independent
  of resource feasibility.

## Claim C3

The current static RankAwareProxy objective fails as a main method.

Status: Supported as a failure ablation.

Evidence:
- RankAwareProxy uses a static segmented cut-load proxy, not singular spectra or
  dynamic TT-Cross rank estimates.
- On HM8 it raises memory to 5012.2 MB and requests to 45.16M, worse than
  Local, BlockPAM, and CouplingAwarePAM.
- On HM12 it has 10/10 OOM.
- Proxy--actual correlations are weak or inconsistent in the current report:
  correlation with peak memory is -0.15 on HM8 and -0.01 on HM12.

Required experiments:
- If a rank-aware method remains in the paper, redesign it around actual
  cut-rank/spectral cache evidence rather than the present static proxy.
- Use cutwise rank heatmaps and dynamic TT-Cross process logs to identify
  bottleneck cuts.

Do not claim:
- The current rank-aware proxy predicts memory.
- Static cut load is equivalent to dynamic TT-Cross rank burden.

## Claim C4

Weighted linear arrangement is an area-under-cut-profile objective, not a direct
peak-rank objective.

Status: Theoretically supported; empirically consistent with the failure of the
static proxy.

Evidence:
- For weighted cut cost \(C_k^W(\pi)\), the identity
  \(J_{\mathrm{LA}}(\pi)=\sum_k C_k^W(\pi)\) holds.
- TT storage, TT-Round workload, and GPU memory are more sensitive to peak ranks
  or bottleneck cuts than to total cut area alone.
- RankAwareProxy can have favorable static objective values yet fail in dynamic
  TTPI runs, especially on HM12.

Required experiments:
- Report LA, peak-cut, proxy score, actual rank-AUC, and peak memory together.
- Add singular-spectrum diagnostics only when the spectrum estimates are
  representative enough for a mechanism claim.

Do not claim:
- LA is wrong or useless. It remains a legitimate cumulative burden surrogate.

## Claim C5

Empirical/lightweight sensitivity variants are not yet useful replacements for
the current deterministic CouplingAwarePAM.

Status: Supported on HM8 seed 0--2 prestudy.

Evidence:
- SensitivityLiteFD5 completes but is expensive: 6452.1 MB peak memory,
  211.6 s runtime, 61.55M requests, and 11.67 mean max advantage rank.
- SensitivityLiteFirstOrder is also expensive and less stable in performance:
  7565.9 MB, 211.8 s, 53.75M requests, 0.497 success, and 15.67 mean max
  advantage rank.
- Both are substantially more expensive than the current deterministic
  CouplingAwarePAM on HM8.

Required experiments:
- Do not expand these lite variants to HM12 unless their construction objective
  is redesigned.

Do not claim:
- More empirical sensitivity information automatically improves TTPI resource
  usage.

## Claim C6

Ordering can affect control performance only through finite-budget
approximation or action-search error.

Status: Theoretical bridge identified; current performance evidence is
secondary to resource evidence.

Evidence:
- If an ordering-dependent TT approximation has uniform advantage error
  \(\varepsilon_\pi\), its greedy action is within \(2\varepsilon_\pi\) of the
  true one-step greedy advantage.
- HM8 CouplingAwarePAM has better mean success than Local and BlockPAM, but HM12
  completed-run success alone is confounded by OOM filtering. Effective
  conclusions must report OOM rate and completed-run performance together.
- On HM12, CouplingAwarePAM has lower conditional completed-run success than
  BlockPAM (0.481 versus 0.568), but higher operational success when OOM is
  treated as failure (0.433 versus 0.341). This distinction is central to the
  paper's resource-feasibility framing.

Required experiments:
- For final claims, report effective success with OOM-as-failure alongside
  completed-run success.
- Keep resource and performance claims separate unless both agree across shared
  seeds.

Do not claim:
- Better resource metrics automatically imply better learned control.
- Raw rank-AUC/query/runtime averages are directly comparable across methods
  with different OOM rates without a censoring caveat.
