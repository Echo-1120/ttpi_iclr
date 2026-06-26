# Claims Ledger

Chinese synchronized version: `notes/claims_zh.md`.

## Paper Framing

Working title:
Rank-Aware, Block-Preserving Mode Reordering for Tensor-Train Policy Iteration.

Core position:
Mode ordering is a first-class algorithmic variable in TTPI because TT ranks are
defined by ordering-induced sequential unfoldings. Physics-aware coupling is a
useful prior, but the paper should treat it as one ingredient in a rank-aware
ordering framework rather than as the whole contribution.

Safe main claim:
At fixed TTPI approximation budgets, action-mode ordering can change
intermediate matricization ranks, TT-Cross request burden, peak GPU memory, and
wall-clock time.

Do not claim as a headline:
- Physics-aware ordering is always better.
- A lower physics linear-arrangement objective directly implies higher return.
- The current pilot results prove policy-quality improvement.

## Claim C1

Mode ordering affects TTPI resource consumption under fixed approximation
budgets.

Status: Partially supported.

Evidence:
- HM8 smoke diagnostics show Random and BadSplit increase peak memory relative
  to Local. Uploaded pilot numbers show Local around 648.7 MB, Random around
  1485.1 MB, and BadSplit around 1902.0 MB.
- TT-Cross call count alone is not explanatory because smoke variants have
  nearly the same call count.
- Reward-side TT-Cross rank profiles differ across orderings: Local and
  OppositePair peak around rank 3, while Random and BadSplit peak higher in the
  uploaded pilot.

Required experiments:
- HM8 seeds 0-9 with state40/action50/iter30.
- HM12 and HM16 core baselines.
- Rank-memory and query-memory correlation analysis.
- Cross-process transient rank logging before and after TT-Round.

Do not claim:
- Ordering always improves control performance.
- PAM improves final policy quality before multi-seed evidence exists.

## Claim C2

Block-preserving PAM is more robust than free or bad physical splits.

Status: Not yet supported.

Evidence:
- Implementation exists for block-preserving PAM.

Required experiments:
- Local, BadSplit, Free PAM, Block PAM, Rank-aware PAM on HM8/HM12.
- Index-permuted HardMove stress test.

Do not claim:
- Block constraints are universally optimal.

## Claim C3

Rank-aware objectives better predict TTPI memory than simple LA alone.

Status: Open.

Evidence:
- Smoke data are too small and include only four orderings.

Required experiments:
- Compare LA, peak-cut, rank-aware proxy, and spectral-cache objectives.
- Include singular-spectrum diagnostics.

Do not claim:
- Static singular spectra directly equal dynamic TT-Cross ranks.

## Claim C4

Weighted linear arrangement is an area-under-cut-profile objective, not a direct
peak-rank objective.

Status: Theoretically supported; empirical support pending.

Evidence:
- For weighted cut cost \(C_k^W(\pi)\), the identity
  \(J_{\mathrm{LA}}(\pi)=\sum_k C_k^W(\pi)\) holds.
- TT storage, TT-Round workload, and GPU memory are more sensitive to peak ranks
  or peak cut burden than to total cut area alone.

Required experiments:
- Plot LA, peak-cut, rank-aware score, and peak memory together.
- Verify whether peak-cut and rank-aware objectives correlate better with memory
  than LA across HM8/HM12/HM16.

Do not claim:
- LA is wrong or useless. It is a legitimate cumulative burden surrogate.

## Claim C5

Ordering can affect control performance only through finite-budget
approximation or action-search error.

Status: Theoretical bridge identified; empirical support open.

Evidence:
- If an ordering-dependent TT approximation has uniform advantage error
  \(\varepsilon_\pi\), its greedy action is within \(2\varepsilon_\pi\) of the
  true one-step greedy advantage.
- Current pilot runs have zero success and identical pathological returns, so
  they cannot support performance claims.

Required experiments:
- Fix evaluation pathologies or intentionally label short runs as diagnostics.
- Run multi-seed, longer-horizon HM8/HM12/HM16 after confirming diagnostic
  metrics behave as expected.

Do not claim:
- Better resource metrics automatically imply better learned control.
