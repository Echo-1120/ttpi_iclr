# Claims Ledger

## Claim C1

Mode ordering significantly affects TTPI resource consumption.

Status: Partially supported.

Evidence:
- HM8 smoke diagnostics show Random and BadSplit increase peak memory relative
  to Local.
- TT-Cross call count alone is not explanatory because all smoke variants have
  the same call count.
- Reward-model TT-Cross rank profiles differ across orderings.

Required experiments:
- HM8 seeds 0-9 with state40/action50/iter30.
- HM12 and HM16 core baselines.
- Rank-memory and query-memory correlation analysis.

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
