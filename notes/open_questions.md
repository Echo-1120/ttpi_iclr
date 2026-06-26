# Open Questions

- Are the current zero-success, identical-return pilot metrics expected because
  the runs are intentionally short, or do they reveal an evaluation bug?
- Does lower peak-cut objective consistently reduce peak memory beyond HM8
  smoke runs?
- Do full TT rank profiles differ after enough policy iterations?
- Do transient TT-Cross ranks before TT-Round explain memory spikes better than
  post-iteration ranks?
- Does index permutation make Local fail while block-aware PAM recovers?
- Is cross-coupled HardMove better explained by sensitivity or rank-aware
  objectives?
- How expensive is spectral-cache rank awareness compared with the segment
  coupling proxy?
- Is the ablation command manifest complete relative to the summary CSVs, or are
  some pilot runs being launched outside the manifest?
