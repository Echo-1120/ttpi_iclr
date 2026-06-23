# PAM Feasibility Review Based on TTPI

## 1. Main Reference Paper

Primary reference:

> Shetty, Xue, Calinon. **Generalized Policy Iteration Using Tensor Approximation for Hybrid Control**. ICLR 2024.

This paper introduces TTPI, an approximate dynamic programming algorithm that uses Tensor Train (TT) representations for the value function and advantage function, and uses TTGO to retrieve policies in hybrid action spaces.

For our work, TTPI is the base algorithm, not the competing method. PAM should be positioned as an extension or system-level improvement for TTPI-style tensorized control.

## 2. TTPI Baseline: What It Already Solves

TTPI already contributes:

1. TT approximation of value and advantage functions.
2. Hybrid action support through TTGO policy retrieval.
3. Model-based ADP for hybrid robotic control.
4. Simulation evidence on Catch-Point and Hard-Move.
5. Real-robot evidence on non-prehensile planar pushing.

Therefore, PAM should not claim:

- A new TTPI algorithm from scratch.
- First use of TT for optimal control.
- First solution to hybrid action control.
- First non-prehensile planar pushing controller.

The defensible gap is narrower:

> TTPI uses TT representations, but it does not systematically study or exploit the effect of physical mode ordering on TT rank, memory, training time, and scalability.

## 3. Proposed Innovation: PAM

Working definition:

> PAM, Physics-Aware Mode Reordering, arranges TT state/action/parameter modes according to physical coupling structure, so that strongly coupled variables remain local in the TT chain. This reduces TT rank growth, peak memory, and training time in tensorized hybrid control.

The clean version of the contribution is:

1. Identify TT mode ordering as a practical scalability bottleneck in TTPI.
2. Formalize physical mode locality as a weighted ordering problem.
3. Show that preserving physically coupled modes reduces TT rank and memory.
4. Validate on Hard-Move and, if possible, parameter-augmented planar pushing.

The theoretical intuition:

For a tensor `T(i_1, ..., i_D)`, TT rank `r_k` is the rank of the unfolding across cut `k`.

If strongly coupled variables are separated across many TT cuts, the unfolding rank must encode long-range dependency and rank grows. If coupled variables are adjacent, dependency remains local and rank stays low.

This is the core argument PAM can defend.

## 4. Current Code Evidence on RTX 5080

Current branch:

- `repro/hardmove-5080`

Current hardware:

- RTX 5080, 16 GB
- Paper used RTX 3090, 24 GB

Current main evidence:

- `repro/results/pam_full_results.csv`
- `repro/EXPERIMENT_SECTION.md`
- `repro/results/local_3seed_scaling_summary.csv`

### 4.1 Hard-Move Ordering Evidence

HM8, 5 seeds:

| Order | S x mu | Advantage rank | Policy rank | Peak memory | Time |
|---|---:|---:|---:|---:|---:|
| Local | 0.62 +/- 0.05 | 7.0 | 9.0 | 2.8 GB | 170 s |
| BadSplit | 0.60 +/- 0.12 | 49.0 | 51.0 | 9.4 GB | 673 s |
| Random | 0.54 +/- 0.11 | 50.0 | 52.0 | 9.6 GB | 673 s |
| OppositePair | 0.57 +/- 0.10 | 7.0 | 9.0 | 2.8 GB | 159 s |

Interpretation:

- BadSplit and Random inflate TT rank by roughly 5-6x.
- BadSplit and Random use roughly 3x GPU memory.
- BadSplit and Random take roughly 4x wall-clock time.
- OppositePair keeps `(acc_i, sw_i)` adjacency and stays low-rank, so the key factor is local physical coupling, not a specific fixed order.

This is already enough to support the basic PAM principle on Hard-Move.

### 4.2 Scaling Evidence

Local-only scaling, 3 seeds:

| n_act | S x mu | Max rank | Peak memory |
|---:|---:|---:|---:|
| 8 | 0.620 +/- 0.066 | 9.0 +/- 0.0 | 2.86 +/- 1.33 GB |
| 12 | 0.517 +/- 0.036 | 14.0 +/- 1.0 | 5.86 +/- 0.00 GB |
| 16 | 0.567 +/- 0.041 | 17.0 +/- 3.6 | 5.87 +/- 0.00 GB |

Negative controls:

- HM12 BadSplit: OOM, rank near ceiling.
- HM12 Random: OOM, rank at ceiling.
- HM16 BadSplit/Random: complete in current reduced setting but rank saturates near ceiling and performance is poor.

Interpretation:

- PAM/locality improves feasibility under a constrained 16 GB GPU budget.
- Random/BadSplit can fail before Local, even with the same TTPI code and same task.

This is the strongest current evidence.

## 5. Current Weaknesses

### 5.1 Local Is Not Yet a Full Algorithm

Current Local ordering is manually specified from known Hard-Move structure:

```text
[acc_0, sw_0, acc_1, sw_1, ..., acc_N, sw_N]
```

This is valid as a physical locality baseline, but not yet enough to claim an automatic PAM algorithm.

To claim algorithmic novelty, we need one of:

1. A manually defined physical coupling graph plus an ordering solver.
2. A finite-difference sensitivity graph estimated from dynamics/reward.
3. A trajectory/statistics-based coupling graph that does not rely mainly on hand boosting.

Recommended formal objective:

```text
min_pi sum_{i,j} w_ij |pi(i) - pi(j)|
```

where `w_ij` is the physical coupling strength and `pi(i)` is the TT-chain position of mode `i`.

### 5.2 Planar Pushing Is Not Yet Controller-Level Evidence

Current planar pushing evidence is mostly rank-proxy:

- The script tests TT-Cross rank/storage for a one-step parameter/contact proxy.
- It does not yet prove full TTPI rollout success under parameter uncertainty.

This can support the claim:

> PAM reduces rank/storage in parameter-augmented contact-mode proxy functions.

It cannot yet support:

> PAM produces a robust planar pushing controller.

For the stronger claim, we need rollout metrics:

- success rate
- final position error
- final orientation error
- rank
- memory
- time
- multi-seed results

### 5.3 DPRP and LaX Are Pilot Only

DPRP and LaX currently should not be part of the main claim.

Current LaX result:

- Same-callback improvement exists in one case.
- Best-over-training does not yet improve over No-LaX.
- Only single-seed pilot evidence exists.

DPRP:

- Code path exists.
- Smoke test exists.
- Paper-level Local-vs-Random pruning evidence is not complete.

Recommended decision:

- Keep DPRP/LaX as future work or appendix unless multi-seed paper-budget evidence is produced.

## 6. Is PAM Feasible?

Current answer:

> Yes, PAM is feasible as a TTPI scalability improvement focused on mode ordering and physical locality.

Evidence already supports:

1. Mode ordering strongly affects TT rank.
2. Physical local ordering keeps ranks low.
3. Bad/random ordering inflates rank, memory, and training time.
4. OOM can occur for bad ordering under the same hardware budget where Local remains feasible.

Evidence does not yet fully support:

1. Fully automatic physical structure discovery.
2. Robust parameter-augmented TTPI controller.
3. Broad generalization beyond Hard-Move.

Therefore the current viable paper/work direction is:

> Physics-aware mode ordering for scalable tensorized hybrid control.

Not yet:

> Fully robust parameter-augmented TTPI for contact-rich manipulation.

## 7. Is the Workload Sufficient?

Current workload is sufficient for a serious research extension if organized correctly.

Completed workload:

1. Reproduced and stabilized TTPI-related Hard-Move experiments on RTX 5080.
2. Built ordering variants: Local, BadSplit, Random, OppositePair.
3. Ran multi-seed HM8 comparisons.
4. Ran HM8/HM12/HM16 scaling comparisons.
5. Collected rank, memory, time, and performance evidence.
6. Built planar pushing proxy experiments.
7. Collected literature around TTPI, TT control, hybrid action RL, and non-prehensile manipulation.

Still required workload before strong conclusion:

1. Implement a clean PAM ordering generator.
2. Re-run Hard-Move with generated PAM ordering, not just hand-coded Local.
3. Add at least one second task with controller-level metrics or a very clear proxy-only claim.
4. Produce an ablation table:
   - Original TTPI natural order
   - PAM order
   - BadSplit
   - Random
   - OppositePair or block-preserving control
5. Write the related-work gap around TTPI and tensor mode ordering.

## 8. RTX 5080 vs Renting Server

Do not rent a server yet just to explore.

The current 5080 is enough to establish logical feasibility because:

1. HM8/HM12/HM16 Local runs already complete under reduced settings.
2. BadSplit/Random already show rank/memory failure modes.
3. The main unknown is not raw compute; it is whether PAM can be formalized and produce consistent evidence.

Rent a server only after these conditions are met:

### Server-rental gate A: Algorithmic gate

The PAM ordering generator is implemented and produces a concrete ordering from a coupling graph or sensitivity estimate.

Required output:

- `order`
- `coupling_matrix`
- ordering objective value
- comparison against Local/BadSplit/Random objective values

### Server-rental gate B: 5080 smoke gate

The generated PAM ordering runs successfully on RTX 5080 for at least:

- HM8 seed 0
- HM12 seed 0

Required metrics:

- rank lower than BadSplit/Random
- memory within 5080 budget
- no immediate OOM
- performance not collapsed

### Server-rental gate C: conclusion gate

The expected result must be binary enough that additional compute only fills in data:

Rent server only if the remaining question is:

> Does this already-defined PAM setup hold across more seeds / higher resolution / paper-budget settings?

Do not rent server if the remaining question is:

> What exactly is PAM?

or:

> Which experiment should prove it?

Those are design questions, not compute questions.

## 9. What Server Would Be Useful For

If gates A-C are met, renting a larger GPU is useful for:

1. HM16 paper-budget runs.
2. More seeds for Random/BadSplit at HM12/HM16.
3. Higher grid resolution.
4. Planar pushing controller-level experiments.
5. Paper-quality final tables.

Recommended minimum server:

- 24 GB VRAM: comparable to paper RTX 3090.
- 48 GB VRAM: preferable if doing planar pushing parameter augmentation.

Do not target CPU-only servers.

## 10. Immediate Next Steps

Priority order:

1. Clean the repository state:
   - ignore `.DS_Store`
   - avoid committing local macOS artifacts

2. Implement formal PAM ordering:
   - start with a manual physical coupling graph for Hard-Move
   - solve or approximate weighted minimum linear arrangement
   - record objective values

3. Re-run low-cost Hard-Move validation on RTX 5080:
   - HM8 seed 0
   - HM8 seeds 0-4 if stable
   - HM12 seed 0

4. Compare:
   - Local
   - generated PAM
   - BadSplit
   - Random
   - OppositePair

5. Decide whether Planar Pushing will be:
   - proxy-only evidence, or
   - full controller-level evidence

6. Rent server only if generated PAM already works on 5080 and the remaining need is more memory or more seeds.

## 11. Current Decision

Current decision:

> Continue on RTX 5080. Do not rent a server yet.

Reason:

The current evidence already shows PAM-like locality is feasible, but the method definition is not yet strong enough. More GPU will not fix an under-specified innovation claim.

Once PAM ordering is formalized and passes 5080 smoke tests, renting a 24-48 GB GPU becomes justified because the experiment will then be data-generation rather than method discovery.
