# PAM v1 Implementation Notes

## Purpose

This update starts the formal PAM implementation for TTPI HardMove experiments.

PAM is implemented as a physics-aware TT action-mode ordering layer. It does not
modify the TTPI core algorithm. The goal is to test whether a physical coupling
graph can generate TT mode orderings that reduce TT rank, GPU memory, and
training time compared with random or destructive action-mode layouts.

## Method Definition

PAM builds a weighted graph over HardMove action modes:

```text
[acc_0, sw_0, acc_1, sw_1, ..., acc_{n-1}, sw_{n-1}]
```

The graph currently uses:

- strong same-actuator coupling: `(acc_i, sw_i)`
- weaker neighboring-actuator coupling
- weaker opposite-actuator coupling when `n_actuator` is even

The ordering objective is weighted minimum linear arrangement:

```text
sum_{i < j} W[i, j] * abs(pos(i) - pos(j))
```

Lower objective means strongly coupled modes are placed closer in the TT chain.

## Added Files

- `repro/pam_ordering.py`
  - Builds HardMove physical coupling matrices.
  - Generates `local`, `badsplit`, `random`, `opposite_pair`, `pam_spectral`,
    `pam_greedy`, `pam_spectral_refined`, and `pam_greedy_refined` orderings.
  - Computes ordering objective and adjacency score.

- `repro/scripts/build_pam_ordering.py`
  - CLI for inspecting PAM orderings without GPU.
  - Writes `repro/results/pam_ordering_HM{n}.json`.

- `repro/scripts/run_pam_ordering_smoke.sh`
  - Server-side smoke test for Local/PAM/baseline orderings.

- `repro/PAM_FEASIBILITY_REVIEW.md`
  - Feasibility memo for PAM based on the TTPI ICLR 2024 paper and current code evidence.

- `repro/PAM_V1_IMPLEMENTATION.md`
  - This implementation note.

## Modified Files

- `repro/scripts/run_hardmove.py`
  - Adds `--action-order`.
  - Reorders the TT action domain according to the selected ordering.
  - Maps TT-order actions back to physical action order before calling HardMove dynamics/reward.
  - Applies the same inverse mapping during evaluation and trajectory plotting.
  - Adds order metadata to result JSON.
  - Adds `order{action_order}` to output filenames to avoid overwriting results.

- `repro/README.md`
  - Adds PAM v1 usage notes.

## Local Validation

Static validation passed:

```bash
python3 -m compileall -q repro/pam_ordering.py repro/scripts/build_pam_ordering.py repro/scripts/run_hardmove.py
```

Ordering generation was checked locally:

```bash
python3 repro/scripts/build_pam_ordering.py --n-actuator 8
python3 repro/scripts/build_pam_ordering.py --n-actuator 12
python3 repro/scripts/build_pam_ordering.py --n-actuator 16
```

HM8 objective summary:

| Order | Objective | Adjacency |
|---|---:|---:|
| opposite_pair | 336 | 88 |
| pam_spectral | 336 | 89 |
| pam_greedy | 336 | 91 |
| local | 448 | 87 |
| random | 836 | 10 |
| badsplit | 988 | 15 |

This confirms that generated PAM orderings improve the defined physical locality
objective over Local and are substantially better than Random/BadSplit.

## Server Run Commands

On the GPU server:

```bash
cd /home/s110/code/ttpi_iclr
git checkout repro/hardmove-5080
git pull
```

Inspect generated orderings:

```bash
python repro/scripts/build_pam_ordering.py --n-actuator 8
python repro/scripts/build_pam_ordering.py --n-actuator 12
python repro/scripts/build_pam_ordering.py --n-actuator 16
```

Run HM8 smoke:

```bash
bash repro/scripts/run_pam_ordering_smoke.sh
```

Override task size if needed:

```bash
N_ACTUATOR=12 SEED=0 N_ITER=30 bash repro/scripts/run_pam_ordering_smoke.sh
```

Single-run example:

```bash
python repro/scripts/run_hardmove.py \
  --n-actuator 8 --n-state 40 --n-action 50 \
  --n-iter 30 --callback-freq 10 --n-test 50 --seed 0 \
  --rmax-v 60 --rmax-a 60 \
  --max-batch-v 5000 --max-batch-a 20000 \
  --eps-cross-v 1e-3 --eps-cross-a 1e-3 \
  --nswp-v 5 --nswp-a 10 --n-samples 50 \
  --action-order pam_greedy \
  --device cuda
```

## Expected Validation

The first server run should answer:

1. Does `pam_greedy` run without action-order mapping errors?
2. Does `pam_greedy` produce ranks close to `opposite_pair` and below Random/BadSplit?
3. Does `pam_greedy` avoid OOM in HM8/HM12 smoke?
4. Is performance not collapsed compared with Local?

If HM8/HM12 smoke passes, the next step is multi-seed:

```text
HM8: local, pam_greedy, pam_spectral, opposite_pair, random, badsplit; seeds 0-4
HM12: local, pam_greedy, random, badsplit; seeds 0-2
```

Do not rent a larger GPU until generated PAM orderings pass HM8/HM12 smoke on the RTX 5080.
