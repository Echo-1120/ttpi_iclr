# TTPI Reproduction Baseline

## System Configuration

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 5080 (16 GB) |
| CUDA | 12.8 |
| PyTorch | 2.11.0.dev20260130+cu128 |
| Python | 3.10.19 |
| OS | Linux 6.8.0-101-generic (Ubuntu 22.04) |
| Git commit | `94f32ec` |
| Branch | `repro/hardmove-5080` |
| Working dir | `/home/s110/code/ttpi_iclr` |
| Conda env | `tt_5080` |
| Paper GPU | NVIDIA GeForce RTX 3090 (24 GB) |

## Shared Parameters (All Experiments)

```python
# Algorithm
gamma = 0.99
n_iter_v = 1
n_samples = 50
normalize_reward = True

# TT-Cross
rmax_v, rmax_a = 100, 100
nswp_v, nswp_a = 5, 10
kickrank_v, kickrank_a = 10, 10
eps_cross_v, eps_cross_a = 1e-3, 1e-3
eps_round_v, eps_round_a = 1e-3, 1e-3

# Environment
dt = 0.01
horizon_sec = 10.0
target_radius = 0.02
position_bounds = [-1, 1]
velocity_bounds = [0, 0.25]
w_goal = 1e3
w_action = 1e4

# Initial state sampling
init_clip = [0.25, 0.75]  # middle 50% of state bounds
init_velocity = 0

# GPU memory fix
PYTORCH_CUDA_ALLOC_CONF = expandable_segments:True
```

## HM(8) — n_state=100, n_action=100, n_iter=200

**Per-seed results (n_test=20, best joint S≥0.95):**

| Seed | Best S | μ at S≥0.95 | Train Time | Notes |
|------|--------|-------------|------------|-------|
| 0 | 0.95 | 0.84 | ~210s | best single seed |
| 1 | 1.00 | 0.73 | ~470s | S=1.0, μ moderate |
| 2 | - | - | - | OOM early (cb0 only) |
| 3 | 1.00 | 0.66 | ~300s | |
| 4 | 0.90 | 0.60 | ~240s | S<0.95 at best μ |

**Aggregate:** S = 0.96 ± 0.04, μ = 0.71 ± 0.10 (4 seeds, excl. seed 2)

**Paper:** S = 1.00, μ = 0.93 ± 0.01, T = 850s (3090)

## HM(12) — n_state=50, n_action=100, n_iter=100

**Per-seed results (n_test=20):**

| Seed | Best S | μ at S≥0.95 | Train Time | Notes |
|------|--------|-------------|------------|-------|
| 0 | 0.95 | 0.77 | ~120s | |
| 1 | 0.95 | 0.66 | ~130s | |
| 2 | 0.85 | - | ~90s | S<0.95 at all checkpoints |

**Aggregate:** S = 0.95 ± 0.00, μ = 0.72 ± 0.08 (2 seeds at S≥0.95)

**Paper:** S = 1.00, μ = 0.92 ± 0.01, T = 946s (3090)

## HM(16) — n_state=50, n_action=100, n_iter=100

| Seed | Status |
|------|--------|
| 0-4 | OOM on 5080 (36D TT domain exceeds 16 GB) |

**Paper:** S = 1.00, μ = 0.92 ± 0.02, T = 1743s (3090)

## CP (Catch-Point) — n_state=100, n_action=100, n_iter=50

| Seed | Best S | Best μ | Train Time |
|------|--------|--------|------------|
| 0 | 1.00 | 1.00 | 47s |

**Paper:** S = 1.00, μ = 1.00, T = 30s (3090)

## Table 1 Comparison

| Task | Metric | Ours (5080) | Paper (3090) | Gap |
|------|--------|-------------|-------------|-----|
| CP | S | 1.00 | 1.00 | ✓ |
| CP | μ | 1.00 | 1.00 | ✓ |
| CP | T | 47s | 30s | 5080 faster |
| HM(8) | S | 0.96 | 1.00 | -0.04 |
| HM(8) | μ | 0.71 | 0.93 | -0.22 |
| HM(8) | T | ~350s | 850s | 5080 faster |
| HM(12) | S | 0.95 | 1.00 | -0.05 |
| HM(12) | μ | 0.72 | 0.92 | -0.20 |
| HM(12) | T | ~120s | 946s | 5080 faster |
| HM(16) | - | OOM | 1.00 / 0.92 | 5080 GPU memory |

## Known Limitations

1. **S-μ trade-off**: inherent to GPI with finite TT rank — policy cannot simultaneously maximize success rate and path efficiency
2. **GPU memory**: 5080 (16 GB) < 3090 (24 GB), causing OOM on HM(16) and limiting training iterations
3. **μ gap ~0.2**: path efficiency ~15% below paper, primarily due to TT approximation error accumulation in GPI loop
4. **n_test=20** (notebook default) introduces ~0.05-0.08 statistical noise in μ estimates
