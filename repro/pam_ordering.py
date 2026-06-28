"""Physics-aware TT mode ordering utilities.

PAM represents physical locality as a weighted graph over tensor modes. The
ordering problem is treated as a weighted minimum linear arrangement objective:

    sum_{i < j} W[i, j] * abs(pos[i] - pos[j])

Lower objective means strongly coupled modes are placed closer in the TT chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class OrderingResult:
    name: str
    order: list[int]
    objective: float
    adjacency_score: float
    metadata: dict


def validate_order(order: Iterable[int], n_modes: int) -> list[int]:
    values = [int(x) for x in order]
    expected = list(range(n_modes))
    if sorted(values) != expected:
        raise ValueError(f"Invalid order {values}; expected a permutation of {expected}")
    return values


Matrix = list[list[float]]

ORDERING_NAMES = (
    "local",
    "badsplit",
    "random",
    "reverse_blocks",
    "flip_within_block",
    "opposite_pair",
    "pam_spectral",
    "pam_greedy",
    "pam_spectral_refined",
    "pam_greedy_refined",
    "block_pam",
    "free_pam",
    "peakcut_pam",
    "sensitivity_pam",
    "rankaware_proxy_pam",
    "rankaware_spectral_pam",
    "hybrid_pam",
)


ORDERING_METADATA = {
    "local": ("local", "Local", "primary_baseline", False),
    "random": ("random", "Random", "diagnostic_control", False),
    "badsplit": ("badsplit", "BadSplit", "diagnostic_control", False),
    "reverse_blocks": ("reverse_blocks", "ReverseBlocks", "diagnostic_control", False),
    "flip_within_block": ("flip_within_block", "FlipWithinBlock", "diagnostic_control", False),
    "opposite_pair": (
        "opposite_interleave_legacy",
        "OppositeInterleaveLegacy",
        "legacy_diagnostic",
        True,
    ),
    "pam_spectral": ("pam_spectral", "SpectralPAM", "surrogate_baseline", False),
    "pam_greedy": ("pam_greedy", "GreedyPAM", "surrogate_baseline", False),
    "pam_spectral_refined": ("pam_spectral_refined", "SpectralPAMRefined", "surrogate_baseline", False),
    "pam_greedy_refined": ("pam_greedy_refined", "GreedyPAMRefined", "surrogate_baseline", False),
    "block_pam": ("block_pam", "BlockPAM", "ablation", False),
    "free_pam": ("free_pam", "FreePAM", "ablation", False),
    "peakcut_pam": ("peakcut_pam", "PeakCutPAM", "surrogate_baseline", False),
    "sensitivity_pam": ("sensitivity_pam", "SensitivityPAM", "surrogate_baseline", False),
    "rankaware_proxy_pam": ("rankaware_proxy_pam", "RankAwareProxyPAM", "main_method", False),
    "rankaware_spectral_pam": ("rankaware_spectral_pam", "RankAwareSpectralPAM", "ablation", False),
    "hybrid_pam": ("hybrid_pam", "HybridPAM", "main_method", False),
}


def weighted_linear_arrangement(order: Iterable[int], coupling: Matrix) -> float:
    """Return weighted minimum-linear-arrangement cost for an ordering."""
    order = validate_order(order, len(coupling))
    pos = {mode: idx for idx, mode in enumerate(order)}
    cost = 0.0
    for i in range(len(coupling)):
        for j in range(i + 1, len(coupling)):
            weight = float(coupling[i][j])
            if weight:
                cost += weight * abs(pos[i] - pos[j])
    return float(cost)


def adjacency_score(order: Iterable[int], coupling: Matrix) -> float:
    """Return sum of couplings between adjacent modes; larger is better."""
    order = validate_order(order, len(coupling))
    return float(sum(float(coupling[order[i]][order[i + 1]]) for i in range(len(order) - 1)))


def segmented_cut_costs(order: Iterable[int], coupling: Matrix) -> list[float]:
    """Weighted crossing mass for each TT cut.

    The sum of these cut costs equals the weighted linear arrangement objective,
    because an edge with endpoint distance d crosses exactly d cuts.
    """
    order = validate_order(order, len(coupling))
    pos = {mode: idx for idx, mode in enumerate(order)}
    costs: list[float] = []
    for cut in range(1, len(order)):
        left = {mode for mode, idx in pos.items() if idx < cut}
        cost = 0.0
        for i in left:
            for j in range(len(order)):
                if j not in left:
                    cost += float(coupling[i][j])
        costs.append(float(cost))
    return costs


def weighted_segmented_cut_sum(order: Iterable[int], coupling: Matrix) -> float:
    """Return sum_k cut_cost_k, identical to weighted linear arrangement."""
    return float(sum(segmented_cut_costs(order, coupling)))


def peak_cut_cost(order: Iterable[int], coupling: Matrix) -> float:
    """Maximum weighted edge mass crossing any TT cut."""
    return float(max(segmented_cut_costs(order, coupling), default=0.0))


def rankaware_proxy_cost(
    order: Iterable[int],
    coupling: Matrix,
    lambda_sum: float = 1.0,
    lambda_peak: float = 1.0,
) -> float:
    """Rank-aware surrogate from cut masses."""
    import math

    cut_loads = segmented_cut_costs(order, coupling)
    logs = [math.log1p(max(0.0, x)) for x in cut_loads]
    return float(lambda_sum * sum(logs) + lambda_peak * max(logs, default=0.0))


def available_orderings() -> tuple[str, ...]:
    """Return the unified public ordering registry."""
    return ORDERING_NAMES


def hardmove_action_coupling(
    n_actuator: int,
    pair_weight: float = 10.0,
    neighbor_weight: float = 1.0,
    opposite_weight: float = 2.0,
) -> Matrix:
    """Build a physics coupling graph for HardMove action modes.

    Mode convention follows the original TTPI HardMove action layout:

        [acc_0, sw_0, acc_1, sw_1, ..., acc_{n-1}, sw_{n-1}]

    The strongest edge couples each actuator acceleration with its binary switch.
    Weaker block-level edges encode local neighborhood/opposite actuator structure.
    """
    if n_actuator < 1:
        raise ValueError("n_actuator must be positive")

    n_modes = 2 * n_actuator
    coupling = [[0.0 for _ in range(n_modes)] for _ in range(n_modes)]

    def add(i: int, j: int, weight: float) -> None:
        if i == j or weight <= 0:
            return
        coupling[i][j] = max(coupling[i][j], float(weight))
        coupling[j][i] = max(coupling[j][i], float(weight))

    for actuator in range(n_actuator):
        acc = 2 * actuator
        sw = acc + 1
        add(acc, sw, pair_weight)

    # Block-level physical couplings are applied across all modes in each block.
    for actuator in range(n_actuator):
        neighbor = (actuator + 1) % n_actuator
        for left in (2 * actuator, 2 * actuator + 1):
            for right in (2 * neighbor, 2 * neighbor + 1):
                add(left, right, neighbor_weight)

        if n_actuator % 2 == 0:
            opposite = (actuator + n_actuator // 2) % n_actuator
            for left in (2 * actuator, 2 * actuator + 1):
                for right in (2 * opposite, 2 * opposite + 1):
                    add(left, right, opposite_weight)

    return coupling


def actuator_blocks(n_actuator: int) -> list[list[int]]:
    return [[2 * i, 2 * i + 1] for i in range(n_actuator)]


def block_coupling(coupling: Matrix, blocks: list[list[int]]) -> Matrix:
    n_blocks = len(blocks)
    out = [[0.0 for _ in range(n_blocks)] for _ in range(n_blocks)]
    for i, left in enumerate(blocks):
        for j, right in enumerate(blocks):
            if i == j:
                continue
            out[i][j] = float(sum(coupling[a][b] for a in left for b in right))
    return out


def local_pair_order(n_actuator: int) -> list[int]:
    """Original HardMove physical layout: [acc_i, sw_i] repeated."""
    return list(range(2 * n_actuator))


def badsplit_order(n_actuator: int) -> list[int]:
    """Destructive layout: all accelerations first, all switches second."""
    n_modes = 2 * n_actuator
    return list(range(0, n_modes, 2)) + list(range(1, n_modes, 2))


def reverse_blocks_order(n_actuator: int) -> list[int]:
    """Reverse actuator blocks while preserving [acc_i, sw_i] adjacency."""
    order: list[int] = []
    for actuator in reversed(range(n_actuator)):
        order.extend([2 * actuator, 2 * actuator + 1])
    return order


def flip_within_block_order(n_actuator: int) -> list[int]:
    """Flip every actuator pair: [sw_i, acc_i] repeated."""
    order: list[int] = []
    for actuator in range(n_actuator):
        order.extend([2 * actuator + 1, 2 * actuator])
    return order


def opposite_pair_order(n_actuator: int) -> list[int]:
    """Legacy pair-preserving interleave that places opposite blocks nearby."""
    if n_actuator % 2:
        return local_pair_order(n_actuator)
    half = n_actuator // 2
    order: list[int] = []
    for i in range(half):
        order.extend([2 * i, 2 * i + 1, 2 * (i + half), 2 * (i + half) + 1])
    return order


def random_order(n_actuator: int, seed: int) -> list[int]:
    import random

    values = list(range(2 * n_actuator))
    rng = random.Random(seed)
    rng.shuffle(values)
    return values


def _weighted_degree(coupling: Matrix, mode: int) -> float:
    return float(sum(coupling[mode]))


def _weighted_bfs_order(coupling: Matrix) -> list[int]:
    """Dependency-free spectral substitute: high-degree weighted BFS chain."""
    n_modes = len(coupling)
    if n_modes <= 1:
        return list(range(n_modes))
    remaining = set(range(n_modes))
    start = max(remaining, key=lambda mode: (_weighted_degree(coupling, mode), -mode))
    order = [start]
    remaining.remove(start)
    while remaining:
        best = max(
            remaining,
            key=lambda mode: (
                max(coupling[mode][placed] for placed in order),
                _weighted_degree(coupling, mode),
                -mode,
            ),
        )
        # Place on the side with stronger endpoint coupling.
        if coupling[best][order[0]] > coupling[best][order[-1]]:
            order.insert(0, best)
        else:
            order.append(best)
        remaining.remove(best)
    return order


def spectral_order(coupling: Matrix) -> list[int]:
    """Return a deterministic graph order.

    Uses a Fiedler-vector order when numpy is available; otherwise falls back to
    a dependency-free weighted BFS chain.
    """
    try:
        import numpy as np

        W = np.asarray(coupling, dtype=float)
        if W.shape[0] <= 2:
            return list(range(W.shape[0]))
        D = np.diag(W.sum(axis=1))
        L = D - W
        vals, vecs = np.linalg.eigh(L)
        fiedler = vecs[:, 1]
        return [int(i) for i in np.argsort(fiedler, kind="mergesort")]
    except Exception:
        return _weighted_bfs_order(coupling)


def greedy_adjacent_order(coupling: Matrix) -> list[int]:
    """Greedy chain construction maximizing adjacent coupling at each step."""
    n_modes = len(coupling)
    if n_modes == 1:
        return [0]

    i, j, best_weight = 0, 1, float("-inf")
    for left in range(n_modes):
        for right in range(n_modes):
            if left != right and coupling[left][right] > best_weight:
                i, j, best_weight = left, right, coupling[left][right]
    order = [int(i), int(j)]
    remaining = set(range(n_modes)) - set(order)

    while remaining:
        best_mode = None
        best_side = "right"
        best_gain = -1.0
        left = order[0]
        right = order[-1]
        for mode in remaining:
            left_gain = float(coupling[mode][left])
            right_gain = float(coupling[mode][right])
            if left_gain > best_gain:
                best_mode = mode
                best_side = "left"
                best_gain = left_gain
            if right_gain > best_gain:
                best_mode = mode
                best_side = "right"
                best_gain = right_gain
        assert best_mode is not None
        if best_side == "left":
            order.insert(0, int(best_mode))
        else:
            order.append(int(best_mode))
        remaining.remove(best_mode)
    return order


def improve_by_adjacent_swaps(order: Iterable[int], coupling: Matrix, max_passes: int = 20) -> list[int]:
    """Local adjacent-swap refinement for the MLA objective."""
    current = validate_order(order, len(coupling))
    current_cost = weighted_linear_arrangement(current, coupling)

    for _ in range(max_passes):
        improved = False
        for idx in range(len(current) - 1):
            candidate = list(current)
            candidate[idx], candidate[idx + 1] = candidate[idx + 1], candidate[idx]
            candidate_cost = weighted_linear_arrangement(candidate, coupling)
            if candidate_cost + 1e-12 < current_cost:
                current = candidate
                current_cost = candidate_cost
                improved = True
        if not improved:
            break
    return current


def improve_by_2opt(order: Iterable[int], coupling: Matrix, max_passes: int = 20) -> list[int]:
    """2-opt local search for the weighted linear arrangement objective."""
    current = validate_order(order, len(coupling))
    current_cost = weighted_linear_arrangement(current, coupling)
    n = len(current)
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n + 1):
                candidate = current[:i] + list(reversed(current[i:j])) + current[j:]
                candidate_cost = weighted_linear_arrangement(candidate, coupling)
                if candidate_cost + 1e-12 < current_cost:
                    current = candidate
                    current_cost = candidate_cost
                    improved = True
        if not improved:
            break
    return current


def block_preserving_order(
    n_actuator: int,
    coupling: Matrix,
    allow_internal_flip: bool = True,
) -> list[int]:
    """Order actuator blocks while preserving [acc_i, sw_i] adjacency."""
    blocks = actuator_blocks(n_actuator)
    Wb = block_coupling(coupling, blocks)
    block_order = spectral_order(Wb)
    block_order = improve_by_2opt(block_order, Wb)
    ordered_blocks = [list(blocks[i]) for i in block_order]
    if allow_internal_flip:
        for idx, block in enumerate(ordered_blocks):
            candidate_blocks = [list(b) for b in ordered_blocks]
            candidate_blocks[idx] = list(reversed(block))
            candidate = [mode for b in candidate_blocks for mode in b]
            current = [mode for b in ordered_blocks for mode in b]
            if weighted_linear_arrangement(candidate, coupling) < weighted_linear_arrangement(current, coupling):
                ordered_blocks[idx] = list(reversed(block))
    return [mode for block in ordered_blocks for mode in block]


def free_pam_order(coupling: Matrix) -> list[int]:
    return improve_by_2opt(spectral_order(coupling), coupling)


def sensitivity_weighted_coupling(base: Matrix, n_actuator: int) -> Matrix:
    """Deterministic sensitivity proxy that emphasizes neighboring thrust axes."""
    out = [[float(x) for x in row] for row in base]
    for i in range(n_actuator):
        for j in range(n_actuator):
            if i == j:
                continue
            angular = abs(i - j)
            angular = min(angular, n_actuator - angular)
            weight = 1.0 / (1.0 + angular)
            for left in (2 * i, 2 * i + 1):
                for right in (2 * j, 2 * j + 1):
                    out[left][right] += weight
                    out[right][left] += weight
    return out


def build_hardmove_orders(
    n_actuator: int,
    random_seed: int = 42,
    pair_weight: float = 10.0,
    neighbor_weight: float = 1.0,
    opposite_weight: float = 2.0,
) -> tuple[Matrix, dict[str, OrderingResult]]:
    """Return coupling matrix and standard HardMove ordering set."""
    coupling = hardmove_action_coupling(
        n_actuator=n_actuator,
        pair_weight=pair_weight,
        neighbor_weight=neighbor_weight,
        opposite_weight=opposite_weight,
    )

    candidates = {
        "local": local_pair_order(n_actuator),
        "badsplit": badsplit_order(n_actuator),
        "random": random_order(n_actuator, random_seed),
        "reverse_blocks": reverse_blocks_order(n_actuator),
        "flip_within_block": flip_within_block_order(n_actuator),
        "opposite_pair": opposite_pair_order(n_actuator),
        "pam_spectral": spectral_order(coupling),
        "pam_greedy": greedy_adjacent_order(coupling),
    }
    candidates["pam_spectral_refined"] = improve_by_adjacent_swaps(candidates["pam_spectral"], coupling)
    candidates["pam_greedy_refined"] = improve_by_adjacent_swaps(candidates["pam_greedy"], coupling)
    candidates["block_pam"] = block_preserving_order(n_actuator, coupling, allow_internal_flip=True)
    candidates["free_pam"] = free_pam_order(coupling)
    candidates["peakcut_pam"] = min(
        [
            candidates["local"],
            candidates["pam_spectral"],
            candidates["pam_greedy"],
            candidates["block_pam"],
            candidates["free_pam"],
            candidates["reverse_blocks"],
            random_order(n_actuator, random_seed + 503),
        ],
        key=lambda order: peak_cut_cost(order, coupling),
    )

    sensitivity_coupling = sensitivity_weighted_coupling(coupling, n_actuator)
    candidates["sensitivity_pam"] = block_preserving_order(
        n_actuator, sensitivity_coupling, allow_internal_flip=True
    )
    candidates["rankaware_proxy_pam"] = min(
        [
            candidates["local"],
            candidates["pam_spectral"],
            candidates["pam_greedy"],
            candidates["block_pam"],
            candidates["free_pam"],
            random_order(n_actuator, random_seed + 1009),
        ],
        key=lambda order: rankaware_proxy_cost(order, coupling, lambda_sum=1.0, lambda_peak=2.0),
    )
    candidates["rankaware_spectral_pam"] = candidates["rankaware_proxy_pam"]
    hybrid_coupling = [
        [
            0.5 * float(coupling[i][j]) + 0.5 * float(sensitivity_coupling[i][j])
            for j in range(len(coupling))
        ]
        for i in range(len(coupling))
    ]
    candidates["hybrid_pam"] = block_preserving_order(n_actuator, hybrid_coupling, allow_internal_flip=True)

    results: dict[str, OrderingResult] = {}
    for name, order in candidates.items():
        order = validate_order(order, len(coupling))
        canonical_name, display_name, baseline_category, deprecated_alias = ORDERING_METADATA[name]
        results[name] = OrderingResult(
            name=name,
            order=order,
            objective=weighted_linear_arrangement(order, coupling),
            adjacency_score=adjacency_score(order, coupling),
            metadata={
                "canonical_name": canonical_name,
                "display_name": display_name,
                "baseline_category": baseline_category,
                "deprecated_alias": deprecated_alias,
                "n_actuator": n_actuator,
                "pair_weight": pair_weight,
                "neighbor_weight": neighbor_weight,
                "opposite_weight": opposite_weight,
                "random_seed": random_seed,
                "permutation_seed": random_seed,
                "ordering_seed": random_seed,
                "peak_cut_objective": peak_cut_cost(order, coupling),
                "rankaware_proxy_objective": rankaware_proxy_cost(
                    order, coupling, lambda_sum=1.0, lambda_peak=2.0
                ),
                "block_preserving": all(
                    abs(order.index(2 * i) - order.index(2 * i + 1)) == 1
                    for i in range(n_actuator)
                ),
            },
        )
    return coupling, results
