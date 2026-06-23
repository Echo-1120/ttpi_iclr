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


def local_pair_order(n_actuator: int) -> list[int]:
    """Original HardMove physical layout: [acc_i, sw_i] repeated."""
    return list(range(2 * n_actuator))


def badsplit_order(n_actuator: int) -> list[int]:
    """Destructive layout: all accelerations first, all switches second."""
    n_modes = 2 * n_actuator
    return list(range(0, n_modes, 2)) + list(range(1, n_modes, 2))


def opposite_pair_order(n_actuator: int) -> list[int]:
    """Pair-preserving order that places opposite actuator blocks nearby."""
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

    The name is kept for CLI compatibility. The implementation is dependency-free
    because experiment environments may not have numpy installed on the client.
    """
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
        "opposite_pair": opposite_pair_order(n_actuator),
        "pam_spectral": spectral_order(coupling),
        "pam_greedy": greedy_adjacent_order(coupling),
    }
    candidates["pam_spectral_refined"] = improve_by_adjacent_swaps(candidates["pam_spectral"], coupling)
    candidates["pam_greedy_refined"] = improve_by_adjacent_swaps(candidates["pam_greedy"], coupling)

    results: dict[str, OrderingResult] = {}
    for name, order in candidates.items():
        order = validate_order(order, len(coupling))
        results[name] = OrderingResult(
            name=name,
            order=order,
            objective=weighted_linear_arrangement(order, coupling),
            adjacency_score=adjacency_score(order, coupling),
            metadata={
                "n_actuator": n_actuator,
                "pair_weight": pair_weight,
                "neighbor_weight": neighbor_weight,
                "opposite_weight": opposite_weight,
                "random_seed": random_seed,
            },
        )
    return coupling, results
