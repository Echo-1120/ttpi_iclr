"""Baseline equivalence checks for HardMove action-mode ordering.

These helpers verify that the ordering wrapper preserves the original TTPI
local baseline semantics when ``action_order == local`` and
``env_variant == standard``.
"""

from __future__ import annotations

import random
import copy
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class EquivalenceResult:
    name: str
    passed: bool
    details: dict


def action_inverse_from_order(order: Iterable[int]) -> list[int]:
    """Return inverse map from physical action position to TT-order position."""
    values = [int(x) for x in order]
    expected = list(range(len(values)))
    if sorted(values) != expected:
        raise ValueError(f"Invalid action order {values}; expected a permutation of {expected}")
    return [values.index(i) for i in expected]


def action_mode_sizes(n_actuator: int, n_action: int) -> list[int]:
    return [int(n_action) if mode % 2 == 0 else 2 for mode in range(2 * int(n_actuator))]


def decode_tt_indices_to_physical(index_rows: list[list[int]], action_inverse: list[int]) -> list[list[int]]:
    """Decode TT-ordered action index rows to original physical action order."""
    return [[row[i] for i in action_inverse] for row in index_rows]


def physical_indices_to_tt(index_rows: list[list[int]], action_order: list[int]) -> list[list[int]]:
    """Encode original physical action index rows into a TT action ordering."""
    return [[row[i] for i in action_order] for row in index_rows]


def random_physical_index_rows(
    *,
    n_actuator: int,
    n_action: int,
    n_samples: int,
    seed: int,
) -> list[list[int]]:
    rng = random.Random(seed)
    sizes = action_mode_sizes(n_actuator, n_action)
    return [[rng.randrange(size) for size in sizes] for _ in range(n_samples)]


def check_index_roundtrip(
    *,
    action_order: list[int],
    n_actuator: int,
    n_action: int,
    n_samples: int,
    seed: int,
) -> EquivalenceResult:
    """Check index-level encode/decode roundtrip for a given ordering."""
    action_inverse = action_inverse_from_order(action_order)
    physical_rows = random_physical_index_rows(
        n_actuator=n_actuator,
        n_action=n_action,
        n_samples=n_samples,
        seed=seed,
    )
    tt_rows = physical_indices_to_tt(physical_rows, action_order)
    decoded_rows = decode_tt_indices_to_physical(tt_rows, action_inverse)
    mismatches = [
        {"sample": i, "physical": physical_rows[i], "tt": tt_rows[i], "decoded": decoded_rows[i]}
        for i in range(len(physical_rows))
        if decoded_rows[i] != physical_rows[i]
    ]
    return EquivalenceResult(
        name="index_roundtrip",
        passed=not mismatches,
        details={
            "n_samples": n_samples,
            "n_actuator": n_actuator,
            "n_action": n_action,
            "action_order": action_order,
            "action_inverse": action_inverse,
            "num_mismatches": len(mismatches),
            "first_mismatches": mismatches[:5],
        },
    )


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError("PyTorch is required for HardMove equivalence checks") from exc
    return torch


def _build_hardmove_domains(*, n_actuator: int, n_state: int, n_action: int, device):
    torch = _require_torch()
    l_bound = 1.0
    velocity_max = 0.25 * l_bound
    acc = torch.linspace(-velocity_max, velocity_max, n_action, device=device)
    switch = torch.arange(2, device=device, dtype=torch.float64)
    state_min = torch.tensor([-l_bound, -l_bound, 0.0, 0.0], dtype=torch.float64, device=device)
    state_max = torch.tensor([l_bound, l_bound, velocity_max, velocity_max], dtype=torch.float64, device=device)
    domain_state = [torch.linspace(state_min[i], state_max[i], n_state, device=device) for i in range(4)]
    domain_action_phys = [acc, switch] * n_actuator
    return domain_state, domain_action_phys, state_min, state_max


def _indices_to_values(indices, domains):
    torch = _require_torch()
    return torch.stack([domain[indices[:, site]] for site, domain in enumerate(domains)], dim=1)


def _sample_state_batch(*, n_samples: int, state_min, state_max, seed: int, device):
    torch = _require_torch()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 991)
    values = torch.rand((n_samples, 4), generator=generator, dtype=torch.float64)
    state = state_min.cpu() + values * (state_max.cpu() - state_min.cpu())
    state[:, 2:4] = 0.0
    return state.to(device)


def check_hardmove_local_action_equivalence(
    *,
    n_actuator: int = 8,
    n_state: int = 20,
    n_action: int = 20,
    n_samples: int = 4096,
    seed: int = 0,
    device: str = "cpu",
) -> EquivalenceResult:
    """Compare original local decode with the new local wrapper on random actions."""
    torch = _require_torch()
    from dynamic_systems import HardMove
    from repro.hardmove_variants import transform_hardmove_action
    from repro.pam_ordering import build_hardmove_orders

    torch.set_default_dtype(torch.float64)
    device_obj = torch.device(device)
    domain_state, domain_action_phys, state_min, state_max = _build_hardmove_domains(
        n_actuator=n_actuator,
        n_state=n_state,
        n_action=n_action,
        device=device_obj,
    )
    del domain_state
    _, orders = build_hardmove_orders(n_actuator=n_actuator, random_seed=seed)
    action_order = orders["local"].order
    action_inverse = torch.tensor(action_inverse_from_order(action_order), dtype=torch.long, device=device_obj)
    domain_action_tt = [domain_action_phys[i] for i in action_order]

    physical_rows = random_physical_index_rows(
        n_actuator=n_actuator,
        n_action=n_action,
        n_samples=n_samples,
        seed=seed,
    )
    physical_idx = torch.tensor(physical_rows, dtype=torch.long, device=device_obj)
    tt_idx = physical_idx[:, torch.tensor(action_order, dtype=torch.long, device=device_obj)]

    original_action = _indices_to_values(physical_idx, domain_action_phys)
    new_action_tt = _indices_to_values(tt_idx, domain_action_tt)
    decoded_action = transform_hardmove_action(
        new_action_tt[:, action_inverse],
        n_actuator=n_actuator,
        variant="standard",
        permutation=list(range(n_actuator)),
        cross_coupling_strength=0.0,
    )

    # Code-valued domains catch mode swaps that identical acceleration/switch
    # domains could otherwise hide.
    code_domains_phys = [
        torch.arange(len(domain), dtype=torch.float64, device=device_obj) + 1000.0 * mode
        for mode, domain in enumerate(domain_action_phys)
    ]
    code_domains_tt = [code_domains_phys[i] for i in action_order]
    original_code = _indices_to_values(physical_idx, code_domains_phys)
    decoded_code = _indices_to_values(tt_idx, code_domains_tt)[:, action_inverse]

    dyn = HardMove(dt=0.01, w_goal=1e3, w_action=1e4, n=n_actuator, device=device_obj)
    state = _sample_state_batch(n_samples=n_samples, state_min=state_min, state_max=state_max, seed=seed, device=device_obj)
    original_next = dyn.forward_simulate(state, original_action)
    decoded_next = dyn.forward_simulate(state, decoded_action)
    original_reward = dyn.reward_state_action(state, original_action)
    decoded_reward = dyn.reward_state_action(state, decoded_action)

    action_diff = float((original_action - decoded_action).abs().max().detach().cpu().item())
    code_diff = float((original_code - decoded_code).abs().max().detach().cpu().item())
    next_diff = float((original_next - decoded_next).abs().max().detach().cpu().item())
    reward_diff = float((original_reward - decoded_reward).abs().max().detach().cpu().item())
    passed = action_diff == 0.0 and code_diff == 0.0 and next_diff == 0.0 and reward_diff == 0.0
    return EquivalenceResult(
        name="hardmove_local_action_decode",
        passed=passed,
        details={
            "n_samples": n_samples,
            "n_actuator": n_actuator,
            "n_action": n_action,
            "action_order": action_order,
            "action_inverse": action_inverse.detach().cpu().tolist(),
            "max_action_abs_diff": action_diff,
            "max_code_abs_diff": code_diff,
            "max_next_state_abs_diff": next_diff,
            "max_reward_abs_diff": reward_diff,
        },
    )


def check_ttgo_local_equivalence(
    *,
    n_actuator: int = 4,
    n_state: int = 9,
    n_action: int = 7,
    n_states: int = 32,
    n_ttgo_samples: int = 8,
    rollout_steps: int = 5,
    seed: int = 0,
    device: str = "cpu",
) -> EquivalenceResult:
    """Compare original-local and new-framework-local TTGO outputs."""
    torch = _require_torch()
    try:
        import tntorch as tnt
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError("tntorch is required for TTGO equivalence checks") from exc

    from dynamic_systems import HardMove
    from ttpi import TTPI
    from repro.hardmove_variants import transform_hardmove_action
    from repro.pam_ordering import build_hardmove_orders

    torch.set_default_dtype(torch.float64)
    device_obj = torch.device(device)
    torch.manual_seed(seed)
    domain_state, domain_action_phys, state_min, state_max = _build_hardmove_domains(
        n_actuator=n_actuator,
        n_state=n_state,
        n_action=n_action,
        device=device_obj,
    )
    _, orders = build_hardmove_orders(n_actuator=n_actuator, random_seed=seed)
    action_order = orders["local"].order
    action_inverse = torch.tensor(action_inverse_from_order(action_order), dtype=torch.long, device=device_obj)
    domain_action_new = [domain_action_phys[i] for i in action_order]
    dyn = HardMove(dt=0.01, w_goal=1e3, w_action=1e4, n=n_actuator, device=device_obj)

    def reward_original(state, action):
        return dyn.reward_state_action(state, action)

    def forward_original(state, action):
        return dyn.forward_simulate(state, action)

    def transform_physical_action(action_phys):
        return transform_hardmove_action(
            action_phys,
            n_actuator=n_actuator,
            variant="standard",
            permutation=list(range(n_actuator)),
            cross_coupling_strength=0.0,
        )

    def reward_new(state, action):
        return dyn.reward_state_action(state, transform_physical_action(action[:, action_inverse]))

    def forward_new(state, action):
        return dyn.forward_simulate(state, transform_physical_action(action[:, action_inverse]))

    original = TTPI(
        domain_state=domain_state,
        domain_action=domain_action_phys,
        reward=reward_original,
        forward_model=forward_original,
        n_samples=n_ttgo_samples,
        device=device_obj,
    )
    new = TTPI(
        domain_state=domain_state,
        domain_action=domain_action_new,
        reward=reward_new,
        forward_model=forward_new,
        n_samples=n_ttgo_samples,
        device=device_obj,
    )

    dims = [len(domain) for domain in original.domain_state_action]
    torch.manual_seed(seed + 123)
    policy_model = tnt.rand(dims, ranks_tt=2).tt().to(device_obj)
    original.policy_model = copy.deepcopy(policy_model).to(device_obj)
    new.policy_model = copy.deepcopy(policy_model).to(device_obj)
    original.policy_model_cores = original.policy_model.tt().cores[:]
    new.policy_model_cores = new.policy_model.tt().cores[:]

    state = _sample_state_batch(
        n_samples=n_states,
        state_min=state_min,
        state_max=state_max,
        seed=seed + 17,
        device=device_obj,
    )

    max_action_diff = 0.0
    max_next_diff = 0.0
    max_reward_diff = 0.0
    state_original = state.clone()
    state_new = state.clone()
    for _ in range(rollout_steps):
        action_original = original.policy(state_original)
        action_new_tt = new.policy(state_new)
        action_new = transform_physical_action(action_new_tt[:, action_inverse])
        reward_original_value = dyn.reward_state_action(state_original, action_original)
        reward_new_value = dyn.reward_state_action(state_new, action_new)
        next_original = dyn.forward_simulate(state_original, action_original)
        next_new = dyn.forward_simulate(state_new, action_new)

        max_action_diff = max(
            max_action_diff,
            float((action_original - action_new).abs().max().detach().cpu().item()),
        )
        max_next_diff = max(
            max_next_diff,
            float((next_original - next_new).abs().max().detach().cpu().item()),
        )
        max_reward_diff = max(
            max_reward_diff,
            float((reward_original_value - reward_new_value).abs().max().detach().cpu().item()),
        )
        state_original = next_original
        state_new = next_new

    passed = max_action_diff == 0.0 and max_next_diff == 0.0 and max_reward_diff == 0.0
    return EquivalenceResult(
        name="ttgo_local_policy_rollout",
        passed=passed,
        details={
            "n_states": n_states,
            "n_ttgo_samples": n_ttgo_samples,
            "rollout_steps": rollout_steps,
            "n_actuator": n_actuator,
            "n_state": n_state,
            "n_action": n_action,
            "action_order": action_order,
            "action_inverse": action_inverse.detach().cpu().tolist(),
            "max_selected_action_abs_diff": max_action_diff,
            "max_next_state_abs_diff": max_next_diff,
            "max_reward_abs_diff": max_reward_diff,
        },
    )
