"""HardMove environment variant helpers.

This module centralizes action-space transformations used by training and
diagnostic scripts. The legacy ``index_permuted`` name is kept for CLI
compatibility, but its current behavior is an actuator/action relabeling.
"""

from __future__ import annotations

import torch


VARIANT_ALIASES = {
    "standard": "standard",
    "actuator_relabelled": "actuator_relabelled",
    "index_permuted": "actuator_relabelled",
    "cross_coupled": "cross_coupled",
}

VARIANT_DISPLAY_NAMES = {
    "standard": "standard",
    "actuator_relabelled": "actuator_relabelled",
    "index_permuted": "actuator_relabelled",
    "cross_coupled": "cross_coupled",
}


def canonical_env_variant(variant: str) -> str:
    try:
        return VARIANT_ALIASES[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown HardMove env variant {variant!r}") from exc


def variant_display_name(variant: str) -> str:
    return VARIANT_DISPLAY_NAMES.get(variant, canonical_env_variant(variant))


def make_env_permutation(n_actuator: int, seed: int) -> list[int]:
    """Return a reproducible actuator permutation."""
    g_perm = torch.Generator(device="cpu")
    g_perm.manual_seed(int(seed) + int(n_actuator))
    return torch.randperm(int(n_actuator), generator=g_perm).tolist()


def transform_hardmove_action(
    action_phys: torch.Tensor,
    n_actuator: int,
    variant: str,
    permutation: list[int] | torch.Tensor | None = None,
    cross_coupling_strength: float = 0.0,
) -> torch.Tensor:
    """Transform physical HardMove actions for a named stress-test variant."""
    canonical = canonical_env_variant(variant)
    action_out = action_phys.clone()
    if canonical == "standard":
        return action_out

    if canonical == "actuator_relabelled":
        if permutation is None:
            permutation = list(range(n_actuator))
        perm = torch.as_tensor(permutation, device=action_out.device, dtype=torch.long)
        return action_out.reshape(-1, n_actuator, 2)[:, perm, :].reshape(action_out.shape)

    if canonical == "cross_coupled":
        beta = float(cross_coupling_strength)
        if beta == 0.0:
            return action_out
        reshaped = action_out.reshape(-1, n_actuator, 2).clone()
        acc = reshaped[:, :, 0]
        left = torch.roll(acc, shifts=1, dims=1)
        right = torch.roll(acc, shifts=-1, dims=1)
        reshaped[:, :, 0] = acc + beta * 0.5 * (left + right)
        return reshaped.reshape(action_out.shape)

    raise ValueError(f"Unhandled canonical HardMove env variant {canonical!r}")
