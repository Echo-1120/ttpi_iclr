"""Validation helpers for PAM experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from repro.diagnostics import REQUIRED_SUMMARY_FIELDS, validate_required_summary_fields
from repro.pam_ordering import (
    Matrix,
    segmented_cut_costs,
    validate_order,
    weighted_linear_arrangement,
)


def validate_permutation(order: Iterable[int], n_modes: int) -> list[int]:
    """Validate that an ordering is a complete action-mode permutation."""
    return validate_order(order, n_modes)


def validate_block_adjacency(order: Iterable[int], n_actuator: int) -> bool:
    """Require each [acc_i, sw_i] pair to remain adjacent."""
    order = validate_order(order, 2 * n_actuator)
    for actuator in range(n_actuator):
        acc = 2 * actuator
        sw = acc + 1
        if abs(order.index(acc) - order.index(sw)) != 1:
            raise ValueError(f"Actuator block {actuator} is split in order {order}")
    return True


def validate_objective_identity(order: Iterable[int], coupling: Matrix, tol: float = 1e-9) -> bool:
    """Check MLA(order) == sum_k cut_cost_k."""
    lhs = weighted_linear_arrangement(order, coupling)
    rhs = sum(segmented_cut_costs(order, coupling))
    if abs(lhs - rhs) > tol:
        raise ValueError(f"Objective identity failed: MLA={lhs}, cut_sum={rhs}")
    return True


def validate_log_fields(row: dict[str, Any]) -> bool:
    """Validate one global summary row."""
    validate_required_summary_fields(row)
    return True


def validate_summary_csv(path: str | Path) -> bool:
    """Validate required fields for every row in a PAM summary CSV."""
    import csv

    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing_header = [field for field in REQUIRED_SUMMARY_FIELDS if field not in (reader.fieldnames or [])]
        if missing_header:
            raise ValueError(f"Summary CSV missing required columns: {missing_header}")
        for row in reader:
            validate_log_fields(row)
    return True


def validate_standard_run_json(path: str | Path) -> bool:
    """Validate the standard JSON run log shape."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    required_top = ["run_id", "seed", "env", "ordering", "ordering_group", "objectives", "resources", "performance", "tt_cross", "ranks", "status"]
    missing = [key for key in required_top if key not in data]
    if missing:
        raise ValueError(f"Standard JSON missing top-level keys: {missing}")
    for key in ["la", "peak_cut", "rankaware_proxy"]:
        if key not in data["objectives"]:
            raise ValueError(f"Standard JSON missing objectives.{key}")
    for key in ["peak_memory_mb", "runtime_sec"]:
        if key not in data["resources"]:
            raise ValueError(f"Standard JSON missing resources.{key}")
    for key in ["success_rate", "avg_return", "mu"]:
        if key not in data["performance"]:
            raise ValueError(f"Standard JSON missing performance.{key}")
    for key in ["calls", "function_evals_total", "queried_points_total"]:
        if key not in data["tt_cross"]:
            raise ValueError(f"Standard JSON missing tt_cross.{key}")
    return True
