"""Load and normalize TTPI/PAM result artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "oom"}


def normalize_status(row: dict[str, Any]) -> str:
    status = str(row.get("status") or "").strip().lower()
    if status:
        return status
    if as_bool(row.get("oom")):
        return "oom"
    if row.get("error_type") or row.get("error_message"):
        return "error"
    return "ok"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_result_paths(results_root: Path, tag: str | None = None) -> Iterable[Path]:
    if not results_root.exists():
        return []
    paths = sorted(results_root.glob("*.json"))
    if tag:
        paths = [path for path in paths if tag in path.stem]
    return paths


def task_diagnostic_paths(run_id: str, diagnostics_root: Path) -> dict[str, Path]:
    return {
        "rank_profile": diagnostics_root / "rank_profiles" / f"{run_id}.rank_profile.csv",
        "cross_queries": diagnostics_root / "cross_queries" / f"{run_id}.cross_queries.csv",
        "cross_process": diagnostics_root / "cross_process" / f"{run_id}.cross_process.csv",
        "round_events": diagnostics_root / "round_events" / f"{run_id}.round_events.csv",
        "standard_json": diagnostics_root / "run_json" / f"{run_id}.standard.json",
    }


def load_records(
    *,
    results_root: Path | None = None,
    diagnostics_root: Path | None = None,
    tag: str | None = "formal5080",
) -> list[dict[str, Any]]:
    results_root = results_root or ROOT / "repro" / "results"
    diagnostics_root = diagnostics_root or ROOT / "repro" / "diagnostics"
    records: list[dict[str, Any]] = []
    for path in iter_result_paths(results_root, tag=tag):
        try:
            result = load_json(path)
        except Exception as exc:
            records.append(
                {
                    "path": path,
                    "result": {},
                    "run_id": path.stem,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "rank_rows": [],
                    "cross_rows": [],
                    "cross_process_rows": [],
                    "round_rows": [],
                }
            )
            continue
        run_id = result.get("run_id", path.stem)
        diag_paths = task_diagnostic_paths(run_id, diagnostics_root)
        status = normalize_status(result)
        records.append(
            {
                "path": path,
                "result": result,
                "run_id": run_id,
                "env": result.get("env_name", result.get("task", "")),
                "ordering": result.get("action_order_name", ""),
                "seed": as_int(result.get("training_seed", result.get("seed")), 0),
                "status": status,
                "oom": bool(result.get("oom", status == "oom")),
                "error_type": result.get("error_type", ""),
                "error_message": result.get("error_message", ""),
                "diagnostic_paths": diag_paths,
                "rank_rows": read_csv(diag_paths["rank_profile"]),
                "cross_rows": read_csv(diag_paths["cross_queries"]),
                "cross_process_rows": read_csv(diag_paths["cross_process"]),
                "round_rows": read_csv(diag_paths["round_events"]),
                "standard_log": load_json(diag_paths["standard_json"]) if diag_paths["standard_json"].exists() else {},
            }
        )
    return records


def load_summary_rows(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or ROOT / "repro" / "diagnostics" / "pam_ablation_summary.csv"
    rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        normalized = dict(row)
        normalized["status"] = normalize_status(normalized)
        normalized["oom"] = as_bool(normalized.get("oom"))
        rows.append(normalized)
    return rows


def count_files(path: Path, pattern: str = "*") -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob(pattern))
