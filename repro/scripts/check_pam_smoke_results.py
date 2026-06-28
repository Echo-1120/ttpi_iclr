#!/usr/bin/env python3
"""Validate that a PAM smoke manifest produced complete structured outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from repro.scripts.run_pam_ablation import build_run_commands, load_manifest


REQUIRED_RESULT_FIELDS = (
    "run_id",
    "seed",
    "training_seed",
    "environment_seed",
    "permutation_seed",
    "state_sampling_seed",
    "env_name",
    "env_variant",
    "env_variant_canonical",
    "action_order_name",
    "canonical_ordering",
    "baseline_category",
    "ordering_search_time_sec",
    "ttpi_training_time_sec",
    "total_time_sec",
    "status",
    "oom",
    "error_type",
    "error_message",
    "diagnostics",
    "train_data",
)


DIAGNOSTIC_SUBDIRS = {
    "rank_profiles": ".rank_profile.csv",
    "cross_queries": ".cross_queries.csv",
    "cross_process": ".cross_process.csv",
    "round_events": ".round_events.csv",
    "run_json": ".standard.json",
}


def _is_close_total(result: dict) -> bool:
    total = float(result.get("total_time_sec", 0.0))
    expected = float(result.get("ordering_search_time_sec", 0.0)) + float(result.get("ttpi_training_time_sec", 0.0))
    return math.isclose(total, expected, rel_tol=1e-5, abs_tol=1e-3)


def validate_result(item: dict, *, require_ok: bool) -> list[str]:
    errors: list[str] = []
    path = Path(item["result"])
    if not path.exists():
        return [f"missing result JSON: {path}"]

    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - corrupt file path.
        return [f"failed to parse {path}: {exc}"]

    missing = [field for field in REQUIRED_RESULT_FIELDS if field not in result]
    if missing:
        errors.append(f"{path.name}: missing fields {missing}")

    status = str(result.get("status", "")).lower()
    if require_ok and status != "ok":
        errors.append(f"{path.name}: status={status!r}, expected 'ok'")

    if int(result.get("training_seed", -1)) != int(item["training_seed"]):
        errors.append(f"{path.name}: training_seed mismatch")
    if int(result.get("permutation_seed", -1)) != int(item["permutation_seed"]):
        errors.append(f"{path.name}: permutation_seed mismatch")
    if result.get("action_order_name") != item["ordering"]:
        errors.append(f"{path.name}: ordering mismatch")
    if not _is_close_total(result):
        errors.append(f"{path.name}: total_time_sec does not match timing decomposition")

    run_id = result.get("run_id", path.stem)
    diagnostics_root = ROOT / "repro" / "diagnostics"
    for subdir, suffix in DIAGNOSTIC_SUBDIRS.items():
        diag_path = diagnostics_root / subdir / f"{run_id}{suffix}"
        if not diag_path.exists():
            errors.append(f"{path.name}: missing diagnostic file {diag_path}")
        elif diag_path.stat().st_size <= 0:
            errors.append(f"{path.name}: empty diagnostic file {diag_path}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "repro" / "experiments" / "pam_baseline_smoke_manifest.json")
    parser.add_argument("--allow-error-status", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    commands = build_run_commands(manifest, smoke=True)
    if args.limit is not None:
        commands = commands[: args.limit]

    errors: list[str] = []
    seen_results: set[str] = set()
    for item in commands:
        if item["result"] in seen_results:
            errors.append(f"duplicate expected result path: {item['result']}")
        seen_results.add(item["result"])
        errors.extend(validate_result(item, require_ok=not args.allow_error_status))

    if errors:
        print(f"[FAIL] PAM smoke incomplete: {len(errors)} issue(s)")
        for error in errors[:80]:
            print(f" - {error}")
        if len(errors) > 80:
            print(f" - ... {len(errors) - 80} more")
        return 1

    print(f"[OK] PAM smoke complete: {len(commands)} expected runs validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
