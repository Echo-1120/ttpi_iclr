#!/usr/bin/env python3
"""Compatibility CLI for PAM singular-spectrum estimation.

Example:
    python repro/scripts/estimate_spectrum.py \
      --runs repro/results/HM8_state40_action50_iter30_orderlocal_seed0.json \
      --num-states 8 \
      --topk 32 \
      --thresholds 1e-1 5e-2 1e-2
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def expand_runs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = [Path(p) for p in glob.glob(pattern)]
        if matched:
            paths.extend(matched)
        else:
            paths.append(Path(pattern))
    return [p for p in paths if p.exists()]


def infer_run_metadata(run_paths: list[Path]) -> dict:
    if not run_paths:
        raise ValueError("No run JSON files found from --runs")
    first = json.loads(run_paths[0].read_text(encoding="utf-8"))
    n_actuator = first.get("n_actuator")
    n_action = first.get("n_action", 50)
    env_variant = first.get("env_variant", "standard")
    if n_actuator is None:
        match = re.search(r"HM(\d+)", run_paths[0].name)
        if not match:
            raise ValueError(f"Cannot infer n_actuator from {run_paths[0]}")
        n_actuator = int(match.group(1))
    ordering_set = {
        json.loads(path.read_text(encoding="utf-8")).get("action_order_name")
        for path in run_paths
        if path.suffix == ".json"
    }
    orderings = sorted(x for x in ordering_set if x)
    return {
        "n_actuator": int(n_actuator),
        "n_action": int(n_action),
        "env_variant": str(env_variant),
        "orderings": orderings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--num-states", type=int, default=8)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--thresholds", nargs="+", default=["1e-1", "5e-2", "1e-2"])
    parser.add_argument("--action-grid-cap", type=int, default=12)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "repro" / "diagnostics" / "spectra")
    args = parser.parse_args()

    run_paths = expand_runs(args.runs)
    meta = infer_run_metadata(run_paths)
    orderings = ",".join(meta["orderings"]) if meta["orderings"] else "local,block_pam,rankaware_proxy_pam,hybrid_pam"
    cmd = [
        sys.executable,
        str(ROOT / "repro" / "scripts" / "estimate_singular_spectra.py"),
        "--n-actuator",
        str(meta["n_actuator"]),
        "--env-variant",
        meta["env_variant"],
        "--n-action",
        str(meta["n_action"]),
        "--action-grid-cap",
        str(args.action_grid_cap),
        "--orderings",
        orderings,
        "--k-states",
        str(args.num_states),
        "--top-singular",
        str(args.topk),
        "--error-thresholds",
        ",".join(args.thresholds),
        "--device",
        args.device,
        "--out-dir",
        str(args.out_dir),
    ]
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
