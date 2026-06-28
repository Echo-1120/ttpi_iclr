#!/usr/bin/env python3
"""Run small-scale HardMove ordering oracle diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from repro.oracle_ordering import ORACLE_MODES, write_oracle_outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-actuator", type=int, required=True)
    parser.add_argument("--mode", choices=ORACLE_MODES, default="block_with_flips")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "repro" / "diagnostics" / "oracle")
    args = parser.parse_args()

    csv_path, json_path = write_oracle_outputs(
        n_actuator=args.n_actuator,
        mode=args.mode,
        random_seed=args.random_seed,
        out_dir=args.out_dir,
    )
    print(f"[DONE] wrote {csv_path}")
    print(f"[DONE] wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
