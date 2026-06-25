#!/usr/bin/env python3
"""Batch launcher for PAM HardMove ablations."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def add_arg(cmd: list[str], name: str, value) -> None:
    if isinstance(value, bool):
        if value:
            cmd.append(name)
    else:
        cmd.extend([name, str(value)])


def result_path(env_name: str, defaults: dict, ordering: str, seed: int) -> Path:
    task = (
        f"{env_name}"
        f"_state{defaults['n_state']}"
        f"_action{defaults['n_action']}"
        f"_iter{defaults['n_iter']}"
        f"_order{ordering}"
        f"_seed{seed}"
    )
    return ROOT / "repro" / "results" / f"{task}.json"


def build_run_commands(manifest: dict, *, smoke: bool = False) -> list[dict]:
    defaults = dict(manifest.get("defaults", {}))
    seeds = [0] if smoke else list(manifest.get("seeds", [0]))
    envs = list(manifest.get("environments", []))
    orderings = list(manifest.get("orderings", []))
    if smoke:
        envs = [envs[0]]
        orderings = orderings[: min(4, len(orderings))]
        defaults["n_state"] = min(int(defaults.get("n_state", 40)), 20)
        defaults["n_action"] = min(int(defaults.get("n_action", 50)), 20)
        defaults["n_iter"] = min(int(defaults.get("n_iter", 30)), 2)
        defaults["n_test"] = min(int(defaults.get("n_test", 100)), 16)
        defaults["callback_freq"] = 1
        defaults["device"] = defaults.get("device", "cuda")

    commands: list[dict] = []
    for env in envs:
        for seed in seeds:
            for ordering in orderings:
                cmd = [sys.executable, str(ROOT / "repro" / "scripts" / "run_hardmove.py")]
                add_arg(cmd, "--n-actuator", env["n_actuator"])
                add_arg(cmd, "--env-variant", env.get("env_variant", "standard"))
                add_arg(cmd, "--seed", seed)
                add_arg(cmd, "--action-order", ordering)
                for key, value in defaults.items():
                    add_arg(cmd, "--" + key.replace("_", "-"), value)
                commands.append(
                    {
                        "env": env.get("name", f"HM{env['n_actuator']}"),
                        "seed": seed,
                        "ordering": ordering,
                        "cmd": cmd,
                        "result": str(result_path(env.get("name", f"HM{env['n_actuator']}"), defaults, ordering, seed)),
                    }
                )
    return commands


def build_spectrum_commands(manifest: dict, *, smoke: bool = False) -> list[list[str]]:
    spec = manifest.get("singular_spectra", {})
    if not spec.get("enabled", True):
        return []
    envs = list(manifest.get("environments", []))
    if smoke:
        envs = [envs[0]]
    defaults = manifest.get("defaults", {})
    orderings = ",".join(spec.get("orderings", ["local", "block_pam"]))
    commands = []
    for env in envs:
        n_actuator = int(env["n_actuator"])
        k_states = spec.get("k_states_hm16", 8) if n_actuator >= 16 else spec.get("k_states_hm8_hm12", 12)
        if smoke:
            k_states = min(3, int(k_states))
        cmd = [
            sys.executable,
            str(ROOT / "repro" / "scripts" / "estimate_singular_spectra.py"),
            "--n-actuator",
            str(n_actuator),
            "--env-variant",
            env.get("env_variant", "standard"),
            "--n-action",
            str(defaults.get("n_action", 50)),
            "--action-grid-cap",
            str(spec.get("action_grid_cap", 12)),
            "--k-states",
            str(k_states),
            "--top-singular",
            str(spec.get("top_singular", 32)),
            "--orderings",
            orderings,
            "--seed",
            "0",
            "--device",
            "cpu",
        ]
        commands.append(cmd)
    return commands


def write_manifest_csv(commands: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["env", "seed", "ordering", "result", "cmd"])
        writer.writeheader()
        for item in commands:
            writer.writerow({**item, "cmd": " ".join(item["cmd"])})


def run_command(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "repro" / "experiments" / "pam_ablation_manifest.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--run-spectra", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip runs whose result JSON already exists.")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    commands = build_run_commands(manifest, smoke=args.smoke)
    if args.limit is not None:
        commands = commands[: args.limit]
    write_manifest_csv(commands, ROOT / "repro" / "diagnostics" / "pam_ablation_command_manifest.csv")
    print(f"[INFO] generated {len(commands)} training commands")

    if args.execute:
        for item in commands:
            if args.resume and Path(item["result"]).exists():
                print(f"[SKIP] exists {item['result']}")
                continue
            run_command(item["cmd"])
    else:
        for item in commands[:20]:
            print(" ".join(item["cmd"]))
        if len(commands) > 20:
            print(f"[INFO] ... {len(commands) - 20} more commands")

    if args.run_spectra:
        spectrum_commands = build_spectrum_commands(manifest, smoke=args.smoke)
        print(f"[INFO] generated {len(spectrum_commands)} spectrum commands")
        if args.execute:
            for cmd in spectrum_commands:
                run_command(cmd)
        else:
            for cmd in spectrum_commands:
                print(" ".join(cmd))

    if args.make_plots:
        summary_cmd = [sys.executable, str(ROOT / "repro" / "scripts" / "summarize_pam_diagnostics.py")]
        if (ROOT / "repro" / "diagnostics" / "pam_ablation_summary.csv").exists():
            run_command(summary_cmd)
        plot_cmd = [sys.executable, str(ROOT / "repro" / "scripts" / "plot_pam_diagnostics.py")]
        run_command(plot_cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
