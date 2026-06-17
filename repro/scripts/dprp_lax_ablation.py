"""DPRP and LaX ablations for HardMove TTPI.

This script is intentionally separate from the original TTPI implementation.
It wraps training with:
  - DPRP: validation-aware dynamic TT rounding after policy updates.
  - LaX: evaluation-time local action expansion around the TT policy action.

Typical smoke run:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/s110/miniconda3/envs/tt_5080/bin/python repro/scripts/dprp_lax_ablation.py --preset smoke

Paper-budget single-seed run:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/s110/miniconda3/envs/tt_5080/bin/python repro/scripts/dprp_lax_ablation.py --preset paper
"""
import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dynamic_systems import HardMove
from ttpi import TTPI


torch.set_default_dtype(torch.float64)

RESULTS = ROOT / "repro" / "results"
FIGURES = ROOT / "repro" / "figures" / "pam"


def rank_stats(tt_model):
    ranks = [int(x) for x in tt_model.ranks_tt.detach().cpu().tolist()]
    inner = ranks[1:-1] or ranks
    return {
        "ranks": ranks,
        "avg": float(statistics.mean(inner)),
        "max": int(max(ranks)),
    }


def sample_domain_points(domain, n_points, seed, device):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    cols = []
    for values in domain:
        idx = torch.randint(len(values), (n_points,), generator=g, dtype=torch.long)
        cols.append(values.detach().cpu()[idx])
    return torch.stack(cols, dim=1).to(device)


@torch.no_grad()
def dprp_prune_model(
    ttpi,
    model,
    points,
    label,
    target_avg_rank=8.0,
    tol=0.05,
    eps_values=None,
):
    if eps_values is None:
        eps_values = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

    before = rank_stats(model)
    y_ref = ttpi.get_value_a(model, points).detach()
    ref_norm = torch.linalg.norm(y_ref).clamp_min(1e-12)

    accepted = model
    accepted_eps = 0.0
    accepted_rel = 0.0
    accepted_stats = before

    for eps in eps_values:
        cand = model.clone().to(ttpi.device)
        cand.round_tt(eps=eps)
        cand = cand.to(ttpi.device)
        y_cand = ttpi.get_value_a(cand, points).detach()
        rel = float((torch.linalg.norm(y_cand - y_ref) / ref_norm).detach().cpu())
        cand_stats = rank_stats(cand)
        if rel <= tol:
            accepted = cand
            accepted_eps = float(eps)
            accepted_rel = rel
            accepted_stats = cand_stats
            if cand_stats["avg"] <= target_avg_rank:
                break
        else:
            del cand
            break

    return accepted, {
        "label": label,
        "eps": accepted_eps,
        "rel_error": accepted_rel,
        "before_avg": before["avg"],
        "before_max": before["max"],
        "after_avg": accepted_stats["avg"],
        "after_max": accepted_stats["max"],
    }


def build_order(n_act, order, seed=0):
    if order == "Local":
        return list(range(2 * n_act))
    if order == "BadSplit":
        return list(range(0, 2 * n_act, 2)) + list(range(1, 2 * n_act, 2))
    if order == "Random":
        g = torch.Generator(device="cpu")
        g.manual_seed(42 + seed)
        return [int(x) for x in torch.randperm(2 * n_act, generator=g).tolist()]
    raise ValueError(f"Unknown order: {order}")


def make_domains(n_act, ns, na, action_order, device):
    l_bound = 1.0
    vmax = 0.25
    domain_acc = torch.linspace(-vmax, vmax, na, device=device)
    domain_sw = torch.tensor([0.0, 1.0], device=device)
    smin = torch.tensor([-l_bound, -l_bound, 0.0, 0.0], device=device)
    smax = torch.tensor([l_bound, l_bound, vmax, vmax], device=device)
    state_domain = [torch.linspace(smin[i], smax[i], ns, device=device) for i in range(4)]
    action_domain_phys = [domain_acc, domain_sw] * n_act
    action_domain_tt = [action_domain_phys[i] for i in action_order]
    inverse_order = torch.tensor([action_order.index(i) for i in range(2 * n_act)], device=device)
    return state_domain, action_domain_tt, inverse_order, smin, smax


def make_initial_states(seed, n_test, smin, smax, device):
    g = torch.Generator(device="cpu")
    g.manual_seed(12345 + seed)
    init_state = torch.empty((n_test, 4), dtype=torch.float64)
    for i in range(4):
        r = torch.rand(n_test, generator=g, dtype=torch.float64).clip(0.25, 0.75)
        init_state[:, i] = smin[i].cpu() + r * (smax[i].cpu() - smin[i].cpu())
    init_state[:, 2:4] = 0.0
    return init_state.to(device)


@torch.no_grad()
def lax_policy(ttpi, state, radius_scale=0.5):
    base_action = ttpi.policy(state)
    candidates = [base_action]
    for dim, domain in enumerate(ttpi.domain_action):
        if len(domain) <= 2:
            for value in domain:
                cand = base_action.clone()
                cand[:, dim] = value
                candidates.append(cand)
            continue
        step = float((domain[1] - domain[0]).abs().detach().cpu()) * radius_scale
        for sign in (-1.0, 1.0):
            cand = base_action.clone()
            cand[:, dim] = torch.clamp(cand[:, dim] + sign * step, domain[0], domain[-1])
            candidates.append(cand)

    action_batch = torch.stack(candidates, dim=1)
    batch_size, n_candidates, dim_action = action_batch.shape
    state_batch = state[:, None, :].expand(-1, n_candidates, -1)
    state_action = torch.cat((state_batch, action_batch), dim=-1).reshape(
        batch_size * n_candidates, state.shape[-1] + dim_action
    )
    flat_state = state_batch.reshape(batch_size * n_candidates, state.shape[-1])
    flat_action = action_batch.reshape(batch_size * n_candidates, dim_action)
    next_state = ttpi.forward_model(flat_state, flat_action)
    scores = (
        ttpi.reward_normalized(flat_state, flat_action) * ttpi.dt
        + ttpi.gamma * ttpi.get_value_v(ttpi.v_model, next_state)
    ).view(batch_size, n_candidates)
    best = torch.argmax(scores, dim=1)
    return action_batch[torch.arange(batch_size, device=state.device), best]


@torch.no_grad()
def evaluate_policy(ttpi, init_state, horizon, use_lax=False):
    state = init_state.clone()
    n_test = state.shape[0]
    active = torch.ones(n_test, dtype=torch.bool, device=state.device)
    success = torch.zeros(n_test, dtype=torch.bool, device=state.device)
    start_pos = state[:, :2].clone()
    prev_pos = state[:, :2].clone()
    path_len = torch.zeros(n_test, device=state.device)
    mem_before = torch.cuda.memory_allocated() if state.device.type == "cuda" else 0
    mem_extra = 0

    for _ in range(horizon):
        if use_lax:
            action = lax_policy(ttpi, state)
        else:
            action = ttpi.policy(state)
        if state.device.type == "cuda":
            mem_extra = max(mem_extra, torch.cuda.memory_allocated() - mem_before)
        next_state = ttpi.forward_model(state, action)
        next_pos = next_state[:, :2]
        path_len += active.float() * torch.linalg.norm(next_pos - prev_pos, dim=-1)
        reached = active & (torch.linalg.norm(next_pos, dim=-1) <= 0.02)
        success |= reached
        active &= ~reached
        state = next_state
        prev_pos = next_pos

    straight = torch.linalg.norm(start_pos, dim=-1)
    mu_all = (straight / (path_len + 1e-12)).clamp(max=1.0) ** 2
    mu_success = mu_all[success].mean() if success.any() else torch.tensor(0.0, device=state.device)
    success_rate = float(success.float().mean().detach().cpu())
    mu = float(mu_success.detach().cpu())
    return {
        "S": success_rate,
        "mu": mu,
        "S_mu": success_rate * mu,
        "lax_extra_gb": float(mem_extra / 1e9),
    }


def install_dprp(ttpi, dprp_points, events, target_avg_rank, tol):
    orig_pi_update = ttpi.PI_update
    orig_normalize_tt_a = ttpi.normalize_tt_a
    state = {"iter": -1}

    def wrapped_pi_update(n_iter_v=1):
        orig_pi_update(n_iter_v=n_iter_v)
        pruned, rec = dprp_prune_model(
            ttpi,
            ttpi.a_model,
            dprp_points,
            label="A",
            target_avg_rank=target_avg_rank,
            tol=tol,
        )
        ttpi.a_model = pruned.to(ttpi.device)
        rec["iter"] = state["iter"]
        events.append(rec)
        torch.cuda.empty_cache()
        gc.collect()

    def wrapped_normalize_tt_a(model):
        normalized = orig_normalize_tt_a(model).to(ttpi.device)
        pruned, rec = dprp_prune_model(
            ttpi,
            normalized,
            dprp_points,
            label="P",
            target_avg_rank=target_avg_rank,
            tol=tol,
        )
        rec["iter"] = state["iter"]
        events.append(rec)
        return pruned.to(ttpi.device)

    def bump_iter():
        state["iter"] += 1

    ttpi.PI_update = wrapped_pi_update
    ttpi.normalize_tt_a = wrapped_normalize_tt_a
    return bump_iter


def run_case(args, case):
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.manual_seed(case["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(case["seed"])

    action_order = build_order(case["n_act"], case["order"], seed=case["seed"])
    state_domain, action_domain, inverse_order, smin, smax = make_domains(
        case["n_act"], case["ns"], case["na"], action_order, device
    )
    dyn = HardMove(dt=0.01, w_goal=1e3, w_action=1e4, n=case["n_act"], device=device)

    ttpi = TTPI(
        domain_state=state_domain,
        domain_action=action_domain,
        reward=lambda s, a: dyn.reward_state_action(s, a[..., inverse_order.to(a.device)]),
        normalize_reward=True,
        forward_model=lambda s, a: dyn.forward_simulate(s, a[..., inverse_order.to(a.device)]),
        gamma=0.99,
        rmax_v=case["rmax"],
        rmax_a=case["rmax"],
        nswp_v=case["nswp_v"],
        nswp_a=case["nswp_a"],
        kickrank_v=case["kickrank"],
        kickrank_a=case["kickrank"],
        max_batch_v=case["max_batch_v"],
        max_batch_a=case["max_batch_a"],
        eps_cross_v=1e-3,
        eps_cross_a=1e-3,
        eps_round_v=1e-3,
        eps_round_a=1e-3,
        n_samples=case["n_samples"],
        verbose=False,
        device=device,
    )
    ttpi.plt_training_stat = lambda: None
    ttpi.save_model = lambda *a, **kw: None

    init_state = make_initial_states(case["seed"], case["n_test"], smin, smax, device)
    dprp_events = []
    bump_iter = None
    if case["dprp"]:
        dprp_points = sample_domain_points(
            ttpi.domain_state_action, case["dprp_val_points"], seed=9000 + case["seed"], device=device
        )
        bump_iter = install_dprp(
            ttpi,
            dprp_points=dprp_points,
            events=dprp_events,
            target_avg_rank=case["dprp_target_avg_rank"],
            tol=case["dprp_tol"],
        )

    eval_history = []
    rank_history = []
    best_no_lax = {"S_mu": -1.0}
    best_lax = {"S_mu": -1.0}
    last_cb = -1

    def callback(model, callback_count=0):
        nonlocal last_cb, best_no_lax, best_lax
        if bump_iter is not None:
            bump_iter()
        last_cb = callback_count
        no_lax = evaluate_policy(model, init_state, case["horizon"], use_lax=False)
        lax = evaluate_policy(model, init_state, case["horizon"], use_lax=True)
        no_lax["cb"] = int(callback_count)
        lax["cb"] = int(callback_count)
        if no_lax["S_mu"] > best_no_lax["S_mu"]:
            best_no_lax = dict(no_lax)
        if lax["S_mu"] > best_lax["S_mu"]:
            best_lax = dict(lax)

        ar = rank_stats(model.a_model)
        pr = rank_stats(model.policy_model)
        vr = rank_stats(model.v_model)
        rank_row = {
            "cb": int(callback_count),
            "Vr_avg": vr["avg"],
            "Vr_max": vr["max"],
            "Ar_avg": ar["avg"],
            "Ar_max": ar["max"],
            "Pr_avg": pr["avg"],
            "Pr_max": pr["max"],
        }
        eval_history.append({"no_lax": no_lax, "lax": lax})
        rank_history.append(rank_row)
        peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
        print(
            f"  cb{callback_count:>3}: noLaX={no_lax['S_mu']:.4f} "
            f"LaX={lax['S_mu']:.4f} ArAvg={ar['avg']:.1f} PrAvg={pr['avg']:.1f} "
            f"peak={peak:.2f}G",
            flush=True,
        )
        torch.cuda.empty_cache()
        gc.collect()
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    t0 = time.time()
    oom = False
    try:
        ttpi.train(
            resume=False,
            n_iter_max=case["n_iter"],
            n_iter_v=1,
            callback=callback,
            callback_freq=case["callback_freq"],
            verbose=False,
            file_name=None,
        )
    except torch.OutOfMemoryError:
        oom = True
        print("OOM", flush=True)
    elapsed = time.time() - t0

    ar = rank_stats(ttpi.a_model)
    pr = rank_stats(ttpi.policy_model)
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    status = "OOM" if oom else ("OK" if last_cb >= case["n_iter"] - 1 else f"cb{last_cb}")
    return {
        "case": case,
        "status": status,
        "time_s": float(elapsed),
        "peak_gb": float(peak),
        "best_no_lax": best_no_lax,
        "best_lax": best_lax,
        "final_Ar_avg": ar["avg"],
        "final_Ar_max": ar["max"],
        "final_Pr_avg": pr["avg"],
        "final_Pr_max": pr["max"],
        "eval_history": eval_history,
        "rank_history": rank_history,
        "dprp_events": dprp_events,
    }


def make_cases(args):
    if args.preset == "smoke":
        base = {
            "n_act": 8,
            "seed": args.seed,
            "ns": 18,
            "na": 16,
            "n_iter": 2,
            "callback_freq": 1,
            "n_test": 8,
            "horizon": 120,
            "rmax": 30,
            "nswp_v": 2,
            "nswp_a": 3,
            "kickrank": 4,
            "max_batch_v": 1500,
            "max_batch_a": 3000,
            "n_samples": 20,
            "dprp_val_points": 96,
            "dprp_target_avg_rank": 8.0,
            "dprp_tol": 0.08,
        }
    else:
        base = {
            "n_act": 8,
            "seed": args.seed,
            "ns": 30,
            "na": 25,
            "n_iter": 30,
            "callback_freq": 10,
            "n_test": 50,
            "horizon": 1000,
            "rmax": 60,
            "nswp_v": 5,
            "nswp_a": 10,
            "kickrank": 10,
            "max_batch_v": 5000,
            "max_batch_a": 20000,
            "n_samples": 50,
            "dprp_val_points": 512,
            "dprp_target_avg_rank": 8.0,
            "dprp_tol": 0.05,
        }

    requested = set(args.cases.split(","))
    cases = []
    if "local_lax" in requested:
        c = dict(base)
        c.update({"name": "local_lax", "order": "Local", "dprp": False})
        cases.append(c)
    if "local_dprp" in requested:
        c = dict(base)
        c.update({"name": "local_dprp", "order": "Local", "dprp": True})
        cases.append(c)
    if "random_dprp" in requested:
        c = dict(base)
        c.update({"name": "random_dprp", "order": "Random", "dprp": True})
        cases.append(c)
    return cases


def write_outputs(results, out_path):
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": (
            "LaX is an evaluation-time local action expansion. DPRP is validation-aware "
            "dynamic TT rounding; paper claims require paper-budget runs, not smoke runs."
        ),
        "results": results,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)

    summary_path = out_path.with_suffix(".summary.csv")
    with summary_path.open("w") as f:
        f.write(
            "name,order,dprp,status,no_lax_S_mu,lax_S_mu,lax_delta,"
            "best_paired_delta,best_paired_cb,peak_gb,lax_extra_gb,"
            "Ar_avg,Pr_avg,Ar_max,Pr_max,time_s\n"
        )
        for r in results:
            case = r["case"]
            no_lax = r["best_no_lax"]
            lax = r["best_lax"]
            paired = [
                {
                    "cb": item["lax"].get("cb", item["no_lax"].get("cb", -1)),
                    "delta": item["lax"].get("S_mu", 0.0) - item["no_lax"].get("S_mu", 0.0),
                }
                for item in r.get("eval_history", [])
            ]
            best_paired = max(paired, key=lambda item: item["delta"]) if paired else {"cb": -1, "delta": 0.0}
            f.write(
                f"{case['name']},{case['order']},{case['dprp']},{r['status']},"
                f"{no_lax.get('S_mu', 0):.6f},{lax.get('S_mu', 0):.6f},"
                f"{lax.get('S_mu', 0) - no_lax.get('S_mu', 0):.6f},"
                f"{best_paired['delta']:.6f},{best_paired['cb']},"
                f"{r['peak_gb']:.6f},{lax.get('lax_extra_gb', 0):.6f},"
                f"{r['final_Ar_avg']:.6f},{r['final_Pr_avg']:.6f},"
                f"{r['final_Ar_max']},{r['final_Pr_max']},{r['time_s']:.3f}\n"
            )

    plot_lax(results, out_path)
    plot_dprp(results, out_path)
    print(f"Saved results: {out_path}")
    print(f"Saved summary: {summary_path}")


def plot_lax(results, out_path):
    rows = [r for r in results if r["case"]["name"] == "local_lax"]
    if not rows:
        return
    r = rows[0]
    labels = ["No LaX", "LaX"]
    values = [r["best_no_lax"].get("S_mu", 0.0), r["best_lax"].get("S_mu", 0.0)]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.bar(labels, values, color=["#64748b", "#2563eb"], width=0.55)
    ax.set_ylabel(r"Best $S \times \mu$")
    ax.set_title("LaX local refinement ablation")
    ax.set_ylim(0, max(1.0, max(values) + 0.1))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{out_path.stem}_lax_ablation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_dprp(results, out_path):
    rows = [r for r in results if r["case"]["dprp"]]
    if not rows:
        return
    labels = [r["case"]["order"] for r in rows]
    ar = [r["final_Ar_avg"] for r in rows]
    pr = [r["final_Pr_avg"] for r in rows]
    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(x, ar, marker="o", linewidth=2, label="A avg rank", color="#2563eb")
    ax.plot(x, pr, marker="s", linewidth=2, label="P avg rank", color="#dc2626")
    ax.axhspan(5, 8, color="#22c55e", alpha=0.12, label="target 5-8")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average TT rank")
    ax.set_title("DPRP Local vs Random")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / f"{out_path.stem}_dprp_rank.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["smoke", "paper"], default="smoke")
    parser.add_argument(
        "--cases",
        default="local_lax,local_dprp,random_dprp",
        help="Comma-separated subset: local_lax,local_dprp,random_dprp",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    out_name = args.output or f"dprp_lax_{args.preset}_seed{args.seed}.json"
    out_path = RESULTS / out_name
    results = []
    for case in make_cases(args):
        print(f"\n=== {case['name']} order={case['order']} dprp={case['dprp']} ===", flush=True)
        results.append(run_case(args, case))
    write_outputs(results, out_path)


if __name__ == "__main__":
    main()
