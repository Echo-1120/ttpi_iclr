"""Plotting helpers for TTPI/PAM prestudy reports."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        return plt
    except ModuleNotFoundError:
        return None


def missing_artifact(path: Path, title: str, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    note = path.with_suffix(path.suffix + ".missing.txt")
    note.write_text(f"{title}\n\n{message}\n", encoding="utf-8")


def save_both(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def placeholder(path: Path, title: str, message: str) -> None:
    plt = _plt()
    if plt is None:
        missing_artifact(path, title, message + " matplotlib is not installed in this Python environment.")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    save_both(fig, path)
    plt.close(fig)


def boxplot_by_method(rows: list[dict[str, Any]], metric: str, path: Path, title: str, ylabel: str) -> None:
    plt = _plt()
    if plt is None:
        missing_artifact(path, title, f"Cannot draw {metric}; matplotlib is not installed.")
        return
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (int, float)):
            groups[str(row.get("ordering"))].append(float(value))
    groups = {key: vals for key, vals in groups.items() if vals}
    if not groups:
        placeholder(path, title, f"No numeric rows for {metric}.")
        return
    labels = list(groups)
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(labels)), 4.4))
    ax.boxplot([groups[label] for label in labels], labels=labels, showmeans=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    save_both(fig, path)
    plt.close(fig)


def bar_by_method(rows: list[dict[str, Any]], metric: str, path: Path, title: str, ylabel: str) -> None:
    plt = _plt()
    if plt is None:
        missing_artifact(path, title, f"Cannot draw {metric}; matplotlib is not installed.")
        return
    values = [(str(row.get("ordering")), row.get(metric)) for row in rows if isinstance(row.get(metric), (int, float))]
    if not values:
        placeholder(path, title, f"No numeric rows for {metric}.")
        return
    labels, ys = zip(*values)
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(labels)), 4.4))
    ax.bar(labels, ys)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    save_both(fig, path)
    plt.close(fig)


def scatter(rows: list[dict[str, Any]], x: str, y: str, path: Path, title: str) -> None:
    plt = _plt()
    if plt is None:
        missing_artifact(path, title, f"Cannot draw {x} vs {y}; matplotlib is not installed.")
        return
    points = [
        (float(row[x]), float(row[y]), str(row.get("ordering")))
        for row in rows
        if isinstance(row.get(x), (int, float)) and isinstance(row.get(y), (int, float))
    ]
    if not points:
        placeholder(path, title, f"No numeric rows for {x} vs {y}.")
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    labels = sorted({label for _, _, label in points})
    for label in labels:
        xs = [px for px, py, plabel in points if plabel == label]
        ys = [py for px, py, plabel in points if plabel == label]
        ax.scatter(xs, ys, label=label, alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(alpha=0.25)
    if len(labels) <= 10:
        ax.legend(fontsize=8)
    save_both(fig, path)
    plt.close(fig)


def rank_trajectory(records: list[dict[str, Any]], path: Path, title: str, typical_seed: int | None = None) -> None:
    plt = _plt()
    if plt is None:
        missing_artifact(path, title, "Cannot draw rank trajectory; matplotlib is not installed.")
        return
    series: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        ordering = str(record.get("ordering"))
        for row in record.get("rank_rows", []):
            iteration = int(float(row.get("iteration") or 0))
            value = row.get("adv_rank_max")
            if value not in (None, ""):
                series[ordering][iteration].append(float(value))
    if not series:
        placeholder(path, title, "No rank profile rows found.")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ordering, by_iter in sorted(series.items()):
        xs = sorted(by_iter)
        ys = [median(by_iter[x]) for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=ordering)
    suffix = f" (typical seed {typical_seed})" if typical_seed is not None else ""
    ax.set_title(title + suffix)
    ax.set_xlabel("PI iteration")
    ax.set_ylabel("median advantage max rank")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save_both(fig, path)
    plt.close(fig)


def cutwise_rank_heatmap(records: list[dict[str, Any]], path: Path, title: str) -> None:
    plt = _plt()
    if plt is None:
        missing_artifact(path, title, "Cannot draw cutwise rank heatmap; matplotlib is not installed.")
        return
    rows = []
    labels = []
    for record in records:
        maxima: dict[int, float] = {}
        for row in record.get("rank_rows", []):
            for key, value in row.items():
                if key.startswith("adv_rank_") and key[len("adv_rank_") :].isdigit():
                    idx = int(key[len("adv_rank_") :])
                    maxima[idx] = max(maxima.get(idx, 0.0), float(value))
        if maxima:
            labels.append(f"{record.get('ordering')}:s{record.get('seed')}")
            rows.append([maxima.get(i, 0.0) for i in range(1, max(maxima) + 1)])
    if not rows:
        placeholder(path, title, "No per-cut rank columns found.")
        return
    width = max(len(row) for row in rows)
    matrix = [row + [0.0] * (width - len(row)) for row in rows[:40]]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.22 * len(matrix))))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("cut")
    ax.set_ylabel("run")
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(labels[:40], fontsize=6)
    fig.colorbar(im, ax=ax, label="max adv rank")
    save_both(fig, path)
    plt.close(fig)
