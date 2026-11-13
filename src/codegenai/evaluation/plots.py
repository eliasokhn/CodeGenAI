"""Comparison charts for the MBPP evaluation (matplotlib, headless)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {"A": "A: Baseline", "B": "B: Fine-tuned", "C": "C: Base + corr.", "D": "D: FT + corr."}
COLOURS = {"A": "#7f8c8d", "B": "#2980b9", "C": "#f39c12", "D": "#27ae60"}


def _bar_with_ci(ax, metrics: dict[str, dict[str, Any]], key: str, ci_key: str, title: str) -> None:
    confs = [c for c in ("A", "B", "C", "D") if c in metrics]
    vals = [metrics[c][key] * 100 for c in confs]
    los = [(metrics[c][key] - metrics[c][ci_key][0]) * 100 for c in confs]
    his = [(metrics[c][ci_key][1] - metrics[c][key]) * 100 for c in confs]
    ax.bar([LABELS[c] for c in confs], vals, yerr=[los, his], capsize=4, color=[COLOURS[c] for c in confs])
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("%")
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=15)


def plot_success_rates(metrics: dict[str, dict[str, Any]], path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    _bar_with_ci(axes[0], metrics, "pass_at_1", "pass_at_1_ci", "pass@1 (initial generation, hidden tests)")
    _bar_with_ci(axes[1], metrics, "final_success_rate", "final_success_ci", "Final success (after correction)")
    fig.suptitle("MBPP: success rates with 95% Wilson CIs")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_status_breakdown(metrics: dict[str, dict[str, Any]], path: Path) -> Path:
    confs = [c for c in ("A", "B", "C", "D") if c in metrics]
    statuses = sorted({s for c in confs for s in metrics[c]["status_breakdown_final"]})
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bottom = [0.0] * len(confs)
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(statuses):
        vals = [metrics[c]["status_breakdown_final"].get(s, 0) / max(1, metrics[c]["n_problems"]) * 100 for c in confs]
        ax.bar([LABELS[c] for c in confs], vals, bottom=bottom, label=s, color=cmap(i % 10))
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("% of problems")
    ax.set_title("Final outcome categories on hidden tests")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_latency(metrics: dict[str, dict[str, Any]], path: Path) -> Path:
    confs = [c for c in ("A", "B", "C", "D") if c in metrics]
    gen = [metrics[c]["avg_generation_latency_s"] or 0 for c in confs]
    tst = [metrics[c]["avg_testing_latency_s"] or 0 for c in confs]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([LABELS[c] for c in confs], gen, label="generation", color="#3498db")
    ax.bar([LABELS[c] for c in confs], tst, bottom=gen, label="feedback testing", color="#e67e22")
    ax.set_ylabel("seconds / problem")
    ax.set_title("Average latency per problem")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_correction_flow(records_by_config: dict[str, list[dict[str, Any]]], path: Path) -> Path:
    confs = [c for c in ("C", "D") if c in records_by_config]
    if not confs:
        return path
    fig, ax = plt.subplots(figsize=(8, 4))
    cats = ["initial pass", "fixed by correction", "still failing", "broken by correction"]
    width = 0.35
    for i, c in enumerate(confs):
        recs = records_by_config[c]
        n = max(1, len(recs))
        counts = [
            sum(1 for r in recs if r["initial_hidden_passed"] and r["final_hidden_passed"]),
            sum(1 for r in recs if not r["initial_hidden_passed"] and r["final_hidden_passed"]),
            sum(1 for r in recs if not r["initial_hidden_passed"] and not r["final_hidden_passed"]),
            sum(1 for r in recs if r["initial_hidden_passed"] and not r["final_hidden_passed"]),
        ]
        xs = [j + (i - 0.5) * width for j in range(len(cats))]
        ax.bar(xs, [v / n * 100 for v in counts], width=width, label=LABELS[c], color=COLOURS[c])
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats)
    ax.set_ylabel("% of problems")
    ax.set_title("Effect of self-correction (hidden-test outcomes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_all_plots(records_by_config: dict[str, list[dict[str, Any]]], metrics: dict[str, dict[str, Any]],
                   out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "success_rates": plot_success_rates(metrics, out_dir / "success_rates.png"),
        "status_breakdown": plot_status_breakdown(metrics, out_dir / "status_breakdown.png"),
        "latency": plot_latency(metrics, out_dir / "latency.png"),
        "correction_flow": plot_correction_flow(records_by_config, out_dir / "correction_flow.png"),
    }
    return {k: str(v) for k, v in paths.items()}


def plot_training_curves(train_log_rows: list[dict[str, Any]], path: Path) -> Path:
    """Loss curves from ``train_log.jsonl`` (train loss + eval loss over steps)."""
    tr = [(r["step"], r["loss"]) for r in train_log_rows if "loss" in r]
    ev = [(r["step"], r["eval_loss"]) for r in train_log_rows if "eval_loss" in r]
    fig, ax = plt.subplots(figsize=(7, 4))
    if tr:
        ax.plot(*zip(*tr), label="train loss", color="#2980b9")
    if ev:
        ax.plot(*zip(*ev), label="validation loss", color="#c0392b", marker="o")
    ax.set_xlabel("optimisation step")
    ax.set_ylabel("loss")
    ax.set_title("LoRA fine-tuning loss")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
