"""Evaluation metrics with confidence intervals.

All rates are computed from per-problem records (never hand-entered). Confidence intervals:
Wilson score intervals for proportions, paired bootstrap for differences between
configurations evaluated on the same problems.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, round(centre - half, 12)), min(1.0, round(centre + half, 12)))


def mean(xs: Iterable[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def paired_bootstrap_diff(a: Sequence[int], b: Sequence[int], n_boot: int = 2000, seed: int = 42,
                          alpha: float = 0.05) -> dict[str, Any]:
    """Bootstrap CI for mean(b) - mean(a) over paired per-problem binary outcomes."""
    assert len(a) == len(b), "paired bootstrap needs equal-length outcome vectors"
    n = len(a)
    if n == 0:
        return {"diff": None, "ci_low": None, "ci_high": None, "p_value_two_sided": None, "n": 0}
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] - a[i] for i in idx) / n)
    diffs.sort()
    lo = diffs[int(alpha / 2 * n_boot)]
    hi = diffs[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    observed = (sum(b) - sum(a)) / n
    # Two-sided bootstrap p-value for H0: diff == 0 (fraction of resamples on the other side of 0).
    frac_le0 = sum(1 for d in diffs if d <= 0) / n_boot
    frac_ge0 = sum(1 for d in diffs if d >= 0) / n_boot
    p = 2 * min(frac_le0, frac_ge0)
    return {"diff": observed, "ci_low": lo, "ci_high": hi, "p_value_two_sided": min(1.0, p), "n": n}


def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> dict[str, Any]:
    """Exact McNemar test on discordant pairs (b passes & a fails vs. a passes & b fails)."""
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    k, n = min(n01, n10), n01 + n10
    if n == 0:
        return {"n01": n01, "n10": n10, "p_value": 1.0}
    cdf = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return {"n01": n01, "n10": n10, "p_value": min(1.0, 2 * cdf)}


@dataclass
class ConfigMetrics:
    configuration: str
    n_problems: int
    pass_at_1: float  # hidden tests passed by the INITIAL generation
    pass_at_1_ci: tuple[float, float]
    final_success_rate: float  # hidden tests passed by the FINAL code (after correction, if any)
    final_success_ci: tuple[float, float]
    correction_success_rate: Optional[float]  # among problems whose initial code failed hidden tests
    correction_success_ci: Optional[tuple[float, float]]
    feedback_pass_rate_initial: Optional[float]  # visible tests passed on first attempt
    feedback_pass_rate_final: Optional[float]
    syntax_validity_rate: float  # initial generation parses
    runtime_error_rate: float  # initial generation hit a runtime error / timeout on hidden tests
    avg_correction_attempts: float
    avg_generation_latency_s: Optional[float]
    avg_testing_latency_s: Optional[float]
    avg_total_latency_s: Optional[float]
    peak_gpu_memory_mb: Optional[float]
    n_problems_corrected: int
    n_problems_broken_by_correction: int  # initial hidden pass -> final hidden fail (should be ~0)
    status_breakdown_initial: dict[str, int]
    status_breakdown_final: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        return d


def summarise(configuration: str, records: list[dict[str, Any]]) -> ConfigMetrics:
    """Aggregate per-problem result records (see ``mbpp_eval.ProblemResult``) for one configuration."""
    n = len(records)
    init_pass = [int(r["initial_hidden_passed"]) for r in records]
    final_pass = [int(r["final_hidden_passed"]) for r in records]
    syntax_ok = [int(r["initial_syntax_ok"]) for r in records]
    runtime_err = [int(r["initial_hidden_status"] in ("runtime_error", "timeout")) for r in records]
    failed_init = [r for r in records if not r["initial_hidden_passed"]]
    corrected = [r for r in failed_init if r["final_hidden_passed"]]
    broken = sum(1 for r in records if r["initial_hidden_passed"] and not r["final_hidden_passed"])
    fb_init = [r["initial_feedback_passed"] for r in records if r.get("initial_feedback_passed") is not None]
    fb_final = [r["final_feedback_passed"] for r in records if r.get("final_feedback_passed") is not None]
    gpu = [r.get("peak_gpu_memory_mb") for r in records if r.get("peak_gpu_memory_mb")]
    return ConfigMetrics(
        configuration=configuration, n_problems=n,
        pass_at_1=sum(init_pass) / n if n else 0.0, pass_at_1_ci=wilson_interval(sum(init_pass), n),
        final_success_rate=sum(final_pass) / n if n else 0.0, final_success_ci=wilson_interval(sum(final_pass), n),
        correction_success_rate=(len(corrected) / len(failed_init)) if failed_init else None,
        correction_success_ci=wilson_interval(len(corrected), len(failed_init)) if failed_init else None,
        feedback_pass_rate_initial=mean([int(x) for x in fb_init]) if fb_init else None,
        feedback_pass_rate_final=mean([int(x) for x in fb_final]) if fb_final else None,
        syntax_validity_rate=sum(syntax_ok) / n if n else 0.0,
        runtime_error_rate=sum(runtime_err) / n if n else 0.0,
        avg_correction_attempts=mean([r["n_corrections"] for r in records]) or 0.0,
        avg_generation_latency_s=mean([r["generation_latency_s"] for r in records]),
        avg_testing_latency_s=mean([r["testing_latency_s"] for r in records]),
        avg_total_latency_s=mean([r["total_latency_s"] for r in records]),
        peak_gpu_memory_mb=max(gpu) if gpu else None,
        n_problems_corrected=len(corrected), n_problems_broken_by_correction=broken,
        status_breakdown_initial=_count(r["initial_hidden_status"] for r in records),
        status_breakdown_final=_count(r["final_hidden_status"] for r in records),
    )


def _count(xs: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items()))


def compare(records_by_config: dict[str, list[dict[str, Any]]], pairs: list[tuple[str, str]],
            n_boot: int = 2000) -> list[dict[str, Any]]:
    """Paired comparisons of final success between configurations on identical problem sets."""
    out = []
    for a, b in pairs:
        ra, rb = records_by_config.get(a), records_by_config.get(b)
        if not ra or not rb:
            continue
        ia = {r["task_id"]: r for r in ra}
        ib = {r["task_id"]: r for r in rb}
        common = sorted(set(ia) & set(ib))
        va = [int(ia[t]["final_hidden_passed"]) for t in common]
        vb = [int(ib[t]["final_hidden_passed"]) for t in common]
        boot = paired_bootstrap_diff(va, vb, n_boot=n_boot)
        out.append({"baseline": a, "treatment": b, "metric": "final_success_rate", **boot,
                    "mcnemar": mcnemar_exact(va, vb)})
    return out
