"""Metric correctness: pass@1, final success, correction success, CIs, paired comparisons."""
from __future__ import annotations

import math

from codegenai.evaluation.metrics import compare, mcnemar_exact, paired_bootstrap_diff, summarise, wilson_interval


def rec(task_id, init, final, status_i="failed", status_f="failed", n_corr=0, gen=1.0, test=0.5, syntax=True):
    return {
        "task_id": task_id, "initial_hidden_passed": init, "final_hidden_passed": final,
        "initial_hidden_status": "passed" if init else status_i, "final_hidden_status": "passed" if final else status_f,
        "initial_syntax_ok": syntax, "n_corrections": n_corr, "generation_latency_s": gen, "testing_latency_s": test,
        "total_latency_s": gen + test, "peak_gpu_memory_mb": 3000.0, "initial_feedback_passed": init,
        "final_feedback_passed": final,
    }


def test_summarise_counts():
    records = [
        rec(1, True, True), rec(2, False, True, n_corr=1), rec(3, False, False, "runtime_error", "runtime_error", 2),
        rec(4, False, False, "syntax_error", "syntax_error", 2, syntax=False), rec(5, True, False, n_corr=0),
    ]
    m = summarise("D", records)
    assert m.n_problems == 5
    assert math.isclose(m.pass_at_1, 2 / 5) and math.isclose(m.final_success_rate, 2 / 5)
    assert math.isclose(m.correction_success_rate, 1 / 3)  # 3 failed initially, 1 fixed
    assert math.isclose(m.syntax_validity_rate, 4 / 5) and math.isclose(m.runtime_error_rate, 1 / 5)
    assert math.isclose(m.avg_correction_attempts, 1.0)
    assert m.n_problems_corrected == 1 and m.n_problems_broken_by_correction == 1
    assert m.status_breakdown_initial == {"failed": 1, "passed": 2, "runtime_error": 1, "syntax_error": 1}
    assert m.peak_gpu_memory_mb == 3000.0 and math.isclose(m.avg_total_latency_s, 1.5)
    lo, hi = m.pass_at_1_ci
    assert lo < 0.4 < hi


def test_summarise_no_failures_has_no_correction_rate():
    m = summarise("A", [rec(1, True, True), rec(2, True, True)])
    assert m.correction_success_rate is None and m.pass_at_1 == 1.0


def test_wilson_interval_properties():
    assert wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = wilson_interval(50, 100)
    assert 0.40 < lo < 0.5 < hi < 0.60
    lo, hi = wilson_interval(100, 100)
    assert hi == 1.0 and lo > 0.95
    lo2, hi2 = wilson_interval(500, 1000)
    assert (hi2 - lo2) < (hi - lo) or True  # larger n -> narrower (50/100 vs 500/1000)
    assert (wilson_interval(500, 1000)[1] - wilson_interval(500, 1000)[0]) < (wilson_interval(50, 100)[1] - wilson_interval(50, 100)[0])


def test_paired_bootstrap_and_mcnemar():
    a = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0] * 5
    b = [1, 1, 1, 0, 1, 1, 0, 0, 1, 1] * 5
    res = paired_bootstrap_diff(a, b, n_boot=500, seed=1)
    assert math.isclose(res["diff"], (sum(b) - sum(a)) / len(a))
    assert res["ci_low"] <= res["diff"] <= res["ci_high"] and res["ci_low"] > 0 and res["p_value_two_sided"] < 0.05
    same = paired_bootstrap_diff(a, a, n_boot=200)
    assert same["diff"] == 0 and same["p_value_two_sided"] >= 0.99
    mc = mcnemar_exact(a, b)
    assert mc["n01"] == 20 and mc["n10"] == 0 and mc["p_value"] < 0.001
    assert paired_bootstrap_diff([], [], 10)["diff"] is None


def test_compare_uses_common_problems_only():
    recs = {"A": [rec(i, i % 2 == 0, i % 2 == 0) for i in range(20)],
            "D": [rec(i, i % 2 == 0, True) for i in range(15)]}
    out = compare(recs, [("A", "D"), ("A", "B")], n_boot=100)
    assert len(out) == 1 and out[0]["n"] == 15 and out[0]["treatment"] == "D" and out[0]["diff"] > 0
