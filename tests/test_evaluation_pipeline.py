"""Evaluator wiring with a mock model and a fake sandbox: hidden tests only decide the verdict,
feedback uses visible tests only, per-problem records carry every metric field."""
from __future__ import annotations

import json

from codegenai.evaluation.mbpp_eval import CONFIGURATIONS, MBPPEvaluator, write_aggregates
from codegenai.evaluation.metrics import summarise
from codegenai.model.generation import MockCodeGenerator
from codegenai.sandbox import Status
from codegenai.sandbox.docker_runner import Availability
from codegenai.sandbox.report import TestCaseResult, TestReport


class RecordingSandbox:
    """Evaluates MBPP asserts for real... with Python's eval on *trusted test fixtures only*."""

    def __init__(self):
        self.runs = []

    def check_available(self, refresh=False):
        return Availability(True, "ok", "fake", True)

    def run(self, files, entry_point=None, hidden_test_names=None, timeout_override=None):
        self.runs.append(sorted(files))
        hidden_names = hidden_test_names or set()
        correct = "s.replace(ch, '', 1)" in files["solution.py"]
        tests = []
        for fname, src in files.items():
            if not fname.startswith("test_"):
                continue
            prefix = fname[len("test_"):-3]
            n = src.count(f"def test_{prefix}_")
            for i in range(n):
                name = f"test_{prefix}_{i}"
                tests.append(TestCaseResult(name=name, outcome="passed" if correct else "failed",
                                            error_type=None if correct else "AssertionError",
                                            message="" if correct else f"assert failed in {name}",
                                            hidden=name in hidden_names))
        status = Status.PASSED if correct else Status.FAILED
        return TestReport(status=status, tests=tests, total=len(tests), passed=sum(t.outcome == "passed" for t in tests),
                          failed=sum(t.outcome == "failed" for t in tests), verified=correct, summary=status.value)


WRONG = "```python\ndef remove_Occ(s, ch):\n    return s\n```"
RIGHT = "```python\ndef remove_Occ(s, ch):\n    s = s.replace(ch, '', 1)\n    i = s.rfind(ch)\n    return s[:i] + s[i+1:] if i >= 0 else s\n```"


def test_evaluate_problem_configurations(cfg, mbpp_problem, tmp_path):
    sb = RecordingSandbox()
    ev = MBPPEvaluator(cfg, sandbox=sb)
    # Configuration A (no correction): wrong code -> fails hidden tests, no correction attempted
    gen_a = MockCodeGenerator(lambda m: WRONG)
    ra = ev.evaluate_problem(mbpp_problem, gen_a, "A")
    assert not ra.initial_hidden_passed and not ra.final_hidden_passed and ra.n_corrections == 0
    assert ra.initial_feedback_passed is False and ra.hidden_tests_total == 3
    assert len(gen_a.calls) == 1
    # Visible runner used test_visible.py only; hidden verdict used test_hidden.py only
    assert ["solution.py", "test_visible.py"] in sb.runs and ["solution.py", "test_hidden.py"] in sb.runs
    assert not any("test_visible.py" in r and "test_hidden.py" in r for r in sb.runs)

    # Configuration C: wrong then right -> corrected, final passes hidden tests
    answers = iter([WRONG, RIGHT])
    gen_c = MockCodeGenerator(lambda m: next(answers))
    rc = ev.evaluate_problem(mbpp_problem, gen_c, "C")
    assert not rc.initial_hidden_passed and rc.final_hidden_passed and rc.n_corrections == 1 and rc.corrected_on_feedback
    # the correction prompt contained only the visible test, never the other hidden asserts
    corr_prompt = gen_c.calls[1][1]["content"]
    assert mbpp_problem.visible_tests[0] in corr_prompt
    for h in mbpp_problem.hidden_tests[1:]:
        assert h.replace("assert ", "") not in corr_prompt
    assert "test_hidden" not in corr_prompt

    recs = {"A": [ra.to_dict()], "C": [rc.to_dict()]}
    m = summarise("C", recs["C"])
    assert m.pass_at_1 == 0.0 and m.final_success_rate == 1.0 and m.correction_success_rate == 1.0
    agg = write_aggregates(cfg, tmp_path, recs, {"note": "test"})
    assert set(agg["metrics"]) == {"A", "C"} and (tmp_path / "aggregate_metrics.json").exists()
    assert any(c["baseline"] == "A" and c["treatment"] == "C" for c in agg["comparisons"])
    saved = json.loads((tmp_path / "aggregate_metrics.json").read_text(encoding="utf-8"))
    assert saved["metrics"]["C"]["final_success_rate"] == 1.0
    assert CONFIGURATIONS["D"]["lora"] and CONFIGURATIONS["D"]["correction"]
