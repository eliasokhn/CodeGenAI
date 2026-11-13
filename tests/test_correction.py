"""Self-correction loop logic, feedback formatting and hidden-test isolation (mock model + mock sandbox)."""
from __future__ import annotations

import pytest

from codegenai.config import CorrectionConfig
from codegenai.correction import SelfCorrectionLoop, assert_no_hidden_leak, format_feedback
from codegenai.model.generation import MockCodeGenerator
from codegenai.model.prompts import build_correction_messages, build_generation_messages
from codegenai.sandbox import Status
from conftest import make_report

GOOD = "```python\ndef add(a, b):\n    return a + b\n```"
BAD = "```python\ndef add(a, b):\n    return a - b\n```"


def runner_factory(expected_good: str = "return a + b"):
    def _run(code: str):
        if expected_good in code:
            return make_report(Status.PASSED, [("test_visible_0", "passed", None, False)])
        return make_report(Status.FAILED, [("test_visible_0", "failed", "AssertionError", False)])
    return _run


def test_initial_pass_returns_immediately():
    gen = MockCodeGenerator(lambda msgs: GOOD)
    trace = SelfCorrectionLoop(gen, CorrectionConfig()).run("add two numbers", runner_factory(), ["assert add(1,2)==3"])
    assert trace.passed_feedback_tests and trace.n_corrections == 0 and trace.stop_reason == "passed"
    assert len(gen.calls) == 1 and not trace.corrected
    # visible test appears in the generation prompt
    assert "assert add(1,2)==3" in gen.calls[0][1]["content"]


def test_correction_fixes_code_after_feedback():
    answers = iter([BAD, GOOD])
    gen = MockCodeGenerator(lambda msgs: next(answers))
    trace = SelfCorrectionLoop(gen, CorrectionConfig(max_attempts=2)).run("add", runner_factory(), ["assert add(1,2)==3"])
    assert trace.passed_feedback_tests and trace.n_corrections == 1 and trace.corrected
    assert len(trace.attempts) == 2 and trace.attempts[1].feedback is not None
    corr_prompt = gen.calls[1][1]["content"]
    assert "Previous code" in corr_prompt and "return a - b" in corr_prompt and "test_visible_0" in corr_prompt


def test_max_two_corrections_and_exhaustion():
    gen = MockCodeGenerator(lambda msgs: BAD.replace("a - b", f"a - b - {len(msgs[1]['content'])}"))
    trace = SelfCorrectionLoop(gen, CorrectionConfig(max_attempts=2)).run("add", runner_factory(), ["assert add(1,2)==3"])
    assert trace.n_corrections == 2 and len(gen.calls) == 3 and not trace.passed_feedback_tests
    assert trace.stop_reason == "max_attempts_exhausted" and trace.final_status == "failed"


def test_identical_repeated_failure_stops_early():
    gen = MockCodeGenerator(lambda msgs: BAD)
    trace = SelfCorrectionLoop(gen, CorrectionConfig(max_attempts=2)).run("add", runner_factory(), ["assert add(1,2)==3"])
    assert trace.n_corrections == 1 and trace.stop_reason == "identical_code_repeated"


def test_invalid_output_handled():
    answers = iter(["I cannot help with that.", GOOD])
    gen = MockCodeGenerator(lambda msgs: next(answers))
    trace = SelfCorrectionLoop(gen, CorrectionConfig()).run("add", runner_factory(), ["assert add(1,2)==3"])
    assert trace.attempts[0].report["status"] == "syntax_error" and trace.passed_feedback_tests
    assert "does not parse" in trace.attempts[1].feedback


def test_syntax_error_is_caught_before_sandbox():
    calls = []
    def runner(code):
        calls.append(code)
        return make_report(Status.PASSED)
    answers = iter(["```python\ndef add(a, b)\n    return a + b\n```", GOOD])
    gen = MockCodeGenerator(lambda msgs: next(answers))
    SelfCorrectionLoop(gen, CorrectionConfig()).run("add", runner, None and [])
    # with runner but no visible tests listed, still runs runner
    assert len(calls) == 1  # the syntactically broken attempt never reached the sandbox


def test_no_feedback_tests_returns_unverified_single_shot():
    gen = MockCodeGenerator(lambda msgs: BAD)
    trace = SelfCorrectionLoop(gen, CorrectionConfig()).run("add", None)
    assert trace.stop_reason == "no_feedback_tests" and len(gen.calls) == 1
    assert trace.final_status == Status.UNAVAILABLE.value and not trace.passed_feedback_tests


def test_correction_disabled_stops_after_initial():
    gen = MockCodeGenerator(lambda msgs: BAD)
    trace = SelfCorrectionLoop(gen, CorrectionConfig(), enable_correction=False).run("add", runner_factory(), ["t"])
    assert trace.n_corrections == 0 and trace.stop_reason == "correction_disabled"


def test_sandbox_unavailable_stops_loop():
    gen = MockCodeGenerator(lambda msgs: BAD)
    trace = SelfCorrectionLoop(gen, CorrectionConfig()).run("add", lambda code: make_report(Status.UNAVAILABLE), ["t"])
    assert trace.stop_reason == "sandbox_unavailable" and len(gen.calls) == 1


# ----------------------------------------------------------------------------- hidden test isolation
def test_feedback_never_contains_hidden_tests():
    rep = make_report(Status.FAILED, [
        ("test_visible_0", "failed", "AssertionError", False),
        ("test_hidden_0", "failed", "AssertionError", True),
        ("test_hidden_1", "failed", "TypeError", True),
    ])
    rep.tests[1].message = "assert remove_Occ('abcda','a') == 'bcd' SECRET"
    fb = format_feedback(rep)
    assert "test_visible_0" in fb and "test_hidden" not in fb and "SECRET" not in fb
    assert_no_hidden_leak(fb, ["assert remove_Occ('abcda','a') == 'bcd'"], {"test_hidden_0", "test_hidden_1"})
    with pytest.raises(AssertionError):
        assert_no_hidden_leak(fb + " remove_Occ('abcda','a') == 'bcd'", ["assert remove_Occ('abcda','a') == 'bcd'"], set())


def test_only_visible_tests_reach_the_model(mbpp_problem):
    """End-to-end check with mock model + mock runner: hidden asserts never appear in any prompt."""
    seen_prompts = []
    def responder(msgs):
        seen_prompts.append(msgs[1]["content"])
        return BAD
    gen = MockCodeGenerator(responder)
    def runner(code):
        return make_report(Status.FAILED, [("test_visible_0", "failed", "AssertionError", False)])
    SelfCorrectionLoop(gen, CorrectionConfig(max_attempts=2, stop_on_identical_code=False)).run(
        mbpp_problem.text, runner, mbpp_problem.visible_tests)
    disjoint_hidden = [h for h in mbpp_problem.hidden_tests if h not in mbpp_problem.visible_tests]
    assert len(seen_prompts) == 3
    for p in seen_prompts:
        for h in disjoint_hidden:
            assert h.replace("assert ", "") not in p
        assert mbpp_problem.visible_tests[0] in p


def test_format_feedback_special_statuses():
    assert "does not parse" in format_feedback(make_report(Status.SYNTAX_ERROR))
    assert "time limit" in format_feedback(make_report(Status.TIMEOUT))
    assert "unavailable" in format_feedback(make_report(Status.UNAVAILABLE))
    assert "missing" in format_feedback(make_report(Status.MISSING_FUNCTION))
    long_rep = make_report(Status.FAILED, [(f"test_visible_{i}", "failed", "AssertionError", False) for i in range(30)])
    for t in long_rep.tests:
        t.message = "x" * 500
    assert len(format_feedback(long_rep, max_chars=1000)) <= 1000


def test_prompt_builders():
    m = build_generation_messages("do it", ["assert f(1) == 2"], system_prompt="SYS")
    assert m[0] == {"role": "system", "content": "SYS"} and "assert f(1) == 2" in m[1]["content"]
    c = build_correction_messages("do it", "def f(): pass", "1 test failed", ["assert f(1) == 2"])
    assert "Previous code" in c[1]["content"] and "1 test failed" in c[1]["content"]
