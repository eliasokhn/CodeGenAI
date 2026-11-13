"""Feedback-based self-correction with the same Qwen model.

    generate -> run feedback tests -> (pass? return) -> correction prompt with structured
    failures -> regenerate -> retest ... at most ``max_attempts`` corrections.

Only *visible* tests are ever executed during the loop; a separate, final run against the
hidden tests happens in the evaluator **after** the loop has finished and its outcome is never
fed back. Repeated identical code, invalid output and exhausted retries are handled explicitly.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from codegenai.config import CorrectionConfig
from codegenai.correction.feedback import format_feedback
from codegenai.data.extract import check_syntax
from codegenai.logging_utils import get_logger
from codegenai.model.generation import GenerationResult, Generator
from codegenai.model.prompts import DEFAULT_SYSTEM_PROMPT, build_correction_messages, build_generation_messages
from codegenai.sandbox.report import Status, TestReport

log = get_logger(__name__)

# A callable that runs candidate code against the *visible* tests only.
FeedbackRunner = Callable[[str], TestReport]


@dataclass
class Attempt:
    index: int  # 0 = initial generation, 1..n = corrections
    code: str
    raw_text: str
    generation: dict[str, Any]
    report: Optional[dict[str, Any]]
    feedback: Optional[str]  # feedback that was SENT to produce this attempt (None for initial)
    latency_generation_s: float
    latency_testing_s: float
    stop_reason: Optional[str] = None


@dataclass
class CorrectionTrace:
    instruction: str
    attempts: list[Attempt] = field(default_factory=list)
    final_code: str = ""
    final_status: str = Status.UNAVAILABLE.value
    passed_feedback_tests: bool = False
    n_corrections: int = 0
    corrected: bool = False  # initial failed, a later attempt passed the feedback tests
    total_latency_s: float = 0.0
    stop_reason: str = ""
    model_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SelfCorrectionLoop:
    def __init__(self, generator: Generator, cfg: CorrectionConfig,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT, enable_correction: bool = True):
        self.generator = generator
        self.cfg = cfg
        self.system_prompt = system_prompt
        self.enable_correction = enable_correction

    def run(self, instruction: str, feedback_runner: Optional[FeedbackRunner],
            visible_tests: Optional[list[str]] = None) -> CorrectionTrace:
        """Generate (and optionally correct) code for ``instruction``.

        ``feedback_runner`` executes a candidate against the visible tests; ``None`` means no
        reliable tests exist, in which case the result is returned unverified after one shot.
        """
        t_start = time.perf_counter()
        trace = CorrectionTrace(instruction=instruction, model_name=getattr(self.generator, "name", ""))
        messages = build_generation_messages(instruction, visible_tests, self.system_prompt)
        gen = self.generator.generate(messages)
        attempt = self._test(0, gen, None, feedback_runner)
        trace.attempts.append(attempt)
        previous_codes = {attempt.code.strip()}

        max_corr = self.cfg.max_attempts if self.enable_correction else 0
        while True:
            report = attempt.report
            if feedback_runner is None:
                trace.stop_reason = "no_feedback_tests"
                break
            if report is None:
                trace.stop_reason = "no_report"
                break
            status = Status(report["status"])
            if status == Status.PASSED:
                trace.stop_reason = "passed"
                break
            if status == Status.UNAVAILABLE:
                trace.stop_reason = "sandbox_unavailable"
                break
            if trace.n_corrections >= max_corr:
                trace.stop_reason = "max_attempts_exhausted" if max_corr else "correction_disabled"
                break
            feedback = format_feedback(TestReport.from_dict(report), self.cfg.max_feedback_chars)
            prev_code = attempt.code if attempt.code.strip() else attempt.raw_text
            corr_messages = build_correction_messages(instruction, prev_code, feedback, visible_tests,
                                                      self.system_prompt)
            gen = self.generator.generate(corr_messages)
            trace.n_corrections += 1
            attempt = self._test(trace.n_corrections, gen, feedback, feedback_runner)
            trace.attempts.append(attempt)
            if self.cfg.stop_on_identical_code and attempt.code.strip() in previous_codes \
                    and Status(attempt.report["status"]) != Status.PASSED:
                attempt.stop_reason = "identical_code"
                trace.stop_reason = "identical_code_repeated"
                break
            previous_codes.add(attempt.code.strip())

        final = trace.attempts[-1]
        trace.final_code = final.code
        trace.final_status = final.report["status"] if final.report else Status.UNAVAILABLE.value
        trace.passed_feedback_tests = trace.final_status == Status.PASSED.value
        first_passed = trace.attempts[0].report and trace.attempts[0].report["status"] == Status.PASSED.value
        trace.corrected = bool(trace.passed_feedback_tests and not first_passed)
        trace.total_latency_s = time.perf_counter() - t_start
        return trace

    def _test(self, index: int, gen: GenerationResult, feedback: Optional[str],
              runner: Optional[FeedbackRunner]) -> Attempt:
        code = gen.code
        report_dict: Optional[dict[str, Any]] = None
        t_test = 0.0
        if not code.strip():
            # Invalid / empty model output: synthesise a syntax-error report so the loop can react.
            rep = TestReport(status=Status.SYNTAX_ERROR, syntax_error="Empty response: no Python code block found",
                             summary="No code produced")
            report_dict = rep.to_dict()
        elif runner is not None:
            syn = check_syntax(code)
            if not syn.ok:
                rep = TestReport(status=Status.SYNTAX_ERROR, syntax_error=syn.error, summary=f"Syntax error: {syn.error}")
                report_dict = rep.to_dict()
            else:
                t0 = time.perf_counter()
                rep = runner(code)
                t_test = time.perf_counter() - t0
                report_dict = rep.to_dict()
        return Attempt(index=index, code=code, raw_text=gen.raw_text, generation=gen.to_dict(),
                       report=report_dict, feedback=feedback, latency_generation_s=gen.latency_s,
                       latency_testing_s=t_test)
