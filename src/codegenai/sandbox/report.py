"""Structured results of running generated code against tests inside the sandbox."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class Status(str, Enum):
    """Outcome category of one sandbox execution."""

    PASSED = "passed"  # every test passed
    FAILED = "failed"  # at least one assertion failed (wrong output)
    SYNTAX_ERROR = "syntax_error"  # the candidate module does not parse
    MISSING_FUNCTION = "missing_function"  # entry point absent / wrong signature
    RUNTIME_ERROR = "runtime_error"  # exception other than AssertionError
    TIMEOUT = "timeout"  # per-test or whole-run time limit exceeded
    COLLECTION_ERROR = "collection_error"  # the test module itself failed to import
    UNAVAILABLE = "unavailable"  # secure sandbox not available -> execution disabled
    INTERNAL_ERROR = "internal_error"  # sandbox infrastructure failure


@dataclass
class TestCaseResult:
    __test__ = False  # not a pytest test class

    name: str
    outcome: str  # passed | failed | error | timeout
    error_type: Optional[str] = None  # e.g. AssertionError, TypeError, Timeout
    message: str = ""  # trimmed traceback/assertion message
    duration_s: float = 0.0
    hidden: bool = False  # True if this test is a hidden evaluation test

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TestReport:
    __test__ = False  # not a pytest test class

    status: Status
    passed: int = 0
    failed: int = 0
    errors: int = 0
    total: int = 0
    tests: list[TestCaseResult] = field(default_factory=list)
    syntax_error: Optional[str] = None
    summary: str = ""  # one-paragraph human readable summary
    stdout_tail: str = ""
    stderr_tail: str = ""
    execution_time_s: float = 0.0  # wall time of the container run
    verified: bool = False  # True only if the code really ran in the sandbox and all tests passed
    truncated_output: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == Status.PASSED

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TestReport":
        d = dict(d)
        d["status"] = Status(d["status"])
        d["tests"] = [TestCaseResult(**t) for t in d.get("tests", [])]
        return cls(**d)

    @classmethod
    def unavailable(cls, reason: str) -> "TestReport":
        return cls(status=Status.UNAVAILABLE, summary=f"Secure execution unavailable: {reason}", verified=False)

    def failed_tests(self) -> list[TestCaseResult]:
        return [t for t in self.tests if t.outcome != "passed"]

    def visible_failures(self) -> list[TestCaseResult]:
        """Failures that may be shown to the model (never hidden tests)."""
        return [t for t in self.failed_tests() if not t.hidden]


def _looks_like_missing_symbol(text: str) -> bool:
    """True when an error message says a name/function is missing from the candidate module."""
    t = text.replace("\n", " ")
    patterns = (
        "cannot import name", "from 'solution'", "module 'solution' has no attribute",
        "No module named 'solution", "is not defined in solution.py", "is not callable",
    )
    return any(p in t for p in patterns)


def _classify(tests: list[TestCaseResult], collection_error: Optional[str], timed_out: bool,
              syntax_error: Optional[str]) -> Status:
    if syntax_error:
        return Status.SYNTAX_ERROR
    if timed_out:
        return Status.TIMEOUT
    if collection_error:
        ce = str(collection_error)
        if _looks_like_missing_symbol(ce):
            return Status.MISSING_FUNCTION
        if re.search(r"(?:^|[/\s])solution\.py:\d+", ce):
            # The candidate module raised while being imported (module-level crash).
            return Status.RUNTIME_ERROR
        return Status.COLLECTION_ERROR
    if not tests:
        return Status.INTERNAL_ERROR
    failed = [t for t in tests if t.outcome != "passed"]
    if not failed:
        return Status.PASSED
    if any(t.outcome == "timeout" or t.error_type == "Timeout" for t in failed):
        return Status.TIMEOUT
    if any(t.name.startswith("test__entry_point") for t in failed):
        return Status.MISSING_FUNCTION
    if any(t.error_type in {"AttributeError", "ImportError", "NameError"} and _looks_like_missing_symbol(t.message)
           for t in failed):
        return Status.MISSING_FUNCTION
    if all(t.error_type == "AssertionError" for t in failed):
        return Status.FAILED
    if any(t.error_type == "AssertionError" for t in failed):
        return Status.FAILED  # mixed: wrong outputs dominate the classification
    return Status.RUNTIME_ERROR


def build_report(raw: dict[str, Any], hidden_names: Optional[set[str]] = None,
                 execution_time_s: float = 0.0, timed_out: bool = False) -> TestReport:
    """Turn the JSON emitted by ``runner_entry.py`` into a ``TestReport``."""
    hidden_names = hidden_names or set()
    tests: list[TestCaseResult] = []
    for t in raw.get("tests", []):
        name = t.get("name", "?")
        tests.append(TestCaseResult(
            name=name,
            outcome=t.get("outcome", "error"),
            error_type=t.get("error_type"),
            message=t.get("message", "")[:2000],
            duration_s=float(t.get("duration", 0.0)),
            hidden=any(name == h or name.startswith(h + "[") for h in hidden_names),
        ))
    syntax_error = raw.get("syntax_error")
    collection_error = raw.get("collection_error")
    status = _classify(tests, collection_error, timed_out, syntax_error)
    passed = sum(1 for t in tests if t.outcome == "passed")
    failed = sum(1 for t in tests if t.outcome == "failed")
    errors = sum(1 for t in tests if t.outcome in {"error", "timeout"})
    if status == Status.PASSED:
        summary = f"All {passed} test(s) passed."
    elif status == Status.SYNTAX_ERROR:
        summary = f"Syntax error: {syntax_error}"
    elif status == Status.COLLECTION_ERROR:
        summary = f"Test module failed to import: {str(collection_error)[:500]}"
    elif status == Status.TIMEOUT:
        summary = "Execution timed out."
    else:
        summary = f"{passed}/{len(tests)} tests passed; {failed} failed, {errors} errored."
    return TestReport(
        status=status, passed=passed, failed=failed, errors=errors, total=len(tests), tests=tests,
        syntax_error=syntax_error, summary=summary,
        stdout_tail=raw.get("stdout_tail", "")[-2000:], stderr_tail=raw.get("stderr_tail", "")[-2000:],
        execution_time_s=execution_time_s, verified=(status == Status.PASSED),
        truncated_output=bool(raw.get("truncated", False)),
        meta={"collection_error": collection_error} if collection_error else {},
    )
