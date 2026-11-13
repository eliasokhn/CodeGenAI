"""Turn a sandbox ``TestReport`` into model-readable feedback WITHOUT leaking hidden tests.

``format_feedback`` only ever reads ``report.visible_failures()`` plus the status/syntax error.
Hidden test names, messages and assertions are filtered out before anything reaches the model.
"""
from __future__ import annotations

import re

from codegenai.sandbox.report import Status, TestReport

_PATH_RE = re.compile(r"/work/|\\work\\")


def _clean(msg: str, limit: int) -> str:
    msg = _PATH_RE.sub("", msg)
    msg = re.sub(r"\n{3,}", "\n\n", msg).strip()
    if len(msg) > limit:
        msg = msg[: limit - 20].rstrip() + "\n... (truncated)"
    return msg


def format_feedback(report: TestReport, max_chars: int = 2500) -> str:
    """Human/model readable summary of the *visible* failures in a report."""
    if report.status == Status.UNAVAILABLE:
        return "Tests could not be executed (secure sandbox unavailable)."
    if report.status == Status.SYNTAX_ERROR:
        return f"The code does not parse.\n{report.syntax_error}"
    if report.status == Status.TIMEOUT and not report.visible_failures():
        return "Execution exceeded the time limit (possible infinite loop or very slow algorithm)."
    if report.status == Status.COLLECTION_ERROR:
        err = _clean(str(report.meta.get("collection_error", ""))[-1200:], 1200)
        return f"The test module could not import your code:\n{err}"
    fails = report.visible_failures()
    if not fails:
        if report.status == Status.MISSING_FUNCTION:
            return "The required function is missing or has the wrong name/signature."
        return report.summary
    per_test = max(300, (max_chars - 200) // max(1, len(fails)))
    lines = [f"{len(fails)} test(s) failed:"]
    for t in fails:
        head = f"- {t.name}: {t.error_type or t.outcome}"
        body = _clean(t.message, per_test)
        lines.append(head + ("\n" + _indent(body) if body else ""))
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[: max_chars - 20] + "\n... (truncated)"


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + ln for ln in text.splitlines())


def assert_no_hidden_leak(feedback: str, hidden_tests: list[str], hidden_names: set[str]) -> None:
    """Defensive check used by tests and the evaluation loop: hidden content must not appear."""
    for h in hidden_tests:
        core = h.replace("assert ", "").strip()
        if core and core in feedback:
            raise AssertionError(f"Hidden test leaked into feedback: {core[:80]}")
    for name in hidden_names:
        if name in feedback:
            raise AssertionError(f"Hidden test name leaked into feedback: {name}")
