"""In-container test runner (stdlib + pytest only). NEVER import this on the host.

Reads a JSON payload from stdin::

    {"files": {"solution.py": "...", "test_solution.py": "..."},
     "per_test_timeout": 5, "entry_point": "func_name"}

writes the files into the working directory (a tmpfs mount), checks the candidate module for
syntax errors, runs pytest with an in-process result collector and finally prints one line
``__CODEGENAI_REPORT__`` followed by a JSON report. All output is size-limited.
"""
from __future__ import annotations

import io
import json
import os
import sys
import traceback

MAX_MSG = 1500
MAX_TAIL = 4000
SENTINEL = "__CODEGENAI_REPORT__"


class Collector:
    """pytest plugin collecting per-test outcomes with error classification."""

    def __init__(self) -> None:
        self.tests: list[dict] = []
        self.collection_error: str | None = None

    def pytest_collectreport(self, report):  # noqa: D401
        if report.failed:
            text = str(report.longrepr)
            self.collection_error = (self.collection_error or "") + text[:MAX_MSG]

    def pytest_runtest_logreport(self, report):
        name = report.nodeid.split("::", 1)[-1]
        if report.when == "call" or (report.when == "setup" and report.failed):
            entry = {"name": name, "outcome": report.outcome, "duration": round(report.duration, 4),
                     "error_type": None, "message": ""}
            if report.failed:
                text = str(report.longrepr)
                entry["message"] = text[-MAX_MSG:]
                etype = None
                # pytest stores the exception repr; try to recover the exception class name.
                repr_crash = getattr(report.longrepr, "reprcrash", None)
                if repr_crash is not None:
                    msg = repr_crash.message
                    if msg.startswith("Failed: Timeout"):
                        etype = "Timeout"
                    else:
                        etype = msg.split(":", 1)[0].strip() if ":" in msg else msg.strip()[:60]
                        if etype == "assert" or etype.startswith("assert "):
                            etype = "AssertionError"
                if etype is None:
                    etype = "Timeout" if "Timeout" in text else "Error"
                if etype in ("AssertionError", "Failed", "assert"):
                    etype = "AssertionError"
                if etype == "Timeout":
                    entry["outcome"] = "timeout"
                elif report.when == "setup":
                    entry["outcome"] = "error"
                elif etype != "AssertionError":
                    entry["outcome"] = "error"
                entry["error_type"] = etype
            self.tests.append(entry)


def main() -> int:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(SENTINEL)
        print(json.dumps({"internal_error": f"bad payload: {exc}"}))
        return 2

    workdir = os.getcwd()
    files = payload.get("files", {})
    for name, content in files.items():
        if "/" in name or "\\" in name or name.startswith("."):
            print(SENTINEL)
            print(json.dumps({"internal_error": f"illegal filename {name!r}"}))
            return 2
        with open(os.path.join(workdir, name), "w", encoding="utf-8") as fh:
            fh.write(content)

    report: dict = {"tests": [], "syntax_error": None, "collection_error": None, "truncated": False}

    # 1. Syntax check of the candidate (compile only, never executed here).
    solution = files.get("solution.py", "")
    try:
        compile(solution, "solution.py", "exec")
    except SyntaxError as exc:
        report["syntax_error"] = f"{type(exc).__name__}: {exc.msg} (line {exc.lineno})"
        print(SENTINEL)
        print(json.dumps(report))
        return 0
    except (ValueError, RecursionError, MemoryError) as exc:
        report["syntax_error"] = f"{type(exc).__name__}: {exc}"
        print(SENTINEL)
        print(json.dumps(report))
        return 0

    # 2. Run pytest in-process with captured output.
    import pytest  # noqa: WPS433 (imported late so a missing pytest is reported as JSON)

    collector = Collector()
    per_test_timeout = float(payload.get("per_test_timeout", 5))
    args = [
        workdir, "-q", "-p", "no:cacheprovider", "--tb=short", "-W", "ignore",
        f"--timeout={per_test_timeout}", "--timeout-method=signal", "-o", "console_output_style=classic",
        "--import-mode=importlib", "--rootdir", workdir,
    ]
    buf_out, buf_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = buf_out, buf_err
    try:
        sys.path.insert(0, workdir)
        exit_code = pytest.main(args, plugins=[collector])
    except BaseException:  # noqa: BLE001 - we must always emit a report
        report["internal_error"] = traceback.format_exc()[-MAX_MSG:]
        exit_code = -1
    finally:
        sys.stdout, sys.stderr = old_out, old_err

    report["tests"] = collector.tests
    report["collection_error"] = collector.collection_error
    report["pytest_exit_code"] = int(exit_code)
    out, err = buf_out.getvalue(), buf_err.getvalue()
    report["truncated"] = len(out) > MAX_TAIL or len(err) > MAX_TAIL
    report["stdout_tail"] = out[-MAX_TAIL:]
    report["stderr_tail"] = err[-MAX_TAIL:]
    print(SENTINEL)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
