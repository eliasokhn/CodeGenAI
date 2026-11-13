"""Build the ``{solution.py, test_*.py}`` bundles that the sandbox executes.

Two flavours:

* KodCode: the upstream pytest file is used verbatim (it does ``from solution import ...``).
* MBPP: each ``assert`` statement becomes its own test function so that per-assert outcomes are
  known; visible and hidden asserts live in *separate* modules and the hidden module's test
  names are reported to the sandbox so they can be flagged ``hidden`` in the report.
"""
from __future__ import annotations

import ast
import textwrap
from typing import Optional

from codegenai.data.schema import MBPPProblem

ENTRY_POINT_TEST = "test__entry_point_exists"

_IMPORT_PREAMBLE = """\
import math, re, heapq, itertools, collections, functools, bisect, string, operator, random
import solution as _solution
from solution import *  # noqa: F401,F403
globals().update({k: v for k, v in vars(_solution).items() if not k.startswith("__")})
"""


def entry_point_test(entry_point: str) -> str:
    return textwrap.dedent(f"""
        def {ENTRY_POINT_TEST}():
            import inspect
            assert hasattr(_solution, {entry_point!r}), "function {entry_point!r} is not defined in solution.py"
            assert callable(getattr(_solution, {entry_point!r})), "{entry_point!r} is not callable"
    """)


def _normalise_assert(stmt: str) -> str:
    """Validate that ``stmt`` is a single Python statement (MBPP asserts are one-liners)."""
    stmt = stmt.strip()
    ast.parse(stmt)  # raises SyntaxError for malformed test strings -> caller decides
    return stmt


def build_assert_module(asserts: list[str], prefix: str, setup_code: str = "",
                        entry_point: Optional[str] = None, include_entry_test: bool = False) -> str:
    """Generate a pytest module with one ``test_{prefix}_{i}`` per assert statement."""
    parts = [_IMPORT_PREAMBLE]
    if setup_code and setup_code.strip():
        parts.append(setup_code.rstrip() + "\n")
    if include_entry_test and entry_point:
        parts.append(entry_point_test(entry_point))
    for i, a in enumerate(asserts):
        body = textwrap.indent(_normalise_assert(a), "    ")
        parts.append(f"\n\ndef test_{prefix}_{i}():\n{body}\n")
    return "\n".join(parts)


def test_names(prefix: str, n: int) -> list[str]:
    return [f"test_{prefix}_{i}" for i in range(n)]


def build_mbpp_bundle(problem: MBPPProblem, candidate_code: str, mode: str) -> tuple[dict[str, str], set[str]]:
    """Return (files, hidden_test_names) for ``mode`` in {"visible", "hidden", "all"}.

    ``visible``: only feedback tests (safe to show the model).
    ``hidden``:  hidden tests only (final verdict).
    ``all``:     both, with hidden names flagged so the report can separate them.
    """
    files = {"solution.py": candidate_code}
    hidden: set[str] = set()
    if mode in ("visible", "all") and problem.visible_tests:
        files["test_visible.py"] = build_assert_module(
            problem.visible_tests, "visible", problem.setup_code, problem.entry_point, include_entry_test=True,
        )
    if mode in ("hidden", "all"):
        hidden_asserts = list(problem.hidden_tests)
        files["test_hidden.py"] = build_assert_module(
            hidden_asserts, "hidden", problem.setup_code, problem.entry_point,
            include_entry_test=(mode == "hidden" or not problem.visible_tests),
        )
        hidden.update(test_names("hidden", len(hidden_asserts)))
        if problem.plus_test:
            files["test_plus.py"] = build_plus_module(problem.plus_test, problem.entry_point)
            hidden.add("test_plus")
    if mode == "hidden":
        hidden.add(ENTRY_POINT_TEST)
    return files, hidden


def build_plus_module(plus_test: str, entry_point: str) -> str:
    """Wrap an MBPP+ (EvalPlus) test script as a single pytest test.

    EvalPlus scripts define ``inputs``/``results``/``assertion`` and a loop that calls the
    candidate as ``candidate``; some versions call ``check(candidate)``. We expose the candidate
    under both names and execute the script inside the test function so assertion errors
    surface as test failures.
    """
    return textwrap.dedent(f"""
        import solution as _solution
        from solution import *  # noqa: F401,F403

        _PLUS_SCRIPT = {plus_test!r}

        def test_plus():
            candidate = getattr(_solution, {entry_point!r})
            ns = {{"candidate": candidate, "__name__": "__plus__"}}
            ns.update({{k: v for k, v in vars(_solution).items() if not k.startswith("__")}})
            exec(compile(_PLUS_SCRIPT, "mbppplus_test.py", "exec"), ns)
            if "check" in ns and callable(ns["check"]):
                ns["check"](candidate)
    """)


def build_kodcode_bundle(solution: str, test_src: str) -> dict[str, str]:
    """Upstream KodCode tests import the module as ``solution``; keep that layout."""
    return {"solution.py": solution, "test_solution.py": test_src}
