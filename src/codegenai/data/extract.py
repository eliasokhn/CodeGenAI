"""Static (never executing) analysis of Python code and pytest files.

Everything here relies on ``ast`` only: parsing untrusted code with ``ast.parse`` compiles
but never runs it, so these helpers are safe to call on model output and dataset rows.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Optional

_CODE_BLOCK_RE = re.compile(r"```(?:python|py|Python)?[ \t]*\r?\n(.*?)```", re.DOTALL)
_ANY_BLOCK_RE = re.compile(r"```[^\n]*\r?\n(.*?)```", re.DOTALL)


def extract_code_blocks(text: str) -> list[str]:
    """Return the contents of all fenced Python code blocks in ``text``."""
    blocks = [b.strip("\r\n") for b in _CODE_BLOCK_RE.findall(text)]
    if not blocks:
        blocks = [b.strip("\r\n") for b in _ANY_BLOCK_RE.findall(text)]
    return [b for b in blocks if b.strip()]


def extract_code(text: str) -> str:
    """Best-effort extraction of Python source from a model response.

    Strategy: concatenate all fenced blocks if any exist (models sometimes split imports and
    the function into two blocks); otherwise fall back to the raw text with any leading prose
    stripped up to the first ``import``/``from``/``def``/``class`` line. An unterminated
    fenced block (``max_new_tokens`` hit) is handled by taking everything after the fence.
    """
    blocks = extract_code_blocks(text)
    if blocks:
        return "\n\n".join(blocks).strip() + "\n"
    stripped = text
    m = re.search(r"```(?:python|py)?[ \t]*\r?\n", text)
    if m:  # unterminated fence
        stripped = text[m.end():]
    lines = stripped.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if re.match(r"^\s*(import |from |def |class |@|#)", line):
            start = i
            break
    code = "\n".join(lines[start:]).strip()
    return code + ("\n" if code else "")


@dataclass
class SyntaxCheck:
    ok: bool
    error: Optional[str] = None
    lineno: Optional[int] = None


def check_syntax(code: str) -> SyntaxCheck:
    try:
        ast.parse(code)
        return SyntaxCheck(True)
    except SyntaxError as exc:  # also covers IndentationError
        return SyntaxCheck(False, f"{type(exc).__name__}: {exc.msg}", exc.lineno)
    except (ValueError, RecursionError, MemoryError) as exc:  # null bytes, absurd nesting
        return SyntaxCheck(False, f"{type(exc).__name__}: {exc}")


@dataclass
class CodeAnalysis:
    syntax_ok: bool
    syntax_error: Optional[str]
    functions: list[str] = field(default_factory=list)  # top-level function names
    classes: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)  # top-level module names
    has_main_guard: bool = False
    has_top_level_io: bool = False  # input()/print() at module level
    n_lines: int = 0
    n_chars: int = 0
    signatures: dict[str, int] = field(default_factory=dict)  # name -> positional arity


def analyze_code(code: str) -> CodeAnalysis:
    """Collect structural facts about a Python module without executing it."""
    n_lines = len([ln for ln in code.splitlines() if ln.strip()])
    syn = check_syntax(code)
    if not syn.ok:
        return CodeAnalysis(False, syn.error, n_lines=n_lines, n_chars=len(code))
    tree = ast.parse(code)
    functions: list[str] = []
    classes: list[str] = []
    imports: set[str] = set()
    signatures: dict[str, int] = {}
    has_main_guard = False
    has_top_level_io = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
            a = node.args
            signatures[node.name] = len(a.posonlyargs) + len(a.args)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
            ):
                has_main_guard = True
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            fn = node.value.func
            if isinstance(fn, ast.Name) and fn.id in {"print", "input"}:
                has_top_level_io = True
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imports.add(node.module.split(".")[0])
    return CodeAnalysis(
        True, None, functions, classes, sorted(imports), has_main_guard, has_top_level_io,
        n_lines, len(code), signatures,
    )


@dataclass
class TestAnalysis:
    syntax_ok: bool
    n_test_functions: int = 0
    n_asserts: int = 0
    n_value_asserts: int = 0  # asserts comparing a call result to a literal/value
    referenced_names: list[str] = field(default_factory=list)  # names called inside asserts
    imports_solution: bool = False
    imported_names: list[str] = field(default_factory=list)  # names imported from solution
    uses_fixtures: bool = False
    uses_mock: bool = False
    has_class_tests: bool = False


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name):
                names.add(f.id)
            elif isinstance(f, ast.Attribute):
                names.add(f.attr)
    return names


def analyze_tests(test_src: str) -> TestAnalysis:
    """Describe a pytest file: how many asserts, which solution names they constrain."""
    try:
        tree = ast.parse(test_src)
    except (SyntaxError, ValueError):
        return TestAnalysis(False)
    out = TestAnalysis(True)
    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "solution":
            out.imports_solution = True
            out.imported_names.extend(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == "solution":
                    out.imports_solution = True
                if a.name.split(".")[0] in {"mock", "unittest"}:
                    out.uses_mock = out.uses_mock or a.name.startswith("unittest.mock")
        if isinstance(node, ast.ImportFrom) and node.module and "mock" in node.module:
            out.uses_mock = True
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            out.n_test_functions += 1
            if node.args.args:
                out.uses_fixtures = True
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            out.has_class_tests = True
        if isinstance(node, ast.Assert):
            out.n_asserts += 1
            referenced |= _called_names(node.test)
            t = node.test
            if isinstance(t, ast.Compare):
                # a comparison whose one side is a call and other side is a literal-ish value;
                # identity checks against None (`is not None`) do not constrain the value.
                sides = [t.left, *t.comparators]
                has_call = any(isinstance(s, ast.Call) for s in sides)
                has_value = any(
                    isinstance(s, (ast.List, ast.Tuple, ast.Dict, ast.Set, ast.UnaryOp))
                    or (isinstance(s, ast.Constant) and s.value is not None)
                    for s in sides
                )
                identity_only = all(isinstance(op, (ast.Is, ast.IsNot)) for op in t.ops)
                if has_call and has_value and not identity_only:
                    out.n_value_asserts += 1
            elif isinstance(t, ast.Call):
                f = t.func
                # pytest.approx / math.isclose style value checks
                if isinstance(f, ast.Attribute) and f.attr in {"isclose", "approx", "allclose"}:
                    out.n_value_asserts += 1
    out.referenced_names = sorted(referenced)
    return out


def choose_entry_point(functions: list[str], test_analysis: Optional[TestAnalysis],
                       test_info: Optional[list[dict]] = None) -> Optional[str]:
    """Pick the primary function a problem is about.

    Preference: upstream ``test_info.function_name`` if it is defined by the solution, then
    the solution-defined name most referenced by the tests, then the last top-level function.
    """
    if not functions:
        return None
    if test_info:
        for info in test_info:
            name = info.get("function_name")
            if name in functions:
                return name
    if test_analysis is not None:
        for name in test_analysis.imported_names:
            if name in functions:
                return name
        for name in test_analysis.referenced_names:
            if name in functions:
                return name
    return functions[-1]


def strip_docstrings_and_comments(code: str) -> str:
    """Normalised code for near-duplicate detection (comments/docstrings removed)."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return re.sub(r"#.*", "", code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(
                getattr(node.body[0], "value", None), ast.Constant
            ) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    try:
        return ast.unparse(tree)
    except Exception:  # pragma: no cover - extremely unusual ASTs
        return code
