"""Mutation-based spot checks of test effectiveness (static mutant generation only).

A test suite that still passes when the reference solution is broken cannot constrain the
model's output. For a sample of records we generate a few deterministic mutants with ``ast`` and
run each against the upstream tests *in the sandbox*; the suite is "effective" for a mutant if at
least one test fails. Execution happens elsewhere (``audit.py``); this module never runs code.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from typing import Optional


@dataclass
class Mutant:
    kind: str
    code: str
    description: str


class _Mutator(ast.NodeTransformer):
    def __init__(self, target_fn: str, kind: str):
        self.target_fn = target_fn
        self.kind = kind
        self.applied = False

    def visit_FunctionDef(self, node: ast.FunctionDef):  # noqa: N802
        if node.name != self.target_fn:
            return self.generic_visit(node)
        if self.kind == "stub":
            doc = node.body[0] if (node.body and isinstance(node.body[0], ast.Expr)
                                   and isinstance(getattr(node.body[0], "value", None), ast.Constant)) else None
            node.body = ([doc] if doc else []) + [ast.Return(value=ast.Constant(value=None))]
            self.applied = True
            return node
        return self.generic_visit(node)

    def visit_Return(self, node: ast.Return):  # noqa: N802
        if self.kind == "return_zero" and not self.applied and node.value is not None:
            self.applied = True
            return ast.Return(value=ast.Constant(value=0))
        return self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare):  # noqa: N802
        if self.kind == "flip_compare" and not self.applied and len(node.ops) == 1:
            flip = {ast.Lt: ast.GtE, ast.Gt: ast.LtE, ast.LtE: ast.Gt, ast.GtE: ast.Lt,
                    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
            op_type = type(node.ops[0])
            if op_type in flip:
                self.applied = True
                node.ops = [flip[op_type]()]
                return node
        return self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp):  # noqa: N802
        if self.kind == "swap_arith" and not self.applied:
            swap = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv}
            op_type = type(node.op)
            if op_type in swap:
                self.applied = True
                node.op = swap[op_type]()
                return node
        return self.generic_visit(node)


def generate_mutants(code: str, entry_point: Optional[str],
                     kinds: tuple[str, ...] = ("stub", "return_zero", "flip_compare", "swap_arith")) -> list[Mutant]:
    """Return syntactically valid mutants of ``code`` (only kinds that actually changed something)."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return []
    fn_names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    target = entry_point if entry_point in fn_names else (fn_names[-1] if fn_names else None)
    if target is None:
        return []
    mutants: list[Mutant] = []
    for kind in kinds:
        mutator = _Mutator(target, kind)
        new_tree = mutator.visit(copy.deepcopy(tree))
        if not mutator.applied:
            continue
        ast.fix_missing_locations(new_tree)
        try:
            src = ast.unparse(new_tree)
            compile(src, "<mutant>", "exec")
        except Exception:  # noqa: BLE001
            continue
        if src.strip() == ast.unparse(tree).strip():
            continue
        mutants.append(Mutant(kind, src + "\n", f"{kind} applied to {target}"))
    return mutants
