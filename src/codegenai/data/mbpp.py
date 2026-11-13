"""MBPP loading with a strict visible/hidden test policy and optional MBPP+ integration.

Policy (see docs/methodology.md):

* The *visible* test(s) are shown in the prompt (standard MBPP prompting – the model must learn
  the expected function name/signature from them) and are the ONLY tests used as correction
  feedback.
* The *hidden* tests are used exclusively for the final verdict. With
  ``hidden_test_policy="all"`` they are all MBPP asserts (standard pass@1); with ``"disjoint"``
  only the asserts the model never saw.
* MBPP ``validation`` is for model selection, ``test`` for final reporting. Training never
  touches MBPP.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from codegenai.config import Config
from codegenai.data.download import load_mbpp, load_mbppplus
from codegenai.data.extract import analyze_code
from codegenai.data.schema import MBPPProblem
from codegenai.logging_utils import get_logger

log = get_logger(__name__)

_CALL_RE = re.compile(r"assert\s+(?:\(?\s*)?(?:[A-Za-z_][\w.]*\()*?([A-Za-z_]\w*)\s*\(")


def infer_entry_point(tests: list[str], reference_code: str) -> str:
    """Function name under test: the name called in the asserts that the reference defines."""
    an = analyze_code(reference_code)
    defined = set(an.functions) if an.syntax_ok else set()
    counts: dict[str, int] = {}
    for t in tests:
        for name in re.findall(r"([A-Za-z_]\w*)\s*\(", t):
            if name in {"assert", "set", "len", "list", "tuple", "str", "int", "float", "sorted", "round",
                        "abs", "math", "isclose", "dict", "all", "any", "sum", "max", "min", "repr", "bool"}:
                continue
            counts[name] = counts.get(name, 0) + (3 if name in defined else 1)
    if counts:
        return max(counts.items(), key=lambda kv: kv[1])[0]
    if an.syntax_ok and an.functions:
        return an.functions[-1]
    return "solution"


def build_problem(row: dict[str, Any], split: str, visible_tests: str = "first",
                  hidden_test_policy: str = "all", include_challenge: bool = False,
                  plus_test: Optional[str] = None) -> MBPPProblem:
    tests = [t.strip() for t in row["test_list"] if t and t.strip()]
    challenge = [t.strip() for t in row.get("challenge_test_list") or [] if t and t.strip()]
    visible = tests[:1] if visible_tests == "first" and tests else []
    if hidden_test_policy == "disjoint":
        hidden = [t for t in tests if t not in visible]
    else:
        hidden = list(tests)
    if include_challenge:
        hidden = hidden + challenge
    return MBPPProblem(
        task_id=int(row["task_id"]), text=row["text"].strip(), reference_code=row["code"].replace("\r\n", "\n"),
        entry_point=infer_entry_point(tests, row["code"]), visible_tests=visible, hidden_tests=hidden,
        setup_code=(row.get("test_setup_code") or "").replace("\r\n", "\n"), challenge_tests=challenge,
        plus_test=plus_test, split=split,
    )


def load_mbpp_problems(cfg: Config, split: Optional[str] = None, limit: Optional[int] = None) -> list[MBPPProblem]:
    ev = cfg.evaluation
    split = split or ev.split
    ds = load_mbpp(cfg, split)
    plus_map: dict[int, dict[str, Any]] = {}
    if ev.use_mbppplus:
        plus_map = load_mbppplus_index(cfg)
    problems: list[MBPPProblem] = []
    unmapped = 0
    for row in ds:
        plus = plus_map.get(int(row["task_id"]))
        plus_test = None
        if plus is not None:
            if verify_plus_mapping(row, plus):
                plus_test = plus["test"]
            else:
                unmapped += 1
        problems.append(build_problem(row, split, ev.visible_tests, ev.hidden_test_policy,
                                      ev.include_challenge_tests, plus_test))
    if ev.use_mbppplus:
        n_plus = sum(1 for p in problems if p.plus_test)
        log.info("MBPP+ coverage on %s: %d/%d problems (%d id-mapping mismatches)", split, n_plus,
                 len(problems), unmapped)
    if limit:
        problems = problems[:limit]
    log.info("Loaded %d MBPP problems from split '%s' (visible=%s, hidden=%s)", len(problems), split,
             ev.visible_tests, ev.hidden_test_policy)
    return problems


def load_mbppplus_index(cfg: Config) -> dict[int, dict[str, Any]]:
    """MBPP+ rows keyed by the MBPP task id (MBPP+ ids are plain ints matching MBPP's ``task_id``)."""
    ds = load_mbppplus(cfg)
    index: dict[int, dict[str, Any]] = {}
    for row in ds:
        tid = row["task_id"]
        if isinstance(tid, str):  # some EvalPlus releases use "Mbpp/<id>"
            tid = int(tid.split("/")[-1])
        index[int(tid)] = row
    return index


def verify_plus_mapping(mbpp_row: dict[str, Any], plus_row: dict[str, Any]) -> bool:
    """The MBPP+ row must describe the same task: same prompt text and same first assert."""
    same_prompt = plus_row.get("prompt", "").strip() == mbpp_row["text"].strip()
    plus_tests = [t.strip() for t in plus_row.get("test_list") or []]
    same_tests = bool(plus_tests) and plus_tests[0] == mbpp_row["test_list"][0].strip()
    return same_prompt or same_tests


def all_mbpp_rows_for_screening(cfg: Config) -> Iterable[dict[str, Any]]:
    """Every MBPP row across all splits (for contamination screening)."""
    for split in ("train", "validation", "test", "prompt"):
        for row in load_mbpp(cfg, split):
            yield {"task_id": row["task_id"], "text": row["text"], "code": row["code"], "split": split}
