"""Benchmark-contamination screening between KodCode training rows and MBPP.

Two complementary detectors are applied to every candidate training row:

1. **Prompt overlap** – normalised token Jaccard similarity and shared word n-grams between the
   KodCode instruction and every MBPP ``text`` (all splits: train/validation/test/prompt).
2. **Solution overlap** – exact / near-exact match between the normalised KodCode solution and
   MBPP reference ``code`` (docstrings and comments stripped, identifiers kept), plus function
   name + arity coincidence combined with any prompt similarity.

The publisher already removed rows with TF-IDF cosine >= 0.70 to HumanEval/MBPP prompts; this
module is an *independent* check with a different similarity family, as required by the audit
protocol. Thresholds live in ``AuditConfig``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from codegenai.data.extract import analyze_code, strip_docstrings_and_comments

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "the", "to", "of", "in", "and", "or", "is", "that", "for", "with", "which", "this",
    "write", "python", "function", "given", "return", "returns", "should", "be", "it", "as", "by",
    "if", "from", "on", "are", "list", "string", "integer", "number", "numbers", "input", "output",
}


def normalise_text(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP]


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def normalise_code(code: str) -> str:
    stripped = strip_docstrings_and_comments(code)
    return re.sub(r"\s+", " ", stripped).strip()


@dataclass
class BenchmarkEntry:
    task_id: int
    split: str
    tokens: set[str]
    grams: set[tuple[str, ...]]
    code_norm: str
    entry_point: Optional[str]
    arity: Optional[int]


@dataclass
class ContaminationHit:
    task_id: int
    split: str
    kind: str  # prompt_jaccard | prompt_ngram | solution_exact | signature+prompt
    score: float
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


@dataclass
class ContaminationResult:
    contaminated: bool
    max_prompt_jaccard: float = 0.0
    max_prompt_jaccard_task: Optional[int] = None
    hits: list[ContaminationHit] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contaminated": self.contaminated,
            "max_prompt_jaccard": round(self.max_prompt_jaccard, 4),
            "max_prompt_jaccard_task": self.max_prompt_jaccard_task,
            "hits": [h.to_dict() for h in self.hits],
        }


def _entry_point_of(code: str) -> tuple[Optional[str], Optional[int]]:
    an = analyze_code(code)
    if not an.syntax_ok or not an.functions:
        return None, None
    name = an.functions[-1]
    return name, an.signatures.get(name)


class ContaminationScreener:
    """Pre-indexes MBPP (all splits) and screens KodCode rows against it."""

    def __init__(self, mbpp_rows: Iterable[dict[str, Any]], jaccard_threshold: float = 0.8,
                 ngram: int = 8):
        self.jaccard_threshold = jaccard_threshold
        self.ngram = ngram
        self.entries: list[BenchmarkEntry] = []
        self._code_index: dict[str, BenchmarkEntry] = {}
        for row in mbpp_rows:
            toks = normalise_text(row["text"])
            name, arity = _entry_point_of(row.get("code", ""))
            entry = BenchmarkEntry(
                task_id=int(row["task_id"]), split=row.get("split", "?"), tokens=set(toks),
                grams=ngrams(toks, ngram), code_norm=normalise_code(row.get("code", "")),
                entry_point=name, arity=arity,
            )
            self.entries.append(entry)
            if entry.code_norm:
                self._code_index[entry.code_norm] = entry

    def screen(self, instruction: str, solution: str) -> ContaminationResult:
        toks = normalise_text(instruction)
        tok_set = set(toks)
        grams = ngrams(toks, self.ngram)
        result = ContaminationResult(False)
        sol_norm = normalise_code(solution)
        sol_name, sol_arity = _entry_point_of(solution)

        exact = self._code_index.get(sol_norm)
        if exact is not None and sol_norm:
            result.hits.append(ContaminationHit(exact.task_id, exact.split, "solution_exact", 1.0))

        for e in self.entries:
            j = jaccard(tok_set, e.tokens)
            if j > result.max_prompt_jaccard:
                result.max_prompt_jaccard, result.max_prompt_jaccard_task = j, e.task_id
            if j >= self.jaccard_threshold:
                result.hits.append(ContaminationHit(e.task_id, e.split, "prompt_jaccard", j))
            shared = grams & e.grams
            if shared and len(toks) >= self.ngram:
                result.hits.append(ContaminationHit(
                    e.task_id, e.split, "prompt_ngram", float(len(shared)),
                    detail=" ".join(next(iter(shared))),
                ))
            if (sol_name and sol_name == e.entry_point and sol_arity == e.arity
                    and j >= self.jaccard_threshold / 2):
                result.hits.append(ContaminationHit(e.task_id, e.split, "signature+prompt", j,
                                                    detail=f"{sol_name}/{sol_arity}"))
        result.contaminated = bool(result.hits)
        return result
