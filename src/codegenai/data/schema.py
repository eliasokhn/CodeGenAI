"""Plain data records shared by the data pipeline, training and evaluation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class KodCodeRecord:
    """One KodCode problem after extraction from the verified dataset + upstream tests."""

    id: str
    uid: str
    subset: str
    style: str
    difficulty: str
    instruction: str  # user turn (problem statement)
    solution: str  # extracted Python source (inside the ```python block)
    assistant_raw: str  # full assistant message as shipped
    system_prompt: str
    test: Optional[str] = None  # upstream pytest file (from KodCode-V1), if retrieved
    test_info: list[dict[str, Any]] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    function_names: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    entry_point: Optional[str] = None  # primary function name the tests exercise
    n_solution_lines: int = 0
    audit: dict[str, Any] = field(default_factory=dict)  # filled during the audit

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KodCodeRecord":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class TrainingExample:
    """Prompt/completion pair in TRL's conversational prompt-completion format."""

    id: str
    prompt: list[dict[str, str]]  # [system, user]
    completion: list[dict[str, str]]  # [assistant]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MBPPProblem:
    """An MBPP problem with an explicit visible/hidden test split."""

    task_id: int
    text: str
    reference_code: str
    entry_point: str
    visible_tests: list[str]  # asserts shown to the model (prompt + correction feedback)
    hidden_tests: list[str]  # asserts used ONLY for the final verdict
    setup_code: str = ""
    challenge_tests: list[str] = field(default_factory=list)
    plus_test: Optional[str] = None  # MBPP+ extended test script (hidden)
    split: str = "test"
    source: str = "google-research-datasets/mbpp"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- JSONL I/O
def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
