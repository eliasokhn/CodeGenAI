"""Convert audited KodCode records into TRL prompt-completion chat examples.

The exact same system prompt and user-turn layout is used at training time and at inference
time (``codegenai.model.prompts``), so the fine-tuned adapter is evaluated in-distribution.
"""
from __future__ import annotations

from typing import Any, Optional

from codegenai.data.schema import KodCodeRecord, TrainingExample
from codegenai.model.prompts import build_generation_messages, format_completion


def record_to_example(rec: KodCodeRecord, system_prompt: str,
                      visible_tests: Optional[list[str]] = None) -> TrainingExample:
    """Build the (prompt, completion) pair for one record.

    ``visible_tests`` (optional upstream KodCode asserts) are rendered with the very same
    template the MBPP evaluation prompt uses, so the adapter sees both prompt forms in training.
    The completion is the reference solution wrapped in a single ```python block – identical to
    the shipped assistant turn's shape – and ends with the chat template's end-of-turn token
    (added by the tokenizer's chat template, not here).
    """
    messages = build_generation_messages(rec.instruction, visible_tests or None, system_prompt=system_prompt)
    completion = [{"role": "assistant", "content": format_completion(rec.solution)}]
    meta: dict[str, Any] = {
        "subset": rec.subset, "difficulty": rec.difficulty, "style": rec.style,
        "entry_point": rec.entry_point, "n_solution_lines": rec.n_solution_lines,
        "n_prompt_tests": len(visible_tests or []),
    }
    return TrainingExample(id=rec.id, prompt=messages, completion=completion, meta=meta)
