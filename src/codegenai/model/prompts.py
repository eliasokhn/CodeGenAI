"""Prompt templates shared by training, inference, evaluation and self-correction.

Keeping these in one place guarantees that the fine-tuned adapter sees exactly the format it
was trained on, and that the correction prompt can never include hidden tests (it only accepts
a pre-filtered feedback string built by ``codegenai.correction.feedback``).
"""
from __future__ import annotations

from typing import Optional

DEFAULT_SYSTEM_PROMPT = (
    "You are a Python coding assistant. Write correct code. Respond with a single code block."
)

USER_TEMPLATE = "{instruction}"

USER_WITH_TESTS_TEMPLATE = (
    "{instruction}\n\n"
    "Your code should satisfy the following test(s):\n"
    "```python\n{tests}\n```"
)

CORRECTION_TEMPLATE = (
    "The previous solution to the task below is incorrect.\n\n"
    "## Task\n{instruction}\n\n"
    "## Previous code\n```python\n{previous_code}\n```\n\n"
    "## Test feedback\n{feedback}\n\n"
    "Fix the code so that it passes the tests. Keep the same function name and signature. "
    "Respond with a single complete ```python code block containing the full corrected module."
)


def build_generation_messages(instruction: str, visible_tests: Optional[list[str]] = None,
                              system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> list[dict[str, str]]:
    if visible_tests:
        user = USER_WITH_TESTS_TEMPLATE.format(instruction=instruction.strip(), tests="\n".join(visible_tests))
    else:
        user = USER_TEMPLATE.format(instruction=instruction.strip())
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]


def build_correction_messages(instruction: str, previous_code: str, feedback: str,
                              visible_tests: Optional[list[str]] = None,
                              system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> list[dict[str, str]]:
    task = instruction.strip()
    if visible_tests:
        task += "\n\nThe code must satisfy:\n```python\n" + "\n".join(visible_tests) + "\n```"
    user = CORRECTION_TEMPLATE.format(instruction=task, previous_code=previous_code.rstrip(), feedback=feedback)
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]


def format_completion(code: str) -> str:
    """Assistant turn format used for SFT targets: exactly one fenced python block."""
    return "```python\n" + code.rstrip("\n") + "\n```"
