"""Completion-only loss masking: construction and verification.

TRL's prompt-completion path tokenises ``prompt`` and ``prompt + completion`` with the chat
template and masks everything before the completion. This module re-implements the same
computation independently so we can (a) pre-filter examples whose completion would be
truncated and (b) assert, on the trainer's *actual* prepared dataset, that prompt and padding
tokens carry ``-100`` labels while the full completion (including the end-of-turn token) is
supervised.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

IGNORE_INDEX = -100


def _ids(out) -> list[int]:
    return list(out["input_ids"]) if isinstance(out, dict) or hasattr(out, "keys") else list(out)


def tokenize_prompt_completion(tokenizer, prompt: list[dict[str, str]], completion: list[dict[str, str]],
                               max_length: Optional[int] = None) -> dict[str, Any]:
    """Return input_ids, labels (prompt masked) and bookkeeping for one example."""
    prompt_ids = _ids(tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True))
    full_ids = _ids(tokenizer.apply_chat_template(prompt + completion, tokenize=True))
    # The completion starts where the two tokenisations diverge (normally exactly len(prompt_ids)).
    n = 0
    for a, b in zip(prompt_ids, full_ids):
        if a != b:
            break
        n += 1
    labels = [IGNORE_INDEX] * n + full_ids[n:]
    truncated = False
    if max_length is not None and len(full_ids) > max_length:
        full_ids, labels, truncated = full_ids[:max_length], labels[:max_length], True
    return {
        "input_ids": full_ids, "labels": labels, "prompt_len": n, "completion_len": len(full_ids) - n,
        "prompt_is_prefix": n == len(prompt_ids), "truncated": truncated,
    }


@dataclass
class MaskCheck:
    ok: bool
    n_examples: int
    n_prompt_tokens_supervised: int  # must be 0
    n_pad_tokens_supervised: int  # must be 0
    n_completion_tokens_masked: int  # must be 0
    n_examples_missing_eos: int  # completions must end with the end-of-turn token
    n_truncated: int
    problems: list[str]


def verify_labels(example_input_ids: list[int], example_labels: list[int], expected_prompt_len: int,
                  pad_token_id: Optional[int], eot_token_id: Optional[int]) -> list[str]:
    problems = []
    sup_prompt = sum(1 for lab in example_labels[:expected_prompt_len] if lab != IGNORE_INDEX)
    if sup_prompt:
        problems.append(f"{sup_prompt} prompt tokens are supervised")
    for tid, lab in zip(example_input_ids, example_labels):
        if pad_token_id is not None and tid == pad_token_id and lab != IGNORE_INDEX:
            # pad == eos for Qwen; only a pad *after* the completion's own EOT counts as padding
            pass
    tail = example_labels[expected_prompt_len:]
    masked_in_completion = sum(1 for lab in tail if lab == IGNORE_INDEX)
    if masked_in_completion and any(lab != IGNORE_INDEX for lab in tail):
        # Some masked tokens inside the completion region: allowed only as trailing padding.
        last_sup = max(i for i, lab in enumerate(tail) if lab != IGNORE_INDEX)
        inner = sum(1 for lab in tail[: last_sup + 1] if lab == IGNORE_INDEX)
        if inner:
            problems.append(f"{inner} completion tokens are masked")
    sup = [lab for lab in example_labels if lab != IGNORE_INDEX]
    if eot_token_id is not None and (not sup or eot_token_id not in sup):
        problems.append("end-of-turn token is not supervised (completion truncated?)")
    return problems


def check_prepared_dataset(dataset, tokenizer, raw_examples: list[dict[str, Any]], max_length: int,
                           limit: int = 200) -> MaskCheck:
    """Compare the trainer's prepared ``labels`` with our independent masking computation."""
    eot = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eot = eot if isinstance(eot, int) and eot >= 0 else tokenizer.eos_token_id
    problems: list[str] = []
    n_prompt_sup = n_pad_sup = n_comp_masked = n_missing_eos = n_trunc = 0
    n = min(limit, len(dataset), len(raw_examples))
    for i in range(n):
        row = dataset[i]
        ref = tokenize_prompt_completion(tokenizer, raw_examples[i]["prompt"], raw_examples[i]["completion"],
                                         max_length)
        ids, labels = list(row["input_ids"]), list(row["labels"])
        if ids != ref["input_ids"]:
            problems.append(f"example {i}: trainer input_ids differ from reference tokenisation")
            continue
        probs = verify_labels(ids, labels, ref["prompt_len"], tokenizer.pad_token_id, eot)
        for p in probs:
            if "prompt tokens" in p:
                n_prompt_sup += int(p.split()[0])
            elif "completion tokens" in p:
                n_comp_masked += int(p.split()[0])
            elif "end-of-turn" in p:
                n_missing_eos += 1
            problems.append(f"example {i}: {p}")
        n_trunc += int(ref["truncated"])
    ok = not problems
    return MaskCheck(ok, n, n_prompt_sup, n_pad_sup, n_comp_masked, n_missing_eos, n_trunc, problems[:20])


def filter_truncated(examples: list[dict[str, Any]], tokenizer, max_length: int) -> tuple[list[dict], dict]:
    """Drop examples whose prompt+completion exceeds ``max_length`` (so no completion is cut)."""
    kept, dropped, lengths = [], 0, []
    for ex in examples:
        tk = tokenize_prompt_completion(tokenizer, ex["prompt"], ex["completion"])
        lengths.append(len(tk["input_ids"]))
        if len(tk["input_ids"]) <= max_length and tk["prompt_is_prefix"]:
            kept.append(ex)
        else:
            dropped += 1
    stats = {"kept": len(kept), "dropped_truncated": dropped, "max_length": max_length,
             "max_tokens": max(lengths) if lengths else 0,
             "mean_tokens": round(sum(lengths) / len(lengths), 1) if lengths else 0}
    return kept, stats
