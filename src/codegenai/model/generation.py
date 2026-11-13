"""Deterministic code generation with latency and GPU-memory accounting.

``CodeGenerator`` wraps a (base or LoRA) causal LM and exposes ``generate(messages)``.
``MockCodeGenerator`` implements the same interface for software tests only; it is never used
for research results (the evaluation CLI refuses it).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional, Protocol

import torch

from codegenai.config import GenerationConfig, ModelConfig
from codegenai.data.extract import extract_code
from codegenai.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class GenerationResult:
    raw_text: str
    code: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    truncated: bool  # hit max_new_tokens without emitting EOS
    peak_gpu_memory_mb: Optional[float] = None
    model_name: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Generator(Protocol):
    name: str

    def generate(self, messages: list[dict[str, str]]) -> GenerationResult: ...


class CodeGenerator:
    """Greedy (``do_sample=False``) generation with the official Qwen chat template."""

    def __init__(self, model, tokenizer, gen_cfg: GenerationConfig, name: str = "qwen"):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = gen_cfg
        self.name = name
        self.device = next(model.parameters()).device
        # Qwen2.5 emits <|im_end|> at the end of a turn; also stop on <|endoftext|>.
        eos = {tokenizer.eos_token_id}
        for tok in ("<|im_end|>", "<|endoftext|>"):
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                eos.add(tid)
        self.eos_token_ids = sorted(t for t in eos if t is not None)

    def _generation_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.cfg.max_new_tokens,
            "do_sample": self.cfg.do_sample,
            "repetition_penalty": self.cfg.repetition_penalty,
            "eos_token_id": self.eos_token_ids,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.cfg.do_sample:
            if self.cfg.temperature is not None:
                kwargs["temperature"] = self.cfg.temperature
            if self.cfg.top_p is not None:
                kwargs["top_p"] = self.cfg.top_p
        else:
            # Explicitly neutralise sampling params so HF does not warn / silently sample.
            kwargs.update({"temperature": None, "top_p": None, "top_k": None})
        return kwargs

    @torch.inference_mode()
    def generate(self, messages: list[dict[str, str]]) -> GenerationResult:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        cuda = self.device.type == "cuda"
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(self.device)
        t0 = time.perf_counter()
        out = self.model.generate(**enc, **self._generation_kwargs())
        if cuda:
            torch.cuda.synchronize()
        latency = time.perf_counter() - t0
        new_tokens = out[0, enc["input_ids"].shape[1]:]
        truncated = len(new_tokens) >= self.cfg.max_new_tokens and int(new_tokens[-1]) not in self.eos_token_ids
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        peak = torch.cuda.max_memory_allocated(self.device) / 2**20 if cuda else None
        return GenerationResult(
            raw_text=text, code=extract_code(text), latency_s=latency,
            prompt_tokens=int(enc["input_ids"].shape[1]), completion_tokens=int(len(new_tokens)),
            truncated=truncated, peak_gpu_memory_mb=peak, model_name=self.name,
        )


class MockCodeGenerator:
    """Deterministic stand-in for unit tests. ``responder`` maps the last user message to text."""

    is_mock = True

    def __init__(self, responder: Callable[[list[dict[str, str]]], str], name: str = "mock"):
        self.responder = responder
        self.name = name
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> GenerationResult:
        self.calls.append(messages)
        t0 = time.perf_counter()
        text = self.responder(messages)
        return GenerationResult(
            raw_text=text, code=extract_code(text), latency_s=time.perf_counter() - t0,
            prompt_tokens=sum(len(m["content"]) // 4 for m in messages), completion_tokens=len(text) // 4,
            truncated=False, model_name=self.name, meta={"mock": True},
        )


def build_generator(model_cfg: ModelConfig, gen_cfg: GenerationConfig, adapter_path=None,
                    name: Optional[str] = None) -> CodeGenerator:
    from codegenai.model.loader import load_model_and_tokenizer

    model, tok = load_model_and_tokenizer(model_cfg, adapter_path)
    label = name or ("qwen-lora" if (adapter_path or model_cfg.adapter_path) else "qwen-base")
    return CodeGenerator(model, tok, gen_cfg, name=label)
