"""Load Qwen2.5-Coder-1.5B-Instruct (optionally 4-bit) with or without a LoRA adapter."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import torch

from codegenai.config import ModelConfig, resolve_path
from codegenai.logging_utils import get_logger

log = get_logger(__name__)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def resolve_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    dt = _DTYPES[name]
    if dt == torch.bfloat16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        log.warning("bf16 not supported on this GPU; falling back to fp16")
        return torch.float16
    if dt in (torch.bfloat16, torch.float16) and not torch.cuda.is_available():
        log.warning("No CUDA device: using float32 on CPU instead of %s", name)
        return torch.float32
    return dt


def bitsandbytes_available() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    try:
        import bitsandbytes as bnb  # noqa: F401

        return True, f"bitsandbytes {bnb.__version__}"
    except Exception as exc:  # noqa: BLE001
        return False, f"bitsandbytes import failed: {exc}"


def build_quantization_config(cfg: ModelConfig):
    """BitsAndBytesConfig for QLoRA (NF4 + double quant), or None when disabled/unavailable."""
    if not cfg.load_in_4bit:
        return None
    ok, why = bitsandbytes_available()
    if not ok:
        log.warning("4-bit loading requested but unavailable (%s); loading in %s instead.", why, cfg.torch_dtype)
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=_DTYPES[cfg.bnb_4bit_compute_dtype],
    )


def load_tokenizer(cfg: ModelConfig):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.revision, trust_remote_code=cfg.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # batched generation with decoder-only models
    return tok


def load_base_model(cfg: ModelConfig, for_training: bool = False):
    """Load the frozen pretrained Qwen model (quantised if configured)."""
    from transformers import AutoModelForCausalLM

    quant = build_quantization_config(cfg)
    dtype = resolve_dtype(cfg.torch_dtype)
    kwargs: dict[str, Any] = {
        "revision": cfg.revision,
        "trust_remote_code": cfg.trust_remote_code,
        "dtype": dtype,
    }
    if quant is not None:
        kwargs["quantization_config"] = quant
    if cfg.attn_implementation:
        kwargs["attn_implementation"] = cfg.attn_implementation
    if cfg.device_map and torch.cuda.is_available():
        kwargs["device_map"] = cfg.device_map
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
    if not for_training:
        model.eval()
    if cfg.device_map is None or not torch.cuda.is_available():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
    log.info("Loaded %s (%s, 4bit=%s) in %.1fs", cfg.model_id, dtype, quant is not None, time.time() - t0)
    return model


def attach_adapter(model, adapter_path: Path | str, is_trainable: bool = False):
    """Load a trained LoRA adapter on top of the base model (weights stay frozen)."""
    from peft import PeftModel

    adapter_path = resolve_path(adapter_path)
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"No adapter_config.json in {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=is_trainable)
    if not is_trainable:
        model.eval()
    log.info("Attached LoRA adapter from %s", adapter_path)
    return model


def load_model_and_tokenizer(cfg: ModelConfig, adapter_path: Optional[Path | str] = None):
    tok = load_tokenizer(cfg)
    model = load_base_model(cfg)
    adapter = adapter_path or cfg.adapter_path
    if adapter:
        model = attach_adapter(model, adapter)
    return model, tok


def adapter_metadata(adapter_path: Path | str) -> dict[str, Any]:
    p = resolve_path(adapter_path) / "adapter_config.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    return {k: cfg.get(k) for k in ("base_model_name_or_path", "r", "lora_alpha", "lora_dropout", "target_modules")}


def count_parameters(model) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
