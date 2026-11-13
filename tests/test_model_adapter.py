"""LoRA adapter construction/loading on a tiny randomly initialised Qwen2 model (no download)."""
from __future__ import annotations

import pytest
import torch

from codegenai.config import LoraConfig as LoraSection
from codegenai.config import ModelConfig
from codegenai.model.generation import MockCodeGenerator
from codegenai.model.loader import adapter_metadata, attach_adapter, count_parameters, resolve_dtype


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    return Qwen2ForCausalLM(cfg).eval()


def test_lora_targets_q_v_only_and_freezes_base(tiny_model, tmp_path):
    from peft import LoraConfig, get_peft_model

    section = LoraSection()
    pm = get_peft_model(tiny_model, LoraConfig(r=section.r, lora_alpha=section.alpha, lora_dropout=section.dropout,
                                               target_modules=section.target_modules, task_type="CAUSAL_LM"))
    trainable = [n for n, p in pm.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    assert all(("q_proj" in n or "v_proj" in n) for n in trainable)
    assert not any("k_proj" in n or "o_proj" in n or "mlp" in n for n in trainable)
    counts = count_parameters(pm)
    assert 0 < counts["trainable"] < 0.1 * counts["total"]
    pm.save_pretrained(str(tmp_path / "adapter"))
    meta = adapter_metadata(tmp_path / "adapter")
    assert meta["r"] == 16 and meta["lora_alpha"] == 32 and set(meta["target_modules"]) == {"q_proj", "v_proj"}


def test_attach_adapter_changes_outputs_only_when_weights_nonzero(tiny_model, tmp_path):
    from peft import LoraConfig, get_peft_model

    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        base_logits = tiny_model(ids).logits.clone()
    pm = get_peft_model(tiny_model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    with torch.no_grad():
        assert torch.allclose(pm(ids).logits, base_logits, atol=1e-5)  # B initialised to zero -> identity
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.add_(torch.randn_like(p) * 0.5)
        changed = pm(ids).logits
    assert not torch.allclose(changed, base_logits, atol=1e-3)
    pm.save_pretrained(str(tmp_path / "adapter2"))
    base = pm.unload() if hasattr(pm, "unload") else tiny_model
    reloaded = attach_adapter(base, tmp_path / "adapter2")
    with torch.no_grad():
        assert torch.allclose(reloaded(ids).logits, changed, atol=1e-4)
    with pytest.raises(FileNotFoundError):
        attach_adapter(base, tmp_path / "nope")


def test_resolve_dtype_cpu_fallback():
    if not torch.cuda.is_available():
        assert resolve_dtype("bfloat16") == torch.float32
    assert resolve_dtype("float32") == torch.float32 and resolve_dtype("auto") == "auto"


def test_quantization_config_disabled_without_cuda():
    from codegenai.model.loader import build_quantization_config

    q = build_quantization_config(ModelConfig(load_in_4bit=True))
    if not torch.cuda.is_available():
        assert q is None
    assert build_quantization_config(ModelConfig(load_in_4bit=False)) is None


def test_mock_generator_is_flagged_and_refused_for_research(cfg, tmp_path):
    from codegenai.evaluation.mbpp_eval import MBPPEvaluator
    from codegenai.sandbox import DisabledSandbox

    gen = MockCodeGenerator(lambda m: "```python\nx=1\n```")
    assert getattr(gen, "is_mock", False)
    ev = MBPPEvaluator(cfg, sandbox=DisabledSandbox("test"))
    with pytest.raises(RuntimeError):
        ev.evaluate_configuration("A", [], gen, tmp_path)
