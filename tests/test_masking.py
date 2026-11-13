"""Completion-only loss masking: fake tokenizer unit tests + real Qwen tokenizer/TRL integration."""
from __future__ import annotations

import pytest

from codegenai.training.masking import IGNORE_INDEX, filter_truncated, tokenize_prompt_completion, verify_labels


class FakeTokenizer:
    """Whitespace tokenizer with a chat template that mimics Qwen's structure."""

    pad_token_id = 0
    eos_token_id = 1
    vocab = {"<pad>": 0, "<|im_end|>": 1, "<|im_start|>": 2}

    def _tok(self, text):
        ids = []
        for w in text.replace("\n", " \n ").split(" "):
            if w == "":
                continue
            ids.append(self.vocab.setdefault(w, len(self.vocab)))
        return ids

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        s = ""
        for m in messages:
            s += f"<|im_start|> {m['role']} \n {m['content']} <|im_end|> \n "
        if add_generation_prompt:
            s += "<|im_start|> assistant \n "
        return self._tok(s) if tokenize else s

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, -1)


def test_tokenize_prompt_completion_masks_prompt_only():
    tok = FakeTokenizer()
    prompt = [{"role": "system", "content": "sys"}, {"role": "user", "content": "write add"}]
    completion = [{"role": "assistant", "content": "```python def add(a, b): return a + b ```"}]
    out = tokenize_prompt_completion(tok, prompt, completion)
    assert out["prompt_is_prefix"] and not out["truncated"]
    n = out["prompt_len"]
    assert all(lab == IGNORE_INDEX for lab in out["labels"][:n])
    assert all(lab != IGNORE_INDEX for lab in out["labels"][n:])
    assert out["labels"][n:] == out["input_ids"][n:]
    # the completion ends with <|im_end|> and that token is supervised
    assert tok.eos_token_id in out["labels"][n:]
    assert verify_labels(out["input_ids"], out["labels"], n, tok.pad_token_id, tok.eos_token_id) == []


def test_verify_labels_detects_problems():
    ids = [5, 6, 7, 8, 1]
    assert "prompt tokens" in verify_labels(ids, [5, -100, 7, 8, 1], 2, 0, 1)[0]
    assert any("completion tokens" in p for p in verify_labels(ids, [-100, -100, 7, -100, 1], 2, 0, 1))
    assert any("end-of-turn" in p for p in verify_labels(ids, [-100, -100, 7, 8, -100], 2, 0, 1))


def test_filter_truncated_drops_long_examples():
    tok = FakeTokenizer()
    short = {"prompt": [{"role": "user", "content": "a b"}], "completion": [{"role": "assistant", "content": "c"}]}
    long = {"prompt": [{"role": "user", "content": " ".join(["w"] * 50)}],
            "completion": [{"role": "assistant", "content": "c"}]}
    kept, stats = filter_truncated([short, long], tok, max_length=30)
    assert kept == [short] and stats["dropped_truncated"] == 1 and stats["kept"] == 1


@pytest.mark.network
def test_qwen_chat_template_masking_and_trl_agreement(cfg):
    """Real Qwen tokenizer: prompt is a strict prefix, completion+<|im_end|> supervised, TRL agrees."""
    from datasets import Dataset
    from transformers import AutoTokenizer

    from codegenai.training.masking import check_prepared_dataset

    tok = AutoTokenizer.from_pretrained(cfg.model.model_id)
    prompt = [{"role": "system", "content": cfg.generation.system_prompt},
              {"role": "user", "content": "Write a function add(a, b) returning the sum."}]
    completion = [{"role": "assistant", "content": "```python\ndef add(a, b):\n    return a + b\n```"}]
    out = tokenize_prompt_completion(tok, prompt, completion, max_length=1024)
    assert out["prompt_is_prefix"], "Qwen template must render the prompt identically with/without completion"
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    sup = [lab for lab in out["labels"] if lab != IGNORE_INDEX]
    assert sup[-2] == im_end or sup[-1] == im_end  # <|im_end|> (then "\n") closes the assistant turn
    decoded_prompt = tok.decode(out["input_ids"][: out["prompt_len"]])
    assert decoded_prompt.endswith("<|im_start|>assistant\n")
    decoded_completion = tok.decode([lab for lab in out["labels"] if lab != IGNORE_INDEX])
    assert decoded_completion.startswith("```python") and "<|im_end|>" in decoded_completion

    # TRL's own prompt-completion preparation (through a real SFTTrainer on a tiny Qwen2 model)
    # must produce exactly the same labels as our independent computation.
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from trl import SFTConfig, SFTTrainer

    ds = Dataset.from_list([{"prompt": prompt, "completion": completion}] * 4)
    tiny = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=2048))
    args = SFTConfig(output_dir=str(cfg.paths.runs_dir / "_mask_check"), max_length=1024, completion_only_loss=True,
                     report_to="none", dataset_num_proc=1, bf16=False, fp16=False, use_cpu=not torch.cuda.is_available(),
                     per_device_train_batch_size=2)
    trainer = SFTTrainer(model=tiny, args=args, train_dataset=ds, processing_class=tok)
    prepared = trainer.train_dataset
    row = prepared[0]
    # and the collator must keep our labels (-100 on prompt/padding) when batching
    batch = trainer.data_collator([prepared[0], prepared[1]])
    assert (batch["labels"][0][: out["prompt_len"]] == -100).all()
    assert int((batch["labels"][0] != -100).sum()) == len(out["labels"]) - out["prompt_len"]
    assert list(row["input_ids"]) == out["input_ids"]
    assert list(row["labels"]) == out["labels"]
    check = check_prepared_dataset(prepared, tok, [{"prompt": prompt, "completion": completion}] * 4, 1024)
    assert check.ok, check.problems
