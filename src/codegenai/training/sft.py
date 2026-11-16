"""SFTTrainer + PEFT LoRA setup, GPU-memory/MLflow callbacks and checkpoint handling."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import torch

from codegenai.config import Config, resolve_path
from codegenai.data.schema import read_jsonl, write_json
from codegenai.logging_utils import get_logger
from codegenai.model.loader import count_parameters, load_base_model, load_tokenizer
from codegenai.training.masking import check_prepared_dataset, filter_truncated

log = get_logger(__name__)


# --------------------------------------------------------------------------- data
def load_examples(path: Path) -> list[dict[str, Any]]:
    rows = []
    for r in read_jsonl(path):
        rows.append({"prompt": r["prompt"], "completion": r["completion"], "id": r.get("id")})
    return rows


def to_hf_dataset(rows: list[dict[str, Any]]):
    from datasets import Dataset

    return Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in rows])


# --------------------------------------------------------------------------- callbacks
class GPUMemoryCallback:
    """Logs peak/allocated CUDA memory with every logging step (TrainerCallback duck-typed)."""

    def __init__(self):
        from transformers import TrainerCallback

        class _CB(TrainerCallback):
            def on_log(self_inner, args, state, control, logs=None, **kwargs):  # noqa: N805
                if torch.cuda.is_available() and logs is not None:
                    logs["gpu_mem_allocated_gb"] = round(torch.cuda.memory_allocated() / 2**30, 3)
                    logs["gpu_mem_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
                    logs["gpu_mem_reserved_gb"] = round(torch.cuda.memory_reserved() / 2**30, 3)

        self.cb = _CB()


class JsonLogCallback:
    """Appends every log/eval record to ``<output_dir>/train_log.jsonl`` for the dashboard."""

    def __init__(self, path: Path):
        from transformers import TrainerCallback

        path.parent.mkdir(parents=True, exist_ok=True)

        class _CB(TrainerCallback):
            def on_log(self_inner, args, state, control, logs=None, **kwargs):  # noqa: N805
                if logs is None:
                    return
                rec = {"step": state.global_step, "epoch": round(state.epoch or 0, 4), "time": time.time(), **logs}
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")

        self.cb = _CB()


# --------------------------------------------------------------------------- MLflow
def setup_mlflow(cfg: Config) -> bool:
    if not cfg.mlflow.enabled or "mlflow" not in cfg.training.report_to:
        return False
    uri = cfg.mlflow.tracking_uri
    if uri.startswith("file:./") or uri.startswith("file:."):
        uri = "file:" + str(resolve_path(uri.split("file:", 1)[1]))
    os.environ["MLFLOW_TRACKING_URI"] = uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg.mlflow.experiment_name
    os.environ.setdefault("MLFLOW_FLATTEN_PARAMS", "1")
    os.environ.setdefault("HF_MLFLOW_LOG_ARTIFACTS", "0")
    return True


# --------------------------------------------------------------------------- version compat
def _training_arg_names() -> set[str]:
    import dataclasses

    from transformers import TrainingArguments

    return {f.name for f in dataclasses.fields(TrainingArguments)}


def _warmup_kwargs(ratio: float) -> dict[str, Any]:
    """transformers<5 uses ``warmup_ratio``; v5 folds the ratio into a float ``warmup_steps``."""
    names = _training_arg_names()
    if "warmup_ratio" in names:
        return {"warmup_ratio": ratio}
    return {"warmup_steps": float(ratio)}


def _checkpointing_kwargs(enabled: bool) -> dict[str, Any]:
    """transformers<5: ``gradient_checkpointing``; v5: ``activation_checkpointing``."""
    names = _training_arg_names()
    kw: dict[str, Any] = {"use_reentrant": False}
    if "activation_checkpointing" in names:
        return {"activation_checkpointing": enabled, "activation_checkpointing_kwargs": kw if enabled else None}
    return {"gradient_checkpointing": enabled, "gradient_checkpointing_kwargs": kw if enabled else None}


# --------------------------------------------------------------------------- trainer
def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists():
        return None
    cks = sorted((p for p in output_dir.glob("checkpoint-*") if p.is_dir()),
                 key=lambda p: int(p.name.split("-")[-1]))
    return cks[-1] if cks else None


def build_trainer(cfg: Config, model=None, tokenizer=None, train_rows=None, eval_rows=None):
    """Create the SFTTrainer with LoRA adapters and completion-only loss."""
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    tcfg, lcfg = cfg.training, cfg.lora
    out_dir = resolve_path(tcfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = tokenizer or load_tokenizer(cfg.model)
    tokenizer.padding_side = "right"  # training: right padding
    train_rows = train_rows if train_rows is not None else load_examples(resolve_path(tcfg.train_file))
    eval_rows = eval_rows if eval_rows is not None else load_examples(resolve_path(tcfg.validation_file))

    # Never train on truncated completions: drop over-long examples up-front.
    if tcfg.drop_truncated_completions:
        train_rows, tstats = filter_truncated(train_rows, tokenizer, tcfg.max_seq_length)
        eval_rows, estats = filter_truncated(eval_rows, tokenizer, tcfg.max_seq_length)
        log.info("Length filter train=%s eval=%s", tstats, estats)
        write_json(out_dir / "length_filter.json", {"train": tstats, "eval": estats})
    train_ds, eval_ds = to_hf_dataset(train_rows), to_hf_dataset(eval_rows)

    if model is None:
        model = load_base_model(cfg.model, for_training=True)
    quantized = getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)
    if quantized:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=tcfg.gradient_checkpointing)
    else:
        for p in model.parameters():  # freeze pretrained weights explicitly
            p.requires_grad_(False)
        if tcfg.gradient_checkpointing:
            if hasattr(model, "activation_checkpointing_enable"):
                model.activation_checkpointing_enable(activation_checkpointing_kwargs={"use_reentrant": False})
            else:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=lcfg.r, lora_alpha=lcfg.alpha, lora_dropout=lcfg.dropout, target_modules=list(lcfg.target_modules),
        bias=lcfg.bias, task_type=lcfg.task_type,
    )
    use_bf16 = tcfg.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = tcfg.fp16 and torch.cuda.is_available() and not use_bf16
    optim = tcfg.optim
    if "8bit" in optim or "paged" in optim:
        try:
            import bitsandbytes  # noqa: F401
        except Exception:  # noqa: BLE001
            log.warning("optimizer %s needs bitsandbytes; falling back to adamw_torch", optim)
            optim = "adamw_torch"
    report_to = list(tcfg.report_to) if setup_mlflow(cfg) or "mlflow" not in tcfg.report_to else \
        [r for r in tcfg.report_to if r != "mlflow"]

    args = SFTConfig(
        output_dir=str(out_dir), run_name=tcfg.run_name,
        learning_rate=tcfg.learning_rate, num_train_epochs=tcfg.num_train_epochs, max_steps=tcfg.max_steps,
        per_device_train_batch_size=tcfg.per_device_train_batch_size,
        per_device_eval_batch_size=tcfg.per_device_eval_batch_size,
        gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
        **_checkpointing_kwargs(tcfg.gradient_checkpointing and not quantized),
        bf16=use_bf16, fp16=use_fp16, seed=tcfg.seed, data_seed=tcfg.seed,
        lr_scheduler_type=tcfg.lr_scheduler_type, **_warmup_kwargs(tcfg.warmup_ratio),
        use_cpu=not torch.cuda.is_available(),
        weight_decay=tcfg.weight_decay, max_grad_norm=tcfg.max_grad_norm, optim=optim,
        logging_steps=tcfg.logging_steps, logging_first_step=True,
        eval_strategy="steps", eval_steps=tcfg.eval_steps,
        save_strategy="steps", save_steps=tcfg.save_steps, save_total_limit=tcfg.save_total_limit,
        load_best_model_at_end=tcfg.load_best_model_at_end, metric_for_best_model=tcfg.metric_for_best_model,
        greater_is_better=False, report_to=report_to or "none",
        max_length=tcfg.max_seq_length, packing=False, completion_only_loss=True,
        dataset_num_proc=None, remove_unused_columns=True, dataloader_pin_memory=torch.cuda.is_available(),
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tokenizer, peft_config=peft_config,
    )
    trainer.add_callback(GPUMemoryCallback().cb)
    trainer.add_callback(JsonLogCallback(out_dir / "train_log.jsonl").cb)
    if tcfg.early_stopping_patience > 0 and tcfg.load_best_model_at_end:
        from transformers import EarlyStoppingCallback

        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=tcfg.early_stopping_patience))

    params = count_parameters(trainer.model)
    log.info("Trainable params: %s / %s (%.3f%%)", f"{params['trainable']:,}", f"{params['total']:,}",
             100 * params["trainable"] / max(1, params["total"]))
    # Sanity: only LoRA parameters are trainable.
    bad = [n for n, p in trainer.model.named_parameters() if p.requires_grad and "lora_" not in n]
    if bad:
        raise RuntimeError(f"Non-LoRA parameters are trainable: {bad[:5]}")

    # Verify completion-only masking on the trainer's actual prepared dataset.
    mask = check_prepared_dataset(trainer.train_dataset, tokenizer, train_rows, tcfg.max_seq_length)
    write_json(out_dir / "mask_check.json", mask.__dict__)
    if not mask.ok:
        raise RuntimeError(f"Loss-mask verification failed: {mask.problems[:3]}")
    log.info("Loss mask verified on %d examples: prompt tokens supervised=%d, completion tokens masked=%d, "
             "missing EOT=%d", mask.n_examples, mask.n_prompt_tokens_supervised, mask.n_completion_tokens_masked,
             mask.n_examples_missing_eos)
    return trainer


def train(cfg: Config) -> dict[str, Any]:
    tcfg = cfg.training
    out_dir = resolve_path(tcfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config.json", cfg.dump())
    trainer = build_trainer(cfg)

    resume = None
    if tcfg.resume_from_checkpoint:
        resume = find_latest_checkpoint(out_dir) if tcfg.resume_from_checkpoint == "auto" \
            else resolve_path(tcfg.resume_from_checkpoint)
        if resume is not None and not Path(resume).exists():
            raise FileNotFoundError(f"checkpoint {resume} not found")
        log.info("Resuming from %s", resume)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    elapsed = time.time() - t0
    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    trainer.processing_class.save_pretrained(str(adapter_dir))
    eval_metrics = trainer.evaluate()
    summary = {
        "run_name": tcfg.run_name, "adapter_dir": str(adapter_dir), "train_time_s": round(elapsed, 1),
        "train_metrics": result.metrics, "eval_metrics": eval_metrics,
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3) if torch.cuda.is_available() else None,
        "lora": cfg.lora.model_dump(), "training": json.loads(cfg.training.model_dump_json()),
        "model": cfg.model.model_dump(), "seed": tcfg.seed,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "global_step": trainer.state.global_step,
    }
    write_json(out_dir / "summary.json", summary)
    log.info("Training finished in %.0fs; adapter saved to %s; eval_loss=%.4f", elapsed, adapter_dir,
             eval_metrics.get("eval_loss", float("nan")))
    return summary
