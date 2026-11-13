"""CLI: fine-tune Qwen2.5-Coder with LoRA/QLoRA on an audited KodCode split.

Examples::

    python -m codegenai.training.train --config configs/train/lora_r16_n2000.yaml
    python -m codegenai.training.train --config configs/train/smoke.yaml       # 4-step smoke test
    python -m codegenai.training.train --config configs/train/lora_r8_n5000.yaml \
        -o training.resume_from_checkpoint=auto
"""
from __future__ import annotations

import argparse
import json
from typing import Optional

from codegenai.config import load_config
from codegenai.logging_utils import setup_logging


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA fine-tuning of Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--config", required=True)
    parser.add_argument("-o", "--override", action="append", default=[], help="section.key=value")
    parser.add_argument("--dry-run", action="store_true", help="build the trainer (incl. mask check) but do not train")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.override)
    setup_logging(log_file=cfg.training.output_dir / "train.log")
    if args.dry_run:
        from codegenai.training.sft import build_trainer

        trainer = build_trainer(cfg)
        print(json.dumps({"train_examples": len(trainer.train_dataset), "eval_examples": len(trainer.eval_dataset),
                          "trainable_params": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)},
                         indent=2))
        return
    from codegenai.training.sft import train

    summary = train(cfg)
    print(json.dumps({k: summary[k] for k in ("run_name", "adapter_dir", "train_time_s", "eval_metrics",
                                              "peak_gpu_memory_gb")}, indent=2, default=str))


if __name__ == "__main__":
    main()
