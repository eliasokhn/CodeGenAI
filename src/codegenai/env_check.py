"""Environment verification: Python, PyTorch/CUDA, bitsandbytes, PEFT/TRL, Docker sandbox, HF access.

Run ``python scripts/verify_env.py`` (or ``python -m codegenai.env_check``). Exit code 0 means
every *required* check passed; optional checks (GPU, bitsandbytes, MBPP+) only warn.
"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Optional

from codegenai.config import load_config


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def _version(mod: str) -> Optional[str]:
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "present")
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc.__class__.__name__}: {exc}"


def run_checks(cfg=None, check_hub: bool = True) -> list[Check]:
    cfg = cfg or load_config()
    checks: list[Check] = []
    py = sys.version_info
    checks.append(Check("python", py >= (3, 10), f"{platform.python_version()} (3.11 recommended)"))
    checks.append(Check("platform", True, platform.platform(), required=False))

    import torch

    cuda = torch.cuda.is_available()
    detail = f"torch {torch.__version__}, cuda={torch.version.cuda}, available={cuda}"
    if cuda:
        props = torch.cuda.get_device_properties(0)
        cap = torch.cuda.get_device_capability(0)
        detail += (f", gpu={props.name}, vram={props.total_memory / 2**30:.1f} GiB, sm_{cap[0]}{cap[1]}, "
                   f"bf16={torch.cuda.is_bf16_supported()}")
        arch_ok = any(f"sm_{cap[0]}{cap[1]}" in a for a in torch.cuda.get_arch_list())
        if not arch_ok:
            detail += (f" | WARNING: this torch build lacks kernels for sm_{cap[0]}{cap[1]} "
                       f"(arch list {torch.cuda.get_arch_list()}); install a CUDA 12.8+ wheel for Blackwell GPUs")
        checks.append(Check("cuda_kernels_for_gpu", arch_ok, detail, required=False))
    checks.append(Check("torch_cuda", cuda, detail, required=False))

    for mod, req in (("transformers", True), ("peft", True), ("trl", True), ("datasets", True),
                     ("accelerate", True), ("gradio", True), ("mlflow", False), ("pytest", True)):
        v = _version(mod)
        checks.append(Check(mod, not str(v).startswith("ERROR"), str(v), required=req))

    bnb = _version("bitsandbytes")
    bnb_ok = not str(bnb).startswith("ERROR") and cuda
    checks.append(Check("bitsandbytes_4bit", bnb_ok, f"{bnb}; required only for QLoRA (model.load_in_4bit=true)",
                        required=False))

    from codegenai.sandbox import DockerSandbox

    avail = DockerSandbox(cfg.sandbox).check_available(refresh=True)
    checks.append(Check("docker_sandbox", avail.available,
                        f"{avail.reason} (server {avail.server_version or '?'}, image {cfg.sandbox.image})"))

    if check_hub:
        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(cfg.model.model_id, revision=cfg.model.revision)
            checks.append(Check("hf_hub_model", True, f"{cfg.model.model_id} @ {info.sha}", required=False))
        except Exception as exc:  # noqa: BLE001
            checks.append(Check("hf_hub_model", False, f"cannot reach Hub: {exc}", required=False))
    return checks


def pin_model_revision(cfg_path: Optional[str]) -> str:
    """Resolve the current commit of the Qwen model and write it into configs/base.yaml."""
    import re

    from huggingface_hub import HfApi

    from codegenai.config import CONFIG_DIR

    cfg = load_config(cfg_path)
    sha = HfApi().model_info(cfg.model.model_id).sha
    base = CONFIG_DIR / "base.yaml"
    text = base.read_text(encoding="utf-8")
    text = re.sub(r"(\n  revision: )null(\s+#[^\n]*)?", rf"\1{sha}\2", text, count=1)
    base.write_text(text, encoding="utf-8")
    return sha


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Verify the CodeGenAI environment.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--offline", action="store_true", help="skip Hugging Face Hub connectivity check")
    parser.add_argument("--pin-model", action="store_true", help="write the current Qwen commit hash into base.yaml")
    args = parser.parse_args(argv)
    if args.pin_model:
        print("pinned model revision:", pin_model_revision(args.config))
    checks = run_checks(load_config(args.config), check_hub=not args.offline)
    if args.json:
        print(json.dumps([asdict(c) for c in checks], indent=2))
    else:
        for c in checks:
            mark = "OK " if c.ok else ("WARN" if not c.required else "FAIL")
            print(f"[{mark}] {c.name:24s} {c.detail}")
    failed = [c for c in checks if c.required and not c.ok]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
