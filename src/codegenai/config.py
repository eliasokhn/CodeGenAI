"""Typed, YAML-backed configuration for every CodeGenAI component.

Configs are plain YAML files. A file may declare ``extends: <relative path>`` to inherit
from another file; the child values are deep-merged over the parent. Command-line
overrides use dotted keys (``training.learning_rate=5e-5``).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field

from codegenai import MODEL_ID

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


# --------------------------------------------------------------------------- sections
class PathsConfig(BaseModel):
    data_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    audit_dir: Path = Path("data/audit")
    runs_dir: Path = Path("runs")
    results_dir: Path = Path("results")
    hf_cache_dir: Optional[Path] = None  # None -> default HF cache (~/.cache/huggingface)


class ModelConfig(BaseModel):
    model_id: str = MODEL_ID
    revision: Optional[str] = None  # pin a HF Hub commit for reproducibility
    torch_dtype: Literal["auto", "bfloat16", "float16", "float32"] = "bfloat16"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    attn_implementation: Optional[str] = None  # e.g. "sdpa", "flash_attention_2"
    device_map: Optional[str] = "auto"
    adapter_path: Optional[Path] = None  # LoRA adapter directory to load (None = base model)
    trust_remote_code: bool = False


class GenerationConfig(BaseModel):
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0
    system_prompt: str = (
        "You are a Python coding assistant. Write correct code. Respond with a single code block."
    )


class KodCodeFilters(BaseModel):
    styles: list[str] = ["instruct"]
    require_verification_passed: bool = True
    require_constrains: bool = True
    require_value_assert: bool = True
    min_asserts: int = 2
    max_eval_similarity: float = 0.6  # stricter than the publisher's 0.70 cut-off
    exclude_subsets: list[str] = []
    exclude_test_shapes: list[str] = [
        "uses_mock", "async_test", "pytest_fixture", "class_based", "setup_module",
    ]
    max_solution_lines: int = 120
    max_solution_chars: int = 4000
    max_prompt_chars: int = 3000
    allowed_imports: list[str] = [
        "collections", "typing", "math", "re", "heapq", "itertools", "functools", "bisect",
        "string", "operator", "random", "datetime", "json", "copy", "statistics", "fractions",
        "decimal", "enum", "dataclasses", "abc", "array", "queue", "sys", "time", "numbers",
        "cmath", "textwrap", "unicodedata", "struct", "calendar", "hashlib", "base64", "zlib",
        "io", "contextlib", "types", "inspect", "ast",
    ]
    disallowed_imports: list[str] = [
        "os", "subprocess", "socket", "requests", "urllib", "http", "shutil", "pathlib",
        "threading", "multiprocessing", "asyncio", "ctypes", "pickle", "importlib", "signal",
        "tempfile", "glob", "flask", "django", "torch", "sklearn", "numpy", "pandas", "scipy",
        "seaborn", "matplotlib", "cv2", "PIL", "networkx", "sympy", "pyarrow", "psutil",
    ]


class DataConfig(BaseModel):
    kodcode_id: str = "F-A-I-L/kodcode-verified-python-235k"
    kodcode_revision: str = "4ba3098799a405943ce4c74a3449a80e4c21841d"
    kodcode_license: str = (
        "CC BY-NC 4.0 (inherited from KodCode/KodCode-V1); non-commercial, attribution required"
    )
    upstream_id: str = "KodCode/KodCode-V1"
    upstream_revision: str = "97d81f0695ddd7c5c02bfe16436db9a17a4a21e5"
    mbpp_id: str = "google-research-datasets/mbpp"
    mbpp_revision: str = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
    mbpp_config: Literal["full", "sanitized"] = "full"
    mbppplus_id: str = "evalplus/mbppplus"
    mbppplus_revision: str = "b2d74c91837c3f2a20c1299ae98133cbe7cfa077"
    candidate_pool_size: int = 30000  # rows scanned (streaming) before filtering
    filters: KodCodeFilters = KodCodeFilters()
    train_sizes: list[int] = [2000, 5000]
    validation_size: int = 300
    split_seed: int = 42


class AuditConfig(BaseModel):
    sample_size: int = 250  # representative sample executed in the sandbox
    mutation_sample_size: int = 60  # subset of the sample used for mutation checks
    execute_in_sandbox: bool = True
    contamination_threshold: float = 0.8  # token-Jaccard on normalised prompt text
    contamination_ngram: int = 8  # shared 8-gram between KodCode prompt and MBPP text
    stratify_by: list[str] = ["subset", "difficulty"]


class LoraConfig(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = ["q_proj", "v_proj"]
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"


class TrainingConfig(BaseModel):
    run_name: str = "lora_r16_n2000"
    train_file: Path = Path("data/processed/train_2000.jsonl")
    validation_file: Path = Path("data/processed/validation.jsonl")
    output_dir: Path = Path("runs/lora_r16_n2000")
    learning_rate: float = 1e-4
    num_train_epochs: float = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 1024
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    seed: int = 42
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    optim: str = "paged_adamw_8bit"
    resume_from_checkpoint: Optional[str] = None  # path or "auto"
    max_steps: int = -1  # >0 overrides epochs (used by smoke tests)
    drop_truncated_completions: bool = True
    report_to: list[str] = ["mlflow"]


class SandboxConfig(BaseModel):
    image: str = "codegenai-sandbox:latest"
    timeout_seconds: int = 20
    per_test_timeout_seconds: int = 5
    memory: str = "256m"
    cpus: float = 1.0
    pids_limit: int = 64
    tmpfs_size: str = "64m"
    max_output_bytes: int = 65536
    docker_host: Optional[str] = None  # None -> environment default
    python_bin: str = "python"


class CorrectionConfig(BaseModel):
    max_attempts: int = 2
    max_feedback_chars: int = 2500
    stop_on_identical_code: bool = True


class EvaluationConfig(BaseModel):
    split: Literal["validation", "test", "prompt", "train"] = "test"
    limit: Optional[int] = None  # evaluate only the first N problems (debug)
    visible_tests: Literal["first", "none"] = "first"  # feedback tests exposed to the model
    hidden_test_policy: Literal["all", "disjoint"] = "all"
    include_challenge_tests: bool = False
    use_mbppplus: bool = False
    configurations: list[str] = ["A", "B", "C", "D"]
    adapter_path: Optional[Path] = None  # LoRA adapter for configurations B and D
    output_dir: Path = Path("results/mbpp_eval")
    bootstrap_samples: int = 2000


class MLflowConfig(BaseModel):
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "codegenai"
    enabled: bool = True


class AppConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False
    adapter_path: Optional[Path] = None
    load_in_4bit: Optional[bool] = None  # None -> follow model.load_in_4bit
    enable_execution: bool = True


class Config(BaseModel):
    project_name: str = "CodeGenAI"
    paths: PathsConfig = PathsConfig()
    model: ModelConfig = ModelConfig()
    generation: GenerationConfig = GenerationConfig()
    data: DataConfig = DataConfig()
    audit: AuditConfig = AuditConfig()
    lora: LoraConfig = LoraConfig()
    training: TrainingConfig = TrainingConfig()
    sandbox: SandboxConfig = SandboxConfig()
    correction: CorrectionConfig = CorrectionConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    mlflow: MLflowConfig = MLflowConfig()
    app: AppConfig = AppConfig()
    source_files: list[str] = Field(default_factory=list)

    def dump(self) -> dict[str, Any]:
        """JSON-serialisable dict of the resolved configuration."""
        return json.loads(self.model_dump_json())


# --------------------------------------------------------------------------- loading
def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml_chain(path: Path, seen: Optional[set[Path]] = None) -> tuple[dict, list[str]]:
    path = Path(path).resolve()
    seen = seen or set()
    if path in seen:
        raise ValueError(f"Circular 'extends' chain at {path}")
    seen.add(path)
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} must be a mapping")
    parent_files: list[str] = []
    merged: dict = {}
    if "extends" in raw:
        parent = (path.parent / raw.pop("extends")).resolve()
        merged, parent_files = _load_yaml_chain(parent, seen)
    merged = _deep_merge(merged, raw)
    return merged, parent_files + [str(path)]


def _coerce(value: str) -> Any:
    """Parse a CLI override value with YAML semantics (so 1e-4, true, [a,b] work)."""
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        return value
    if isinstance(parsed, str):
        # YAML 1.1 (PyYAML) does not treat "5e-5" as a float; Python does.
        try:
            return float(parsed) if any(ch in parsed for ch in "eE.") else int(parsed)
        except ValueError:
            return parsed
    return parsed


def apply_overrides(data: dict, overrides: list[str]) -> dict:
    data = copy.deepcopy(data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like section.key=value")
        key, value = item.split("=", 1)
        node = data
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot override '{key}': '{p}' is not a mapping")
        node[parts[-1]] = _coerce(value)
    return data


def load_config(path: Optional[str | Path] = None, overrides: Optional[list[str]] = None) -> Config:
    """Load a YAML config (default: configs/base.yaml) with inheritance and CLI overrides."""
    path = Path(path) if path is not None else CONFIG_DIR / "base.yaml"
    if not path.is_absolute() and not path.exists():
        candidate = CONFIG_DIR / path
        if candidate.exists():
            path = candidate
    data, files = _load_yaml_chain(path)
    data = apply_overrides(data, overrides or [])
    data["source_files"] = files
    return Config.model_validate(data)


def resolve_path(p: Path | str) -> Path:
    """Resolve relative project paths against the repository root."""
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
