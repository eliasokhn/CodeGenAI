"""Application services: lazily loaded models, sandbox access and result-file readers.

The Gradio UI only talks to this layer, which in turn uses the real project components. No
model output or metric is ever fabricated here: when a model/adapter/result file is missing the
service reports that explicitly.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

from codegenai.config import Config, resolve_path
from codegenai.correction import SelfCorrectionLoop
from codegenai.data.extract import check_syntax
from codegenai.data.schema import read_jsonl
from codegenai.logging_utils import get_logger
from codegenai.model.generation import Generator
from codegenai.sandbox import Status, TestReport, get_sandbox
from codegenai.sandbox.harness import build_assert_module

log = get_logger(__name__)


def parse_user_tests(text: str) -> list[str]:
    """Split a user-supplied block of ``assert`` lines into individual statements (syntax-checked)."""
    tests = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if not s.startswith("assert"):
            s = "assert " + s
        check = check_syntax(s)
        if not check.ok:
            raise ValueError(f"Invalid test statement: {s!r} ({check.error})")
        tests.append(s)
    return tests


class ModelRegistry:
    """Loads base / fine-tuned generators on first use (thread-safe) and caches them."""

    def __init__(self, cfg: Config, adapter_path: Optional[Path] = None, load_in_4bit: Optional[bool] = None):
        self.cfg = cfg
        self.adapter_path = resolve_path(adapter_path) if adapter_path else None
        self.load_in_4bit = load_in_4bit
        self._gens: dict[str, Generator] = {}
        self._lock = threading.Lock()
        self._injected = False

    def inject(self, name: str, generator: Generator) -> None:
        """Register an external generator (tests / smoke runs). Marks the registry as non-production."""
        self._gens[name] = generator
        self._injected = True

    @property
    def available(self) -> list[str]:
        names = ["base"]
        if self.adapter_path and (self.adapter_path / "adapter_config.json").exists():
            names.append("fine-tuned")
        return names

    def adapter_status(self) -> str:
        if not self.adapter_path:
            return "No adapter configured (app.adapter_path). Train one with scripts/train_lora.py."
        if not (self.adapter_path / "adapter_config.json").exists():
            return f"Adapter not found at {self.adapter_path}"
        return f"Adapter: {self.adapter_path}"

    def get(self, name: str) -> Generator:
        with self._lock:
            if name in self._gens:
                return self._gens[name]
            if name == "fine-tuned" and "fine-tuned" not in self.available:
                raise FileNotFoundError(self.adapter_status())
            from codegenai.model.generation import build_generator

            model_cfg = self.cfg.model.model_copy()
            if self.load_in_4bit is not None:
                model_cfg.load_in_4bit = self.load_in_4bit
            t0 = time.time()
            gen = build_generator(model_cfg, self.cfg.generation,
                                  self.adapter_path if name == "fine-tuned" else None,
                                  name="qwen-lora" if name == "fine-tuned" else "qwen-base")
            log.info("Loaded generator '%s' in %.1fs", name, time.time() - t0)
            self._gens[name] = gen
            return gen


class CodeService:
    def __init__(self, cfg: Config, registry: ModelRegistry, sandbox=None, enable_execution: bool = True):
        self.cfg = cfg
        self.registry = registry
        self.sandbox = sandbox or (get_sandbox(cfg.sandbox) if enable_execution else None)
        self.enable_execution = enable_execution and self.sandbox is not None

    # ------------------------------------------------------------------ status
    def sandbox_status(self) -> dict[str, Any]:
        if not self.enable_execution or self.sandbox is None:
            return {"available": False, "reason": "execution disabled by configuration"}
        a = self.sandbox.check_available()
        return {"available": a.available, "reason": a.reason, "image": self.cfg.sandbox.image,
                "server_version": a.server_version}

    # ------------------------------------------------------------------ runners
    def _runner(self, tests: list[str], entry_point: Optional[str]):
        if not tests or not self.enable_execution:
            return None

        def _run(code: str) -> TestReport:
            files = {"solution.py": code,
                     "test_user.py": build_assert_module(tests, "user", "", entry_point, include_entry_test=bool(entry_point))}
            return self.sandbox.run(files, entry_point=entry_point)

        return _run

    @staticmethod
    def _entry_point(tests: list[str]) -> Optional[str]:
        from codegenai.data.mbpp import infer_entry_point

        return infer_entry_point(tests, "") if tests else None

    # ------------------------------------------------------------------ API used by the UI
    def generate(self, instruction: str, model_name: str, tests_text: str = "", correction: bool = False,
                 max_attempts: Optional[int] = None) -> dict[str, Any]:
        """Generate (and optionally self-correct) code; returns a JSON-able trace with verification status."""
        if not instruction or not instruction.strip():
            raise ValueError("Please enter an instruction.")
        tests = parse_user_tests(tests_text)
        gen = self.registry.get(model_name)
        corr_cfg = self.cfg.correction.model_copy()
        if max_attempts is not None:
            corr_cfg.max_attempts = int(max_attempts)
        loop = SelfCorrectionLoop(gen, corr_cfg, self.cfg.generation.system_prompt, enable_correction=correction)
        runner = self._runner(tests, self._entry_point(tests))
        trace = loop.run(instruction, runner, tests or None)
        d = trace.to_dict()
        if runner is None:
            d["verification"] = ("UNVERIFIED: no tests were provided, so the code was not executed." if tests == []
                                 else "UNVERIFIED: code execution is disabled.")
        elif trace.passed_feedback_tests:
            d["verification"] = f"VERIFIED against {len(tests)} user-provided test(s) in the Docker sandbox."
        else:
            d["verification"] = f"NOT VERIFIED: final code fails the user-provided tests ({trace.final_status})."
        d["n_user_tests"] = len(tests)
        return d

    def run_tests(self, code: str, tests_text: str) -> dict[str, Any]:
        tests = parse_user_tests(tests_text)
        if not tests:
            raise ValueError("Provide at least one assert statement.")
        if not self.enable_execution:
            return TestReport.unavailable("execution disabled").to_dict()
        syn = check_syntax(code)
        if not syn.ok:
            return TestReport(status=Status.SYNTAX_ERROR, syntax_error=syn.error, summary=f"Syntax error: {syn.error}").to_dict()
        runner = self._runner(tests, self._entry_point(tests))
        return runner(code).to_dict()

    def compare(self, instruction: str, tests_text: str = "") -> dict[str, Any]:
        out = {}
        for name in ("base", "fine-tuned"):
            if name not in self.registry.available:
                out[name] = {"error": self.registry.adapter_status()}
                continue
            out[name] = self.generate(instruction, name, tests_text, correction=False)
        return out


# ---------------------------------------------------------------------- research artefacts
class ResultsReader:
    """Reads real experiment outputs from disk (never synthesises numbers)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def list_eval_runs(self) -> list[Path]:
        root = resolve_path(self.cfg.paths.results_dir)
        if not root.exists():
            return []
        return sorted(p.parent for p in root.rglob("aggregate_metrics.json"))

    def load_eval(self, run_dir: Path) -> Optional[dict[str, Any]]:
        p = Path(run_dir) / "aggregate_metrics.json"
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def eval_table(self, agg: dict[str, Any]) -> list[list[Any]]:
        rows = []
        for c, m in sorted(agg.get("metrics", {}).items()):
            rows.append([
                c, agg.get("configurations", {}).get(c, {}).get("name", ""), m["n_problems"],
                f"{100 * m['pass_at_1']:.1f} [{100 * m['pass_at_1_ci'][0]:.1f}, {100 * m['pass_at_1_ci'][1]:.1f}]",
                f"{100 * m['final_success_rate']:.1f} [{100 * m['final_success_ci'][0]:.1f}, {100 * m['final_success_ci'][1]:.1f}]",
                "-" if m["correction_success_rate"] is None else f"{100 * m['correction_success_rate']:.1f}",
                f"{100 * m['syntax_validity_rate']:.1f}", f"{100 * m['runtime_error_rate']:.1f}",
                f"{m['avg_correction_attempts']:.2f}",
                "-" if m["avg_generation_latency_s"] is None else f"{m['avg_generation_latency_s']:.2f}",
                "-" if m["avg_testing_latency_s"] is None else f"{m['avg_testing_latency_s']:.2f}",
                "-" if m["peak_gpu_memory_mb"] is None else f"{m['peak_gpu_memory_mb'] / 1024:.2f}",
            ])
        return rows

    def list_training_runs(self) -> list[Path]:
        root = resolve_path(self.cfg.paths.runs_dir)
        if not root.exists():
            return []
        return sorted(p.parent for p in root.glob("*/train_log.jsonl"))

    def training_log(self, run_dir: Path) -> list[dict[str, Any]]:
        p = Path(run_dir) / "train_log.jsonl"
        return list(read_jsonl(p)) if p.exists() else []

    def training_summary(self, run_dir: Path) -> Optional[dict[str, Any]]:
        p = Path(run_dir) / "summary.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def audit_stats(self) -> Optional[dict[str, Any]]:
        p = resolve_path(self.cfg.paths.audit_dir) / "audit_stats.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def split_manifest(self) -> Optional[dict[str, Any]]:
        p = resolve_path(self.cfg.paths.processed_dir) / "manifest.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
