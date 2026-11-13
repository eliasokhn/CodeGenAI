"""Four-way MBPP evaluation: {base, LoRA} x {no correction, correction}.

For every problem and configuration we record:

* the initial generation and its verdict on the *hidden* tests (pass@1),
* the self-correction trace (visible/feedback tests only, max 2 corrections),
* the final code and its verdict on the hidden tests (final success),
* latencies, token counts, peak GPU memory, status categories.

Strict isolation: the correction loop receives a ``feedback_runner`` that can only execute the
visible tests; hidden tests are executed by the evaluator afterwards and never enter a prompt.
Results are written incrementally (JSONL) so interrupted runs can resume.

Usage::

    python -m codegenai.evaluation.mbpp_eval --config configs/eval/mbpp_test.yaml \
        -o evaluation.adapter_path=runs/lora_r16_n2000/adapter
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from codegenai import __version__
from codegenai.config import Config, load_config, resolve_path
from codegenai.correction import SelfCorrectionLoop, assert_no_hidden_leak
from codegenai.data.extract import check_syntax
from codegenai.data.mbpp import load_mbpp_problems
from codegenai.data.schema import MBPPProblem, read_jsonl, write_json
from codegenai.evaluation.metrics import compare, summarise
from codegenai.logging_utils import get_logger, setup_logging
from codegenai.model.generation import Generator
from codegenai.sandbox import Status, TestReport, get_sandbox
from codegenai.sandbox.harness import build_mbpp_bundle

log = get_logger(__name__)

CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "A": {"name": "Baseline", "lora": False, "correction": False},
    "B": {"name": "Fine-tuned", "lora": True, "correction": False},
    "C": {"name": "Baseline + correction", "lora": False, "correction": True},
    "D": {"name": "Fine-tuned + correction", "lora": True, "correction": True},
}


@dataclass
class ProblemResult:
    configuration: str
    task_id: int
    entry_point: str
    initial_code: str
    final_code: str
    initial_syntax_ok: bool
    initial_hidden_status: str
    initial_hidden_passed: bool
    final_hidden_status: str
    final_hidden_passed: bool
    initial_feedback_passed: Optional[bool]
    final_feedback_passed: Optional[bool]
    n_corrections: int
    corrected_on_feedback: bool
    stop_reason: str
    generation_latency_s: float
    testing_latency_s: float  # feedback tests only (part of the user-facing pipeline)
    hidden_testing_latency_s: float
    total_latency_s: float
    prompt_tokens: int
    completion_tokens: int
    truncated_generation: bool
    peak_gpu_memory_mb: Optional[float]
    hidden_tests_passed: int
    hidden_tests_total: int
    attempts: list[dict[str, Any]] = field(default_factory=list)
    hidden_report_final: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MBPPEvaluator:
    def __init__(self, cfg: Config, sandbox=None):
        self.cfg = cfg
        self.sandbox = sandbox or get_sandbox(cfg.sandbox, require=True)

    # ---------------------------------------------------------------- test runners
    def feedback_runner(self, problem: MBPPProblem):
        """Runner that executes ONLY the visible tests (safe to feed back to the model)."""
        if not problem.visible_tests:
            return None

        def _run(code: str) -> TestReport:
            files, hidden = build_mbpp_bundle(problem, code, mode="visible")
            report = self.sandbox.run(files, entry_point=problem.entry_point, hidden_test_names=hidden)
            assert not any(t.hidden for t in report.tests), "hidden test executed inside the feedback runner"
            return report

        return _run

    def hidden_verdict(self, problem: MBPPProblem, code: str) -> TestReport:
        if not code.strip():
            return TestReport(status=Status.SYNTAX_ERROR, syntax_error="empty output", summary="No code produced")
        syn = check_syntax(code)
        if not syn.ok:
            return TestReport(status=Status.SYNTAX_ERROR, syntax_error=syn.error, summary=f"Syntax error: {syn.error}")
        files, hidden = build_mbpp_bundle(problem, code, mode="hidden")
        return self.sandbox.run(files, entry_point=problem.entry_point, hidden_test_names=hidden)

    # ---------------------------------------------------------------- one problem
    def evaluate_problem(self, problem: MBPPProblem, generator: Generator, configuration: str) -> ProblemResult:
        spec = CONFIGURATIONS[configuration]
        loop = SelfCorrectionLoop(generator, self.cfg.correction, self.cfg.generation.system_prompt,
                                  enable_correction=spec["correction"])
        trace = loop.run(problem.text, self.feedback_runner(problem), problem.visible_tests)

        # Hidden-test isolation check: no hidden assert may appear in any feedback that was sent.
        hidden_names = {f"test_hidden_{i}" for i in range(len(problem.hidden_tests))}
        for att in trace.attempts:
            if att.feedback:
                disjoint_hidden = [h for h in problem.hidden_tests if h not in problem.visible_tests]
                assert_no_hidden_leak(att.feedback, disjoint_hidden, hidden_names)

        initial = trace.attempts[0]
        t0 = time.perf_counter()
        init_hidden = self.hidden_verdict(problem, initial.code)
        if trace.n_corrections == 0:
            final_hidden = init_hidden
        else:
            final_hidden = self.hidden_verdict(problem, trace.final_code)
        hidden_latency = time.perf_counter() - t0

        gen_latency = sum(a.latency_generation_s for a in trace.attempts)
        test_latency = sum(a.latency_testing_s for a in trace.attempts)
        peak = max((a.generation.get("peak_gpu_memory_mb") or 0) for a in trace.attempts) or None
        fb_init = initial.report["status"] == Status.PASSED.value if (initial.report and problem.visible_tests) else None
        fb_final = trace.attempts[-1].report["status"] == Status.PASSED.value \
            if (trace.attempts[-1].report and problem.visible_tests) else None
        return ProblemResult(
            configuration=configuration, task_id=problem.task_id, entry_point=problem.entry_point,
            initial_code=initial.code, final_code=trace.final_code,
            initial_syntax_ok=check_syntax(initial.code).ok if initial.code.strip() else False,
            initial_hidden_status=init_hidden.status.value, initial_hidden_passed=init_hidden.ok,
            final_hidden_status=final_hidden.status.value, final_hidden_passed=final_hidden.ok,
            initial_feedback_passed=fb_init, final_feedback_passed=fb_final,
            n_corrections=trace.n_corrections, corrected_on_feedback=trace.corrected, stop_reason=trace.stop_reason,
            generation_latency_s=gen_latency, testing_latency_s=test_latency, hidden_testing_latency_s=hidden_latency,
            total_latency_s=trace.total_latency_s,
            prompt_tokens=sum(a.generation.get("prompt_tokens", 0) for a in trace.attempts),
            completion_tokens=sum(a.generation.get("completion_tokens", 0) for a in trace.attempts),
            truncated_generation=any(a.generation.get("truncated", False) for a in trace.attempts),
            peak_gpu_memory_mb=peak, hidden_tests_passed=final_hidden.passed, hidden_tests_total=final_hidden.total,
            attempts=[asdict(a) for a in trace.attempts], hidden_report_final=final_hidden.to_dict(),
        )

    # ---------------------------------------------------------------- one configuration
    def evaluate_configuration(self, configuration: str, problems: list[MBPPProblem], generator: Generator,
                               out_dir: Path, resume: bool = True) -> list[dict[str, Any]]:
        if getattr(generator, "is_mock", False):
            raise RuntimeError("Mock generators are for software tests only; refusing to produce research results.")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"results_{configuration}.jsonl"
        done: dict[int, dict[str, Any]] = {}
        if resume and path.exists():
            done = {r["task_id"]: r for r in read_jsonl(path)}
            log.info("[%s] resuming: %d problems already evaluated", configuration, len(done))
        t0 = time.time()
        with open(path, "a", encoding="utf-8") as fh:
            for i, problem in enumerate(problems):
                if problem.task_id in done:
                    continue
                res = self.evaluate_problem(problem, generator, configuration).to_dict()
                done[problem.task_id] = res
                fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                fh.flush()
                if (i + 1) % 10 == 0 or i == len(problems) - 1:
                    passed = sum(1 for r in done.values() if r["final_hidden_passed"])
                    log.info("[%s] %d/%d  final-success so far %.1f%%  (%.0fs)", configuration, i + 1,
                             len(problems), 100 * passed / len(done), time.time() - t0)
        return [done[p.task_id] for p in problems if p.task_id in done]


def write_aggregates(cfg: Config, out_dir: Path, records_by_config: dict[str, list[dict[str, Any]]],
                     meta: dict[str, Any]) -> dict[str, Any]:
    metrics = {c: summarise(c, recs).to_dict() for c, recs in records_by_config.items() if recs}
    pairs = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("A", "D")]
    comparisons = compare(records_by_config, pairs, cfg.evaluation.bootstrap_samples)
    agg = {"meta": meta, "configurations": {c: CONFIGURATIONS[c] for c in metrics}, "metrics": metrics,
           "comparisons": comparisons}
    write_json(out_dir / "aggregate_metrics.json", agg)
    try:
        from codegenai.evaluation.plots import make_all_plots

        agg["plots"] = make_all_plots(records_by_config, metrics, out_dir / "plots")
    except Exception as exc:  # noqa: BLE001 - plotting must never invalidate results
        log.warning("Plotting failed: %s", exc)
    write_json(out_dir / "aggregate_metrics.json", agg)
    return agg


def run_evaluation(cfg: Config, configurations: Optional[list[str]] = None, resume: bool = True,
                   generators: Optional[dict[bool, Generator]] = None) -> dict[str, Any]:
    """Evaluate the requested configurations. ``generators`` maps lora(bool) -> Generator."""
    import torch

    ev = cfg.evaluation
    configurations = configurations or ev.configurations
    out_dir = resolve_path(ev.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = load_mbpp_problems(cfg, ev.split, ev.limit)
    evaluator = MBPPEvaluator(cfg)
    needs_lora = any(CONFIGURATIONS[c]["lora"] for c in configurations)
    if needs_lora and not ev.adapter_path and generators is None:
        raise ValueError("Configurations B/D need evaluation.adapter_path (a trained LoRA adapter).")

    meta = {
        "codegenai_version": __version__, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": cfg.model.model_id, "model_revision": cfg.model.revision,
        "adapter_path": str(ev.adapter_path) if ev.adapter_path else None,
        "mbpp": {"repo": cfg.data.mbpp_id, "revision": cfg.data.mbpp_revision, "config": cfg.data.mbpp_config,
                 "split": ev.split, "n_problems": len(problems), "visible_tests": ev.visible_tests,
                 "hidden_test_policy": ev.hidden_test_policy, "mbppplus": ev.use_mbppplus},
        "generation": cfg.generation.model_dump(), "correction": cfg.correction.model_dump(),
        "sandbox": cfg.sandbox.model_dump(), "seed": cfg.training.seed,
        "hardware": {"platform": platform.platform(), "torch": torch.__version__,
                     "cuda": torch.version.cuda if torch.cuda.is_available() else None,
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "config_files": cfg.source_files,
    }
    write_json(out_dir / "problems_manifest.json", [{"task_id": p.task_id, "entry_point": p.entry_point,
                                                     "n_visible": len(p.visible_tests),
                                                     "n_hidden": len(p.hidden_tests), "plus": bool(p.plus_test)}
                                                    for p in problems])
    if ev.adapter_path:
        from codegenai.model.loader import adapter_metadata

        meta["adapter"] = adapter_metadata(ev.adapter_path)

    records_by_config: dict[str, list[dict[str, Any]]] = {}
    gens: dict[bool, Generator] = dict(generators or {})
    for lora in (False, True):
        confs = [c for c in configurations if CONFIGURATIONS[c]["lora"] == lora]
        if not confs:
            continue
        if lora not in gens:
            from codegenai.model.generation import build_generator

            gens[lora] = build_generator(cfg.model, cfg.generation, ev.adapter_path if lora else None)
        for c in confs:
            log.info("=== Configuration %s: %s ===", c, CONFIGURATIONS[c]["name"])
            records_by_config[c] = evaluator.evaluate_configuration(c, problems, gens[lora], out_dir, resume)
        if generators is None:  # free GPU memory before loading the next model
            gens.pop(lora, None)
            if torch.cuda.is_available():
                import gc

                gc.collect()
                torch.cuda.empty_cache()
    # include previously computed configurations on disk for the aggregate
    for c in CONFIGURATIONS:
        p = out_dir / f"results_{c}.jsonl"
        if c not in records_by_config and p.exists():
            records_by_config[c] = list(read_jsonl(p))
    agg = write_aggregates(cfg, out_dir, records_by_config, meta)
    for c, m in agg["metrics"].items():
        log.info("%s %-26s pass@1=%.3f final=%.3f corr_success=%s n=%d", c, CONFIGURATIONS[c]["name"],
                 m["pass_at_1"], m["final_success_rate"], m["correction_success_rate"], m["n_problems"])
    return agg


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate configurations A-D on MBPP.")
    parser.add_argument("--config", default="configs/eval/mbpp_test.yaml")
    parser.add_argument("--configurations", default=None, help="comma separated subset of A,B,C,D")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true", help="recompute metrics/plots from results on disk")
    parser.add_argument("-o", "--override", action="append", default=[])
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.override)
    setup_logging(log_file=resolve_path(cfg.evaluation.output_dir) / "eval.log")
    if args.aggregate_only:
        out_dir = resolve_path(cfg.evaluation.output_dir)
        recs = {c: list(read_jsonl(out_dir / f"results_{c}.jsonl")) for c in CONFIGURATIONS
                if (out_dir / f"results_{c}.jsonl").exists()}
        meta = {}
        mp = out_dir / "aggregate_metrics.json"
        if mp.exists():
            meta = json.loads(mp.read_text(encoding="utf-8")).get("meta", {})
        agg = write_aggregates(cfg, out_dir, recs, meta)
    else:
        confs = args.configurations.split(",") if args.configurations else None
        agg = run_evaluation(cfg, confs, resume=not args.no_resume)
    print(json.dumps({c: {k: m[k] for k in ("n_problems", "pass_at_1", "final_success_rate",
                                             "correction_success_rate", "avg_correction_attempts")}
                      for c, m in agg["metrics"].items()}, indent=2))


if __name__ == "__main__":
    main()
