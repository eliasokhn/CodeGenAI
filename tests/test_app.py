"""UI integration: the Gradio app builds and its service layer is wired to real components (mock model)."""
from __future__ import annotations

import json

import pytest

from codegenai.app.gradio_app import create_app
from codegenai.app.services import CodeService, ModelRegistry, ResultsReader, parse_user_tests
from codegenai.model.generation import MockCodeGenerator
from codegenai.sandbox import Status
from conftest import make_report


class FakeSandbox:
    def __init__(self):
        self.calls = []

    def check_available(self, refresh=False):
        from codegenai.sandbox.docker_runner import Availability
        return Availability(True, "ok", "fake", True)

    def run(self, files, entry_point=None, hidden_test_names=None, timeout_override=None):
        self.calls.append(files)
        passed = "return a + b" in files["solution.py"]
        return make_report(Status.PASSED if passed else Status.FAILED,
                           [("test_user_0", "passed" if passed else "failed", None if passed else "AssertionError", False)])


@pytest.fixture
def service(cfg, tmp_path):
    reg = ModelRegistry(cfg, adapter_path=None)
    answers = iter(["```python\ndef add(a, b):\n    return a - b\n```", "```python\ndef add(a, b):\n    return a + b\n```"])
    reg.inject("base", MockCodeGenerator(lambda m: next(answers, "```python\ndef add(a, b):\n    return a + b\n```")))
    return CodeService(cfg, reg, sandbox=FakeSandbox(), enable_execution=True)


def test_parse_user_tests():
    assert parse_user_tests("assert f(1) == 2\n\n# c\nf(2) == 3") == ["assert f(1) == 2", "assert f(2) == 3"]
    with pytest.raises(ValueError):
        parse_user_tests("assert f( == 1")


def test_generate_with_tests_runs_correction_and_reports_verification(service):
    out = service.generate("add two numbers", "base", "assert add(1, 2) == 3", correction=True)
    assert out["n_corrections"] == 1 and out["passed_feedback_tests"] and out["verification"].startswith("VERIFIED")
    assert len(service.sandbox.calls) == 2 and "test_user.py" in service.sandbox.calls[0]


def test_generate_without_tests_is_unverified(service):
    out = service.generate("add", "base", "", correction=True)
    assert out["verification"].startswith("UNVERIFIED") and out["n_corrections"] == 0 and service.sandbox.calls == []


def test_run_tests_and_syntax_error(service):
    rep = service.run_tests("def add(a, b):\n    return a + b\n", "assert add(1,2)==3")
    assert rep["status"] == "passed"
    rep = service.run_tests("def add(a, b)\n    return 1", "assert add(1,2)==3")
    assert rep["status"] == "syntax_error"


def test_compare_reports_missing_adapter(service):
    out = service.compare("add", "assert add(1,2)==3")
    assert "error" in out["fine-tuned"] and "No adapter" in out["fine-tuned"]["error"]
    assert "final_code" in out["base"]


def test_results_reader_reads_only_real_files(cfg, tmp_path):
    c = cfg.model_copy(deep=True)
    c.paths.results_dir = tmp_path / "results"
    c.paths.runs_dir = tmp_path / "runs"
    c.paths.audit_dir = tmp_path / "audit"
    r = ResultsReader(c)
    assert r.list_eval_runs() == [] and r.list_training_runs() == [] and r.audit_stats() is None
    run = tmp_path / "results" / "x"
    run.mkdir(parents=True)
    agg = {"configurations": {"A": {"name": "Baseline"}}, "metrics": {"A": {
        "n_problems": 3, "pass_at_1": 0.5, "pass_at_1_ci": [0.1, 0.9], "final_success_rate": 0.5, "final_success_ci": [0.1, 0.9],
        "correction_success_rate": None, "syntax_validity_rate": 1.0, "runtime_error_rate": 0.0, "avg_correction_attempts": 0,
        "avg_generation_latency_s": 1.0, "avg_testing_latency_s": None, "peak_gpu_memory_mb": 2048}}}
    (run / "aggregate_metrics.json").write_text(json.dumps(agg), encoding="utf-8")
    assert r.list_eval_runs() == [run]
    table = r.eval_table(r.load_eval(run))
    assert table[0][0] == "A" and table[0][2] == 3 and table[0][-1] == "2.00"


def test_gradio_app_builds(cfg, service):
    demo = create_app(cfg, service)
    import gradio as gr
    assert isinstance(demo, gr.Blocks)
    cfg_json = demo.get_config_file() if hasattr(demo, "get_config_file") else demo.config
    labels = {c.get("props", {}).get("label") for c in cfg_json["components"]}
    assert {"Instruction", "Final code", "Code under test", "Model"} & labels
