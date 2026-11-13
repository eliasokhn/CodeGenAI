"""Gradio application: generation, sandboxed testing, self-correction, model comparison and a
research dashboard that reads real experiment artefacts.

    python -m codegenai.app.gradio_app --config configs/base.yaml -o app.adapter_path=runs/lora_r16_n2000/adapter
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import gradio as gr

from codegenai.app.services import CodeService, ModelRegistry, ResultsReader
from codegenai.config import Config, load_config
from codegenai.logging_utils import get_logger, setup_logging

log = get_logger(__name__)

EVAL_COLUMNS = ["Cfg", "Configuration", "n", "pass@1 % [95% CI]", "Final success % [95% CI]",
                "Correction success %", "Syntax valid %", "Runtime err %", "Avg corrections",
                "Gen latency s", "Test latency s", "Peak GPU GiB"]


def _fmt_report(report: Optional[dict[str, Any]]) -> str:
    if not report:
        return "_No tests executed._"
    lines = [f"**Status:** `{report['status']}` — {report.get('summary', '')}",
             f"**Passed/failed/errors:** {report.get('passed', 0)}/{report.get('failed', 0)}/{report.get('errors', 0)}"
             f" · **sandbox time:** {report.get('execution_time_s', 0):.2f}s · **verified:** {report.get('verified')}"]
    for t in report.get("tests", []):
        icon = "✅" if t["outcome"] == "passed" else "❌"
        msg = f"\n```\n{t['message'][-600:]}\n```" if t.get("message") and t["outcome"] != "passed" else ""
        lines.append(f"- {icon} `{t['name']}` ({t.get('error_type') or t['outcome']}){msg}")
    if report.get("syntax_error"):
        lines.append(f"\n**Syntax error:** `{report['syntax_error']}`")
    return "\n".join(lines)


def _attempt_history(trace: dict[str, Any]) -> str:
    out = []
    for a in trace.get("attempts", []):
        title = "Initial generation" if a["index"] == 0 else f"Correction attempt {a['index']}"
        status = a["report"]["status"] if a.get("report") else "not tested"
        out.append(f"### {title} — `{status}` · gen {a['latency_generation_s']:.2f}s · test {a['latency_testing_s']:.2f}s")
        if a.get("feedback"):
            out.append("**Feedback sent to the model:**\n```\n" + a["feedback"] + "\n```")
        out.append("```python\n" + (a["code"] or a["raw_text"]) + "\n```")
        out.append(_fmt_report(a.get("report")))
    out.append(f"**Stop reason:** `{trace.get('stop_reason')}` · **total latency:** {trace.get('total_latency_s', 0):.2f}s")
    return "\n\n".join(out)


def build_app(cfg: Config, service: CodeService, reader: ResultsReader) -> gr.Blocks:
    sandbox_state = service.sandbox_status()
    exec_note = ("🟢 Docker sandbox available" if sandbox_state["available"]
                 else f"🔴 Code execution disabled: {sandbox_state['reason']}")

    with gr.Blocks(title="CodeGenAI") as demo:
        gr.Markdown(
            f"# CodeGenAI — Qwen2.5-Coder-1.5B + LoRA with sandboxed self-correction\n"
            f"Model: `{cfg.model.model_id}` · {service.registry.adapter_status()} · {exec_note}\n\n"
            "Generated code is executed **only** inside an isolated Docker container (no network, read-only, "
            "non-root, resource-limited). Without tests, outputs are labelled *unverified*."
        )

        # ------------------------------------------------------------ generation
        with gr.Tab("Code Generation"):
            with gr.Row():
                with gr.Column(scale=1):
                    instr = gr.Textbox(label="Instruction", lines=5,
                                       placeholder="Write a function is_prime(n) that returns True if n is prime.")
                    model_sel = gr.Dropdown(service.registry.available, value="base", label="Model")
                    tests_in = gr.Textbox(label="Optional tests (one assert per line; enables verification)", lines=4,
                                          placeholder="assert is_prime(7) == True\nassert is_prime(8) == False")
                    with gr.Row():
                        corr_on = gr.Checkbox(value=True, label="Enable self-correction (needs tests)")
                        max_att = gr.Slider(0, 3, value=cfg.correction.max_attempts, step=1, label="Max correction attempts")
                    gen_btn = gr.Button("Generate", variant="primary")
                with gr.Column(scale=1):
                    code_out = gr.Code(language="python", label="Final code", lines=18)
                    verif = gr.Markdown()
                    latency = gr.Markdown()
            with gr.Accordion("Attempt history (initial code, failed tests, feedback, revisions)", open=False):
                history = gr.Markdown()
            raw_state = gr.State()

            def _generate(instruction, model_name, tests_text, correction, attempts):
                try:
                    trace = service.generate(instruction, model_name, tests_text, correction, int(attempts))
                except Exception as exc:  # noqa: BLE001
                    return "", f"**Error:** {exc}", "", "", None
                gens = [a["latency_generation_s"] for a in trace["attempts"]]
                lat = (f"**Generation latency:** {sum(gens):.2f}s over {len(gens)} call(s) · "
                       f"**corrections:** {trace['n_corrections']} · **final status:** `{trace['final_status']}`")
                peak = trace["attempts"][0]["generation"].get("peak_gpu_memory_mb")
                if peak:
                    lat += f" · **peak GPU memory:** {peak / 1024:.2f} GiB"
                return trace["final_code"], f"**{trace['verification']}**", lat, _attempt_history(trace), trace

            gen_btn.click(_generate, [instr, model_sel, tests_in, corr_on, max_att],
                          [code_out, verif, latency, history, raw_state])

        # ------------------------------------------------------------ testing
        with gr.Tab("Automated Testing"):
            gr.Markdown("Run **any** Python code against your asserts in the sandbox. "
                        "Detects syntax errors, missing functions, wrong outputs, runtime exceptions and timeouts.\n\n"
                        f"Limits: {cfg.sandbox.timeout_seconds}s total, {cfg.sandbox.per_test_timeout_seconds}s/test, "
                        f"{cfg.sandbox.memory} RAM, {cfg.sandbox.cpus} CPU, no network, no filesystem writes outside tmpfs.")
            with gr.Row():
                code_in = gr.Code(language="python", label="Code under test", lines=16,
                                  value="def add(a, b):\n    return a + b\n")
                tests_t = gr.Textbox(label="Tests (one assert per line)", lines=16,
                                     value="assert add(1, 2) == 3\nassert add(-1, 1) == 0")
            run_btn = gr.Button("Run tests in sandbox", variant="primary")
            test_md = gr.Markdown()
            test_json = gr.JSON(label="Structured report")

            def _run(code, tests_text):
                try:
                    rep = service.run_tests(code, tests_text)
                except Exception as exc:  # noqa: BLE001
                    return f"**Error:** {exc}", None
                return _fmt_report(rep), rep

            run_btn.click(_run, [code_in, tests_t], [test_md, test_json])

        # ------------------------------------------------------------ self-correction
        with gr.Tab("Self-Correction"):
            gr.Markdown("Full loop: generate → test → structured feedback → regenerate (max 2 corrections). "
                        "Only the tests you provide are used as feedback; the model never sees anything else.")
            with gr.Row():
                sc_instr = gr.Textbox(label="Instruction", lines=4)
                sc_tests = gr.Textbox(label="Feedback tests (required, one assert per line)", lines=4)
            with gr.Row():
                sc_model = gr.Dropdown(service.registry.available, value="base", label="Model")
                sc_btn = gr.Button("Run self-correction", variant="primary")
            sc_summary = gr.Markdown()
            with gr.Row():
                sc_initial = gr.Code(language="python", label="Initial code", lines=14)
                sc_final = gr.Code(language="python", label="Final code", lines=14)
            sc_history = gr.Markdown()

            def _correct(instruction, tests_text, model_name):
                try:
                    if not tests_text.strip():
                        raise ValueError("Self-correction requires feedback tests.")
                    trace = service.generate(instruction, model_name, tests_text, True, cfg.correction.max_attempts)
                except Exception as exc:  # noqa: BLE001
                    return f"**Error:** {exc}", "", "", ""
                summ = (f"**{trace['verification']}** · corrections: {trace['n_corrections']} · "
                        f"corrected: {trace['corrected']} · stop: `{trace['stop_reason']}` · "
                        f"total {trace['total_latency_s']:.2f}s")
                return summ, trace["attempts"][0]["code"], trace["final_code"], _attempt_history(trace)

            sc_btn.click(_correct, [sc_instr, sc_tests, sc_model], [sc_summary, sc_initial, sc_final, sc_history])

        # ------------------------------------------------------------ comparison
        with gr.Tab("Model Comparison"):
            gr.Markdown("Same prompt, greedy decoding, pretrained vs. LoRA fine-tuned (no correction).")
            cmp_instr = gr.Textbox(label="Instruction", lines=4)
            cmp_tests = gr.Textbox(label="Optional tests (one assert per line)", lines=3)
            cmp_btn = gr.Button("Compare", variant="primary")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Pretrained Qwen2.5-Coder-1.5B-Instruct")
                    cmp_base = gr.Code(language="python", lines=16)
                    cmp_base_md = gr.Markdown()
                with gr.Column():
                    gr.Markdown("### LoRA fine-tuned")
                    cmp_ft = gr.Code(language="python", lines=16)
                    cmp_ft_md = gr.Markdown()

            def _compare(instruction, tests_text):
                try:
                    res = service.compare(instruction, tests_text)
                except Exception as exc:  # noqa: BLE001
                    return "", f"**Error:** {exc}", "", ""
                outs = []
                for name in ("base", "fine-tuned"):
                    r = res[name]
                    if "error" in r:
                        outs += ["", f"_{r['error']}_"]
                    else:
                        a = r["attempts"][0]
                        outs += [r["final_code"], f"{r['verification']} · latency {a['latency_generation_s']:.2f}s · "
                                                  f"{a['generation']['completion_tokens']} tokens"]
                return outs[0], outs[1], outs[2], outs[3]

            cmp_btn.click(_compare, [cmp_instr, cmp_tests], [cmp_base, cmp_base_md, cmp_ft, cmp_ft_md])

        # ------------------------------------------------------------ dashboard
        with gr.Tab("Research Dashboard"):
            gr.Markdown("All numbers below are read from files produced by the evaluation, training and audit "
                        "scripts (`results/`, `runs/`, `data/audit/`). Nothing is hard-coded.")
            runs = reader.list_eval_runs()
            run_dd = gr.Dropdown([str(p) for p in runs], value=str(runs[-1]) if runs else None,
                                 label="Evaluation run (directory with aggregate_metrics.json)", allow_custom_value=True)
            refresh = gr.Button("Refresh")
            eval_meta = gr.Markdown()
            eval_table = gr.Dataframe(headers=EVAL_COLUMNS, label="Configurations A–D", interactive=False)
            eval_cmp = gr.JSON(label="Paired comparisons (bootstrap 95% CI, McNemar)")
            with gr.Row():
                plot1 = gr.Image(label="Success rates", type="filepath")
                plot2 = gr.Image(label="Correction flow", type="filepath")
            with gr.Row():
                plot3 = gr.Image(label="Outcome categories", type="filepath")
                plot4 = gr.Image(label="Latency", type="filepath")

            def _load_eval(run_dir):
                if not run_dir:
                    return "_No evaluation results found. Run `python scripts/evaluate_mbpp.py`._", [], None, None, None, None, None
                agg = reader.load_eval(Path(run_dir))
                if not agg:
                    return f"_No aggregate_metrics.json in {run_dir}_", [], None, None, None, None, None
                meta = agg.get("meta", {})
                md = (f"**Model:** `{meta.get('model_id')}` · **adapter:** `{meta.get('adapter_path')}` · "
                      f"**MBPP split:** {meta.get('mbpp', {}).get('split')} (n={meta.get('mbpp', {}).get('n_problems')}) · "
                      f"**visible tests:** {meta.get('mbpp', {}).get('visible_tests')} · "
                      f"**hidden policy:** {meta.get('mbpp', {}).get('hidden_test_policy')} · "
                      f"**created:** {meta.get('created_at')} · **GPU:** {meta.get('hardware', {}).get('gpu')}")
                plots = agg.get("plots", {})
                pick = lambda k: plots.get(k) if plots.get(k) and Path(plots[k]).exists() else None  # noqa: E731
                return (md, reader.eval_table(agg), agg.get("comparisons", []), pick("success_rates"),
                        pick("correction_flow"), pick("status_breakdown"), pick("latency"))

            run_dd.change(_load_eval, [run_dd], [eval_meta, eval_table, eval_cmp, plot1, plot2, plot3, plot4])
            refresh.click(_load_eval, [run_dd], [eval_meta, eval_table, eval_cmp, plot1, plot2, plot3, plot4])

            gr.Markdown("## Training curves")
            truns = reader.list_training_runs()
            trun_dd = gr.Dropdown([str(p) for p in truns], value=str(truns[-1]) if truns else None,
                                  label="Training run", allow_custom_value=True)
            train_md = gr.Markdown()
            train_plot = gr.LinePlot(x="step", y="loss", color="series", title="Loss", height=320)

            def _load_train(run_dir):
                import pandas as pd

                if not run_dir:
                    return "_No training runs found in runs/._", pd.DataFrame({"step": [], "loss": [], "series": []})
                rows = reader.training_log(Path(run_dir))
                summ = reader.training_summary(Path(run_dir))
                data = [{"step": r["step"], "loss": r["loss"], "series": "train"} for r in rows if "loss" in r]
                data += [{"step": r["step"], "loss": r["eval_loss"], "series": "validation"} for r in rows if "eval_loss" in r]
                md = f"**{len(rows)} log records**"
                if summ:
                    md += (f" · final eval_loss: {summ.get('eval_metrics', {}).get('eval_loss', float('nan')):.4f} · "
                           f"train time: {summ.get('train_time_s')}s · peak GPU: {summ.get('peak_gpu_memory_gb')} GiB · "
                           f"LoRA r={summ.get('lora', {}).get('r')} · examples={summ.get('training', {}).get('train_file')}")
                else:
                    md += " · _summary.json not found (run still in progress or interrupted)_"
                return md, pd.DataFrame(data) if data else pd.DataFrame({"step": [], "loss": [], "series": []})

            trun_dd.change(_load_train, [trun_dd], [train_md, train_plot])

            gr.Markdown("## Dataset audit")
            audit_md = gr.Markdown()
            audit_json = gr.JSON(label="audit_stats.json")

            def _load_audit():
                stats = reader.audit_stats()
                man = reader.split_manifest()
                if not stats:
                    return "_No audit found. Run `python scripts/audit_kodcode.py`._", None
                ex = stats.get("execution_audit", {})
                md = (f"**Accepted:** {stats['accepted']} · **rejected:** {stats['rejected']} · "
                      f"**pool:** {stats['pool_size']} · **contamination hits:** {stats['contamination']['hits']} · "
                      f"**sandbox reproduction rate:** {ex.get('reproduction_rate', 'n/a')} on {ex.get('sample_size', 0)} "
                      f"· **mutation kill rate:** {ex.get('mutation', {}).get('kill_rate', 'n/a')}")
                if man:
                    md += "\n\n**Splits:** " + ", ".join(f"{k}={v['rows']}" for k, v in man["files"].items())
                return md, stats

            demo.load(_load_eval, [run_dd], [eval_meta, eval_table, eval_cmp, plot1, plot2, plot3, plot4])
            demo.load(_load_train, [trun_dd], [train_md, train_plot])
            demo.load(_load_audit, [], [audit_md, audit_json])
    return demo


def create_app(cfg: Config, service: Optional[CodeService] = None) -> gr.Blocks:
    if service is None:
        registry = ModelRegistry(cfg, cfg.app.adapter_path or cfg.evaluation.adapter_path, cfg.app.load_in_4bit)
        service = CodeService(cfg, registry, enable_execution=cfg.app.enable_execution)
    return build_app(cfg, service, ResultsReader(cfg))


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Launch the CodeGenAI Gradio app.")
    parser.add_argument("--config", default=None)
    parser.add_argument("-o", "--override", action="append", default=[])
    args = parser.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config, args.override)
    demo = create_app(cfg)
    demo.queue().launch(server_name=cfg.app.host, server_port=cfg.app.port, share=cfg.app.share)


if __name__ == "__main__":
    main()
