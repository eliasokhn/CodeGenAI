# CodeGenAI — Python Code Generation with LoRA Fine-Tuning and Automated Self-Correction

Natural-language instruction → **Qwen2.5-Coder-1.5B-Instruct** (base or LoRA/QLoRA fine-tuned on
execution-verified **KodCode**) → Python function → **pytest inside a hardened Docker sandbox** →
structured failure feedback → up to two **self-correction** rounds with the same model → verified
output. Evaluated on **MBPP** in four configurations (A baseline, B fine-tuned, C baseline + correction,
D fine-tuned + correction) with confidence intervals, plus a Gradio demo and research dashboard.

* Methodology: [docs/methodology.md](docs/methodology.md)
* Dataset audit: [docs/dataset_audit.md](docs/dataset_audit.md)
* Security model: [docs/security.md](docs/security.md)

## Repository layout

```
configs/                YAML configs (base.yaml + experiment files; `extends:` inheritance, -o overrides)
docker/sandbox/         Dockerfile of the execution sandbox
scripts/                CLI entry points (thin wrappers around the package modules)
src/codegenai/
  config.py             typed config (pydantic) loaded from YAML
  env_check.py          environment verification (GPU/CUDA/bnb/Docker/Hub)
  data/                 download (pinned revisions), extract (AST), contamination, mutation, audit, splits, mbpp
  model/                Qwen loading (4-bit NF4 optional), LoRA adapters, deterministic generation, prompts
  training/             SFTTrainer + PEFT LoRA, completion-only loss masking + verification, MLflow/GPU callbacks
  sandbox/              Docker runner (hardened), in-container pytest runner, harness builders, reports
  correction/           feedback formatting (hidden-test safe) and the self-correction loop
  evaluation/           MBPP four-way evaluation, metrics with CIs, plots
  app/                  Gradio application + services
tests/                  pytest suite (unit + Docker integration + tokenizer/TRL integration)
data/raw|audit|processed  downloads, audit outputs, training splits (generated)
runs/  results/  mlruns/  training runs, evaluation results, MLflow store (generated)
```

## 1. Setup

Requirements: Python 3.11, Docker (Linux containers; Docker Desktop on Windows / WSL2), an NVIDIA GPU
for training (tested target: RTX 5070 Ti 12 GB, 32 GB RAM). Everything except training and
GPU inference also runs on CPU.

```bash
# Windows (PowerShell) or Linux/WSL2
uv venv --python 3.11 .venv            # or: python3.11 -m venv .venv
# Windows: .venv\Scripts\activate      Linux/WSL2: source .venv/bin/activate

# PyTorch: pick the wheel for your CUDA. RTX 50-series (Blackwell, sm_120) needs CUDA 12.8+:
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -r requirements-gpu.txt   # requirements-gpu.txt = bitsandbytes
pip install -e .

# Sandbox image (required for ANY code execution)
docker build -t codegenai-sandbox:latest -f docker/sandbox/Dockerfile .

python scripts/verify_env.py           # add --pin-model to write the current Qwen commit into configs/base.yaml
```

`verify_env.py` reports Python, torch/CUDA (incl. whether the torch build has kernels for your GPU's
compute capability), bitsandbytes, PEFT/TRL/Transformers, the Docker sandbox and Hub access.
**WSL2 note:** on Windows prefer running training inside WSL2 Ubuntu (CUDA + bitsandbytes are most
reliable there); the Docker sandbox works from either side as long as Docker Desktop runs.

## 2. Data: download, audit, splits

```bash
python scripts/download_data.py                       # KodCode verified JSONL (594 MB) + MBPP, pinned revisions
python scripts/audit_kodcode.py                       # full audit incl. sandbox execution (~20 min, needs Docker)
python scripts/audit_kodcode.py --no-execute          # static-only audit
python scripts/build_splits.py                        # validation.jsonl, train_2000.jsonl, train_5000.jsonl (+ manifest)
```

Outputs in `data/audit/`: `accepted.jsonl`, `rejected.jsonl` (with reasons), `contamination_hits.jsonl`,
`execution_results.jsonl`, `mutation_results.jsonl`, `audit_stats.json`; provenance in
`data/raw/provenance.json`. The audit retrieves the upstream pytest files from `KodCode/KodCode-V1`
(column-projected parquet reads, ~650 s), re-executes reference solutions in the sandbox, runs
mutation checks and screens every row against all MBPP splits. See [docs/dataset_audit.md](docs/dataset_audit.md).

## 3. Training (LoRA / QLoRA)

Training is **not** started automatically. Headline experiments (data size × adapter width):

```bash
python scripts/train_lora.py --config configs/train/smoke.yaml            # 4 optimisation steps, sanity check (run first)
python scripts/train_lora.py --config configs/train/lora_qv_n5000.yaml    # 5k examples,  LoRA on q,v           (~1-2 h on the 5070 Ti, est.)
python scripts/train_lora.py --config configs/train/lora_all_n5000.yaml   # 5k examples,  LoRA on all projections
python scripts/train_lora.py --config configs/train/lora_qv_n20000.yaml   # 20k examples, LoRA on q,v           (~4-6 h, est.)
python scripts/train_lora.py --config configs/train/lora_all_n20000.yaml  # 20k examples, LoRA on all projections
# rank ablation (optional): configs/train/lora_r{8,16}_n{2000,5000}.yaml
# resume:  -o training.resume_from_checkpoint=auto
# dry run (loads model, builds trainer, verifies loss mask, no training):  --dry-run
# no bitsandbytes? train LoRA in bf16 instead of QLoRA:  -o model.load_in_4bit=false -o training.optim=adamw_torch
mlflow ui --backend-store-uri mlruns                                      # training curves, params, GPU memory
```

Hyper-parameters (all in YAML): QLoRA 4-bit NF4 + double quant, r=16, α=32, dropout 0.05, lr 1e-4,
3 epochs (upper bound), batch 1 × accumulation 8, max length 1,024, gradient checkpointing, bf16,
seed 42. Validation loss on the 300 held-out KodCode rows is computed every 100 steps; the final
adapter is the lowest-validation-loss checkpoint (`load_best_model_at_end`) and early stopping
(patience 3) ends runs that stop improving. Half of the training prompts carry one upstream KodCode
assert in the same template as the evaluation prompt (`data.train_prompt_test_fraction`). The
trainer refuses to start if any non-LoRA parameter is trainable or if the completion-only loss mask
is wrong (`runs/<run>/mask_check.json`). Each run writes `train_log.jsonl`, `summary.json` (incl.
`best_model_checkpoint`), `config.json`, checkpoints and `adapter/`.

## 4. Evaluation (MBPP, configurations A–D)

Model selection happens on MBPP **validation** only; the **test** split is run once, with the chosen
adapter. If no adapter beats the baseline on validation, that is the reported result (pre-registered
fallbacks are listed in [docs/methodology.md](docs/methodology.md)).

```bash
# 1. baseline on MBPP validation (90 problems) -- once
python scripts/evaluate_mbpp.py --config configs/eval/mbpp_validation.yaml --configurations A
# 2. every adapter on validation (configuration B); use a separate output dir per adapter
for r in lora_qv_n5000 lora_all_n5000 lora_qv_n20000 lora_all_n20000; do
  python scripts/evaluate_mbpp.py --config configs/eval/mbpp_validation.yaml --configurations B \
    -o evaluation.adapter_path=runs/$r/adapter -o evaluation.output_dir=results/mbpp_validation_$r
done
# 3. final numbers on MBPP test (500 problems), all four configurations, selected adapter only
python scripts/evaluate_mbpp.py --config configs/eval/mbpp_test.yaml -o evaluation.adapter_path=runs/<selected>/adapter
# subsets / resume / MBPP+
python scripts/evaluate_mbpp.py --config configs/eval/mbpp_test.yaml --configurations A,C
python scripts/evaluate_mbpp.py --config configs/eval/mbpp_test.yaml --aggregate-only
python scripts/evaluate_mbpp.py --config configs/eval/mbpp_test.yaml -o evaluation.use_mbppplus=true -o evaluation.output_dir=results/mbppplus_test
```

Per-problem records `results/<run>/results_{A,B,C,D}.jsonl` (code of every attempt, feedback, test
reports, latencies, GPU memory), `aggregate_metrics.json` (pass@1, final success, correction success,
syntax validity, runtime-error rate, average corrections, latency, peak memory, Wilson CIs, paired
bootstrap + McNemar comparisons, metadata/seeds/revisions) and `plots/*.png`.

## 5. Gradio application

```bash
python scripts/run_app.py -o app.adapter_path=runs/lora_r16_n2000/adapter        # http://127.0.0.1:7860
python scripts/run_app.py -o model.load_in_4bit=false                             # CPU / no bitsandbytes
```

Tabs: Code Generation (model choice, optional tests, correction controls, latency), Automated Testing
(sandboxed pytest with structured report), Self-Correction (attempt history with feedback), Model
Comparison (base vs. LoRA side by side), Research Dashboard (reads `results/`, `runs/`, `data/audit/`;
shows "not found" instead of placeholder numbers).

## 6. Tests

```bash
pytest -q                     # unit + Docker integration (skipped if Docker is down) + Hub tokenizer test (skipped offline)
pytest -q -m "not docker and not network"
```

Covers config loading, extraction/AST analysis, filters and static checks, contamination screening,
mutation generation, nested/disjoint splits, report classification, harness generation, sandbox
hardening flags and real-container isolation probes, correction logic and hidden-test isolation,
loss masking (fake tokenizer + real Qwen tokenizer/TRL trainer agreement), LoRA adapter targeting and
loading, metrics/CIs, evaluator wiring and the Gradio app.

## Status of this repository (what has actually been run)

Executed on the development laptop (CPU only, Docker Desktop) on 2026-11-13:

* **Data**: KodCode verified JSONL + MBPP downloaded at pinned revisions; full audit with sandbox
  re-execution (250/250 reproduced) and mutation checks (97.8 % kill rate); 22,940 accepted rows;
  nested splits `validation` 300 / `train_2000` / `train_5000` / `train_20000` built, 39 % of prompts
  carrying one upstream assert ([docs/dataset_audit.md](docs/dataset_audit.md)).
* **Sandbox**: image built; isolation probes (non-root, no network, read-only fs, no Docker socket,
  timeouts, output caps) pass inside the real container.
* **Tests**: `pytest` 65/65 passing (unit, Docker integration, real Qwen tokenizer + TRL trainer masking).
* **Trainer**: `--dry-run` with the real Qwen2.5-Coder-1.5B-Instruct weights on CPU: 2,179,072 trainable
  LoRA parameters (0.141 %) for `q,v`, 18,464,768 (1.18 %) for all projections, loss mask verified on
  200 real examples, 0.8 % of the 5k set dropped by the 1,024-token limit; short CPU runs with a tiny
  Qwen2 stand-in exercised training/eval/checkpoint/early-stopping/adapter saving on the real splits.
* **Evaluation pipeline**: real-model CPU smoke run (configs A and C, 3 MBPP validation problems) produced
  `results_A/C.jsonl`, `aggregate_metrics.json` and plots; all three problems were genuine model failures
  (3 problems is a plumbing check, not a result).
* **App**: Gradio UI built and served against the real audit/split artefacts.

**Not executed here** (needs the RTX 5070 Ti machine): the LoRA/QLoRA training runs and the full MBPP
A–D evaluation. No training or evaluation numbers are claimed until those runs exist in `runs/` and
`results/`; launch them with the commands above.

## Licences and attribution

* KodCode (`KodCode/KodCode-V1`, Xu et al. 2025) and the derivative `F-A-I-L/kodcode-verified-python-235k`
  are **CC BY-NC 4.0** — non-commercial use only, attribution required; adapters trained on them
  inherit this restriction.
* MBPP: CC BY 4.0. MBPP+ (EvalPlus): Apache-2.0. Qwen2.5-Coder: Apache-2.0.
* Project code: MIT.
