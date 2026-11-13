# Methodology

## Research question

Does (a) LoRA fine-tuning of Qwen2.5-Coder-1.5B-Instruct on execution-verified KodCode problems and
(b) feedback-based self-correction with sandboxed unit tests improve the functional correctness of
Python code generated from natural-language instructions, and are the two effects additive?

## Pipeline

```
KodCode-verified (234,555 rows) -> metadata filters -> deterministic 30k pool -> upstream tests
   -> static checks -> contamination screen -> sandbox re-execution + mutation checks (sample)
   -> accepted.jsonl -> nested splits: validation (300) | train_2000 ⊂ train_5000
   -> SFTTrainer + PEFT LoRA (completion-only loss)       -> adapter
Instruction -> Qwen (base | +adapter) -> code -> sandbox (visible tests) -> structured feedback
   -> correction prompt (<= 2 rounds) -> final code -> sandbox (hidden tests) -> verdict
```

## Data

* **Training source**: `F-A-I-L/kodcode-verified-python-235k` @ `4ba3098…` (CC BY-NC 4.0, derivative
  of `KodCode/KodCode-V1`). Only `style == "instruct"` rows whose upstream tests passed on the
  publisher's side **and** in our own sandbox. Filters, rejection reasons and statistics:
  `data/audit/audit_stats.json`, narrative in `docs/dataset_audit.md`.
* **Upstream tests**: retrieved per `question_id` from `KodCode/KodCode-V1` @ `97d81f0…` (parquet
  column projection) and used (1) to re-verify reference solutions, (2) for mutation checks,
  (3) for execution-graded KodCode validation. They are *not* shown to the model during training
  (the SFT target is the solution only).
* **Splits**: one seeded, stratified (subset × difficulty) ordering of the accepted records; the
  first 300 form the KodCode validation set; `train_2000` is a prefix of `train_5000` so the
  2k-vs-5k comparison isolates data quantity. `data/processed/manifest.json` stores SHA-256 hashes
  and id lists.
* **Evaluation**: `google-research-datasets/mbpp` (`full` config) @ `4bb6404…`. The **validation**
  split (90 problems) is used for model selection (choosing between r=8/16 and 2k/5k adapters);
  the **test** split (500 problems) is reserved for the final A–D comparison and is run once per
  adapter chosen on validation. MBPP is never used for training. MBPP+ (`evalplus/mbppplus`) can be
  enabled with `evaluation.use_mbppplus=true`; task-id mapping is verified by matching prompt text
  or the first assert, and the MBPP+ `test` script is executed as an additional **hidden** test.

### Contamination control

The publisher removed rows with TF-IDF cosine ≥ 0.70 to HumanEval/MBPP prompts. Independently we
(1) tighten the publisher's `quality.eval_similarity` to ≤ 0.60, (2) compute normalised-token
Jaccard (≥ 0.8 rejects) and shared 8-gram overlap between each KodCode instruction and every MBPP
`text` (all four splits), (3) compare normalised solution code with MBPP reference code (exact
match rejects), and (4) reject function-name+arity coincidences with moderate prompt similarity.
Unresolved risk: semantic paraphrases below these thresholds, and the fact that Qwen2.5-Coder's
pre-training corpus may itself contain MBPP; the latter affects all four configurations equally and
is a limitation of the baseline model, not of our fine-tuning data.

## Prompting and decoding

Identical for every configuration:

* system prompt: `You are a Python coding assistant. Write correct code. Respond with a single code block.`
* user turn: MBPP `text` + `Your code should satisfy the following test(s):` + the **first** MBPP assert
  (standard MBPP practice; the model must learn the function name from it).
* Qwen chat template, greedy decoding (`do_sample=False`), `max_new_tokens=512`, seed 42.

The training completion uses the same system prompt and the same “single ```python block” answer
format, so the adapter is evaluated in-distribution.

### Visible vs. hidden tests

| Role | Content | Who sees it |
|---|---|---|
| Visible / feedback tests | `test_list[0]` | prompt + correction feedback |
| Hidden tests (`hidden_test_policy=all`, default) | all `test_list` asserts (+ MBPP+ script if enabled) | evaluator only |
| Hidden tests (`disjoint`) | `test_list[1:]` | evaluator only |

pass@1 and final success are always judged on the hidden tests. With the default policy the visible
assert is part of the hidden set (standard pass@1); per-assert outcomes are stored so the disjoint
view can be recomputed offline.

## Self-correction

Up to two correction rounds. The correction prompt contains the task, the previous code and a
structured description of the *visible* failures only (test name, exception type, trimmed message).
Loop stops on: pass, max attempts, identical code regenerated, sandbox unavailable, invalid output
(no code block → synthetic “does not parse” feedback). Each attempt (code, feedback, report,
latencies) is stored.

## Configurations and metrics

| | LoRA | Correction |
|---|---|---|
| A Baseline | – | – |
| B Fine-tuned | ✓ | – |
| C Baseline + correction | – | ✓ |
| D Fine-tuned + correction | ✓ | ✓ |

Per problem we record initial/final hidden verdicts and status categories (`passed`, `failed`,
`syntax_error`, `missing_function`, `runtime_error`, `timeout`), feedback-test outcomes, number of
corrections, token counts, generation / feedback-test / hidden-test latency and peak CUDA memory.
Aggregates: pass@1, final success, correction success (fixed ÷ initially failing), syntax validity,
runtime-error rate, average corrections, latencies, peak GPU memory, with 95 % Wilson intervals;
paired bootstrap (2,000 resamples) and exact McNemar tests for A→B, A→C, B→D, C→D, A→D.
Dataset-size and rank effects (2k vs 5k, r=8 vs r=16) are compared on MBPP validation and KodCode
validation loss; only the selected adapter is evaluated on MBPP test.

## Training

QLoRA: 4-bit NF4, double quantisation, bf16 compute; LoRA r=16, α=32, dropout 0.05 on `q_proj`,
`v_proj`; lr 1e-4 cosine with 3 % warm-up; 3 epochs; batch 1 × grad-accum 8; max length 1,024;
gradient checkpointing; paged 8-bit AdamW; seed 42. Base weights frozen (asserted at trainer
construction: every trainable parameter name contains `lora_`). Completion-only loss: TRL's
prompt-completion path masks the prompt; `codegenai.training.masking` recomputes the mask
independently and the trainer refuses to start if any prompt token is supervised or the `<|im_end|>`
token of a completion is missing. Examples whose prompt+completion exceed `max_seq_length` are
dropped (never truncated). MLflow logs hyper-parameters, loss, eval loss and GPU memory.

## Reproducibility

Pinned dataset revisions, seeded hash-based pooling and stratified ordering, SHA-256 manifests,
resolved configs saved next to every run/evaluation, `results_<cfg>.jsonl` written incrementally and
resumable, aggregate metrics regenerable with `--aggregate-only`.

## Known limitations

* 1.5B model, greedy decoding: pass@1 only (no pass@k sampling).
* MBPP asserts are few (3) and sometimes under-specified; MBPP+ mitigates this when enabled.
* KodCode and MBPP are both “function-level” Python; results may not transfer to larger programs.
* Non-commercial licence of KodCode restricts downstream use of the adapter.
