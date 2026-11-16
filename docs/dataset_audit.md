# KodCode dataset audit — findings

Executed on 2026-11-13 on the development laptop (CPU + Docker Desktop) with
`python scripts/audit_kodcode.py` (full run incl. sandbox execution; 571 s after the upstream
test index was cached, 1,236 s on the first run that built it). Raw outputs: `data/audit/`
(`audit_stats.json`, `accepted.jsonl`, `rejected.jsonl`, `contamination_hits.jsonl`,
`execution_results.jsonl`, `mutation_results.jsonl`), provenance: `data/raw/provenance.json`.

## Sources (pinned)

| Source | Revision | Rows | Licence |
|---|---|---|---|
| `F-A-I-L/kodcode-verified-python-235k` (`kodcode_verified_python.jsonl`) | `4ba3098799a405943ce4c74a3449a80e4c21841d` | 234,555 | CC BY-NC 4.0 (inherited) |
| `KodCode/KodCode-V1` (upstream tests, 15 parquet shards, column-projected) | `97d81f0695ddd7c5c02bfe16436db9a17a4a21e5` | 484,097 | CC BY-NC 4.0 |
| `google-research-datasets/mbpp` (`full`) — screening + evaluation only | `4bb6404fdc6cacfda99d4ac4205087b89d32030c` | 374/90/500/10 | CC BY 4.0 |

Schema confirmed: `messages` (system/user/assistant), `id`, `uid`, `subset`, `style`, `difficulty`,
`verification{passed, verdict, …}`, `quality{n_asserts, value_assert, constrains, eval_similarity, …}`.
Upstream rows carry `question`, `solution`, `test` (pytest file importing `solution`), `test_info`.

## Funnel

| Stage | Rows | Notes |
|---|---|---|
| Verified JSONL | 234,555 | |
| Stage 1 – metadata filters | **128,642** kept | removed: `style=complete` 70,684; `quality.constrains=false` 30,323; `eval_similarity>0.60` 1,991; prompt > 3,000 chars 1,668; excluded test shapes (mock/async/fixture/class/setup_module) 757; `n_asserts<2` 323; no value assert 93; solution > 4,000 chars 74 |
| Stage 2 – deterministic pool | 30,000 | blake2b(`seed:id`) ordering, seed 42 |
| Stage 3 – upstream tests retrieved | 30,000 / 30,000 | 0 ids missing |
| Stage 4 – static checks + dedup + contamination | **22,940 accepted**, 7,060 rejected | see table below |
| Stage 5 – sandbox execution (sample) | 0 additional removals | 250/250 reproduced, 0 vacuous suites |

Static rejection reasons (stage 4):

| Reason | Rows | What it catches |
|---|---|---|
| `test_no_value_asserts` | 1,773 | upstream suite has no assert comparing a call result to a value (our AST recount disagrees with the publisher's `value_assert` flag on these rows) |
| `near_duplicate` | 1,648 | identical normalised solution (docstrings/comments stripped) or identical normalised instruction already accepted |
| `no_top_level_function` | 1,639 | class-only or script-style answers (we train function generation) |
| `top_level_io` | 691 | module-level `print`/`input` (stdin/stdout programs) |
| `disallowed_import` / `test_disallowed_import` | 463 / 100 | os, subprocess, socket, threading, numpy/pandas/torch, … (not supported in the sandbox or unsafe) |
| `instruction_solution_misaligned` | 447 | <15 % of instruction keywords appear in the code and function name tokens absent from the instruction |
| `unsupported_import` | 240 | third-party or unusual stdlib module outside the allow-list |
| `test_imports_undefined_name` | 28 | test imports a name the solution does not define |
| `tests_do_not_reference_solution_functions` | 24 | asserts never call a solution function |
| `mbpp_contamination` | 7 | see below |

`solution_differs_from_upstream` never fired: every extracted assistant code block was byte-identical
(after whitespace stripping) to the upstream `solution`, confirming the publisher's claim on this pool.

## Independent re-execution and test effectiveness

* **Execution**: a stratified sample (subset × difficulty) of **250** accepted records had its reference
  solution executed against the upstream pytest file inside our Docker sandbox (Python 3.11, pytest 8.4.2,
  no network, 5 s/test). **250/250 passed** (mean 0.99 s per run). Publisher verdicts therefore reproduce
  on this slice; the publisher's own audit reported 599/600.
* **Mutation checks**: for 60 of those records we generated up to four AST mutants each (body stub
  `return None`, first `return` → `0`, flipped comparison, swapped arithmetic operator) and re-ran the
  tests: **184 mutants, 180 killed (97.8 %)**, **0 stub mutants survived**. The four surviving mutants are
  operator/comparison flips on code paths the suite does not reach; no suite was vacuous. (The publisher
  measured 0.33 % stub survival corpus-wide; our `quality.constrains` + value-assert filters remove the
  rows where that risk concentrates.)

## Contamination screening (vs. all MBPP splits)

Independent of the publisher's TF-IDF filter: normalised-token Jaccard (threshold 0.8), shared 8-grams,
exact normalised-solution match, and function name+arity coincidence with Jaccard ≥ 0.4.

* Max prompt Jaccard observed over 24,000+ screened rows: **0.50** (p95 = 0.23, median = 0.10) — no
  prompt-level hit and no shared 8-gram with any MBPP problem.
* **7 rows rejected**: 5 whose normalised solution is identical to the MBPP *train* reference of
  task 958 (`int_to_roman`, the canonical value/symbol-table implementation that KodCode reproduces
  verbatim in several problems), and 2 whose entry-point name and arity
  coincide with an MBPP function (tasks 395 test, 733 train) at Jaccard ≈ 0.4. All seven are
  excluded from training. No hit involved MBPP validation/test *prompts*.
* Unresolved risk: semantic paraphrases below these thresholds, and MBPP's presence in Qwen's
  pre-training data, which affects all configurations equally (see methodology).

## Accepted data profile (22,940 rows)

| | |
|---|---|
| Subsets | Filter 8,130 · Taco 2,837 · Leetcode 2,366 · Algorithm 1,789 · Prefill 1,572 · Code_Contests 1,508 · Codeforces 1,473 · Data_Structure 1,213 · Apps 898 · Package 597 · Evol 463 · Docs 94 |
| Difficulty | easy 13,694 (59.7 %) · medium 5,046 (22.0 %) · hard 4,200 (18.3 %) |
| Solution length (non-blank lines) | median 16, p95 33, max 107 |
| Instruction length (chars) | median 605, p95 1,940 |
| Upstream asserts per problem | median 7, p95 13 |
| Distinct entry-point names | 11,990 |
| Top imports | collections, typing, re, heapq, math, random, itertools (all stdlib) |
| Tokens (Qwen chat template, system+user+assistant), n=3,000 | prompt median 167 (p95 530); total median 339, p95 775, max 1,731; **0.6 % exceed 1,024** (dropped at training time, never truncated) |

## Splits (`data/processed/manifest.json`)

Single seeded (42) stratified ordering → `validation` 300 (held-out KodCode, with tests in
`validation_records.jsonl`), `train_2000` ⊂ `train_5000` ⊂ `train_20000` (nested, asserted; the 20k
set fits inside the 22,940 accepted rows without a further audit). Subset proportions are preserved
within ±1 row per stratum (e.g. Filter 35.5 % / 35.4 % / 35.4 %). Files are TRL prompt-completion
conversations with the project system prompt; SHA-256 per file in the manifest.

**Prompt-test augmentation.** A deterministic 50 % of ids are selected to show one upstream assert in
the user turn (same template as the MBPP evaluation prompt). Only self-contained asserts qualify,
so the realised share is 39.4 % (`train_20000`: 7,879 rows; `train_5000`: 1,944; `validation`: 108).
The remaining selected rows keep the instruction-only prompt.

## Representative-sample review

The 250 executed records double as the manual-review sample: ids and per-record status are in
`execution_results.jsonl`; `accepted.jsonl` keeps `audit.alignment_score`, `audit.n_asserts`,
`audit.n_value_asserts`, `audit.contamination` and (for sampled rows) `audit.execution` /
`audit.mutation` per record.

## Caveats

* The alignment heuristic is lexical; 447 rejections are conservative, not proof of misalignment.
* Near-duplicate detection is exact on normalised text; paraphrased duplicates may remain.
* Only `instruct` prompts are used; `complete` (stub+doctest) rows are out of scope by design.
* KodCode is synthetic (GPT-generated problems/tests) and skews easy; this is inherited.
