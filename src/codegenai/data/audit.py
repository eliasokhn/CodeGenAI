"""KodCode quality audit: metadata filters, static validation, independent re-execution,
mutation checks, contamination screening and statistics.

Pipeline (``run_audit``)::

    verified JSONL (234k rows)
      -> stage 1  metadata filters (style, verification, quality flags, subset, lengths)
      -> stage 2  deterministic candidate pool (hash-ordered sample of filtered ids)
      -> stage 3  upstream test retrieval (KodCode-V1, column projection)
      -> stage 4  static checks: code block extraction, syntax, standalone function, imports,
                  test analysis (asserts, value asserts, references solution), instruction/
                  solution alignment, near-duplicate detection, MBPP contamination screening
      -> stage 5  sandbox execution of a stratified sample (reference vs. upstream tests) and
                  mutation spot checks
      -> outputs  accepted.jsonl, rejected.jsonl (with reasons), audit_stats.json, manifest.json

Publisher verification metadata is treated as a *hint*, never as proof: a record is only
accepted when our own static checks pass, and the execution sample tells us how often the
publisher's "passed" verdict reproduces in our sandbox.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from codegenai.config import Config, load_config, resolve_path
from codegenai.data.contamination import ContaminationScreener
from codegenai.data.download import (
    build_upstream_test_index, download_kodcode_jsonl, iter_kodcode_rows, load_upstream_index,
    stable_hash, write_provenance,
)
from codegenai.data.extract import (
    analyze_code, analyze_tests, choose_entry_point, extract_code_blocks, strip_docstrings_and_comments,
)
from codegenai.data.mutation import generate_mutants
from codegenai.data.schema import KodCodeRecord, sha256_text, write_json, write_jsonl
from codegenai.logging_utils import get_logger, setup_logging
from codegenai.sandbox import DockerSandbox, Status, TestReport, get_sandbox
from codegenai.sandbox.harness import build_kodcode_bundle

log = get_logger(__name__)


# --------------------------------------------------------------------------- stage 1: metadata
def metadata_reject_reason(row: dict[str, Any], cfg: Config) -> Optional[str]:
    f = cfg.data.filters
    q = row.get("quality") or {}
    v = row.get("verification") or {}
    if row.get("style") not in f.styles:
        return f"style={row.get('style')}"
    if f.require_verification_passed and not (v.get("passed") is True and v.get("verdict") == "pass"):
        return "verification_not_passed"
    if f.require_constrains and not q.get("constrains"):
        return "tests_do_not_constrain_solution"
    if f.require_value_assert and not q.get("value_assert"):
        return "no_value_assert"
    if int(q.get("n_asserts") or 0) < f.min_asserts:
        return f"n_asserts<{f.min_asserts}"
    if float(q.get("eval_similarity") or 0.0) > f.max_eval_similarity:
        return f"eval_similarity>{f.max_eval_similarity}"
    if row.get("subset") in set(f.exclude_subsets):
        return f"subset_excluded:{row.get('subset')}"
    for shape in f.exclude_test_shapes:
        if q.get(shape):
            return f"test_shape:{shape}"
    msgs = row.get("messages") or []
    roles = [m.get("role") for m in msgs]
    if roles != ["system", "user", "assistant"]:
        return f"bad_roles:{roles}"
    if len(msgs[1]["content"]) > f.max_prompt_chars:
        return "prompt_too_long"
    if len(msgs[2]["content"]) > f.max_solution_chars + 20:
        return "solution_too_long"
    return None


def stage1_filter(jsonl_path: Path, cfg: Config) -> tuple[list[dict[str, Any]], collections.Counter]:
    """Pass over the full verified file; return rows passing metadata filters + reason counts."""
    kept: list[dict[str, Any]] = []
    reasons: collections.Counter = collections.Counter()
    total = 0
    for row in iter_kodcode_rows(jsonl_path):
        total += 1
        reason = metadata_reject_reason(row, cfg)
        if reason is None:
            kept.append(row)
        else:
            reasons[reason.split(":")[0]] += 1
    reasons["__total_rows__"] = total
    reasons["__kept__"] = len(kept)
    log.info("Stage 1: %d/%d rows pass metadata filters", len(kept), total)
    return kept, reasons


def stage2_pool(rows: list[dict[str, Any]], cfg: Config) -> list[dict[str, Any]]:
    """Deterministic candidate pool: order by blake2b(id) and take the first N."""
    rows = sorted(rows, key=lambda r: stable_hash(f"{cfg.data.split_seed}:{r['id']}"))
    pool = rows[: cfg.data.candidate_pool_size]
    log.info("Stage 2: candidate pool of %d rows (seed=%d)", len(pool), cfg.data.split_seed)
    return pool


# --------------------------------------------------------------------------- stage 4: static
@dataclass
class StaticVerdict:
    record: Optional[KodCodeRecord]
    reject_reason: Optional[str] = None
    flags: dict[str, Any] = field(default_factory=dict)


def _alignment_score(instruction: str, record_names: list[str], solution: str) -> float:
    """Crude instruction/solution alignment: fraction of instruction keywords that appear in code."""
    import re

    words = {w for w in re.findall(r"[a-z]{4,}", instruction.lower())}
    stop = {"function", "python", "write", "given", "return", "returns", "should", "that", "with", "which",
            "this", "from", "into", "list", "string", "integer", "example", "implement", "where", "each",
            "there", "then", "than", "them", "they", "will", "have", "when", "also", "only", "such", "your",
            "input", "output", "value", "values", "number", "numbers", "create", "takes", "take", "contains"}
    words -= stop
    instr_l = instruction.lower()
    # A function whose (snake_case) name tokens appear in the instruction is by construction on-topic.
    for name in record_names:
        parts = [p for p in name.lower().split("_") if len(p) >= 3]
        if parts and all(p in instr_l for p in parts):
            return 1.0
    if len(words) < 5:  # too short to judge; do not reject on this heuristic
        return 1.0
    code_l = solution.lower()
    hit = sum(1 for w in words if w in code_l or w.rstrip("s") in code_l
              or any(w in n.lower() for n in record_names))
    return hit / len(words)


def static_check(row: dict[str, Any], upstream: Optional[dict[str, Any]], cfg: Config) -> StaticVerdict:
    f = cfg.data.filters
    msgs = row["messages"]
    instruction, assistant = msgs[1]["content"], msgs[2]["content"]
    blocks = extract_code_blocks(assistant)
    if len(blocks) != 1:
        return StaticVerdict(None, f"assistant_blocks={len(blocks)}")
    solution = blocks[0].rstrip() + "\n"
    an = analyze_code(solution)
    if not an.syntax_ok:
        return StaticVerdict(None, "solution_syntax_error")
    if not an.functions:
        return StaticVerdict(None, "no_top_level_function")
    if an.has_top_level_io:
        return StaticVerdict(None, "top_level_io")
    if an.n_lines > f.max_solution_lines:
        return StaticVerdict(None, "solution_too_many_lines")
    bad = sorted(set(an.imports) & set(f.disallowed_imports))
    if bad:
        return StaticVerdict(None, f"disallowed_import:{bad[0]}")
    unknown = sorted(set(an.imports) - set(f.allowed_imports))
    if unknown:
        return StaticVerdict(None, f"unsupported_import:{unknown[0]}")
    if upstream is None:
        return StaticVerdict(None, "upstream_test_missing")
    test_src = upstream.get("test") or ""
    ta = analyze_tests(test_src)
    if not ta.syntax_ok:
        return StaticVerdict(None, "test_syntax_error")
    if not ta.imports_solution:
        return StaticVerdict(None, "test_does_not_import_solution")
    if ta.n_asserts < f.min_asserts:
        return StaticVerdict(None, "test_too_few_asserts")
    if ta.n_value_asserts == 0:
        return StaticVerdict(None, "test_no_value_asserts")
    if not (set(ta.referenced_names) | set(ta.imported_names)) & set(an.functions):
        return StaticVerdict(None, "tests_do_not_reference_solution_functions")
    missing = [n for n in ta.imported_names if n not in an.functions and n not in an.classes and n != "*"]
    if missing:
        return StaticVerdict(None, "test_imports_undefined_name")
    tan = analyze_code(test_src)
    bad_t = sorted(set(tan.imports) & set(f.disallowed_imports))
    if bad_t:
        return StaticVerdict(None, f"test_disallowed_import:{bad_t[0]}")
    if (upstream.get("solution") or "").strip() != solution.strip():
        # The publisher claims byte-identity with upstream; verify it and reject divergent rows.
        return StaticVerdict(None, "solution_differs_from_upstream")
    entry = choose_entry_point(an.functions, ta, upstream.get("test_info"))
    align = _alignment_score(instruction, an.functions, solution)
    if align < 0.15:
        return StaticVerdict(None, "instruction_solution_misaligned")
    rec = KodCodeRecord(
        id=row["id"], uid=row["uid"], subset=row["subset"], style=row["style"], difficulty=row["difficulty"],
        instruction=instruction.strip(), solution=solution, assistant_raw=assistant,
        system_prompt=msgs[0]["content"], test=test_src, test_info=upstream.get("test_info") or [],
        verification=row.get("verification") or {}, quality=row.get("quality") or {},
        function_names=an.functions, imports=an.imports, entry_point=entry, n_solution_lines=an.n_lines,
        audit={"alignment_score": round(align, 3), "n_asserts": ta.n_asserts,
               "n_value_asserts": ta.n_value_asserts, "n_test_functions": ta.n_test_functions},
    )
    return StaticVerdict(rec)


# --------------------------------------------------------------------------- stage 5: execution
def stratified_sample(records: list[KodCodeRecord], n: int, keys: list[str], seed: int) -> list[KodCodeRecord]:
    """Proportional stratified sample over the product of ``keys`` (at least 1 per stratum)."""
    rng = random.Random(seed)
    strata: dict[tuple, list[KodCodeRecord]] = collections.defaultdict(list)
    for r in records:
        strata[tuple(getattr(r, k) for k in keys)].append(r)
    total = len(records)
    out: list[KodCodeRecord] = []
    for key, items in sorted(strata.items()):
        k = max(1, round(n * len(items) / total))
        rng.shuffle(items)
        out.extend(items[:k])
    rng.shuffle(out)
    return out[:n]


def execute_reference(sb: DockerSandbox, rec: KodCodeRecord) -> TestReport:
    return sb.run(build_kodcode_bundle(rec.solution, rec.test or ""), entry_point=rec.entry_point)


def mutation_check(sb: DockerSandbox, rec: KodCodeRecord) -> dict[str, Any]:
    mutants = generate_mutants(rec.solution, rec.entry_point)
    results = []
    for m in mutants:
        rep = sb.run(build_kodcode_bundle(m.code, rec.test or ""), entry_point=rec.entry_point)
        results.append({"kind": m.kind, "killed": rep.status != Status.PASSED, "status": rep.status.value})
    killed = sum(1 for r in results if r["killed"])
    return {"n_mutants": len(results), "killed": killed, "results": results,
            "stub_survived": any(r["kind"] == "stub" and not r["killed"] for r in results)}


# --------------------------------------------------------------------------- statistics
def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    vs = sorted(values)
    pick = lambda q: vs[min(len(vs) - 1, int(q * (len(vs) - 1)))]  # noqa: E731
    return {"min": vs[0], "p25": pick(0.25), "median": pick(0.5), "p75": pick(0.75), "p95": pick(0.95),
            "max": vs[-1], "mean": round(statistics.fmean(vs), 2)}


def token_lengths(records: list[KodCodeRecord], cfg: Config, max_records: int = 3000) -> dict[str, Any]:
    """Token statistics under the Qwen chat template (requires the tokenizer; skipped offline)."""
    try:
        from transformers import AutoTokenizer

        from codegenai.data.formatting import record_to_example
        from codegenai.training.masking import tokenize_prompt_completion

        tok = AutoTokenizer.from_pretrained(cfg.model.model_id, revision=cfg.model.revision)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"tokenizer unavailable: {exc}"}
    prompt_lens, total_lens = [], []
    for rec in records[:max_records]:
        ex = record_to_example(rec, cfg.generation.system_prompt)
        tk = tokenize_prompt_completion(tok, ex.prompt, ex.completion)
        prompt_lens.append(tk["prompt_len"])
        total_lens.append(len(tk["input_ids"]))
    msl = cfg.training.max_seq_length
    return {
        "n": len(total_lens), "prompt_tokens": _percentiles(prompt_lens), "total_tokens": _percentiles(total_lens),
        f"fraction_over_{msl}": round(sum(1 for t in total_lens if t > msl) / max(1, len(total_lens)), 4),
        "tokenizer": cfg.model.model_id,
    }


def compute_stats(records: list[KodCodeRecord]) -> dict[str, Any]:
    by = lambda k: dict(collections.Counter(getattr(r, k) for r in records))  # noqa: E731
    return {
        "n": len(records),
        "by_subset": by("subset"), "by_difficulty": by("difficulty"), "by_style": by("style"),
        "solution_lines": _percentiles([r.n_solution_lines for r in records]),
        "solution_chars": _percentiles([len(r.solution) for r in records]),
        "instruction_chars": _percentiles([len(r.instruction) for r in records]),
        "n_asserts": _percentiles([r.audit.get("n_asserts", 0) for r in records]),
        "functions_per_solution": _percentiles([len(r.function_names) for r in records]),
        "top_imports": dict(collections.Counter(i for r in records for i in r.imports).most_common(20)),
        "distinct_entry_points": len({r.entry_point for r in records}),
    }


# --------------------------------------------------------------------------- orchestration
def run_audit(cfg: Config, execute: Optional[bool] = None, limit_pool: Optional[int] = None) -> dict[str, Any]:
    t_start = time.time()
    audit_dir = resolve_path(cfg.paths.audit_dir)
    raw_dir = resolve_path(cfg.paths.data_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    execute = cfg.audit.execute_in_sandbox if execute is None else execute

    jsonl = download_kodcode_jsonl(cfg)
    kept, stage1_reasons = stage1_filter(jsonl, cfg)
    pool = stage2_pool(kept, cfg)
    if limit_pool:
        pool = pool[:limit_pool]

    upstream_path = raw_dir / "kodcode_v1_upstream_subset.jsonl"
    build_upstream_test_index(cfg, (r["id"] for r in pool), upstream_path)
    upstream = load_upstream_index(upstream_path)

    # MBPP (all splits) for contamination screening
    from codegenai.data.mbpp import all_mbpp_rows_for_screening

    screener = ContaminationScreener(all_mbpp_rows_for_screening(cfg), cfg.audit.contamination_threshold,
                                     cfg.audit.contamination_ngram)

    accepted: list[KodCodeRecord] = []
    rejected: list[dict[str, Any]] = []
    reasons: collections.Counter = collections.Counter()
    seen_solutions: dict[str, str] = {}
    seen_instructions: dict[str, str] = {}
    contamination_hits: list[dict[str, Any]] = []
    max_jaccard: list[float] = []
    for row in pool:
        verdict = static_check(row, upstream.get(row["id"]), cfg)
        if verdict.record is None:
            reasons[verdict.reject_reason.split(":")[0]] += 1
            rejected.append({"id": row["id"], "reason": verdict.reject_reason})
            continue
        rec = verdict.record
        sol_key = sha256_text(strip_docstrings_and_comments(rec.solution))
        ins_key = sha256_text(" ".join(rec.instruction.lower().split()))
        if sol_key in seen_solutions or ins_key in seen_instructions:
            reasons["near_duplicate"] += 1
            rejected.append({"id": rec.id, "reason": f"near_duplicate_of:{seen_solutions.get(sol_key) or seen_instructions.get(ins_key)}"})
            continue
        cont = screener.screen(rec.instruction, rec.solution)
        max_jaccard.append(cont.max_prompt_jaccard)
        rec.audit["contamination"] = cont.to_dict()
        if cont.contaminated:
            reasons["mbpp_contamination"] += 1
            rejected.append({"id": rec.id, "reason": "mbpp_contamination", "hits": cont.to_dict()["hits"][:5]})
            contamination_hits.append({"id": rec.id, **cont.to_dict()})
            continue
        seen_solutions[sol_key] = rec.id
        seen_instructions[ins_key] = rec.id
        accepted.append(rec)
    log.info("Stage 4: %d accepted, %d rejected", len(accepted), len(rejected))

    # Stage 5: sandbox re-execution and mutation checks on a stratified sample
    execution: dict[str, Any] = {"executed": False}
    if execute:
        sb = get_sandbox(cfg.sandbox, require=False)
        if sb.check_available().available:
            sample = stratified_sample(accepted, cfg.audit.sample_size, cfg.audit.stratify_by, cfg.data.split_seed)
            exec_results: list[dict[str, Any]] = []
            failed_ids: set[str] = set()
            t0 = time.time()
            for i, rec in enumerate(sample):
                rep = execute_reference(sb, rec)
                exec_results.append({"id": rec.id, "status": rep.status.value, "passed": rep.passed,
                                     "failed": rep.failed, "time_s": round(rep.execution_time_s, 2),
                                     "summary": rep.summary[:300]})
                rec.audit["execution"] = exec_results[-1]
                if rep.status != Status.PASSED:
                    failed_ids.add(rec.id)
                if (i + 1) % 25 == 0:
                    log.info("  executed %d/%d (%.0fs)", i + 1, len(sample), time.time() - t0)
            mut_sample = [r for r in sample if r.id not in failed_ids][: cfg.audit.mutation_sample_size]
            mutation_results: list[dict[str, Any]] = []
            weak_ids: set[str] = set()
            for rec in mut_sample:
                m = mutation_check(sb, rec)
                m["id"] = rec.id
                mutation_results.append(m)
                rec.audit["mutation"] = {k: v for k, v in m.items() if k != "results"}
                if m["stub_survived"]:
                    weak_ids.add(rec.id)
            # Records whose reference fails in OUR sandbox or whose tests are vacuous are rejected.
            before = len(accepted)
            accepted = [r for r in accepted if r.id not in failed_ids and r.id not in weak_ids]
            for rid in failed_ids:
                rejected.append({"id": rid, "reason": "reference_failed_in_sandbox"})
                reasons["reference_failed_in_sandbox"] += 1
            for rid in weak_ids:
                rejected.append({"id": rid, "reason": "vacuous_tests_stub_survived"})
                reasons["vacuous_tests_stub_survived"] += 1
            statuses = collections.Counter(r["status"] for r in exec_results)
            execution = {
                "executed": True, "sample_size": len(sample), "statuses": dict(statuses),
                "reproduction_rate": round(statuses.get("passed", 0) / max(1, len(sample)), 4),
                "mean_exec_time_s": round(statistics.fmean([r["time_s"] for r in exec_results]), 2) if exec_results else None,
                "mutation": {
                    "sample_size": len(mutation_results),
                    "total_mutants": sum(m["n_mutants"] for m in mutation_results),
                    "killed": sum(m["killed"] for m in mutation_results),
                    "stub_survived": len(weak_ids),
                    "kill_rate": round(sum(m["killed"] for m in mutation_results)
                                       / max(1, sum(m["n_mutants"] for m in mutation_results)), 4),
                },
                "removed_after_execution": before - len(accepted),
                "sandbox_image": cfg.sandbox.image,
            }
            write_jsonl(audit_dir / "execution_results.jsonl", exec_results)
            write_jsonl(audit_dir / "mutation_results.jsonl", mutation_results)
        else:
            execution = {"executed": False, "reason": sb.check_available().reason}
            log.warning("Sandbox unavailable: execution/mutation audit skipped (%s)", execution["reason"])

    stats = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t_start, 1),
        "config": {"filters": cfg.data.filters.model_dump(), "audit": cfg.audit.model_dump(),
                   "candidate_pool_size": cfg.data.candidate_pool_size, "seed": cfg.data.split_seed},
        "stage1_metadata_filter": dict(stage1_reasons),
        "pool_size": len(pool),
        "upstream_rows_retrieved": sum(1 for r in pool if r["id"] in upstream),
        "static_rejections": dict(reasons),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "contamination": {
            "threshold_jaccard": cfg.audit.contamination_threshold, "ngram": cfg.audit.contamination_ngram,
            "hits": len(contamination_hits), "max_prompt_jaccard_observed": round(max(max_jaccard), 4) if max_jaccard else None,
            "prompt_jaccard_percentiles": _percentiles(max_jaccard),
            "mbpp_splits_screened": ["train", "validation", "test", "prompt"],
        },
        "execution_audit": execution,
        "accepted_stats": compute_stats(accepted),
        "token_stats": token_lengths(accepted, cfg),
    }
    write_jsonl(audit_dir / "accepted.jsonl", (r.to_dict() for r in accepted))
    write_jsonl(audit_dir / "rejected.jsonl", rejected)
    write_jsonl(audit_dir / "contamination_hits.jsonl", contamination_hits)
    write_json(audit_dir / "audit_stats.json", stats)
    write_provenance(cfg, jsonl, {"audit": {"accepted": len(accepted), "rejected": len(rejected),
                                            "stats_file": str(audit_dir / "audit_stats.json")}})
    log.info("Audit complete: %d accepted, %d rejected in %.0fs -> %s", len(accepted), len(rejected),
             stats["elapsed_s"], audit_dir)
    return stats


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Audit the KodCode dataset before training.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-execute", action="store_true", help="skip sandbox execution/mutation checks")
    parser.add_argument("--limit-pool", type=int, default=None, help="debug: only audit the first N pool rows")
    parser.add_argument("-o", "--override", action="append", default=[])
    args = parser.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config, args.override)
    stats = run_audit(cfg, execute=not args.no_execute, limit_pool=args.limit_pool)
    print(json.dumps({k: stats[k] for k in ("accepted", "rejected", "static_rejections", "execution_audit")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
