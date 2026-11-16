"""Reproducible training/validation splits from the audited KodCode records.

* A held-out KodCode validation subset (``validation_size`` rows) is drawn first and never
  overlaps any training subset.
* Training subsets are **nested**: ``train_2000`` is a prefix of ``train_5000`` under a single
  stratified, seeded ordering, so the 2k-vs-5k comparison isolates dataset size.
* Stratification by (subset, difficulty) keeps the distributions of all subsets similar.
* Every output file ships with a manifest (row counts, SHA-256, id lists, config, seed).
"""
from __future__ import annotations

import argparse
import collections
import random
from pathlib import Path
from typing import Any, Optional

from codegenai.config import Config, load_config, resolve_path
from codegenai.data.download import stable_hash
from codegenai.data.extract import extract_standalone_asserts
from codegenai.data.formatting import record_to_example
from codegenai.data.schema import KodCodeRecord, read_jsonl, sha256_file, write_json, write_jsonl
from codegenai.logging_utils import get_logger, setup_logging

log = get_logger(__name__)


def stratified_order(records: list[KodCodeRecord], keys: list[str], seed: int) -> list[KodCodeRecord]:
    """Interleave strata proportionally so every prefix of the order is ~stratified."""
    rng = random.Random(seed)
    strata: dict[tuple, list[KodCodeRecord]] = collections.defaultdict(list)
    for r in records:
        strata[tuple(getattr(r, k) for k in keys)].append(r)
    for items in strata.values():
        rng.shuffle(items)
    total = len(records)
    # Weighted round-robin: emit from the stratum whose emitted share lags its target share most.
    emitted = {k: 0 for k in strata}
    order: list[KodCodeRecord] = []
    while len(order) < total:
        best = None
        best_score = None
        for k, items in strata.items():
            if emitted[k] >= len(items):
                continue
            target = len(items) / total
            score = emitted[k] / max(1, len(order)) - target if order else -target
            if best_score is None or score < best_score:
                best, best_score = k, score
        order.append(strata[best][emitted[best]])
        emitted[best] += 1
    return order


def build_splits(cfg: Config, accepted_path: Optional[Path] = None) -> dict[str, Any]:
    audit_dir = resolve_path(cfg.paths.audit_dir)
    out_dir = resolve_path(cfg.paths.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = accepted_path or (audit_dir / "accepted.jsonl")
    records = [KodCodeRecord.from_dict(d) for d in read_jsonl(accepted_path)]
    if not records:
        raise RuntimeError(f"No accepted records in {accepted_path}; run the audit first.")
    keys = cfg.audit.stratify_by
    order = stratified_order(records, keys, cfg.data.split_seed)

    n_val = cfg.data.validation_size
    sizes = sorted(cfg.data.train_sizes)
    needed = n_val + sizes[-1]
    if len(order) < needed:
        log.warning("Only %d accepted records for %d requested (validation %d + train %d); the largest "
                    "training subset will be truncated.", len(order), needed, n_val, sizes[-1])
    validation = order[:n_val]
    train_pool = order[n_val:]
    val_ids = {r.id for r in validation}

    manifest: dict[str, Any] = {
        "seed": cfg.data.split_seed, "stratify_by": keys, "source": str(accepted_path),
        "source_sha256": sha256_file(accepted_path), "n_accepted": len(records), "files": {},
        "system_prompt": cfg.generation.system_prompt, "format": "trl prompt-completion (conversational)",
        "prompt_test_fraction": cfg.data.train_prompt_test_fraction,
        "prompt_max_tests": cfg.data.train_prompt_max_tests,
    }

    def _prompt_tests(rec: KodCodeRecord) -> list[str]:
        """Upstream asserts shown in the prompt for a deterministic ``train_prompt_test_fraction`` of ids."""
        frac = cfg.data.train_prompt_test_fraction
        if frac <= 0 or not rec.test:
            return []
        if stable_hash(f"{cfg.data.split_seed}:prompt:{rec.id}") % 1000 >= int(frac * 1000):
            return []
        return extract_standalone_asserts(rec.test, rec.entry_point, cfg.data.train_prompt_max_tests)

    def _write(name: str, recs: list[KodCodeRecord]) -> None:
        path = out_dir / f"{name}.jsonl"
        examples = [record_to_example(r, cfg.generation.system_prompt, _prompt_tests(r)) for r in recs]
        n = write_jsonl(path, (ex.to_dict() for ex in examples))
        ids_path = out_dir / f"{name}_ids.json"
        write_json(ids_path, [r.id for r in recs])
        manifest["files"][name] = {
            "path": str(path), "rows": n, "sha256": sha256_file(path), "ids_file": str(ids_path),
            "rows_with_prompt_tests": sum(1 for ex in examples if ex.meta["n_prompt_tests"]),
            "by_subset": dict(collections.Counter(r.subset for r in recs)),
            "by_difficulty": dict(collections.Counter(r.difficulty for r in recs)),
        }
        log.info("Wrote %s (%d rows, %d with prompt tests)", path, n, manifest["files"][name]["rows_with_prompt_tests"])

    _write("validation", validation)
    # Validation records also keep their tests for execution-graded validation.
    write_jsonl(out_dir / "validation_records.jsonl", (r.to_dict() for r in validation))
    for size in sizes:
        subset = train_pool[:size]
        assert not ({r.id for r in subset} & val_ids), "validation/train overlap"
        _write(f"train_{size}", subset)
    # nested check
    for a, b in zip(sizes, sizes[1:]):
        ids_a = [r.id for r in train_pool[:a]]
        ids_b = [r.id for r in train_pool[:b]]
        assert ids_b[:a] == ids_a, "training subsets are not nested"
    manifest["nested"] = True
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build reproducible KodCode training/validation splits.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--accepted", default=None, help="path to accepted.jsonl (default: audit dir)")
    parser.add_argument("-o", "--override", action="append", default=[])
    args = parser.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config, args.override)
    manifest = build_splits(cfg, Path(args.accepted) if args.accepted else None)
    for name, info in manifest["files"].items():
        print(f"{name:12s} rows={info['rows']:5d} sha256={info['sha256'][:12]} subsets={info['by_subset']}")


if __name__ == "__main__":
    main()
