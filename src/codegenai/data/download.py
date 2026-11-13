"""Programmatic, revision-pinned dataset downloads with local caching and provenance.

Three sources are used:

* ``F-A-I-L/kodcode-verified-python-235k`` – SFT rows (system/user/assistant ``messages``).
* ``KodCode/KodCode-V1`` – upstream problems; we retrieve the pytest ``test`` files per id so
  the reference solutions can be re-executed independently in our sandbox.
* ``google-research-datasets/mbpp`` (and optionally ``evalplus/mbppplus``) – evaluation only.

The upstream dataset is 2.6 GB of parquet, most of it GPT trial transcripts we never need.
``build_upstream_test_index`` therefore reads only the required columns through
``HfFileSystem`` (HTTP range requests) and keeps just the requested ids.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from codegenai.config import Config, load_config, resolve_path
from codegenai.data.schema import read_jsonl, sha256_file, write_json
from codegenai.logging_utils import get_logger, setup_logging

log = get_logger(__name__)

KODCODE_FILENAME = "kodcode_verified_python.jsonl"
UPSTREAM_COLUMNS = [
    "question_id", "style", "subset", "question", "solution", "test", "test_info",
    "gpt_difficulty", "benchmark_similarity", "benchmark_instruction", "benchmark_task_id",
    "filter_reason",
]


def _cache_dir(cfg: Config) -> Optional[str]:
    return str(resolve_path(cfg.paths.hf_cache_dir)) if cfg.paths.hf_cache_dir else None


# --------------------------------------------------------------------------- KodCode (verified)
def download_kodcode_jsonl(cfg: Config) -> Path:
    """Download the verified KodCode JSONL at the pinned revision; return the local path."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=cfg.data.kodcode_id,
        filename=KODCODE_FILENAME,
        repo_type="dataset",
        revision=cfg.data.kodcode_revision,
        cache_dir=_cache_dir(cfg),
    )
    log.info("KodCode verified JSONL at %s", path)
    return Path(path)


def iter_kodcode_rows(path: Path) -> Iterator[dict[str, Any]]:
    yield from read_jsonl(path)


def stable_hash(text: str) -> int:
    """Deterministic 64-bit hash (Python's ``hash`` is salted per process)."""
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


# --------------------------------------------------------------------------- KodCode-V1 upstream
def _upstream_parquet_files(cfg: Config) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(cfg.data.upstream_id, repo_type="dataset", revision=cfg.data.upstream_revision)
    return sorted(f for f in files if f.startswith("data/train-") and f.endswith(".parquet"))


def build_upstream_test_index(cfg: Config, ids: Iterable[str], out_path: Path,
                              force: bool = False) -> Path:
    """Retrieve upstream KodCode-V1 rows (tests, solutions, metadata) for the given ids.

    Reads only ``UPSTREAM_COLUMNS`` from each parquet shard via HTTP range requests and
    writes the matching rows to ``out_path`` (JSONL). Re-runs are incremental: ids already
    present are skipped unless ``force`` is set.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    out_path = Path(out_path)
    wanted = set(ids)
    have: set[str] = set()
    if out_path.exists() and not force:
        have = {r["question_id"] for r in read_jsonl(out_path)}
    missing = wanted - have
    log.info("Upstream index: %d wanted, %d cached, %d to fetch", len(wanted), len(have & wanted), len(missing))
    if not missing:
        return out_path

    fs = HfFileSystem()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    found = 0
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as fh:
        for rel in _upstream_parquet_files(cfg):
            url = f"datasets/{cfg.data.upstream_id}@{cfg.data.upstream_revision}/{rel}"
            log.info("Scanning %s", rel)
            with fs.open(url, "rb") as f:
                pf = pq.ParquetFile(f)
                for rg in range(pf.num_row_groups):
                    # First read ids only, then fetch the heavy columns just for matching rows.
                    id_col = pf.read_row_group(rg, columns=["question_id"]).column(0).to_pylist()
                    hits = [i for i, qid in enumerate(id_col) if qid in missing]
                    if not hits:
                        continue
                    table = pf.read_row_group(rg, columns=UPSTREAM_COLUMNS).take(hits).to_pylist()
                    for row in table:
                        row["test_info"] = row.get("test_info") or []
                        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                        found += 1
                    missing -= {r["question_id"] for r in table}
            if not missing:
                break
    log.info("Fetched %d upstream rows in %.1fs (%d ids not found)", found, time.time() - t0, len(missing))
    return out_path


def load_upstream_index(path: Path) -> dict[str, dict[str, Any]]:
    if not Path(path).exists():
        return {}
    return {r["question_id"]: r for r in read_jsonl(path)}


# --------------------------------------------------------------------------- MBPP / MBPP+
def load_mbpp(cfg: Config, split: str):
    """Load an MBPP split at the pinned revision (``datasets.Dataset``)."""
    from datasets import load_dataset

    return load_dataset(
        cfg.data.mbpp_id, cfg.data.mbpp_config, split=split,
        revision=cfg.data.mbpp_revision, cache_dir=_cache_dir(cfg),
    )


def load_mbppplus(cfg: Config):
    from datasets import load_dataset

    return load_dataset(
        cfg.data.mbppplus_id, split="test", revision=cfg.data.mbppplus_revision, cache_dir=_cache_dir(cfg),
    )


# --------------------------------------------------------------------------- manifest
def write_provenance(cfg: Config, kodcode_path: Path, extra: Optional[dict[str, Any]] = None) -> Path:
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": {
            "kodcode_verified": {
                "repo_id": cfg.data.kodcode_id,
                "revision": cfg.data.kodcode_revision,
                "file": KODCODE_FILENAME,
                "sha256": sha256_file(kodcode_path),
                "license": cfg.data.kodcode_license,
                "attribution": "KodCode (Xu et al., 2025), https://kodcode-ai.github.io/ ; "
                               "derivative release F-A-I-L/kodcode-verified-python-235k",
            },
            "kodcode_v1_upstream": {
                "repo_id": cfg.data.upstream_id,
                "revision": cfg.data.upstream_revision,
                "license": "CC BY-NC 4.0",
                "columns": UPSTREAM_COLUMNS,
            },
            "mbpp": {
                "repo_id": cfg.data.mbpp_id,
                "revision": cfg.data.mbpp_revision,
                "config": cfg.data.mbpp_config,
                "license": "CC BY 4.0",
                "usage": "validation split = model selection; test split = final evaluation only",
            },
            "mbppplus": {
                "repo_id": cfg.data.mbppplus_id,
                "revision": cfg.data.mbppplus_revision,
                "license": "Apache-2.0",
                "enabled": cfg.evaluation.use_mbppplus,
            },
        },
    }
    if extra:
        manifest.update(extra)
    out = resolve_path(cfg.paths.data_dir) / "provenance.json"
    write_json(out, manifest)
    log.info("Wrote provenance manifest to %s", out)
    return out


# --------------------------------------------------------------------------- CLI
def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Download CodeGenAI datasets (pinned revisions).")
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-mbpp", action="store_true")
    parser.add_argument("-o", "--override", action="append", default=[], help="section.key=value")
    args = parser.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config, args.override)

    kod = download_kodcode_jsonl(cfg)
    n = sum(1 for _ in open(kod, "rb"))
    log.info("KodCode verified rows: %d", n)
    if not args.skip_mbpp:
        for split in ("train", "validation", "test", "prompt"):
            ds = load_mbpp(cfg, split)
            log.info("MBPP %s/%s: %d rows", cfg.data.mbpp_config, split, len(ds))
        if cfg.evaluation.use_mbppplus:
            plus = load_mbppplus(cfg)
            log.info("MBPP+ rows: %d", len(plus))
    write_provenance(cfg, kod, {"kodcode_rows": n})


if __name__ == "__main__":
    main()
