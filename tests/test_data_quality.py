"""Dataset processing: metadata filters, static checks, contamination, mutation, splits."""
from __future__ import annotations

import json


from codegenai.data.audit import metadata_reject_reason, static_check, stratified_sample
from codegenai.data.contamination import ContaminationScreener, jaccard, ngrams, normalise_text
from codegenai.data.formatting import record_to_example
from codegenai.data.mbpp import build_problem, infer_entry_point, verify_plus_mapping
from codegenai.data.mutation import generate_mutants
from codegenai.data.schema import KodCodeRecord, read_jsonl
from codegenai.data.splits import build_splits, stratified_order


def _row(**over):
    base = {
        "id": "Algorithm_7_I", "uid": "kodcode/Algorithm_7_I", "subset": "Algorithm", "style": "instruct",
        "difficulty": "easy",
        "messages": [{"role": "system", "content": "sys"},
                     {"role": "user", "content": "Write a function add(a, b) that returns the sum of two integers."},
                     {"role": "assistant", "content": "```python\ndef add(a, b):\n    return a + b\n```"}],
        "verification": {"passed": True, "verdict": "pass"},
        "quality": {"constrains": True, "value_assert": True, "n_asserts": 3, "eval_similarity": 0.3,
                    "uses_mock": False, "async_test": False, "pytest_fixture": False, "class_based": False,
                    "setup_module": False},
    }
    base.update(over)
    return base


UPSTREAM = {"question_id": "Algorithm_7_I", "solution": "def add(a, b):\n    return a + b\n",
            "test": ("from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n\n"
                     "def test_neg():\n    assert add(-1, -1) == -2\n"),
            "test_info": [{"function_name": "add"}]}


def test_metadata_filters(cfg):
    assert metadata_reject_reason(_row(), cfg) is None
    assert metadata_reject_reason(_row(style="complete"), cfg).startswith("style=")
    assert metadata_reject_reason(_row(verification={"passed": False, "verdict": "fail"}), cfg) == "verification_not_passed"
    q = dict(_row()["quality"]); q["constrains"] = False
    assert metadata_reject_reason(_row(quality=q), cfg) == "tests_do_not_constrain_solution"
    q = dict(_row()["quality"]); q["eval_similarity"] = 0.65
    assert "eval_similarity" in metadata_reject_reason(_row(quality=q), cfg)
    q = dict(_row()["quality"]); q["uses_mock"] = True
    assert metadata_reject_reason(_row(quality=q), cfg) == "test_shape:uses_mock"


def test_static_check_accepts_clean_row_and_rejects_bad_ones(cfg):
    v = static_check(_row(), UPSTREAM, cfg)
    assert v.record is not None and v.record.entry_point == "add" and v.record.function_names == ["add"]
    assert static_check(_row(), None, cfg).reject_reason == "upstream_test_missing"
    # disallowed import
    bad = _row(); bad["messages"][2]["content"] = "```python\nimport os\ndef add(a, b):\n    return a + b\n```"
    up = dict(UPSTREAM); up["solution"] = "import os\ndef add(a, b):\n    return a + b\n"
    assert static_check(bad, up, cfg).reject_reason.startswith("disallowed_import")
    # solution differs from upstream
    up = dict(UPSTREAM); up["solution"] = "def add(a, b):\n    return b + a\n"
    assert static_check(_row(), up, cfg).reject_reason == "solution_differs_from_upstream"
    # tests don't import solution
    up = dict(UPSTREAM); up["test"] = "def test_x():\n    assert 1 + 2 == 3\n"
    assert static_check(_row(), up, cfg).reject_reason == "test_does_not_import_solution"
    # two code blocks
    bad = _row(); bad["messages"][2]["content"] = "```python\nx=1\n```\n```python\ny=2\n```"
    assert static_check(bad, UPSTREAM, cfg).reject_reason == "assistant_blocks=2"
    # syntax error
    bad = _row(); bad["messages"][2]["content"] = "```python\ndef add(a, b)\n    return a + b\n```"
    assert static_check(bad, UPSTREAM, cfg).reject_reason == "solution_syntax_error"


def test_contamination_screener_detects_mbpp_overlap():
    mbpp = [{"task_id": 11, "split": "test", "text": "Write a python function to remove first and last occurrence of a given character from the string.",
             "code": "def remove_Occ(s,ch):\n    return s\n"}]
    sc = ContaminationScreener(mbpp, jaccard_threshold=0.8, ngram=8)
    hit = sc.screen("Write a python function to remove first and last occurrence of a given character from the string.",
                    "def other(x):\n    return x\n")
    assert hit.contaminated and any(h.kind == "prompt_jaccard" for h in hit.hits)
    exact = sc.screen("Totally different task about graphs.", "def remove_Occ(s,ch):\n    # c\n    return s\n")
    assert exact.contaminated and any(h.kind == "solution_exact" for h in exact.hits)
    clean = sc.screen("Compute the maximum subarray sum using Kadane's algorithm.", "def kadane(a):\n    return max(a)\n")
    assert not clean.contaminated and clean.max_prompt_jaccard < 0.8
    assert jaccard(set(), set()) == 0.0 and ngrams(["a", "b"], 3) == {("a", "b")}
    assert "python" not in normalise_text("Write a python function")


def test_generate_mutants_are_valid_and_distinct():
    code = "def f(a, b):\n    if a < b:\n        return a + b\n    return a - b\n"
    muts = generate_mutants(code, "f")
    kinds = {m.kind for m in muts}
    assert {"stub", "return_zero", "flip_compare", "swap_arith"} <= kinds
    for m in muts:
        compile(m.code, "<m>", "exec")
        assert m.code.strip() != code.strip()
    stub = next(m for m in muts if m.kind == "stub")
    assert "return None" in stub.code
    assert generate_mutants("x = 1\n", None) == []
    assert generate_mutants("def f(:\n", "f") == []


def test_stratified_sample_and_order_cover_strata():
    recs = []
    for i in range(60):
        recs.append(KodCodeRecord(id=f"r{i}", uid=f"k/r{i}", subset="A" if i % 3 else "B", style="instruct",
                                  difficulty="easy" if i % 2 else "hard", instruction="x", solution="def f(): pass\n",
                                  assistant_raw="", system_prompt=""))
    s = stratified_sample(recs, 12, ["subset", "difficulty"], 42)
    assert len(s) == 12 and {(r.subset, r.difficulty) for r in s} == {(r.subset, r.difficulty) for r in recs}
    order = stratified_order(recs, ["subset", "difficulty"], 42)
    assert len(order) == 60 and len({r.id for r in order}) == 60
    # determinism
    assert [r.id for r in stratified_order(recs, ["subset", "difficulty"], 42)] == [r.id for r in order]
    # early prefix is roughly stratified
    prefix = order[:20]
    assert sum(1 for r in prefix if r.subset == "B") in range(4, 10)


def test_build_splits_nested_and_disjoint(tmp_path, cfg, sample_record):
    recs = []
    for i in range(40):
        d = sample_record.to_dict()
        d["id"] = f"id{i}"; d["subset"] = "A" if i % 2 else "B"
        recs.append(d)
    acc = tmp_path / "accepted.jsonl"
    acc.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    c = cfg.model_copy(deep=True)
    c.paths.processed_dir = tmp_path / "processed"
    c.data.validation_size = 10
    c.data.train_sizes = [10, 25]
    man = build_splits(c, acc)
    val = json.loads((tmp_path / "processed" / "validation_ids.json").read_text())
    t10 = json.loads((tmp_path / "processed" / "train_10_ids.json").read_text())
    t25 = json.loads((tmp_path / "processed" / "train_25_ids.json").read_text())
    assert len(val) == 10 and len(t10) == 10 and len(t25) == 25
    assert not set(val) & set(t25) and t25[:10] == t10 and man["nested"]
    row = next(read_jsonl(tmp_path / "processed" / "train_10.jsonl"))
    assert [m["role"] for m in row["prompt"]] == ["system", "user"] and row["completion"][0]["role"] == "assistant"
    assert row["completion"][0]["content"].startswith("```python\n") and row["completion"][0]["content"].endswith("```")


def test_record_to_example_uses_consistent_prompt(sample_record, cfg):
    ex = record_to_example(sample_record, cfg.generation.system_prompt)
    assert ex.prompt[0]["content"] == cfg.generation.system_prompt
    assert ex.prompt[1]["content"] == sample_record.instruction
    assert sample_record.solution.strip() in ex.completion[0]["content"]


def test_mbpp_problem_visible_hidden_split():
    row = {"task_id": 11, "text": "Write a python function to remove first and last occurrence of a given character from the string.",
           "code": "def remove_Occ(s,ch): \r\n    return s\r\n",
           "test_list": ['assert remove_Occ("hello","l") == "heo"', 'assert remove_Occ("abcda","a") == "bcd"',
                         'assert remove_Occ("PHP","P") == "H"'],
           "test_setup_code": "", "challenge_test_list": ['assert remove_Occ("","l") == ""']}
    p = build_problem(row, "test", "first", "all")
    assert p.entry_point == "remove_Occ" and p.visible_tests == row["test_list"][:1] and len(p.hidden_tests) == 3
    p2 = build_problem(row, "test", "first", "disjoint", include_challenge=True)
    assert len(p2.hidden_tests) == 3 and row["test_list"][0] not in p2.hidden_tests
    p3 = build_problem(row, "test", "none", "all")
    assert p3.visible_tests == []
    assert infer_entry_point(["assert set(similar_elements((3, 4),(5, 4))) == set((4,))"], "def similar_elements(a,b): pass") == "similar_elements"
    assert verify_plus_mapping(row, {"prompt": row["text"], "test_list": []})
    assert not verify_plus_mapping(row, {"prompt": "other", "test_list": ["assert x == 1"]})
