from pathlib import Path

import pytest

from codegenai.config import CONFIG_DIR, apply_overrides, load_config
from codegenai.data.extract import (
    analyze_code, analyze_tests, check_syntax, choose_entry_point, extract_code, extract_code_blocks,
    strip_docstrings_and_comments,
)


def test_base_config_loads_initial_settings(cfg):
    assert cfg.model.model_id == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert cfg.lora.r == 16 and cfg.lora.alpha == 32 and cfg.lora.dropout == 0.05
    assert cfg.lora.target_modules == ["q_proj", "v_proj"]
    assert cfg.training.learning_rate == 1e-4 and cfg.training.num_train_epochs == 3
    assert cfg.training.per_device_train_batch_size == 1 and cfg.training.gradient_accumulation_steps == 8
    assert cfg.training.max_seq_length == 1024 and cfg.training.gradient_checkpointing and cfg.training.seed == 42
    assert cfg.model.load_in_4bit and cfg.model.bnb_4bit_quant_type == "nf4" and cfg.model.bnb_4bit_use_double_quant
    assert cfg.generation.do_sample is False and cfg.generation.max_new_tokens == 512
    assert cfg.correction.max_attempts == 2


@pytest.mark.parametrize("name,r,size", [("lora_r8_n2000", 8, 2000), ("lora_r16_n5000", 16, 5000)])
def test_experiment_configs_extend_base(name, r, size):
    c = load_config(CONFIG_DIR / "train" / f"{name}.yaml")
    assert c.lora.r == r and c.lora.alpha == 2 * r
    assert c.training.train_file == Path(f"data/processed/train_{size}.jsonl")
    assert c.training.learning_rate == 1e-4  # inherited
    assert len(c.source_files) == 2


def test_overrides_are_yaml_typed():
    data = apply_overrides({"training": {"learning_rate": 1e-4}}, ["training.learning_rate=5e-5", "lora.r=8",
                                                                    "training.report_to=[]"])
    assert data["training"]["learning_rate"] == 5e-5 and data["lora"]["r"] == 8 and data["training"]["report_to"] == []
    c = load_config(None, ["evaluation.limit=5", "model.load_in_4bit=false"])
    assert c.evaluation.limit == 5 and c.model.load_in_4bit is False


def test_extract_code_blocks_and_fallbacks():
    text = "Here you go:\n```python\ndef f():\n    return 1\n```\nDone."
    assert extract_code_blocks(text) == ["def f():\n    return 1"]
    assert extract_code(text) == "def f():\n    return 1\n"
    # unterminated fence (max_new_tokens hit)
    assert extract_code("```python\ndef g():\n    return 2").startswith("def g()")
    # no fence: strip prose
    assert extract_code("Sure! Here is the code\nimport math\ndef h():\n    return math.pi\n").startswith("import math")
    # multiple blocks are concatenated
    multi = "```python\nimport re\n```\n```python\ndef k(s):\n    return re.sub('a','b',s)\n```"
    assert "import re" in extract_code(multi) and "def k" in extract_code(multi)
    assert extract_code("") == ""


def test_check_syntax_and_analyze_code():
    assert check_syntax("def f():\n    return 1\n").ok
    bad = check_syntax("def f(:\n  pass")
    assert not bad.ok and "SyntaxError" in bad.error
    an = analyze_code("import os\nfrom collections import Counter\n\ndef f(a, b=2):\n    return a\n\nclass C:\n    pass\n"
                      "print(f(1))\nif __name__ == '__main__':\n    pass\n")
    assert an.functions == ["f"] and an.classes == ["C"] and an.imports == ["collections", "os"]
    assert an.has_main_guard and an.has_top_level_io and an.signatures["f"] == 2


def test_analyze_tests_counts_value_asserts_and_references():
    src = ("from solution import foo, bar\nimport pytest\n\n"
           "def test_a():\n    assert foo(1) == 2\n\n"
           "def test_b():\n    assert bar([1,2]) == [2,1]\n    assert foo(0)\n\n"
           "def test_c(tmp_path):\n    assert foo(3) is not None\n")
    ta = analyze_tests(src)
    assert ta.imports_solution and sorted(ta.imported_names) == ["bar", "foo"]
    assert ta.n_test_functions == 3 and ta.n_asserts == 4 and ta.n_value_asserts == 2
    assert "foo" in ta.referenced_names and ta.uses_fixtures
    assert not analyze_tests("def test_x(:\n pass").syntax_ok


def test_choose_entry_point_prefers_test_info_then_tests():
    ta = analyze_tests("from solution import helper, main_fn\n\ndef test():\n    assert main_fn(1) == 1\n")
    assert choose_entry_point(["helper", "main_fn"], ta, [{"function_name": "main_fn"}]) == "main_fn"
    assert choose_entry_point(["helper", "main_fn"], ta, None) == "helper"  # first imported name defined
    assert choose_entry_point([], ta, None) is None


def test_extract_standalone_asserts():
    from codegenai.data.extract import extract_standalone_asserts

    src = (
        "import math\nfrom solution import area, helper\nimport pytest\n\n"
        "@pytest.fixture\ndef radius():\n    return 2\n\n"
        "def test_with_fixture(radius):\n    assert area(radius) == 4\n\n"          # fixture -> skipped
        "def test_local_var():\n    r = 3\n    assert area(r) == 9\n\n"             # local variable -> skipped
        "def test_pure():\n    '''doc'''\n    assert area(2) == 4\n    assert math.isclose(area(1.5), 2.25)\n\n"
        "def test_other():\n    assert helper(1) == 1\n"                            # wrong function -> skipped
    )
    out = extract_standalone_asserts(src, "area", max_tests=5)
    assert out == ["assert area(2) == 4", "assert math.isclose(area(1.5), 2.25)"]
    assert extract_standalone_asserts(src, "area", max_tests=1) == ["assert area(2) == 4"]
    assert extract_standalone_asserts(src, None) == []
    assert extract_standalone_asserts("def test_x(:\n", "area") == []
    assert extract_standalone_asserts("from solution import f\n\ndef test_x():\n    assert f([i for i in range(3)]) == 3\n", "f") == []


@pytest.mark.parametrize("name,targets,size", [
    ("lora_qv_n5000", ["q_proj", "v_proj"], 5000),
    ("lora_all_n5000", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 5000),
    ("lora_qv_n20000", ["q_proj", "v_proj"], 20000),
    ("lora_all_n20000", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 20000),
])
def test_headline_experiment_configs(name, targets, size):
    c = load_config(CONFIG_DIR / "train" / f"{name}.yaml")
    assert c.lora.target_modules == targets and c.lora.r == 16 and c.lora.alpha == 32
    assert c.training.train_file == Path(f"data/processed/train_{size}.jsonl")
    assert c.training.run_name == name and c.training.output_dir == Path(f"runs/{name}")
    assert c.training.load_best_model_at_end and c.training.early_stopping_patience == 3
    assert c.data.train_prompt_test_fraction == 0.5 and 20000 in c.data.train_sizes


def test_strip_docstrings_and_comments():
    code = 'def f(x):\n    """doc"""\n    # comment\n    return x  # trailing\n'
    out = strip_docstrings_and_comments(code)
    assert "doc" not in out and "comment" not in out and "return x" in out
