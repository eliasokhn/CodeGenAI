"""Shared fixtures: config, sample records, mock generators, sandbox availability."""
from __future__ import annotations

import os
import socket

import pytest

from codegenai.config import load_config
from codegenai.data.schema import KodCodeRecord, MBPPProblem
from codegenai.sandbox import DockerSandbox
from codegenai.sandbox.report import Status, TestCaseResult, TestReport


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def _docker_ok(cfg) -> bool:
    try:
        return DockerSandbox(cfg.sandbox).check_available().available
    except Exception:  # noqa: BLE001
        return False


def _network_ok() -> bool:
    if os.environ.get("CODEGENAI_OFFLINE") == "1":
        return False
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3).close()
        return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    cfg = load_config()
    docker = _docker_ok(cfg)
    network = _network_ok()
    skip_docker = pytest.mark.skip(reason="Docker sandbox not available")
    skip_net = pytest.mark.skip(reason="no network / offline")
    for item in items:
        if "docker" in item.keywords and not docker:
            item.add_marker(skip_docker)
        if "network" in item.keywords and not network:
            item.add_marker(skip_net)


@pytest.fixture
def sample_record() -> KodCodeRecord:
    return KodCodeRecord(
        id="Algorithm_1_I", uid="kodcode/Algorithm_1_I", subset="Algorithm", style="instruct", difficulty="easy",
        instruction="Write a function max_product(nums) returning the maximum product of two distinct elements.",
        solution="def max_product(nums):\n    nums = sorted(nums)\n    return max(nums[0] * nums[1], nums[-1] * nums[-2])\n",
        assistant_raw="```python\ndef max_product(nums):\n    nums = sorted(nums)\n    return max(nums[0] * nums[1], nums[-1] * nums[-2])\n```",
        system_prompt="You are a Python coding assistant.",
        test=("from solution import max_product\n\n"
              "def test_pos():\n    assert max_product([1, 5, 3, 7, 9]) == 63\n\n"
              "def test_neg():\n    assert max_product([-10, -3, 1, 2]) == 30\n"),
        test_info=[{"function_name": "max_product"}], function_names=["max_product"], entry_point="max_product",
        n_solution_lines=3,
    )


@pytest.fixture
def mbpp_problem() -> MBPPProblem:
    return MBPPProblem(
        task_id=11, text="Write a python function to remove first and last occurrence of a given character from the string.",
        reference_code="def remove_Occ(s, ch):\n    s = s.replace(ch, '', 1)\n    i = s.rfind(ch)\n    return s[:i] + s[i+1:] if i >= 0 else s\n",
        entry_point="remove_Occ",
        visible_tests=['assert remove_Occ("hello","l") == "heo"'],
        hidden_tests=['assert remove_Occ("hello","l") == "heo"', 'assert remove_Occ("abcda","a") == "bcd"',
                      'assert remove_Occ("PHP","P") == "H"'],
        split="validation",
    )


def make_report(status: Status, tests: list[tuple[str, str, str, bool]] | None = None) -> TestReport:
    """Helper: tests = [(name, outcome, error_type, hidden)]."""
    tcs = [TestCaseResult(name=n, outcome=o, error_type=e, message=f"{e}: {n} failed", hidden=h)
           for n, o, e, h in (tests or [])]
    rep = TestReport(status=status, tests=tcs, total=len(tcs),
                     passed=sum(1 for t in tcs if t.outcome == "passed"),
                     failed=sum(1 for t in tcs if t.outcome == "failed"),
                     errors=sum(1 for t in tcs if t.outcome not in ("passed", "failed")),
                     summary=status.value, verified=(status == Status.PASSED))
    return rep
