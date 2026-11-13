"""Sandbox: report classification, harness generation, run-time hardening flags, Docker integration."""
from __future__ import annotations

import ast

import pytest

from codegenai.sandbox import DisabledSandbox, DockerSandbox, Status, TestReport
from codegenai.sandbox.harness import ENTRY_POINT_TEST, build_assert_module, build_kodcode_bundle, build_mbpp_bundle
from codegenai.sandbox.report import build_report


def _raw(tests, **extra):
    return {"tests": [{"name": n, "outcome": o, "error_type": e, "message": m, "duration": 0.01}
                      for n, o, e, m in tests], **extra}


def test_build_report_classification():
    assert build_report(_raw([("test_a", "passed", None, "")])).status == Status.PASSED
    assert build_report(_raw([("test_a", "failed", "AssertionError", "assert 1 == 2")])).status == Status.FAILED
    assert build_report(_raw([("test_a", "error", "TypeError", "TypeError: bad")])).status == Status.RUNTIME_ERROR
    assert build_report(_raw([("test_a", "timeout", "Timeout", "Failed: Timeout >5.0s")])).status == Status.TIMEOUT
    assert build_report(_raw([(ENTRY_POINT_TEST, "failed", "AssertionError", "function 'f' is not defined in solution.py")])).status == Status.MISSING_FUNCTION
    assert build_report(_raw([], syntax_error="SyntaxError: x")).status == Status.SYNTAX_ERROR
    assert build_report(_raw([], collection_error="ImportError: cannot import name 'f' from 'solution'")).status == Status.MISSING_FUNCTION
    assert build_report(_raw([], collection_error="E   solution.py:1: in <module>\nE   ValueError: boom")).status == Status.RUNTIME_ERROR
    assert build_report(_raw([], collection_error="SyntaxError in test_solution.py")).status == Status.COLLECTION_ERROR
    assert build_report(_raw([]), timed_out=True).status == Status.TIMEOUT
    mixed = build_report(_raw([("test_a", "failed", "AssertionError", "x"), ("test_b", "error", "ValueError", "y")]))
    assert mixed.status == Status.FAILED and mixed.passed == 0 and mixed.failed == 1 and mixed.errors == 1


def test_report_hidden_flagging_and_roundtrip():
    rep = build_report(_raw([("test_hidden_0", "failed", "AssertionError", "secret"), ("test_visible_0", "passed", None, "")]),
                       hidden_names={"test_hidden_0"})
    assert [t.hidden for t in rep.tests] == [True, False]
    assert rep.visible_failures() == [] and len(rep.failed_tests()) == 1
    d = rep.to_dict()
    back = TestReport.from_dict(d)
    assert back.status == rep.status and back.tests[0].hidden and not back.verified
    assert TestReport.unavailable("x").status == Status.UNAVAILABLE


def test_assert_module_generation_is_valid_python(mbpp_problem):
    src = build_assert_module(mbpp_problem.hidden_tests, "hidden", "", mbpp_problem.entry_point, include_entry_test=True)
    tree = ast.parse(src)
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert names == [ENTRY_POINT_TEST, "test_hidden_0", "test_hidden_1", "test_hidden_2"]
    with pytest.raises(SyntaxError):
        build_assert_module(["assert f( == 1"], "x")


def test_mbpp_bundle_modes_separate_visible_and_hidden(mbpp_problem):
    files, hidden = build_mbpp_bundle(mbpp_problem, "def remove_Occ(s, ch): return s", "visible")
    assert set(files) == {"solution.py", "test_visible.py"} and hidden == set()
    assert mbpp_problem.hidden_tests[1] not in files["test_visible.py"]
    files, hidden = build_mbpp_bundle(mbpp_problem, "x", "hidden")
    assert set(files) == {"solution.py", "test_hidden.py"} and hidden == {"test_hidden_0", "test_hidden_1", "test_hidden_2", ENTRY_POINT_TEST}
    files, hidden = build_mbpp_bundle(mbpp_problem, "x", "all")
    assert set(files) == {"solution.py", "test_visible.py", "test_hidden.py"}
    kod = build_kodcode_bundle("def f(): pass", "from solution import f")
    assert set(kod) == {"solution.py", "test_solution.py"}


def test_docker_command_enforces_hardening(cfg):
    sb = DockerSandbox(cfg.sandbox)
    cmd = sb.docker_command("codegenai-test")
    joined = " ".join(cmd)
    assert cmd[1] == "run" and "--rm" in cmd
    assert "--network none" in joined
    assert "--read-only" in cmd
    assert "--user 1000:1000" in joined
    assert "--cap-drop ALL" in joined
    assert "--security-opt no-new-privileges" in joined
    assert f"--pids-limit {cfg.sandbox.pids_limit}" in joined
    assert f"--memory {cfg.sandbox.memory}" in joined and "--memory-swap" in cmd
    assert f"--cpus {cfg.sandbox.cpus}" in joined
    assert "--tmpfs /work:rw,noexec,nosuid,nodev" in joined and "--tmpfs /tmp:rw,noexec" in joined
    assert "--init" in cmd and cmd[-1] == cfg.sandbox.image
    # no bind mounts, no docker socket, no privileged mode, no secrets
    assert "-v" not in cmd and "--mount" not in cmd and "docker.sock" not in joined
    assert "--privileged" not in cmd and "--cap-add" not in cmd
    for part in cmd:
        assert "HF_TOKEN" not in part and "API_KEY" not in part


def test_illegal_filenames_rejected(cfg):
    sb = DockerSandbox(cfg.sandbox)
    sb._availability = type(sb.check_available())(True, "ok", "x", True)
    with pytest.raises(ValueError):
        sb.run({"../evil.py": "x", "solution.py": "y"})
    with pytest.raises(ValueError):
        sb.run({"solution.py": "y", "payload.sh": "x"})


def test_disabled_sandbox_never_executes():
    sb = DisabledSandbox("no docker")
    rep = sb.run({"solution.py": "import os; os.system('echo pwned')", "test_x.py": "def test(): pass"})
    assert rep.status == Status.UNAVAILABLE and not rep.verified and not sb.check_available().available


def test_parse_sentinel():
    assert DockerSandbox._parse("noise\n__CODEGENAI_REPORT__\n{\"tests\": []}\n") == {"tests": []}
    assert DockerSandbox._parse("garbage") is None
    assert DockerSandbox._parse("__CODEGENAI_REPORT__\nnot json") is None


# ----------------------------------------------------------------------------- integration
@pytest.mark.docker
class TestDockerIntegration:
    TEST = "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n\ndef test_neg():\n    assert add(-1, 1) == 0\n"

    @pytest.fixture(scope="class")
    def sb(self, cfg):
        return DockerSandbox(cfg.sandbox)

    def test_pass_and_fail(self, sb):
        ok = sb.run(build_kodcode_bundle("def add(a, b):\n    return a + b\n", self.TEST), entry_point="add")
        assert ok.status == Status.PASSED and ok.verified and ok.passed == 2
        bad = sb.run(build_kodcode_bundle("def add(a, b):\n    return a - b\n", self.TEST))
        assert bad.status == Status.FAILED and bad.failed == 2 and bad.passed == 0
        assert all(t.error_type == "AssertionError" for t in bad.failed_tests())
        partial = sb.run(build_kodcode_bundle("def add(a, b):\n    return 3\n", self.TEST))
        assert partial.status == Status.FAILED and partial.passed == 1 and partial.failed == 1

    def test_error_categories(self, sb):
        assert sb.run(build_kodcode_bundle("def add(a, b)\n    return a + b\n", self.TEST)).status == Status.SYNTAX_ERROR
        assert sb.run(build_kodcode_bundle("def plus(a, b):\n    return a + b\n", self.TEST)).status == Status.MISSING_FUNCTION
        assert sb.run(build_kodcode_bundle("def add(a, b):\n    return a + undefined\n", self.TEST)).status == Status.RUNTIME_ERROR
        assert sb.run(build_kodcode_bundle("def add(a, b):\n    while True:\n        pass\n", self.TEST)).status == Status.TIMEOUT

    def test_isolation_properties(self, sb):
        probe = (
            "import os, socket\n"
            "def add(a, b):\n"
            "    assert os.getuid() != 0, 'running as root'\n"
            "    try:\n        socket.create_connection(('1.1.1.1', 80), timeout=2)\n        raise AssertionError('network reachable')\n"
            "    except OSError:\n        pass\n"
            "    try:\n        open('/etc/marker', 'w').write('x')\n        raise AssertionError('root fs writable')\n"
            "    except OSError:\n        pass\n"
            "    assert not os.path.exists('/var/run/docker.sock')\n"
            "    return a + b\n"
        )
        rep = sb.run(build_kodcode_bundle(probe, self.TEST))
        assert rep.status == Status.PASSED, rep.to_dict()

    def test_output_flood_is_truncated(self, sb, cfg):
        flood = "def add(a, b):\n    print('x' * 1000000)\n    return a + b\n"
        rep = sb.run(build_kodcode_bundle(flood, self.TEST))
        assert rep.status == Status.PASSED and len(rep.stdout_tail) <= 2000

    def test_mbpp_hidden_flags(self, sb, mbpp_problem):
        code = "def remove_Occ(s, ch):\n    s = s.replace(ch, '', 1)\n    i = s.rfind(ch)\n    return s[:i] + s[i+1:] if i >= 0 else s\n"
        files, hidden = build_mbpp_bundle(mbpp_problem, code, "all")
        rep = sb.run(files, entry_point=mbpp_problem.entry_point, hidden_test_names=hidden)
        assert rep.status == Status.PASSED and sum(t.hidden for t in rep.tests) == 3
