"""Hardened Docker execution of untrusted Python + pytest.

Security model (enforced on every run):

* ``--network none``                     no network at all
* ``--read-only`` + tmpfs ``/work``/``/tmp`` immutable image; writable scratch is RAM-only, noexec
* ``--user 1000:1000``                   unprivileged user (never root)
* ``--cap-drop ALL`` + ``no-new-privileges``  no capabilities, no setuid escalation
* ``--pids-limit`` / ``--memory`` / ``--cpus`` / ``--ulimit``  resource limits (fork bombs, memory bombs)
* wall-clock timeout + ``docker kill``   whole-run limit, plus pytest-timeout per test
* host-side output cap                   container stdout is read incrementally and truncated
* payload over stdin                     no bind mounts, no Docker socket, no environment secrets

If the daemon or the image is unavailable the runner reports ``Status.UNAVAILABLE``; it never
falls back to executing code on the host.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from codegenai.config import PROJECT_ROOT, SandboxConfig
from codegenai.logging_utils import get_logger
from codegenai.sandbox.report import Status, TestReport, build_report

log = get_logger(__name__)

SENTINEL = "__CODEGENAI_REPORT__"
DOCKERFILE = PROJECT_ROOT / "docker" / "sandbox" / "Dockerfile"


class SandboxUnavailable(RuntimeError):
    """Raised when secure execution cannot be guaranteed."""


@dataclass
class Availability:
    available: bool
    reason: str = ""
    server_version: str = ""
    image_present: bool = False


def _docker_bin() -> Optional[str]:
    return shutil.which("docker")


def _run_docker(args: list[str], timeout: float = 30, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_docker_bin() or "docker", *args], capture_output=True, text=True, timeout=timeout, env=env,
    )


class DockerSandbox:
    """Run ``{solution.py, test_*.py}`` bundles inside the hardened container."""

    def __init__(self, cfg: SandboxConfig):
        self.cfg = cfg
        self._env = dict(os.environ)
        if cfg.docker_host:
            self._env["DOCKER_HOST"] = cfg.docker_host
        self._availability: Optional[Availability] = None

    # ------------------------------------------------------------------ availability
    def check_available(self, refresh: bool = False) -> Availability:
        if self._availability is not None and not refresh:
            return self._availability
        if _docker_bin() is None:
            self._availability = Availability(False, "docker CLI not found on PATH")
            return self._availability
        try:
            info = _run_docker(["info", "--format", "{{.ServerVersion}}|{{.OSType}}"], env=self._env)
        except (subprocess.TimeoutExpired, OSError) as exc:
            self._availability = Availability(False, f"docker info failed: {exc}")
            return self._availability
        if info.returncode != 0:
            self._availability = Availability(False, f"docker daemon unreachable: {info.stderr.strip()[:300]}")
            return self._availability
        version, _, ostype = info.stdout.strip().partition("|")
        if ostype.strip() != "linux":
            self._availability = Availability(False, f"Linux containers required, daemon reports '{ostype}'", version)
            return self._availability
        img = _run_docker(["image", "inspect", self.cfg.image, "--format", "{{.Id}}"], env=self._env)
        if img.returncode != 0:
            self._availability = Availability(
                False, f"image '{self.cfg.image}' not found; build it with "
                       f"`docker build -t {self.cfg.image} -f docker/sandbox/Dockerfile .`", version, False,
            )
            return self._availability
        self._availability = Availability(True, "ok", version, True)
        return self._availability

    def build_image(self, dockerfile: Path = DOCKERFILE, context: Path = PROJECT_ROOT, timeout: float = 900) -> bool:
        res = _run_docker(["build", "-t", self.cfg.image, "-f", str(dockerfile), str(context)],
                          timeout=timeout, env=self._env)
        if res.returncode != 0:
            log.error("Sandbox image build failed:\n%s", res.stderr[-3000:])
            return False
        self._availability = None
        return self.check_available(refresh=True).available

    # ------------------------------------------------------------------ command line
    def docker_command(self, container_name: str) -> list[str]:
        """The exact ``docker run`` argument vector (exposed for security unit tests)."""
        c = self.cfg
        tmpfs_opts = f"rw,noexec,nosuid,nodev,size={c.tmpfs_size},uid=1000,gid=1000,mode=0700"
        return [
            _docker_bin() or "docker", "run", "--rm", "-i", "--name", container_name,
            "--network", "none",
            "--read-only",
            "--tmpfs", f"/work:{tmpfs_opts}",
            "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=16m,uid=1000,gid=1000,mode=0700",
            "--user", "1000:1000",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", str(c.pids_limit),
            "--memory", c.memory, "--memory-swap", c.memory,
            "--cpus", str(c.cpus),
            "--ulimit", "nofile=256:256",
            "--ulimit", "fsize=10485760:10485760",
            "--ulimit", f"cpu={c.timeout_seconds}:{c.timeout_seconds}",
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--env", "PYTHONHASHSEED=0",
            "--env", "HOME=/work",
            "--workdir", "/work",
            "--init",
            c.image,
        ]

    # ------------------------------------------------------------------ execution
    def run(self, files: dict[str, str], entry_point: Optional[str] = None,
            hidden_test_names: Optional[set[str]] = None,
            timeout_override: Optional[float] = None) -> TestReport:
        """Execute ``files`` (must contain ``solution.py`` and at least one ``test_*.py``)."""
        avail = self.check_available()
        if not avail.available:
            return TestReport.unavailable(avail.reason)
        for name in files:
            if "/" in name or "\\" in name or name.startswith(".") or not name.endswith(".py"):
                raise ValueError(f"Illegal sandbox filename: {name!r}")
        payload = json.dumps({
            "files": files, "entry_point": entry_point,
            "per_test_timeout": self.cfg.per_test_timeout_seconds,
        }).encode("utf-8")
        name = f"codegenai-{uuid.uuid4().hex[:12]}"
        cmd = self.docker_command(name)
        timeout = timeout_override or self.cfg.timeout_seconds
        t0 = time.time()
        timed_out = False
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        truncated = False

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env=self._env)

        def _reader(stream, sink: list[bytes]) -> None:
            nonlocal truncated
            total = 0
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break
                total += len(chunk)
                if total <= self.cfg.max_output_bytes:
                    sink.append(chunk)
                else:
                    truncated = True  # keep draining so the child never blocks on a full pipe
            stream.close()

        threads = [threading.Thread(target=_reader, args=(proc.stdout, stdout_chunks), daemon=True),
                   threading.Thread(target=_reader, args=(proc.stderr, stderr_chunks), daemon=True)]
        for t in threads:
            t.start()
        try:
            proc.stdin.write(payload)
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            proc.wait(timeout=timeout + 5)  # a little slack: the in-container cpu ulimit fires first
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill(name)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        for t in threads:
            t.join(timeout=5)
        elapsed = time.time() - t0
        stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace")
        stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")

        raw = self._parse(stdout)
        if raw is None:
            if timed_out:
                raw = {"tests": [], "stdout_tail": stdout[-2000:], "stderr_tail": stderr[-2000:]}
            else:
                report = TestReport(
                    status=Status.INTERNAL_ERROR, summary="Sandbox produced no report",
                    stdout_tail=stdout[-2000:], stderr_tail=stderr[-2000:], execution_time_s=elapsed,
                    meta={"returncode": proc.returncode, "command": " ".join(cmd[:4])},
                )
                # Container OOM-kill shows up as rc 137
                if proc.returncode in (137, 139):
                    report.status = Status.RUNTIME_ERROR
                    report.summary = f"Process killed (exit {proc.returncode}): memory limit or crash."
                return report
        report = build_report(raw, hidden_test_names, execution_time_s=elapsed, timed_out=timed_out)
        report.truncated_output = report.truncated_output or truncated
        report.meta.update({"container": name, "returncode": proc.returncode, "image": self.cfg.image})
        return report

    def _kill(self, name: str) -> None:
        try:
            _run_docker(["kill", name], timeout=15, env=self._env)
        except Exception as exc:  # noqa: BLE001
            log.warning("docker kill %s failed: %s", name, exc)

    @staticmethod
    def _parse(stdout: str) -> Optional[dict]:
        if SENTINEL not in stdout:
            return None
        tail = stdout.rsplit(SENTINEL, 1)[1].strip()
        line = tail.splitlines()[0] if tail else ""
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None


class DisabledSandbox:
    """Stand-in used when secure execution is impossible: every run is reported as unverified."""

    def __init__(self, reason: str):
        self.reason = reason

    def check_available(self, refresh: bool = False) -> Availability:
        return Availability(False, self.reason)

    def run(self, files: dict[str, str], entry_point: Optional[str] = None,
            hidden_test_names: Optional[set[str]] = None, timeout_override: Optional[float] = None) -> TestReport:
        return TestReport.unavailable(self.reason)


def get_sandbox(cfg: SandboxConfig, require: bool = False) -> DockerSandbox | DisabledSandbox:
    sb = DockerSandbox(cfg)
    avail = sb.check_available()
    if avail.available:
        return sb
    log.warning("Secure sandbox unavailable (%s). Code execution is DISABLED.", avail.reason)
    if require:
        raise SandboxUnavailable(avail.reason)
    return DisabledSandbox(avail.reason)
