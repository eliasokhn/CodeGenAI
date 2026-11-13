# CodeGenAI secure test sandbox.
# Build (from the repository root):  docker build -t codegenai-sandbox:latest -f docker/sandbox/Dockerfile .
#
# Hardening is applied at *run* time by codegenai.sandbox.docker_runner (no network, read-only
# root fs, tmpfs work dir, non-root user, all capabilities dropped, no-new-privileges, pid/memory/
# cpu limits, timeouts). The image itself only ships Python, pytest and the runner script.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Pinned test dependencies. numpy<2 is needed only by the MBPP+ harness (and matches KodCode's
# verification environment). No network libraries are installed.
RUN pip install --no-cache-dir "pytest==8.4.2" "pytest-timeout==2.4.0" "numpy<2" \
    && pip uninstall -y pip setuptools wheel >/dev/null 2>&1 || true

# Unprivileged runtime user (uid/gid 1000). The runner is root-owned and read-only.
RUN groupadd -g 1000 sandbox && useradd -m -u 1000 -g 1000 -s /usr/sbin/nologin sandbox \
    && mkdir -p /opt/runner /work && chown 1000:1000 /work
COPY src/codegenai/sandbox/runner_entry.py /opt/runner/runner_entry.py
RUN chmod 0755 /opt/runner && chmod 0644 /opt/runner/runner_entry.py

USER 1000:1000
WORKDIR /work
# -I: isolated mode (ignores PYTHON* env vars, user site, and does not add the script dir / cwd
#     to sys.path implicitly -- the runner adds the work dir itself).
ENTRYPOINT ["python", "-I", "/opt/runner/runner_entry.py"]
