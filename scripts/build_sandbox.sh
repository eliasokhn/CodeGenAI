#!/usr/bin/env bash
# Build the hardened sandbox image used for all code execution.
set -euo pipefail
cd "$(dirname "$0")/.."
docker build -t codegenai-sandbox:latest -f docker/sandbox/Dockerfile .
docker image inspect codegenai-sandbox:latest --format "built {{.Id}} ({{.Size}} bytes)"
