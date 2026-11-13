# Build the hardened sandbox image used for all code execution (PowerShell).
Set-Location (Join-Path $PSScriptRoot "..")
docker build -t codegenai-sandbox:latest -f docker/sandbox/Dockerfile .
docker image inspect codegenai-sandbox:latest --format "built {{.Id}}"
