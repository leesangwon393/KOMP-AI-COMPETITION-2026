#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert (3,10)<=sys.version_info[:2]<(3,13), "Use Python3.10–3.12 (e.g. PYTHON_BIN=python3.11)"'
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-cuda.txt
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c 'import torch,torchvision; print(torch.__version__,torchvision.__version__); assert torch.cuda.is_available(), "CUDA GPU is unavailable"; print(torch.cuda.get_device_name(0))'
.venv/bin/python verify_release.py
printf 'Setup complete. Start with: bash start_server.sh\n'
