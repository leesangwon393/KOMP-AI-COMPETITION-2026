#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
    printf 'Run bash setup_server.sh first.\n' >&2
    exit 1
fi
export TORCH_HOME="${TORCH_HOME:-$PWD/torch_cache}"
mkdir -p checks
.venv/bin/python verify_release.py
.venv/bin/python -c 'import shutil,torch; assert torch.cuda.is_available(), "CUDA unavailable"; free=shutil.disk_usage(".").free; print("Available GiB:",round(free/2**30,1)); assert free>=40*2**30, "Need at least40GiB free for up to32 runs/checkpoints"'
.venv/bin/python verify_architectures.py --device cuda --size 448 --batch-size 4 --data Data --report checks/cuda_preflight.json
exec .venv/bin/python -u run_suite.py --data Data --output runs/layer_search --device cuda "$@"
