#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DATA_DIR="${DATA_DIR:-$PWD/Data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/runs/transformer_hybrid}"
PHASE="${PHASE:-primary}"
BATCH_SIZE="${BATCH_SIZE:-4}"
"$PYTHON_BIN" verify_release.py
"$PYTHON_BIN" -c 'import json,sys; from pathlib import Path; import torch; from komap_phase.runtime import environment; p=Path("checks/cuda_preflight.json"); assert p.exists(), "Run full CUDA preflight first (README)"; r=json.loads(p.read_text()); assert r.get("passed") and r.get("full_protocol_cuda"), "Require passing CUDA448 preflight for all4 configs"; assert r.get("batch")==int(sys.argv[1]), "Selected batch differs from preflight; rerun it"; current=environment(torch.device("cuda")); assert all(r["environment"].get(k)==current.get(k) for k in ("code_sha256","torch","torchvision","cuda_build","device","gpu","dependencies")), "Preflight source/environment changed; rerun"' "$BATCH_SIZE"
exec "$PYTHON_BIN" -u run_suite.py --data "$DATA_DIR" --output "$OUTPUT_DIR" --device cuda --phase "$PHASE" --batch-size "$BATCH_SIZE" --resume
