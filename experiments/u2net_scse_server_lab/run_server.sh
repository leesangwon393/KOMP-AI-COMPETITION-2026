#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
profile="${1:-all}"
output="${2:-runs/u2_attention_all_batch4}"
batch="${3:-4}"
resume="${4:-}"
case "$profile" in all|core|scse) ;; *) echo 'profile: all / core / scse' >&2; exit 2;; esac
case "$batch" in 2|4) ;; *) echo 'batch size: 2 / 4' >&2; exit 2;; esac
if [[ -n "$resume" && "$resume" != '--resume' ]]; then echo 'fourth argument: --resume' >&2; exit 2; fi
if [[ ! -x .venv/bin/python ]]; then echo 'Run bash setup.sh first.' >&2; exit 2; fi
.venv/bin/python verify_release.py
# A failing preflight stops before any real-data training. FP32 and batch stay fixed.
.venv/bin/python verify.py --device cuda --size 64 --batch-size 2 \
  --output "checks/${profile}_batch${batch}/small"
.venv/bin/python verify.py --device cuda --size 448 --batch-size "$batch" \
  --ids U2_ECA_SCSE_DEEP U2_ECA_CBAM_DEEP --output "checks/${profile}_batch${batch}/fullsize"
args=(--profile "$profile" --seeds 42 43 44 --batch-size "$batch" --output "$output" --device cuda)
if [[ "$resume" == '--resume' ]]; then args+=(--resume); fi
.venv/bin/python -u server_suite.py "${args[@]}"
