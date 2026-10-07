#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-cuda.txt
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c 'import torch, torchvision, timm, segmentation_models_pytorch as smp; print("torch",torch.__version__,"CUDA",torch.version.cuda,"available",torch.cuda.is_available(),"SMP",smp.__version__,"timm",timm.__version__); assert torch.cuda.is_available(), "NVIDIA driver / cu128 compatibility must be checked"'
