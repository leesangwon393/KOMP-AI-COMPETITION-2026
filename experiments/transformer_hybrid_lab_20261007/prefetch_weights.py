#!/usr/bin/env python3
"""Download official weights before the long queue; preserve cache for fresh runs."""
import hashlib
import json
from pathlib import Path
import torch
from torchvision.models import ResNet101_Weights, Swin_T_Weights
from komap_phase.config import ROOT
from komap_phase.runtime import write_json


def main():
    rows = []
    for weights in (ResNet101_Weights.IMAGENET1K_V2, Swin_T_Weights.IMAGENET1K_V1):
        state = weights.get_state_dict(progress=True,check_hash=True)
        del state
        path = Path(torch.hub.get_dir())/'checkpoints'/weights.url.rsplit('/',1)[-1]
        rows.append(dict(weights=str(weights), filename=path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size))
    (ROOT/'checks').mkdir(exist_ok=True)
    write_json(ROOT/'checks/pretrained_weights.json',rows)
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
