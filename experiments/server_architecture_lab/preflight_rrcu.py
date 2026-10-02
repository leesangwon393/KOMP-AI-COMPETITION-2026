"""Check shared initialization, first crops, and one real C13 optimizer step."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from komap_lab.config import load_config
from komap_lab.data import NativePatchDataset, manifest
from komap_lab.loss import R022Loss
from komap_lab.models import SegmentationModel
from komap_lab.runtime import device_for, seed_everything, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parent
    baseline, candidate = [load_config(root / 'configs' / f'{name}.json') for name in ('B', 'C13')]
    seed_everything(42)
    model_b = SegmentationModel(baseline, pretrained=False)
    rng_b = torch.get_rng_state().clone()
    state_b = {k: hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest()
               for k, v in model_b.state_dict().items()}
    parameters_b = sum(p.numel() for p in model_b.parameters())
    del model_b
    seed_everything(42)
    model_c = SegmentationModel(candidate, pretrained=False)
    assert torch.equal(rng_b, torch.get_rng_state()), 'Model construction changed sampling RNG'
    common = [k for k in model_c.state_dict() if k in state_b]
    assert all(hashlib.sha256(model_c.state_dict()[k].cpu().numpy().tobytes()).hexdigest()
               == state_b[k] for k in common), 'Unchanged model tensors differ'
    parameters_c = sum(p.numel() for p in model_c.parameters())
    del model_c
    dataset = NativePatchDataset(args.data)
    torch.set_rng_state(rng_b)
    first = [dataset[i] for i in range(4)]
    crop_hashes = [hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest() for x, y, _ in first]
    torch.set_rng_state(rng_b)
    second = [dataset[i] for i in range(4)]
    assert crop_hashes == [hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()
                          for x, y, _ in second], 'Initial sampling differs'
    seed_everything(42)
    device = device_for(args.device)
    model = SegmentationModel(candidate, pretrained=True).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    images = torch.stack([x for x, _, _ in first]).to(device)
    targets = torch.stack([y for _, y, _ in first]).to(device)
    started = time.perf_counter()
    output = model(images)
    assert tuple(output.shape) == (4, 4, 224, 224)
    loss = R022Loss(**candidate['loss']).to(device)(output, targets)
    assert torch.isfinite(loss), 'Nonfinite loss'
    loss.backward()
    gradient = model.decoder.blocks[0].recurrent_conv.weight.grad
    assert gradient is not None and gradient.isfinite().all() and gradient.abs().sum() > 0
    optimizer.step()
    if device.type == 'mps':
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    model.eval()
    with torch.inference_mode():
        assert tuple(model(images[:1]).shape) == (1, 4, 448, 448)
    report = {'passed': True, 'device': str(device), 'shared_tensors_equal': len(common),
              'sampling_rng_equal': True, 'first_crop_hashes': crop_hashes,
              'baseline_parameters': parameters_b, 'candidate_parameters': parameters_c,
              'batch_size': 4, 'input': [4, 1, 448, 448], 'target': [4, 224, 224],
              'first_step_loss': float(loss.detach().cpu()), 'first_step_seconds': elapsed,
              'dataset_counts': {k: len(v) for k, v in manifest(args.data).items()}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
