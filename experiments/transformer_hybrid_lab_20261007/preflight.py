#!/usr/bin/env python3
"""Real architecture, gradient, identity, RNG, freeze and optimizer checks."""
import argparse
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from komap_phase.config import ROOT, load_config
from komap_phase.hybrid import SegmentationModel, make_optimizer, weights_hash
from komap_phase.models import SegmentationModel as Baseline
from komap_phase.evaluation import evaluate
from komap_phase.objectives import PhaseLoss
from komap_phase.runtime import device_for, seed_everything, environment, write_json


def nonzero(parameters):
    return any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in parameters)


def run_checks(device_name='cpu', size=64, batch=2, config_path=None, report_path=None):
    if size < 64 or size % 64 or batch < 2:
        raise ValueError('size must be a multiple of64, batch>=2')
    device = device_for(device_name)
    paths = [Path(config_path)] if config_path else [ROOT/'configs'/f'{ident}.json' for ident in ('B','HT_BOT','HT_PYR','HT_DEC16')]
    report = dict(environment=environment(device), passed=False, size=size, batch=batch,
        pretrained_download=False, actual_alloy_training=False, checks=[])
    for path in paths:
        c = load_config(path)
        c['train'].update(patch=size,target=size//2,stride=size//4,batch_size=batch)
        c['evaluation']['batch_size'] = batch
        seed_everything(c['train']['seed'])
        baseline = Baseline(c, pretrained=False).eval()
        baseline_hash = weights_hash(baseline.state_dict())
        expected_cpu_rng = torch.get_rng_state().clone()
        image_cpu = torch.rand(batch,1,size,size)
        with torch.inference_mode():
            expected = baseline(image_cpu)
        del baseline
        gc.collect()
        seed_everything(c['train']['seed'])
        model = SegmentationModel(c, pretrained=False)
        assert model.initialization['common_initial_sha256'] == baseline_hash
        assert torch.equal(torch.get_rng_state(), expected_cpu_rng), 'Branch init changed sampling RNG'
        model.eval()
        with torch.inference_mode():
            actual = model(image_cpu)
        assert torch.equal(actual, expected), 'Zero residual changed baseline output'
        del expected, actual
        model = model.to(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        model.set_epoch(5); model.train()
        if model.swin is not None:
            assert not model.swin.training and not any(p.requires_grad for p in model.swin.parameters())
        model.set_epoch(6); model.train()
        if model.swin is not None:
            assert model.swin.training and all(p.requires_grad for p in model.swin.parameters())
        optimizer = make_optimizer(model,c)
        groups = [g.get('name','all') for g in optimizer.param_groups]
        image = image_cpu.to(device)
        target = torch.zeros(batch,size//2,size//2,dtype=torch.long,device=device)
        mid = size//4
        target[:,:mid,mid:] = 1; target[:,mid:,:mid] = 3; target[:,mid:,mid:] = 2
        criterion = PhaseLoss(c).to(device)
        branches = list(model.fusions.values()) + ([model.decoder_attention] if model.decoder_attention is not None else [])
        upstream = list(model.fusions.parameters()) + (list(model.swin.parameters()) if model.swin is not None else [])
        if model.decoder_attention is not None:
            upstream += list(model.decoder_attention.blocks.parameters())
        outputs = {id(p) for b in branches for p in b.output.parameters()}
        upstream = [p for p in upstream if id(p) not in outputs]
        losses = []
        step_seconds = []
        for step in range(2):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            sampling_rng = torch.get_rng_state().clone()
            logits, embeddings = model(image, return_features=True)
            assert torch.equal(sampling_rng, torch.get_rng_state()), 'Forward consumed sampling RNG'
            loss, _ = criterion(logits,target,embeddings,epoch=6)
            assert logits.shape == (batch,4,size//2,size//2) and torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
            for branch in branches:
                assert nonzero(branch.output.parameters()), 'Residual output has no gradient'
            if upstream:
                assert nonzero(upstream) == (step == 1), 'Unexpected upstream gradient through zero projection'
            if model.swin is not None:
                assert nonzero(model.swin.parameters()) == (step == 1), 'Swin gradient missing after fusion opens'
            if model.decoder_attention is not None:
                assert nonzero(model.decoder_attention.blocks.parameters()) == (step == 1), 'Decoder attention gradient missing'
            optimizer.step()
            losses.append(float(loss.detach()))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_seconds.append(time.perf_counter()-started)
        # Actual sliding-window validation, using a synthetic center-sized image.
        synthetic = SimpleNamespace(images=[(image_cpu[0,0,:size//2,:size//2].numpy()*255).astype('uint8')],
            labels=[target[0].cpu().numpy().astype('uint8')], paths=[Path('synthetic_image.png')])
        validation = evaluate(model,synthetic,c,device,diagnostic=True)
        assert 0 <= validation['miou'] <= 1
        peak = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None
        total = torch.cuda.get_device_properties(device).total_memory if device.type == 'cuda' else None
        row = dict(id=c['id'], passed=True, parameters=sum(p.numel() for p in model.parameters()),
            common_weights_and_initial_eval_equal=True, initialization_and_forward_sampling_rng_preserved=True,
            freeze5_unfreeze6=True, zero_residual_and_second_step_upstream_gradients=True,
            optimizer_groups=groups, losses=losses, training_step_seconds=step_seconds,
            steady_step_images_per_second=batch/step_seconds[-1], sliding_window_validation=True,
            peak_cuda_reserved_bytes=peak, gpu_total_bytes=total,
            cuda_headroom_passed=(peak<=.8*total) if total else None)
        report['checks'].append(row)
        print(json.dumps(row),flush=True)
        del model, optimizer, image, image_cpu, target, criterion, logits, loss, branches, upstream
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        if total and peak > .8*total:
            if report_path:
                Path(report_path).parent.mkdir(parents=True,exist_ok=True); write_json(report_path,report)
            raise RuntimeError('CUDA preflight exceeded80% VRAM; use a larger GPU for the unchanged protocol')
    report['passed'] = True
    report['full_protocol_cuda'] = device.type == 'cuda' and size == 448 and batch in (4,8,16) and len(paths) == 4
    if report_path:
        Path(report_path).parent.mkdir(parents=True,exist_ok=True); write_json(report_path,report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    p.add_argument('--size', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--config', type=Path)
    p.add_argument('--report', type=Path, default=ROOT/'checks/preflight.json')
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    run_checks(args.device,args.size,args.batch_size,args.config,args.report)
