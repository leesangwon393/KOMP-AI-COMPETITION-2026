#!/usr/bin/env python3
"""Real synthetic training across Swin freeze5/unfreeze6, interruption and replay."""
import argparse
import copy
import gc
import json
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import torch
from komap_phase.config import ROOT, load_config
from komap_phase.data import PALETTE
from komap_phase import training
from komap_phase.runtime import environment, write_json
from run_suite import completed, guard_equal


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a,b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a, (list,tuple)):
        return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a == b


def run(device_name, report):
    device = torch.device(device_name)
    c = load_config(ROOT/'configs/HT_BOT.json')
    c['model']['initialization'] = 'none'
    if device.type == 'cuda':
        c['train']['deterministic'] = True  # Synthetic replay check, separate from the training recipe.
    c['train'].update(epochs=6, patch=64, target=32, stride=16, samples_per_epoch=2, batch_size=2)
    c['evaluation']['batch_size'] = 2
    with tempfile.TemporaryDirectory(prefix='komap_hybrid_resume_') as tmp:
        root = Path(tmp); data = root/'Data'
        generator = np.random.default_rng(5678)
        for split in ('train','valid'):
            for kind in ('images','masks'):
                (data/split/kind).mkdir(parents=True)
            for index in range(2 if split=='train' else 1):
                image = generator.integers(0,256,(64,64),dtype=np.uint8)
                labels = np.zeros((64,64),dtype=np.uint8)
                labels[:32,32:] = 1; labels[32:,:32] = 3; labels[32:,32:] = 2
                Image.fromarray(image).save(data/split/'images'/f'{index}_image.png')
                Image.fromarray(PALETTE[labels]).save(data/split/'masks'/f'{index}_mask.png')
        training.train(c,data,root/'continuous',device)
        original = training.evaluate
        calls = 0
        def interrupt(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 6:
                raise KeyboardInterrupt('Test: interrupt epoch6 before committing checkpoint')
            return original(*args, **kwargs)
        training.evaluate = interrupt
        try:
            try:
                training.train(c,data,root/'resumed',device)
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError('Expected interruption')
        finally:
            training.evaluate = original
        checkpoint = torch.load(root/'resumed/last.pt',map_location='cpu',weights_only=True,mmap=True)
        assert checkpoint['epoch'] == 5 and checkpoint['branch_rng']['swin_frozen']
        assert [g['name'] for g in checkpoint['optimizer']['param_groups']] == ['cnn','decoder','fusion','swin']
        assert checkpoint['optimizer']['param_groups'][-1]['lr'] > 0
        del checkpoint; gc.collect()
        training.train(c,data,root/'resumed',device,resume=True)
        a = torch.load(root/'continuous/last.pt',map_location='cpu',weights_only=True,mmap=True)
        b = torch.load(root/'resumed/last.pt',map_location='cpu',weights_only=True,mmap=True)
        assert a['epoch'] == b['epoch'] == 6 and not b['branch_rng']['swin_frozen']
        for key in ('model','optimizer','scheduler','rng','branch_rng','initialization','contrast_rng'):
            assert equal_tree(a[key],b[key]), 'Replay mismatch at '+key
        for x,y in zip(a['history'],b['history']):
            for key in x:
                if key != 'epoch_seconds':
                    assert x[key] == y[key], 'History mismatch at '+key
        assert completed(root/'resumed',c,b['data_manifest'],environment(device))
        bad = copy.deepcopy(c);bad['train']['seed']+=1
        try:
            completed(root/'resumed',bad,b['data_manifest'],environment(device))
        except ValueError:
            pass
        else:
            raise AssertionError('Completed guard accepted changed config')
        guard_equal(root/'identity.json',dict(a=1))
        try:
            guard_equal(root/'identity.json',dict(a=2))
        except ValueError:
            pass
        else:
            raise AssertionError('Identity silently overwritten')
        result = dict(passed=True, environment=environment(device), variant='HT_BOT',
            synthetic_size=64, initialization='none', epochs=6, training_epoch_steps=13,
            freeze5_unfreeze6_resumed=True, model_optimizer_scheduler_rng_bitwise_equal=True,
            history_except_time_equal=True, completed_config_and_identity_guards=True,
            note='Synthetic integration only; not full-size CUDA or actual alloy performance')
    report.parent.mkdir(parents=True,exist_ok=True); write_json(report,result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--report',type=Path,default=ROOT/'checks/resume.json')
    args = p.parse_args();torch.set_num_threads(args.threads)
    run(args.device,args.report)
