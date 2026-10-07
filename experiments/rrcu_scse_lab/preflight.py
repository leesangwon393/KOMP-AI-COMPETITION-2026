"""Matched initial state/crop checks plus real-data MPS optimizer preflight.

These steps are discarded; fresh training starts independently after passing.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import torch

from komap_phase.config import ROOT, load_config
from komap_phase.data import NativePatchDataset, manifest
from komap_phase.loss import R022Loss
from komap_phase.models import SegmentationModel
from komap_phase.runtime import device_for, environment, seed_everything, write_json


def digest(x):
    return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--device',choices=('mps','cuda'),default='mps')
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    os.environ['TORCH_HOME']=str(args.cache.resolve())
    torch.set_num_threads(4)
    device=device_for(args.device)
    identity=manifest(args.data)
    dataset=NativePatchDataset(args.data)
    results=[]
    reference_state=reference_rng=reference_crops=None
    for name in ('B','R01','R02','R03'):
        c=load_config(ROOT/f'configs/{name}.json')
        seed_everything(c['train']['seed'])
        model=SegmentationModel(c,pretrained=True)
        if name=='B':
            reference_state={k:digest(v) for k,v in model.state_dict().items()}
            reference_rng=torch.get_rng_state().clone()
        else:
            state=model.state_dict()
            common=set(reference_state)&set(state)
            assert all(reference_state[k]==digest(state[k]) for k in common)
            assert torch.equal(reference_rng,torch.get_rng_state())
            if name in ('R02','R03'): assert len(common)==len(reference_state)
        samples=[dataset[i] for i in range(4)]
        hashes=[digest(x)+':'+digest(y) for x,y,_ in samples]
        if reference_crops is None: reference_crops=hashes
        else: assert hashes==reference_crops
        images=torch.stack([s[0] for s in samples]).to(device)
        targets=torch.stack([s[1] for s in samples]).to(device)
        model.to(device).train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=c['train']['lr'],weight_decay=c['train']['weight_decay'])
        criterion=R022Loss(**c['loss']).to(device)
        losses=[]
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits=model(images)
            assert tuple(logits.shape)==(4,4,224,224)
            loss=criterion(logits,targets)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad.isfinite().all() for p in model.parameters() if p.grad is not None)
            if name!='B':
                unit=model.decoder.blocks[3] if name=='R01' else model.decoder.blocks[3].attention.recurrent
                module=unit.recurrent_conv if name=='R01' or step==1 else unit.output
                assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode(): predicted=model(images[:1])
        assert tuple(predicted.shape)==(1,4,448,448) and predicted.isfinite().all()
        row={'id':name,'passed':True,'parameters':sum(p.numel() for p in model.parameters()),
             'losses':losses,'common_initial_tensors':len(reference_state) if name=='B' else len(common),
             'crop_rng_equal':True,'shape':[4,1,448,448]}
        results.append(row)
        print(json.dumps(row),flush=True)
        del model,optimizer,criterion,images,targets,logits,loss,predicted,samples
        gc.collect()
        if device.type=='mps': torch.mps.empty_cache()
        else: torch.cuda.empty_cache()
    report={'passed':True,'device':str(device),'environment':environment(device),'checks':results,
            'dataset_counts':{k:len(v) for k,v in identity.items()},'data_manifest':identity,'first_crop_hashes':reference_crops,
            'note':'Optimizer preflight only; no trained weights or quality result retained'}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.report,report)


if __name__=='__main__': main()
