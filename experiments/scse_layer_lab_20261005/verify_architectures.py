#!/usr/bin/env python3
"""Verify legacy equivalence, paired RNG, identity gates and live gradients."""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import torch
from komap_phase.config import ROOT
from komap_phase.data import NativePatchDataset, manifest
from komap_phase.models import SegmentationModel
from komap_phase.objectives import PhaseLoss
from komap_phase.runtime import device_for, seed_everything, write_json, environment
from plan import reproduction, positions, mechanisms, combinations, spatial_candidates


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def candidates():
    return [c for c in reproduction() if c['train']['seed']==42] + positions() + [c for stage in (0,3,4) for c in mechanisms(stage)] + combinations(['C3','C2']) + [c for stage in range(5) for c in spatial_candidates(stage)]


def verify(device_name='cpu', size=64, batch_size=2, data=None):
    torch.set_num_threads(4)
    device=device_for(device_name)
    spec=importlib.util.spec_from_file_location('legacy_rrcu_models',ROOT/'source_reference/komap_phase/models.py')
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    originals={name:json.loads((ROOT/f'source_reference/{name}.json').read_text()) for name in ('B','R03')}
    legacy_hashes={}
    for name,c in originals.items():
        seed_everything(42)
        old=legacy.SegmentationModel(c,pretrained=False)
        legacy_hashes[name]={k:digest(v) for k,v in old.state_dict().items()}
        del old
    dataset=NativePatchDataset(data) if data else None
    if data:
        assert {k:len(v) for k,v in manifest(data).items()}=={'train':70,'valid':20}
    reference_rng=reference_crops=reference_pred=None
    image=torch.rand(batch_size,1,size,size,generator=torch.Generator().manual_seed(731)).to(device)
    target=torch.zeros(batch_size,size//2,size//2,dtype=torch.long,device=device)
    mid=size//4
    target[:,:mid,mid:]=1;target[:,mid:,:mid]=3;target[:,mid:,mid:]=2
    results=[]
    for original in candidates():
        c=json.loads(json.dumps(original))
        c['train'].update(patch=size,target=size//2,stride=size//4)
        seed_everything(42)
        model=SegmentationModel(c,pretrained=False)
        state={k:digest(v) for k,v in model.state_dict().items()}
        assert all(state[k]==v for k,v in legacy_hashes['B'].items()), c['id']
        if c['id']=='C3':
            assert state==legacy_hashes['R03'], 'New C3 differs from legacy R03 initial state'
        rng=torch.get_rng_state().clone()
        if reference_rng is None:reference_rng=rng
        else:assert torch.equal(rng,reference_rng), 'Add-on consumed the baseline sampling RNG'
        if dataset:
            samples=[dataset[index] for index in range(4)]
            hashes=[digest(x)+':'+digest(y) for x,y,_ in samples]
            if reference_crops is None:reference_crops=hashes
            else:assert hashes==reference_crops,'Different crop/augmentation pairing'
            del samples
        model.to(device).eval()
        with torch.inference_mode():
            prediction=model(image).cpu()
        if reference_pred is None:reference_pred=prediction.clone()
        else:assert torch.equal(prediction,reference_pred),'Zero initialized gates do not preserve baseline output'
        model.train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=c['train']['lr'])
        criterion=PhaseLoss(c).to(device)
        losses=[]
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits,embeddings=model(image,return_features=True)
            loss,_=criterion(logits,target,embeddings,epoch=10)
            assert torch.isfinite(loss) and tuple(logits.shape)==tuple(target.shape[:1])+(4,)+tuple(target.shape[1:])
            loss.backward()
            assert all(p.grad.isfinite().all() for p in model.parameters() if p.grad is not None)
            for u in c['model']['units']:
                block=model.decoder.final if u['stage']==4 else model.decoder.blocks[u['stage']]
                branches=[block.attention.recurrent] if block.attention.recurrent is not None else []
                if block.attention.recurrent_spatial is not None:branches.append(block.attention.recurrent_spatial)
                for branch in branches:
                    group=branch.output if step==0 or not hasattr(branch,'recurrent_conv') else branch.recurrent_conv
                    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in group.parameters()), (c['id'],u,step)
                    if step==1 and hasattr(branch,'project') and not hasattr(branch,'recurrent_conv'):
                        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in branch.project.parameters())
                    if step==1 and hasattr(branch,'local'):
                        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in branch.local.parameters())
                        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in branch.context.parameters())
                    if step==1 and u['sharing']=='unshared' and hasattr(branch,'unshared_convs'):
                        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in branch.unshared_convs.parameters())
            optimizer.step();losses.append(float(loss.detach()))
        row=dict(id=c['id'],units=c['model']['units'],passed=True,losses=losses,
                 parameters=sum(p.numel() for p in model.parameters()),legacy_base_tensors_equal=True,
                 legacy_R03_tensors_equal=c['id']=='C3',sampling_rng_equal=True,
                 synthetic_size=size,batch_size=batch_size)
        results.append(row);print(json.dumps(row),flush=True)
        del model,optimizer,criterion,loss,logits,embeddings,prediction
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    return dict(passed=True,checks=results,environment=environment(device),
                data_crop_pairing_verified=dataset is not None,
                note='Fresh synthetic optimizer checks, not quality training; pretrained downloads not required')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda','mps'),default='cpu')
    parser.add_argument('--size',type=int,default=64)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--report',type=Path,default=ROOT/'checks/architectures.json')
    args=parser.parse_args()
    if args.size<64 or args.size%64 or args.batch_size<2:parser.error('Require size multiple64 and batch>=2')
    result=verify(args.device,args.size,args.batch_size,args.data)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.report,result)
