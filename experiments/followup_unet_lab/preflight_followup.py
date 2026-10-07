"""Verify matched initialization/crops and actual batch4 Context448 optimizer steps."""
import argparse
import hashlib
import json
import torch
from komap_phase.config import ROOT, load_config
from komap_phase.data import NativePatchDataset, manifest
from komap_phase.models import SegmentationModel
from komap_phase.objectives import PhaseLoss
from komap_phase.runtime import code_hash, device_for, seed_everything, write_json, environment
from experiment_queue import SCREEN, validate_contrasts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=('cpu','mps','cuda'),default='mps')
    p.add_argument('--data',type=__import__('pathlib').Path,required=True)
    p.add_argument('--report',type=__import__('pathlib').Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);device=device_for(a.device)
    contrasts=validate_contrasts()
    dataset=NativePatchDataset(a.data)
    reference_state=reference_rng=reference_crops=None
    checks=[]
    for name in SCREEN:
        c=load_config(ROOT/f'configs/{name}.json');seed_everything(c['train']['seed'])
        model=SegmentationModel(c,pretrained=True)
        state=model.state_dict();rng=torch.get_rng_state().clone()
        if name=='B':
            reference_state={k:v.clone() for k,v in state.items()};reference_rng=rng
        common=set(state)&set(reference_state)
        if not torch.equal(rng,reference_rng) or any(not torch.equal(state[k],reference_state[k]) for k in common):
            raise RuntimeError(f'{name} changed shared initialization or sampling RNG')
        samples=[dataset[index] for index in range(4)]
        crops=[hashlib.sha256(image.numpy().tobytes()+target.numpy().tobytes()).hexdigest() for image,target,_ in samples]
        if name=='B':reference_crops=crops
        if crops!=reference_crops:raise RuntimeError(f'{name} changed first four crops')
        images=torch.stack([s[0] for s in samples]).to(device)
        targets=torch.stack([s[1] for s in samples]).to(device)
        model=model.to(device).train();criterion=PhaseLoss(c).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=c['train']['lr'],weight_decay=c['train']['weight_decay'])
        losses=[]
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits,aux=model(images,return_features=True)
            loss,parts=criterion(logits,targets,aux,epoch=1)
            if logits.shape!=(4,4,224,224) or not torch.isfinite(loss):raise RuntimeError('Bad training shape/loss')
            loss.backward()
            gradients=[p.grad for p in model.parameters() if p.grad is not None]
            if any(not g.isfinite().all() for g in gradients):raise RuntimeError('Nonfinite gradient')
            for attribute in ('connectivity_head',):
                module=getattr(model,attribute)
                if module is not None and not any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()):
                    raise RuntimeError('Missing auxiliary gradient')
            if model.decoder.frequency_fusion is not None:
                for generator in (model.decoder.frequency_fusion.low_kernel,model.decoder.frequency_fusion.high_kernel):
                    if not any(p.grad is not None and p.grad.abs().sum()>0 for p in generator.parameters()):
                        raise RuntimeError('Missing frequency generator gradient')
            optimizer.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():prediction=model(images)
        if prediction.shape!=(4,4,448,448) or not prediction.isfinite().all():raise RuntimeError('Bad evaluation shape')
        entry={'id':name,'passed':True,'common_initial_tensors':len(common),'crop_rng_equal':True,
               'shape':list(images.shape),'losses':losses}
        checks.append(entry);print(json.dumps(entry),flush=True)
        del model,state,optimizer,criterion,images,targets,logits,aux,loss,prediction,gradients,samples
        if device.type=='mps':torch.mps.empty_cache()
        if device.type=='cuda':torch.cuda.empty_cache()
    report={'passed':True,'device':str(device),'environment':environment(device),'code_sha256':code_hash(),
            'data_manifest':manifest(a.data),'checks':checks,'contrasts':contrasts,
            'note':'Optimizer preflight weights discarded; these losses are not quality results'}
    a.report.parent.mkdir(parents=True,exist_ok=True);write_json(a.report,report)


if __name__=='__main__':main()
