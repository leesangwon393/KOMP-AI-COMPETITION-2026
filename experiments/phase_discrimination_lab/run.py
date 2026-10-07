#!/usr/bin/env python3
"""Explicit server commands: list, check, train, evaluate, compare."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from komap_phase.config import ROOT, load_config, protocol
from komap_phase.data import FullResolutionDataset, manifest
from komap_phase.evaluation import evaluate
from komap_phase.models import SegmentationModel
from komap_phase.objectives import PhaseLoss
from komap_phase.runtime import device_for, environment, seed_everything, write_json
from komap_phase.training import train


def check(args):
    device = device_for(args.device)
    paths = [args.config] if args.config else sorted((ROOT/'configs').glob('*.json'))
    report = {'environment':environment(device),'checks':[],
              'note':'Synthetic forward/backward and optimizer only; no real-data training, no weight download'}
    for path in paths:
        original = load_config(path)
        c = copy.deepcopy(original)
        c['train'].update(patch=args.size,target=args.size//2,stride=args.size//4)
        seed_everything(c['train']['seed'])
        model = SegmentationModel(c,pretrained=False).to(device).train()
        criterion = PhaseLoss(c).to(device)
        image = torch.rand(args.batch_size,1,args.size,args.size,device=device)
        # Four broad regions preserve classes at every decoder supervision scale.
        target = torch.zeros(args.batch_size,args.size//2,args.size//2,dtype=torch.long,device=device)
        mid = args.size//4
        target[:,:mid,mid:]=1
        target[:,mid:,:mid]=3
        target[:,mid:,mid:]=2
        optimizer = torch.optim.AdamW(model.parameters(),lr=c['train']['lr'])
        details = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits, embeddings = model(image,return_features=True)
            loss, stats = criterion(logits,target,embeddings,epoch=10)
            if logits.shape != (args.batch_size,4,args.size//2,args.size//2) or not torch.isfinite(loss):
                raise RuntimeError('Invalid logits/loss')
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or any(not g.isfinite().all() for g in gradients):
                raise RuntimeError('Invalid backward gradients')
            if c['model']['contrast'] and (not stats['contrast_points'] or
                    not any(p.grad is not None and p.grad.abs().sum()>0 for p in model.projections.parameters())):
                raise RuntimeError('Contrast supervision/gradient missing')
            if c['model']['bem'] and step==1 and not any(p.grad is not None and p.grad.abs().sum()>0 for p in model.bem.branches.parameters()):
                raise RuntimeError('BEM branch gradient missing after identity initialization')
            if c['aux']['contact_weight']>1 and not stats['contact_fraction']:
                raise RuntimeError('Contact test did not exercise the weighted region')
            optimizer.step()
            details.append({'loss':float(loss.detach()),**stats})
        model.eval()
        with torch.inference_mode():
            pred = model(image)
        if pred.shape != (args.batch_size,4,args.size,args.size) or not pred.isfinite().all():
            raise RuntimeError('Invalid inference output')
        entry = {'id':c['id'],'status':'passed','synthetic_size':args.size,
                 'parameters':sum(p.numel() for p in model.parameters()),'steps':details}
        report['checks'].append(entry)
        print(json.dumps(entry),flush=True)
        del model,criterion,optimizer,image,target,logits,embeddings,loss,pred,gradients
        if device.type=='cuda':
            torch.cuda.empty_cache()
    if args.data:
        report['dataset_counts'] = {k:len(v) for k,v in manifest(args.data,patch=load_config(paths[0])['train']['target']).items()}
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        write_json(args.report,report)


def evaluate_checkpoint(args):
    state = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if state.get('phase')!='phase_segmentation':
        raise ValueError('Use a checkpoint from this package')
    c = state['config']
    device = device_for(args.device)
    model = SegmentationModel(c,pretrained=False).to(device)
    model.load_state_dict(state['model'],strict=True)
    dataset = FullResolutionDataset(args.data,args.split)
    if args.split=='valid':
        current = []
        for path,image in zip(dataset.paths,dataset.images):
            mask = args.data/'valid/masks'/path.name.replace('_image','_mask')
            current.append({'image':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                            'shape':list(image.shape),'mask_sha256':hashlib.sha256(mask.read_bytes()).hexdigest()})
        if current!=state['data_manifest']['valid']:
            raise ValueError('Validation data changed')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a new/empty evaluation output')
    args.output.mkdir(parents=True,exist_ok=True)
    result = evaluate(model,dataset,c,device,d4=args.d4,export=args.output/'masks',diagnostic=True)
    result.update(id=c['id'],checkpoint_epoch=state['epoch'],split=args.split)
    write_json(args.output/'metrics.json',result)
    print(json.dumps({'id':c['id'],'miou':result.get('miou'),'output':str(args.output)}))


def compare(args):
    summaries = [json.loads((p/'summary.json').read_text()) for p in args.runs]
    configs = [load_config(p/'config.json') for p in args.runs]
    manifests = [json.loads((p/'data_manifest.json').read_text()) for p in args.runs]
    environments = [json.loads((p/'environment.json').read_text()) for p in args.runs]
    if any(protocol(c)!=protocol(configs[0]) for c in configs[1:]) or any(m!=manifests[0] for m in manifests[1:]):
        raise ValueError('Data/seed/epochs/base architecture or evaluation protocol differs')
    for key in ('code_sha256','device','torch','torchvision','cuda_build','gpu','dependencies'):
        if any(e.get(key)!=environments[0].get(key) for e in environments[1:]):
            raise ValueError(f'Environment differs: {key}')
    rows,paired = [],{}
    for view in ('single','d4'):
        if not all(view in s for s in summaries):
            if view=='d4' and all(view not in s for s in summaries):
                continue
            raise ValueError('Incomplete evaluation views')
        reference = summaries[0][view]
        deltas = []
        for path,summary in zip(args.runs,summaries):
            result = summary[view]
            if result['filenames']!=reference['filenames']:
                raise ValueError('Validation ordering differs')
            delta = np.asarray(result['per_image_miou'])-np.asarray(reference['per_image_miou'])
            deltas.append({'run':str(path),'delta_per_image':delta.tolist()})
            rows.append({'id':summary['id'],'run':str(path),'view':view,'best_epoch':summary['best_epoch'],
                'miou':result['miou'],'strict_absent_zero_miou':result['miou_absent_zero'],
                'delta_miou_pp':float(delta.mean()*100),'al3ni_iou':result['class_iou'][1],
                'eutectic_iou':result['class_iou'][2],
                'eutectic_precision':result['eutectic']['precision'],'eutectic_recall':result['eutectic']['recall'],
                'thin_skeleton_recall':result['eutectic']['thin_skeleton_recall'],
                **result['phase_confusion'],'contact_error_rate':result['contact']['error_rate'],
                'seconds_per_image':result['seconds_per_image'],'parameters_total':summary['parameters_total'],
                'parameters_inference':summary['parameters_inference'],'peak_cuda_memory_bytes':summary['peak_cuda_memory_bytes']})
        paired[view] = {'filenames':reference['filenames'],'deltas_vs_first_run':deltas}
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a new/empty comparison output')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'comparison.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output/'paired_deltas.json',paired)
    write_json(args.output/'comparison.json',rows)
    print(json.dumps(rows,indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--threads',type=int,default=4)
    sub = p.add_subparsers(dest='command',required=True)
    sub.add_parser('list')
    ch = sub.add_parser('check')
    ch.add_argument('--config',type=Path)
    ch.add_argument('--data',type=Path)
    ch.add_argument('--device',choices=('cpu','cuda','mps'),default='cpu')
    ch.add_argument('--size',type=int,default=64)
    ch.add_argument('--batch-size',type=int,default=2)
    ch.add_argument('--report',type=Path)
    tr = sub.add_parser('train')
    tr.add_argument('--config',type=Path,required=True)
    tr.add_argument('--data',type=Path,required=True)
    tr.add_argument('--output',type=Path,required=True)
    tr.add_argument('--device',choices=('cpu','cuda','mps'),default='cuda')
    tr.add_argument('--resume',action='store_true')
    ev = sub.add_parser('evaluate')
    ev.add_argument('--checkpoint',type=Path,required=True)
    ev.add_argument('--data',type=Path,required=True)
    ev.add_argument('--output',type=Path,required=True)
    ev.add_argument('--device',choices=('cpu','cuda','mps'),default='cuda')
    ev.add_argument('--split',choices=('valid','test'),default='valid')
    ev.add_argument('--d4',action='store_true')
    cm = sub.add_parser('compare')
    cm.add_argument('--runs',type=Path,nargs='+',required=True)
    cm.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    if args.threads<1:
        p.error('threads must be positive')
    torch.set_num_threads(args.threads)
    if args.command=='list':
        for path in sorted((ROOT/'configs').glob('*.json')):
            c = load_config(path)
            print(c['id'],c['description'])
    elif args.command=='check':
        if args.size<64 or args.size%64 or args.batch_size<2:
            p.error('Synthetic size must be a multiple of64, at least64')
        check(args)
    elif args.command=='train':
        train(load_config(args.config),args.data,args.output,device_for(args.device),args.resume)
    elif args.command=='evaluate':
        evaluate_checkpoint(args)
    else:
        compare(args)


if __name__=='__main__':
    main()
