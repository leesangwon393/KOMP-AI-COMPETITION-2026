import csv
import json
import math
import signal
import time

import torch
from torch.utils.data import DataLoader

from .data import FullResolutionDataset, manifest
from .augmentations import AugmentedNativePatchDataset as NativePatchDataset
from .evaluation import evaluate
from .models import SegmentationModel
from .objectives import PhaseLoss
from .runtime import (capture_rng, environment, restore_rng, resume_guard, run_directory,
                      save_checkpoint, seed_everything, write_json)


def save_history(output, history):
    write_json(output/'history.json', history)
    if history:
        temp = output/'history.csv.tmp'
        with temp.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        temp.replace(output/'history.csv')


def train(config, data, output, device, resume=False):
    seed_everything(config['train']['seed'], config['train']['deterministic'])
    def terminate(_signum, _frame):
        raise KeyboardInterrupt('SIGTERM requested; resume from last completed epoch')
    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        with run_directory(output, resume) as output:
            t = config['train']
            identity = manifest(data, patch=t['target'])
            previous = torch.load(output/'last.pt', map_location='cpu', weights_only=True) if resume else None
            if previous:
                resume_guard(previous, config, identity, 'phase_segmentation', device)
            model = SegmentationModel(config, pretrained=not resume and config['model']['initialization']=='imagenet').to(device)
            criterion = PhaseLoss(config).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=t['lr'], weight_decay=t['weight_decay'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t['epochs'])
            dataset = NativePatchDataset(data, context=t['patch'], target=t['target'], stride=t['stride'],
                rare_fraction=t['rare_fraction'], enriched_probability=t['enriched_probability'],
                rare_classes=t['rare_classes'], samples_per_epoch=t['samples_per_epoch'],
                augmentation=config['augmentation'], seed=t['seed'])
            valid = FullResolutionDataset(data, 'valid')
            generator = torch.Generator().manual_seed(t['seed'])
            loader = DataLoader(dataset, batch_size=t['batch_size'], shuffle=True, num_workers=0,
                                generator=generator, pin_memory=device.type=='cuda')
            history, best, best_epoch, start, best_model = [], -1., 0, 1, None
            if previous:
                model.load_state_dict(previous['model'], strict=True)
                optimizer.load_state_dict(previous['optimizer'])
                scheduler.load_state_dict(previous['scheduler'])
                history, best, best_epoch = previous['history'], previous['best_miou'], previous['best_epoch']
                best_model, start = previous['best_model'], previous['epoch']+1
                restore_rng(previous['rng'], generator)
                criterion.generator.set_state(previous['contrast_rng'])
                dataset.extra_generator.set_state(previous['augmentation_rng'])
            base = {'config':config, 'data_manifest':identity, 'environment':environment(device),
                    'phase':'phase_segmentation', 'initialization':{'kind':config['model']['initialization']}}
            for filename, value in [('config.json',config), ('data_manifest.json',identity), ('environment.json',base['environment'])]:
                write_json(output/filename, value)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            print(json.dumps({'id':config['id'], 'device':str(device), 'start_epoch':start,
                'parameters_total':sum(p.numel() for p in model.parameters()),
                'parameters_inference':sum(p.numel() for n,p in model.named_parameters() if not n.startswith('projections.')),
                'enriched_sources_by_class':{str(k):len(v) for k,v in dataset.enriched.items()},
                'enriched_tiles_by_class':{str(k):sum(map(len,v.values())) for k,v in dataset.enriched.items()}}), flush=True)
            try:
                for epoch in range(start, t['epochs']+1):
                    started, examples, loss_sum = time.perf_counter(), 0, 0.
                    totals = {k:0. for k in ('seg_loss','contrast_loss','contrast_lambda','contrast_points','contact_fraction')}
                    model.train()
                    for step, (images, targets, _) in enumerate(loader, 1):
                        images, targets = images.to(device), targets.to(device)
                        optimizer.zero_grad(set_to_none=True)
                        logits, embeddings = model(images, return_features=True)
                        loss, parts = criterion(logits, targets, embeddings, epoch)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f'Nonfinite loss at epoch{epoch} step{step}')
                        loss.backward()
                        optimizer.step()
                        examples += len(images)
                        loss_sum += float(loss.detach())*len(images)
                        for key,value in parts.items():
                            totals[key] += value*len(images)
                        if step==1 or step%10==0 or step==len(loader):
                            write_json(output/'status.json', {'state':'training','epoch':epoch,'epochs':t['epochs'],
                                'step':step,'steps':len(loader),'train_total_loss':loss_sum/examples,
                                **{k:v/examples for k,v in totals.items()},'best_miou':best if best>=0 else None,'best_epoch':best_epoch})
                    write_json(output/'status.json', {'state':'validating','epoch':epoch,'epochs':t['epochs'],'best_epoch':best_epoch})
                    validation = evaluate(model, valid, config, device)
                    score = validation['miou']
                    if not math.isfinite(score):
                        raise FloatingPointError('Nonfinite validation score')
                    record = {'epoch':epoch,'train_total_loss':loss_sum/examples,
                        **{k:v/examples for k,v in totals.items()},'valid_single_miou':score,
                        'primary_si_iou':validation['class_iou'][0], 'al3ni_iou':validation['class_iou'][1],
                        'eutectic_si_iou':validation['class_iou'][2], 'al_iou':validation['class_iou'][3],
                        'lr':optimizer.param_groups[0]['lr'],'epoch_seconds':time.perf_counter()-started}
                    history.append(record)
                    scheduler.step()
                    if score>best:
                        best,best_epoch = score,epoch
                        best_model = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    state = {**base,'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                        'scheduler':scheduler.state_dict(),'rng':capture_rng(generator),
                        'contrast_rng':criterion.generator.get_state(),'augmentation_rng':dataset.extra_generator.get_state(),'history':history,
                        'best_miou':best,'best_epoch':best_epoch,'best_model':best_model}
                    save_checkpoint(output/'last.pt',state)
                    save_checkpoint(output/'best.pt',{**base,'epoch':best_epoch,'model':best_model,'best_miou':best,'best_epoch':best_epoch})
                    save_history(output,history)
                    print(json.dumps({**record,'best_miou':best,'best_epoch':best_epoch}),flush=True)
                # last.pt is authoritative, including best weights, after an interrupted epoch transaction.
                if best_model is None:
                    raise RuntimeError('No completed epoch checkpoint')
                model.load_state_dict(best_model,strict=True)
                save_checkpoint(output/'best.pt',{**base,'epoch':best_epoch,'model':best_model,'best_miou':best,'best_epoch':best_epoch})
                save_history(output,history)
                final = {'id':config['id'],'best_epoch':best_epoch,'parameters_total':sum(p.numel() for p in model.parameters()),
                         'parameters_inference':sum(p.numel() for n,p in model.named_parameters() if not n.startswith('projections.'))}
                write_json(output/'status.json', {'state':'final_evaluation','best_epoch':best_epoch})
                for d4 in (False,True) if config['evaluation']['final_d4'] else (False,):
                    name = 'd4' if d4 else 'single'
                    result = evaluate(model,valid,config,device,d4=d4,export=output/'valid_masks'/name,diagnostic=True)
                    write_json(output/f'valid_{name}.json',result)
                    final[name] = result
                final['peak_cuda_memory_bytes'] = torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
                write_json(output/'summary.json',final)
                write_json(output/'status.json',{'state':'completed','epochs':t['epochs'],'best_epoch':best_epoch,
                    'best_single_miou':best,'final_d4_miou':final.get('d4',{}).get('miou')})
            except (Exception,KeyboardInterrupt) as exc:
                write_json(output/'status.json',{'state':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                    'error':str(exc),'resume_from_last_completed_epoch':True})
                raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
