#!/usr/bin/env python3
"""Rebuild paired same-seed comparisons, diagnostics, bootstrap and seed aggregates."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from komap_phase.runtime import write_json
from komap_phase.config import protocol


def summarize(root):
    root = Path(root)
    entries = []
    for path in sorted(root.glob('seed*/*/summary.json')):
        run = path.parent
        status = json.loads((run/'status.json').read_text())
        if status.get('state') != 'completed':
            continue
        config = json.loads((run/'config.json').read_text())
        seed = config['train']['seed']
        reference = root/f'seed{seed}'/'B'
        if not (reference/'summary.json').exists():
            raise ValueError('Missing same-seed baseline')
        a,b = json.loads(path.read_text()),json.loads((reference/'summary.json').read_text())
        base_config = json.loads((reference/'config.json').read_text())
        if protocol(config) != protocol(base_config):
            raise ValueError('Paired protocols differ')
        for name in ('data_manifest.json','environment.json'):
            left,right = json.loads((run/name).read_text()),json.loads((reference/name).read_text())
            keys = ('code_sha256','device','torch','torchvision','cuda_build','gpu','dependencies') if name=='environment.json' else left.keys()
            if any(left.get(k)!=right.get(k) for k in keys):
                raise ValueError('Paired data/environment differs')
        for view in ('single','d4'):
            r, baseline = a[view], b[view]
            if r['filenames'] != baseline['filenames']:
                raise ValueError('Different Valid ordering')
            delta = np.asarray(r['per_image_miou'])-np.asarray(baseline['per_image_miou'])
            generator = np.random.default_rng(20261005)
            means = delta[generator.integers(0,len(delta),size=(10000,len(delta)))].mean(1)*100
            lo,hi = np.percentile(means,[2.5,97.5])
            extra={}
            if config['id'].startswith('CS_'):
                stage=config['model']['units'][0]['stage']
                for label,ident in [('channel_only','C'+str(stage)),('spatial_only','S_'+config['id'][3:])]:
                    paired=root/f'seed{seed}'/ident/'summary.json'
                    extra['delta_vs_'+label+'_pp']=None
                    if paired.exists():
                        value=json.loads(paired.read_text())[view]
                        assert r['filenames']==value['filenames']
                        extra['delta_vs_'+label+'_pp']=(r['miou']-value['miou'])*100
            entries.append(dict(id=config['id'],seed=seed,view=view,best_epoch=a['best_epoch'],
                miou_pct=r['miou']*100,delta_vs_same_seed_B_pp=float(delta.mean()*100),
                image_bootstrap95_lo_pp=float(lo),image_bootstrap95_hi_pp=float(hi),
                primary_si_iou_pct=r['class_iou'][0]*100,al3ni_iou_pct=r['class_iou'][1]*100,
                eutectic_iou_pct=r['class_iou'][2]*100,al_iou_pct=r['class_iou'][3]*100,
                eutectic_precision=r['eutectic']['precision'],eutectic_recall=r['eutectic']['recall'],
                thin_skeleton_recall=r['eutectic']['thin_skeleton_recall'],contact_error=r['contact']['error_rate'],
                seconds_per_image=r['seconds_per_image'],parameters=a['parameters_total'],
                peak_cuda_memory_bytes=a['peak_cuda_memory_bytes'],units=json.dumps(config['model']['units']),
                delta_vs_channel_only_pp=extra.get('delta_vs_channel_only_pp'),
                delta_vs_spatial_only_pp=extra.get('delta_vs_spatial_only_pp')))
    destination = root/'reports'
    destination.mkdir(exist_ok=True)
    if entries:
        with (destination/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(entries[0]));writer.writeheader();writer.writerows(entries)
    aggregates=[]
    for ident in sorted({r['id'] for r in entries}):
        values=[r for r in entries if r['id']==ident and r['view']=='d4']
        deltas=np.array([r['delta_vs_same_seed_B_pp'] for r in values])
        aggregates.append(dict(id=ident,seeds=[r['seed'] for r in values],n=len(values),
            mean_d4_pct=float(np.mean([r['miou_pct'] for r in values])),
            mean_delta_pp=float(deltas.mean()),seed_delta_std_pp=float(deltas.std(ddof=1)) if len(deltas)>1 else None))
    write_json(destination/'comparison.json',entries)
    write_json(destination/'seed_aggregates.json',aggregates)
    lines=['# 층별 attention 실험 결과','',
           'Valid20, 이미지별4클래스 IoU 평균, 빈 클래스=1. D4는 Single로 선택한 동일 checkpoint의8-view TTA.',
           '구조 선택에도 같은 Valid를 사용했다. 이미지 bootstrap은 학습 seed 변동을 표현하지 않으며 공식 Test 결과가 아니다.','',
           '| ID | Seeds | D4 평균(%) | 같은 seed B 대비 평균(%p) | seed별 Δ 표준편차(%p) |',
           '|---|---|---:|---:|---:|']
    for row in aggregates:
        std='—' if row['seed_delta_std_pp'] is None else f"{row['seed_delta_std_pp']:.4f}"
        lines.append(f"| {row['id']} | {row['seeds']} | {row['mean_d4_pct']:.4f} | {row['mean_delta_pp']:+.4f} | {std} |")
    (destination/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(destination), completed_runs=len(entries)//2)))
    return entries


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    summarize(parser.parse_args().output)
