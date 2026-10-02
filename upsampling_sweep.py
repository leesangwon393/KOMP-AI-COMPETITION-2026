#!/usr/bin/env python3
"""Queue three independent final-stage decoder ablations after seed validation."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'runs/upsampling_sweep_v1'
METHODS = ('fullres_skip','pixelshuffle','dysample')


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def common_digest(model):
    digest = hashlib.sha256()
    for name,tensor in model.state_dict().items():
        if name.startswith(('detail.','upsampler.')):
            continue
        if name == 'decoder0.block.0.weight':
            tensor = tensor[:,:64]
        digest.update(name.encode())
        digest.update(tensor.cpu().numpy().tobytes())
    return digest.hexdigest()


def diagnostics(mask_dir):
    import numpy as np
    from PIL import Image
    from scipy import ndimage as ndi
    from mimu.data import decode_mask
    rows = []
    for path in sorted((ROOT/'Data/valid/masks').glob('*.png')):
        gt = decode_mask(np.asarray(Image.open(path).convert('RGB'))) == 2
        pred = decode_mask(np.asarray(Image.open(mask_dir/path.name).convert('RGB'))) == 2
        distance = ndi.distance_transform_edt(gt)
        mask = gt.copy()
        skeleton = np.zeros_like(gt)
        cross = ndi.generate_binary_structure(2,1)
        while mask.any():
            eroded = ndi.binary_erosion(mask,structure=cross)
            skeleton |= mask & ~ndi.binary_dilation(eroded,structure=cross)
            mask = eroded
        thin = skeleton & (distance <= 2)
        labels,_ = ndi.label(gt,structure=np.ones((3,3)))
        eligible,fragmented,missed = 0,0,0
        for index,slices in enumerate(ndi.find_objects(labels),1):
            component = labels[slices] == index
            if component.sum() < 16:
                continue
            eligible += 1
            _,pieces = ndi.label(component & pred[slices],structure=np.ones((3,3)))
            fragmented += int(pieces > 1)
            missed += int(pieces == 0)
        tp = int((gt & pred).sum())
        rows.append({'file':path.name,'tp':tp,'fp':int((pred & ~gt).sum()),'fn':int((gt & ~pred).sum()),
                     'thin_skeleton_total':int(thin.sum()),'thin_skeleton_hit':int((thin & pred).sum()),
                     'components':eligible,'fragmented':fragmented,'missed':missed})
    totals = {key:sum(row[key] for row in rows) for key in rows[0] if key != 'file'}
    return {'per_image':rows,'totals':totals,
            'precision':totals['tp']/max(1,totals['tp']+totals['fp']),
            'recall':totals['tp']/max(1,totals['tp']+totals['fn']),
            'thin_skeleton_recall':totals['thin_skeleton_hit']/max(1,totals['thin_skeleton_total']),
            'fragmented_fraction':totals['fragmented']/max(1,totals['components']),
            'missed_fraction':totals['missed']/max(1,totals['components']),
            'note':'Pooled-pixel diagnostics. Thin = morphological GT skeleton with distance to background <=2px; not a measured physical width. Components >=16px. Intersection fragmentation is only a proxy; examine missed components and precision together.'}


def report():
    base = ROOT/'runs/resnet34_unet_scse_ce_dice'
    entries = [('baseline',read(base/'validation_metrics.json'),read(base/'d4_tta/metrics.json'),base/'d4_tta')]
    for method in METHODS:
        status = read(OUT/method/'status.json')
        if status.get('state') == 'completed':
            entries.append((method,status['single'],status['d4'],OUT/method/'d4_tta'))
    lines = ['# 마지막 업샘플링 비교', '',
             'Seed 42, 100 epoch, CE/Dice 기준선에서 각 변경을 독립 적용. clDice는 추가하지 않음. 최고 일반 검증 체크포인트에 D4 평가.', '',
             '| 방법 | 일반 mIoU % | D4 mIoU % | Eutectic IoU % | 차이 %p |', '|---|---:|---:|---:|---:|']
    summary = []
    for method,single,d4,masks in entries:
        lines.append(f'| {method} | {single["miou"]*100:.4f} | {d4["miou"]*100:.4f} | {d4["class_iou"][2]*100:.4f} | {(d4["miou"]-entries[0][2]["miou"])*100:+.4f} |')
        diag_path = OUT/(method+'_morphology.json')
        if not diag_path.exists():
            diag_path.write_text(json.dumps(diagnostics(masks),indent=2))
        summary.append({'method':method,'single':single,'d4':d4,'diagnostics':read(diag_path)})
    lines += ['', '| 방법 | Eut precision % | Eut recall % | 얇은 skeleton recall % | 분절 component % | 누락 component % |', '|---|---:|---:|---:|---:|---:|']
    for row in summary:
        d=row['diagnostics']
        lines.append('| '+row['method']+' | '+' | '.join(f'{d[k]*100:.3f}' for k in ('precision','recall','thin_skeleton_recall','fragmented_fraction','missed_fraction'))+' |')
    lines += ['', '형태 지표는 전체 픽셀/성분을 합친 보조 진단입니다. 얇은 skeleton은 GT skeleton 중 배경까지 거리 2px 이하로 정의했습니다. 분절은 GT와 예측의 교집합이 여러 조각인 경우이며 물리적인 연결성의 증거가 아닙니다.',
              'Validation 20장, 단일 seed의 탐색 결과입니다. 일반화 개선이나 업샘플링의 인과적 문제를 이 결과만으로 확정하지 않습니다.']
    (OUT/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--worker',choices=METHODS)
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.report_only:
        report();return
    if args.launch:
        with (OUT/'queue.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve())],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print('Queue PID',p.pid);return
    if args.worker:
        import torch
        import research_sweep as sweep
        from mimu.upsampling_variants import UpsamplingUNet
        torch.set_num_threads(4)
        torch.hub.set_dir(str(ROOT/'.torch_cache/hub'))
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable')
        # Verify actual device forward/backward before starting a full run.
        test=UpsamplingUNet(args.worker,pretrained=False).to('mps').train()
        y=test(torch.rand(2,1,224,224,device='mps'))
        y.square().mean().backward()
        assert torch.isfinite(y).all()
        assert all(torch.isfinite(p.grad).all() for p in test.parameters() if p.grad is not None)
        del test,y
        torch.mps.empty_cache()
        sweep.ResearchUNet=UpsamplingUNet
        sweep.model_digest=common_digest
        sweep.WEIGHTS[args.worker]=0.
        sweep.run_method(args.worker,SimpleNamespace(seed=42,device='mps',epochs=100,output=OUT,
                         architecture='resnet34-unet-scse-last-stage-'+args.worker))
        return
    with (OUT/'queue.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        caffeine=subprocess.Popen(['/usr/bin/caffeinate','-i','-s','-w',str(os.getpid())])
        try:
            (OUT/'status.json').write_text(json.dumps({'state':'waiting_for_seed_validation','pid':os.getpid()}))
            # Acquire the existing queue's lock to prevent concurrent GPU training.
            with (ROOT/'runs/cldice_seed_validation/queue.lock').open('a') as prior:
                fcntl.flock(prior,fcntl.LOCK_EX)
                for name in ['upsampling_sweep.py','mimu/upsampling_variants.py','research_sweep.py','mimu/resnet34_unet.py','mimu/data.py','mimu/loss.py','train.py','evaluate_d4.py']:
                    dest=OUT/'source_snapshot'/name
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    if not dest.exists():shutil.copy2(ROOT/name,dest)
                report()
                failed=[]
                for method in METHODS:
                    (OUT/'status.json').write_text(json.dumps({'state':'running','method':method,'pid':os.getpid()}))
                    for attempt in range(2):
                        if read(OUT/method/'status.json').get('state')=='completed':break
                        print('START',method,attempt+1,time.ctime(),flush=True)
                        subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',method],cwd=ROOT)
                    if read(OUT/method/'status.json').get('state')!='completed':failed.append(method)
                    report()
                (OUT/'status.json').write_text(json.dumps({'state':'completed' if not failed else 'completed_with_failures','failures':failed}))
        finally:caffeine.terminate()


if __name__=='__main__':main()
