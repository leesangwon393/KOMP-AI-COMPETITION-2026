#!/usr/bin/env python3
"""100-epoch context and HRNet ablations, resumable sequential execution."""
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

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/context_hrnet_v1'
METHODS=('context448','hrnet_w18')


def read(path):return json.loads(path.read_text()) if path.exists() else {}


def cross_evaluate():
    import torch
    from mimu.context_hrnet import ContextUNet
    from mimu.research_variants import ResearchUNet
    from mimu.data import FullResolutionDataset
    from train import evaluate,write_json
    import research_sweep as sweep
    from evaluate_d4 import predict_image
    torch.set_num_threads(4)
    ds=FullResolutionDataset(ROOT/'Data','valid')
    for method,checkpoint in [('baseline',ROOT/'runs/resnet34_unet_scse_ce_dice/best.pt'),('context448',OUT/'context448/best.pt')]:
        folder=OUT/'cross_eval'/method
        if (folder/'d4_tta/metrics.json').exists():continue
        model=ContextUNet(False) if method=='context448' else ResearchUNet('baseline',False)
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'])
        model.pretrained=True
        model.to('mps').eval()
        ds.size,ds.stride=448,224
        single=evaluate(model,ds,torch.device('mps'),4,None,'one',folder/'valid_masks')
        # Evaluate both baseline and context checkpoints with the same larger inference tiles.
        original=sweep.predict_image
        sweep.predict_image=lambda m,im,size,stride,device,batch:predict_image(m,im,448,224,device,4)
        try:d4=sweep.d4_eval(model,ds,folder/'d4_tta',torch.device('mps'))
        finally:sweep.predict_image=original
        write_json(folder/'metrics.json',{'single':single,'d4':d4,'tile':448,'stride':224,
                                         'selection':'Checkpoint selected using 224 tile validation; 448 evaluation is post hoc, no checkpoint reselection.'})
        del model
        torch.mps.empty_cache()
    # Correct the training/inference context mismatch; retain original scores as diagnostics.
    from evaluate_context_halo import main as halo_evaluate
    halo_evaluate()


def report():
    from upsampling_sweep import diagnostics
    base=ROOT/'runs/resnet34_unet_scse_ce_dice'
    entries=[('baseline224',read(base/'validation_metrics.json'),read(base/'d4_tta/metrics.json'),base/'d4_tta')]
    for method in METHODS:
        s=read(OUT/method/'status.json')
        if s.get('state')=='completed':entries.append((method,s['single'],s['d4'],OUT/method/'d4_tta'))
    lines=['# 문맥 및 HRNet 비교', '',
           '모든 학습은 seed 42, 100 epoch, batch 4, 420 central patches/epoch, 동일 CE/Dice와 AdamW/cosine 설정. 기본 검증 tile 224/stride 112로 checkpoint 선택.',
           'context448은 원본 448 입력의 중앙 224에만 loss를 적용합니다. 기준선과 정답 패치·샘플링 RNG·업데이트 수를 맞추되 입력 연산량은 4배 면적입니다. 이미지 가장자리의 외부 문맥은 reflect padding입니다.',
           'HRNet은 timm 1.0.22의 ImageNet W18 backbone + 1/4 해상도 multi-resolution concat segmentation head입니다. HRNet-OCR 전체 모델 재현은 아닙니다.', '',
           '| 방법 (평가 tile 224) | 일반 mIoU % | D4 mIoU % | Eut IoU % | 얇은 skeleton recall % |', '|---|---:|---:|---:|---:|']
    summary=[]
    for name,single,d4,masks in entries:
        diagnostic=OUT/(name+'_morphology.json')
        if not diagnostic.exists():diagnostic.write_text(json.dumps(diagnostics(masks),indent=2))
        d=read(diagnostic)
        lines.append(f'| {name} | {single["miou"]*100:.4f} | {d4["miou"]*100:.4f} | {d4["class_iou"][2]*100:.4f} | {d["thin_skeleton_recall"]*100:.4f} |')
        summary.append({'method':name,'single':single,'d4':d4,'diagnostics':d})
    lines+=['','**동일 checkpoint의 448 inference 비교**','','| 모델 | 일반 mIoU % | D4 mIoU % | Eut IoU % |','|---|---:|---:|---:|']
    cross={}
    for method in ('baseline','context448'):
        d=read(OUT/'cross_eval'/method/'metrics.json')
        if d:
            cross[method]=d
            lines.append(f'| {method} | {d["single"]["miou"]*100:.4f} | {d["d4"]["miou"]*100:.4f} | {d["d4"]["class_iou"][2]*100:.4f} |')
    lines+=['','문맥 크기를 늘리면 feature 통계와 연산량도 변하므로 완벽한 단일 요인 분리는 아닙니다.',
            'Validation 20장, 단일 seed의 탐색 실험입니다. 448 평가는 224 검증으로 선택한 동일 checkpoint에 시행하며 재선택하지 않습니다. 상세 오답 비교는 error_audit/error_panels.png를 참고하세요.']
    (OUT/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')
    (OUT/'summary.json').write_text(json.dumps({'main':summary,'cross_eval448':cross},indent=2))


def worker(method):
    import torch
    import research_sweep as sweep
    from mimu.context_hrnet import build_model,ContextDataset
    torch.set_num_threads(4)
    torch.hub.set_dir(str(ROOT/'.torch_cache/hub'))
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    sweep.ResearchUNet=build_model
    sweep.WEIGHTS[method]=0.
    if method=='context448':sweep.NativePatchDataset=ContextDataset
    sweep.run_method(method,SimpleNamespace(seed=42,device='mps',epochs=100,output=OUT,
                     architecture='resnet34-unet-scse-context448' if method=='context448' else 'hrnet-w18-native-multires-head',
                     extra_config={'train_input_size':448 if method=='context448' else 224,'supervised_size':224,
                                   'pretrained_source':'torchvision-resnet34' if method=='context448' else 'timm/hrnet_w18.ms_in1k',
                                   'context_padding':'reflect' if method=='context448' else None}))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--launch',action='store_true')
    p.add_argument('--worker',choices=METHODS)
    p.add_argument('--cross-eval',action='store_true')
    p.add_argument('--report-only',action='store_true')
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.worker:worker(args.worker);return
    if args.cross_eval:cross_evaluate();return
    if args.report_only:report();return
    if args.launch:
        with (OUT/'queue.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve())],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print('Queue PID',child.pid);return
    with (OUT/'queue.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        caffeine=subprocess.Popen(['/usr/bin/caffeinate','-i','-s','-w',str(os.getpid())])
        try:
            for name in ['context_hrnet_sweep.py','mimu/context_hrnet.py','research_sweep.py','mimu/research_variants.py','mimu/resnet34_unet.py','mimu/data.py','mimu/loss.py','mimu/metrics.py','train.py','evaluate_d4.py','analyze_error_regions.py']:
                dest=OUT/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():shutil.copy2(ROOT/name,dest)
            provenance={'timm':'1.0.22','weights':'timm/hrnet_w18.ms_in1k',
                        'weights_sha256':hashlib.sha256((ROOT/'.torch_cache/hrnet/hrnet_w18.ms_in1k.safetensors').read_bytes()).hexdigest()}
            (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
            report();failures=[]
            for method in METHODS:
                (OUT/'status.json').write_text(json.dumps({'state':'running','method':method,'pid':os.getpid()}))
                for attempt in range(2):
                    if read(OUT/method/'status.json').get('state')=='completed':break
                    print('START',method,attempt+1,time.ctime(),flush=True)
                    subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',method],cwd=ROOT)
                if read(OUT/method/'status.json').get('state')!='completed':failures.append(method)
                report()
            if 'context448' not in failures:
                (OUT/'status.json').write_text(json.dumps({'state':'cross_evaluating448','pid':os.getpid()}))
                result=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--cross-eval'],cwd=ROOT)
                if result.returncode:failures.append('cross_eval448')
            report()
            (OUT/'status.json').write_text(json.dumps({'state':'completed' if not failures else 'completed_with_failures','failures':failures}))
        finally:caffeine.terminate()


if __name__=='__main__':main()
