"""Re-run the best DySample architecture with a full 200-epoch cosine schedule."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'runs/dysample_200_v1'


def worker():
    import torch
    import research_sweep as sweep
    from mimu.upsampling_variants import UpsamplingUNet
    from upsampling_sweep import common_digest, diagnostics
    torch.set_num_threads(4)
    torch.hub.set_dir(str(ROOT/'.torch_cache/hub'))
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable')
    sweep.ResearchUNet = UpsamplingUNet
    sweep.model_digest = common_digest
    sweep.WEIGHTS['dysample'] = 0.
    sweep.run_method('dysample', SimpleNamespace(seed=42, device='mps', epochs=200, output=OUT,
        architecture='resnet34-unet-scse-last-stage-dysample',
        extra_config={'experiment':'fresh initialization, cosine T_max=200; no warm start from 100-epoch checkpoint'}))
    old = json.loads((ROOT/'runs/upsampling_sweep_v1/dysample/status.json').read_text())
    new = json.loads((OUT/'dysample/status.json').read_text())
    old_config = json.loads((ROOT/'runs/upsampling_sweep_v1/dysample/config.json').read_text())
    new_config = json.loads((OUT/'dysample/config.json').read_text())
    if old_config['initial_common_weights_sha256'] != new_config['initial_common_weights_sha256']:
        raise AssertionError('Initial common weights differ from original run')
    morph = diagnostics(OUT/'dysample/d4_tta')
    (OUT/'dysample/morphology.json').write_text(json.dumps(morph, indent=2))
    old_morph = json.loads((ROOT/'runs/upsampling_sweep_v1/dysample_morphology.json').read_text())
    lines = ['# DySample 100 vs 200 epoch', '',
             '동일 seed42, 모델/패치/loss/optimizer. 새로 학습하며 cosine T_max를 100→200으로 변경.',
             '일반 검증 mIoU로 best 선택 후 D4 평가. 학습 기간과 학습률 경로가 함께 달라지는 비교.', '',
             '| Epoch budget | Best epoch | D4 mIoU % | Eut IoU % | Eut precision % | Eut recall % | Thin recall % |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for budget, state, diag in [(100, old, old_morph), (200, new, morph)]:
        values = [state['d4']['miou'], state['d4']['class_iou'][2], diag['precision'], diag['recall'], diag['thin_skeleton_recall']]
        lines.append(f'| {budget} | {state["best_epoch"]} | '+' | '.join(f'{v*100:.4f}' for v in values)+' |')
    lines += ['', '단일 seed, 반복 사용한 Valid20 기준이며 독립 Test 성능이 아니다. 기존 결과는 보존한다.']
    (OUT/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.worker: return worker()
    if args.launch:
        import torch
        if not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable')
        with (OUT/'queue.log').open('a') as log:
            p = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve())], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print('Queue PID', p.pid)
        return
    with (OUT/'queue.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return
        def status(**data):
            tmp = OUT/'status.tmp'
            tmp.write_text(json.dumps({'pid':os.getpid(), **data}, indent=2))
            tmp.replace(OUT/'status.json')
        caffeine = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-s', '-w', str(os.getpid())])
        try:
            for name in ['dysample_200.py', 'research_sweep.py', 'upsampling_sweep.py', 'mimu/upsampling_variants.py',
                         'mimu/resnet34_unet.py', 'mimu/data.py', 'mimu/loss.py', 'mimu/metrics.py',
                         'mimu/research_variants.py', 'train.py', 'evaluate_d4.py']:
                dest = OUT/'source_snapshot'/name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists(): shutil.copy2(ROOT/name, dest)
            status(state='waiting_for_context_corrected', target_epochs=200)
            with (ROOT/'runs/context_corrected_v1/queue.lock').open('a') as previous:
                fcntl.flock(previous, fcntl.LOCK_EX)
                status(state='running', target_epochs=200)
                subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--worker'], cwd=ROOT, check=True)
                status(state='completed')
        except BaseException as exc:
            status(state='failed', error=repr(exc))
            raise
        finally: caffeine.terminate()


if __name__ == '__main__': main()
