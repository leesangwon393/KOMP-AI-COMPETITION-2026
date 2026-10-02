#!/usr/bin/env python3
"""Sequential paired sampling experiment; waits for the existing GPU queue."""
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
OUT = ROOT/'runs/contact_sampling_v1'
METHODS = {'baseline': 0., 'contact25': .25}


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def worker(method):
    import torch
    import research_sweep as sweep
    from mimu.contact_sampling import ContactPatchDataset
    from mimu.research_variants import ResearchUNet
    torch.set_num_threads(4)
    torch.hub.set_dir(str(ROOT/'.torch_cache/hub'))
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable')
    sweep.NativePatchDataset = lambda *a, **kw: ContactPatchDataset(*a, contact_probability=METHODS[method], **kw)
    sweep.ResearchUNet = lambda _, pretrained: ResearchUNet('baseline', pretrained=pretrained)
    sweep.WEIGHTS[method] = 0.
    sweep.run_method(method, SimpleNamespace(seed=42, device='mps', epochs=100, output=OUT,
        extra_config={'sampling': {'rich': .5, 'contact': METHODS[method], 'all': .5-METHODS[method]},
                      'contact_definition': '8-neighbor Al3Ni/Eutectic endpoints, train GT only; image uniform then coordinate uniform; center crop clamped to image'}))


def report():
    import numpy as np
    from upsampling_sweep import diagnostics
    result = {}
    lines = ['# 접촉 경계 sampling 비교', '',
             'Seed 42, 100 epoch, 동일 모델/CE-Dice/420 patches/batch 4. 일반 검증으로 checkpoint 선택 후 D4 평가.', '',
             '| 방법 | D4 mIoU % | Eut precision % | Eut recall % | 얇은 skeleton recall % | Al3Ni→Eut % | Eut→Al3Ni % |',
             '|---|---:|---:|---:|---:|---:|---:|']
    digests = []
    for method in METHODS:
        state = read(OUT/method/'status.json')
        if state.get('state') != 'completed':
            continue
        config = read(OUT/method/'config.json')
        digests.append(config['initial_common_weights_sha256'])
        path = OUT/method/'morphology.json'
        if not path.exists():
            path.write_text(json.dumps(diagnostics(OUT/method/'d4_tta'), indent=2))
        d = read(path)
        cm = np.array(state['d4']['confusion_matrix_sum'])
        values = [state['d4']['miou'], d['precision'], d['recall'], d['thin_skeleton_recall'],
                  cm[1,2]/cm[1].sum(), cm[2,1]/cm[2].sum()]
        lines.append('| '+method+' | '+' | '.join(f'{v*100:.4f}' for v in values)+' |')
        result[method] = {'status': state, 'diagnostics': d}
    if len(set(digests)) > 1:
        raise AssertionError('Initial weights differ between arms')
    lines += ['', 'mIoU는 이미지 평균, 나머지는 픽셀/형태 합산 진단이다. 단일 seed 탐색이며 Valid 선택 편향이 있다.',
              '점수 상승만으로 구조 보존 개선으로 판단하지 않는다. Eutectic recall과 얇은 skeleton recall을 함께 확인한다.']
    (OUT/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')
    (OUT/'summary.json').write_text(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--launch', action='store_true')
    p.add_argument('--worker', choices=METHODS)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args.worker)
        return
    if args.launch:
        import torch
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable; launch outside sandbox')
        with (OUT/'queue.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve())], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print('Queue PID', child.pid)
        return
    with (OUT/'queue.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        def status(**kw):
            tmp = OUT/'status.tmp'
            tmp.write_text(json.dumps({'pid': os.getpid(), **kw}, indent=2))
            tmp.replace(OUT/'status.json')
        caffeine = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-s', '-w', str(os.getpid())])
        try:
            for name in ['contact_sampling_sweep.py', 'mimu/contact_sampling.py', 'research_sweep.py',
                         'mimu/data.py', 'mimu/research_variants.py', 'mimu/resnet34_unet.py',
                         'mimu/loss.py', 'mimu/metrics.py', 'train.py', 'evaluate_d4.py', 'upsampling_sweep.py']:
                dest = OUT/'source_snapshot'/name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copy2(ROOT/name, dest)
            status(state='waiting_for_context_queue')
            # Hold its lock throughout our run so a resumed older queue cannot overlap.
            with (ROOT/'runs/context_hrnet_v1/queue.lock').open('a') as previous:
                fcntl.flock(previous, fcntl.LOCK_EX)
                for method in METHODS:
                    status(state='running', method=method)
                    subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--worker', method],
                                   cwd=ROOT, check=True)
                    report()
                status(state='completed')
        except BaseException as exc:
            status(state='failed', error=repr(exc))
            raise
        finally:
            caffeine.terminate()


if __name__ == '__main__':
    main()
