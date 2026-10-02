#!/usr/bin/env python3
"""Paired baseline/clDice validation at prespecified seeds 123 and 2026."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'runs/cldice_seed_validation'
SEEDS = (123, 2026)


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def report():
    rows = []
    base = ROOT/'runs/resnet34_unet_scse_ce_dice'
    old = read(ROOT/'runs/research_sweep_v1/cldice/status.json')
    rows.append({'seed': 42, 'method': 'baseline', 'single': read(base/'validation_metrics.json'),
                 'd4': read(base/'d4_tta/metrics.json')})
    rows.append({'seed': 42, 'method': 'cldice', 'single': old['single'], 'd4': old['d4']})
    for seed in SEEDS:
        for method in ('baseline', 'cldice'):
            status = read(OUT/str(seed)/method/'status.json')
            if status.get('state') == 'completed':
                rows.append({'seed': seed, 'method': method, 'single': status['single'], 'd4': status['d4']})
    lines = ['# clDice seed 재현성 검증', '',
             'Seed 42는 기존 결과, seed 123/2026은 사전 지정한 추가 실험입니다. 각 100 epoch, 동일한 데이터·sampling·optimizer·평가 조건을 유지합니다.',
             '각 seed에서 일반 검증 최고 체크포인트를 선택한 뒤 D4 평가합니다. 얼리 스탑핑은 사용하지 않습니다.', '',
             '| Seed | 모델 | 일반 mIoU % | D4 mIoU % | D4 Eutectic IoU % |',
             '|---|---|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["seed"]} | {row["method"]} | {100*row["single"]["miou"]:.4f} | {100*row["d4"]["miou"]:.4f} | {100*row["d4"]["class_iou"][2]:.4f} |')
    pairs = []
    for seed in (42,)+SEEDS:
        pair = {r['method']: r for r in rows if r['seed'] == seed}
        if len(pair) == 2:
            pairs.append({'seed': seed, 'miou_delta_pp': 100*(pair['cldice']['d4']['miou']-pair['baseline']['d4']['miou']),
                          'eutectic_delta_pp': 100*(pair['cldice']['d4']['class_iou'][2]-pair['baseline']['d4']['class_iou'][2])})
    lines += ['', f'완료된 seed 쌍: {len(pairs)}/3.']
    if len(pairs) == 3:
        lines += ['', '평균 ± 표본 표준편차 (3 seeds):']
        for method in ('baseline', 'cldice'):
            values = [100*r['d4']['miou'] for r in rows if r['method'] == method]
            eut = [100*r['d4']['class_iou'][2] for r in rows if r['method'] == method]
            lines.append(f'- {method}: D4 mIoU {statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}%, Eutectic {statistics.mean(eut):.4f} ± {statistics.stdev(eut):.4f}%.')
        lines.append(f'- clDice 평균 차이: mIoU {statistics.mean(p["miou_delta_pp"] for p in pairs):+.4f}%p, Eutectic {statistics.mean(p["eutectic_delta_pp"] for p in pairs):+.4f}%p.')
    lines += ['', '동일한 검증 20장에 대한 결과로 공식 Test 성능은 아닙니다. seed 3개로 확정적인 통계적 우위를 주장하지 않습니다.']
    (OUT/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')
    (OUT/'summary.json').write_text(json.dumps({'rows': rows, 'paired_deltas': pairs}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--worker', choices=('baseline', 'cldice'))
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.launch:
        with (OUT/'queue.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve())], cwd=ROOT,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print('Background queue PID:', child.pid)
        return
    if args.worker:
        import torch
        from research_sweep import run_method
        torch.set_num_threads(4)
        torch.hub.set_dir(str(ROOT/'.torch_cache/hub'))
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable')
        run_method(args.worker, SimpleNamespace(seed=args.seed, device='mps', epochs=100, output=OUT/str(args.seed)))
        return
    with (OUT/'queue.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Queue already running', flush=True)
            return
        caffeine = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-s', '-w', str(os.getpid())])
        try:
            sources = ['seed_validation.py', 'research_sweep.py', 'mimu/research_variants.py',
                       'mimu/resnet34_unet.py', 'mimu/data.py', 'mimu/loss.py', 'train.py', 'evaluate_d4.py']
            for name in sources:
                dest = OUT/'source_snapshot'/name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copy2(ROOT/name, dest)
            report()
            failures = []
            for seed in SEEDS:
                for method in ('baseline', 'cldice'):
                    (OUT/'status.json').write_text(json.dumps({'state': 'running', 'seed': seed, 'method': method, 'pid': os.getpid()}))
                    for attempt in range(2):
                        if read(OUT/str(seed)/method/'status.json').get('state') == 'completed':
                            break
                        print('START', seed, method, 'attempt', attempt+1, time.ctime(), flush=True)
                        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--worker', method, '--seed', str(seed)], cwd=ROOT)
                    if read(OUT/str(seed)/method/'status.json').get('state') != 'completed':
                        failures.append([seed, method])
                    report()
            (OUT/'status.json').write_text(json.dumps({'state': 'completed' if not failures else 'completed_with_failures', 'failures': failures}))
        finally:
            caffeine.terminate()


if __name__ == '__main__':
    main()
