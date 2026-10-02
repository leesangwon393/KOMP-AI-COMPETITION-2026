#!/usr/bin/env python3
"""Keep the authorized research queue running and save an overnight summary."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs/research_sweep_v1'
METHODS = ['contrast', 'snake', 'boundary', 'cldice']


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def running(pid):
    if not pid:
        return False
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True)
    return result.returncode == 0 and 'research_sweep.py' in result.stdout


def report():
    names = {'contrast': 'Pixel Contrast', 'snake': 'DSConv 응용',
             'boundary': '경계 보조 학습', 'cldice': 'clDice'}
    base = read(ROOT / 'runs/resnet34_unet_scse_ce_dice/d4_tta/metrics.json')
    lines = ['# 밤사이 실험 결과', '',
             '동일한 100 epoch 설정으로 논문에서 착안한 구성 요소를 개별 비교했습니다. 논문 전체 재현은 아닙니다.', '',
             '| 방법 | 상태 | 최고 epoch | 일반 mIoU | D4 mIoU | D4 Eutectic IoU | 기준 대비 D4 차이 |',
             '|---|---|---:|---:|---:|---:|---:|']
    single = read(ROOT / 'runs/resnet34_unet_scse_ce_dice/validation_metrics.json')
    lines.append(f'| 기존 scSE | 완료 | 78 | {single["miou"]*100:.4f}% | {base["miou"]*100:.4f}% | {base["class_iou"][2]*100:.4f}% | — |')
    completed = []
    for method in METHODS:
        status = read(OUT / method / 'status.json')
        if status.get('state') == 'completed':
            d4 = status['d4']
            completed.append((method, d4['miou']))
            lines.append(f'| {names[method]} | 완료 | {status["best_epoch"]} | {status["single"]["miou"]*100:.4f}% | {d4["miou"]*100:.4f}% | {d4["class_iou"][2]*100:.4f}% | {(d4["miou"]-base["miou"])*100:+.4f}%p |')
        else:
            lines.append(f'| {names[method]} | {status.get("state", "미완료")} | — | — | — | — | — |')
    if completed:
        best = max(completed, key=lambda item: item[1])
        lines += ['', f'신규 실험 최고: {names[best[0]]}, D4 mIoU {best[1]*100:.4f}%.',
                  '기존 모델을 넘어섰는지와 별개로 seed 42 한 번의 결과이며, 검증 20장에 대한 반복 선택 편향이 있습니다. 작은 차이를 확정적인 개선으로 해석하지 않습니다.']
    lines += ['', '학습을 조기 종료하지 않았습니다. patience 15/20/30의 절약 epoch와 놓치는 성능은 RESULTS.md 및 early_stopping_review.json에 사후 계산했습니다.',
              '최고 체크포인트는 일반 검증 mIoU로 선택한 뒤 D4로 평가했습니다. 클래스별 상세 결과와 이미지 단위 bootstrap은 RESULTS.md 및 summary.json을 참고하세요.',
              '오류 발생 시 같은 설정과 저장된 체크포인트로 최대 2회 재시도하고 다음 실험으로 넘어갑니다. 실행 내역은 overnight.log에 기록됩니다.']
    (OUT / 'OVERNIGHT_REPORT_KO.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--existing-pid', type=int, default=4708)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.launch:
        pidfile = OUT / 'overnight.pid'
        if pidfile.exists():
            old = subprocess.run(['ps', '-p', pidfile.read_text().strip(), '-o', 'command='], capture_output=True, text=True)
            if old.returncode == 0 and 'overnight_research.py' in old.stdout:
                print('Supervisor already running:', pidfile.read_text().strip())
                return
        with (OUT / 'overnight.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                      '--existing-pid', str(args.existing_pid)], cwd=ROOT,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        pidfile.write_text(str(child.pid))
        print('Detached supervisor PID:', child.pid)
        return
    caffeine = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-s', '-w', str(os.getpid())])
    print('Supervisor started', time.ctime(), 'waiting for existing PID', args.existing_pid, flush=True)
    try:
        while running(args.existing_pid):
            time.sleep(30)
        for method in METHODS:
            for attempt in range(2):
                if read(OUT / method / 'status.json').get('state') == 'completed':
                    break
                print('Resume', method, 'attempt', attempt + 1, time.ctime(), flush=True)
                result = subprocess.run([sys.executable, '-u', 'research_sweep.py', '--device', 'mps',
                                         '--methods', method], cwd=ROOT)
                print('Exit', result.returncode, flush=True)
        from research_sweep import create_report
        create_report(OUT)
        report()
        done = [m for m in METHODS if read(OUT / m / 'status.json').get('state') == 'completed']
        (OUT / 'status.json').write_text(json.dumps({'state': 'completed' if len(done) == 4 else 'completed_with_failures',
                                                   'sequence': METHODS, 'completed_methods': done}, indent=2))
        print('Overnight queue finished', time.ctime(), done, flush=True)
    finally:
        caffeine.terminate()


if __name__ == '__main__':
    main()
