"""Paired 100-epoch runs selected with matched 448-input/224-center validation."""
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
OUT = ROOT/'runs/context_corrected_v1'
METHODS = ('baseline', 'context448')


def evaluate_halo(model, dataset, device, batch_size=4, criterion=None, absent='one', export=None, tta=False):
    import numpy as np
    from PIL import Image
    from evaluate_context_halo import predict
    from mimu.data import PALETTE
    from mimu.metrics import confusion_matrix, summarize
    from train import write_json
    model.eval()
    matrices = []
    if export is not None:
        export.mkdir(parents=True, exist_ok=True)
    for path, image, target in zip(dataset.paths, dataset.images, dataset.original_labels):
        prob = predict(model, image, device, tta=tta)[1 if tta else 0]
        pred = prob.argmax(0).numpy().astype(np.uint8)
        matrices.append(confusion_matrix(target, pred))
        if export is not None:
            Image.fromarray(PALETTE[pred]).save(export/path.name.replace('_image', '_mask'))
    result = summarize(matrices, absent)
    result['confusion_matrix_sum'] = np.sum(matrices, axis=0).tolist()
    result['filenames'] = [p.name for p in dataset.paths]
    result['inference'] = {'input':448, 'output_center':224, 'stride':112, 'padding':'reflect', 'd4':tta}
    if export is not None:
        write_json(export/'metrics.json', result)
    return result


def worker(method):
    import torch
    import research_sweep as sweep
    from mimu.context_hrnet import ContextDataset, ContextUNet
    torch.set_num_threads(4)
    torch.hub.set_dir(str(ROOT/'.torch_cache/hub'))
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable')
    if method == 'context448':
        sweep.NativePatchDataset = ContextDataset
        sweep.ResearchUNet = lambda _, pretrained: ContextUNet(pretrained)
    sweep.WEIGHTS[method] = 0.
    sweep.evaluate = evaluate_halo
    sweep.d4_eval = lambda model, dataset, output, device: evaluate_halo(model, dataset, device, export=output, tta=True)
    sweep.run_method(method, SimpleNamespace(seed=42, device='mps', epochs=100, output=OUT,
        extra_config={'train_input':224 if method == 'baseline' else 448, 'supervised_center':224,
                      'evaluator':'448 input, central224, stride112, reflect padding; image mean mIoU absent=one',
                      'checkpoint_selection':'best single-view halo validation each epoch; D4 only after selection'}))


def report():
    import numpy as np
    from upsampling_sweep import diagnostics
    lines = ['# 문맥 확대: 매 epoch halo 검증으로 선택', '',
             '두 모델 모두 448 입력/중앙224 출력/stride112/reflect padding으로 검증·선택. Seed42, 각100 epoch.', '',
             '| 모델 | Best epoch | D4 mIoU % | Eut IoU % | Eut precision % | Eut recall % | Thin recall % | Al3Ni→Eut % | Eut→Al3Ni % |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    digests = []
    for method in METHODS:
        p = OUT/method/'status.json'
        if not p.exists(): continue
        s = json.loads(p.read_text())
        if s['state'] != 'completed': continue
        digests.append(json.loads((OUT/method/'config.json').read_text())['initial_common_weights_sha256'])
        p = OUT/method/'morphology.json'
        if not p.exists(): p.write_text(json.dumps(diagnostics(OUT/method/'d4_tta'), indent=2))
        d = json.loads(p.read_text())
        cm = np.array(s['d4']['confusion_matrix_sum'])
        vals = [s['d4']['miou'], s['d4']['class_iou'][2], d['precision'], d['recall'],
                d['thin_skeleton_recall'], cm[1,2]/cm[1].sum(), cm[2,1]/cm[2].sum()]
        lines.append(f'| {method} | {s["best_epoch"]} | '+' | '.join(f'{v*100:.4f}' for v in vals)+' |')
    if len(set(digests)) > 1: raise AssertionError('Initial weights mismatch')
    lines += ['', 'mIoU/클래스 IoU는 이미지 평균. 나머지는 합산 진단. 단일 seed 탐색이며 Valid 선택 편향이 있다.',
              '대조군도 halo 검증으로 새로 checkpoint를 선택한다. 입력 면적·BN 통계가 함께 달라지므로 문맥만의 인과 효과는 아니다.']
    (OUT/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--worker', choices=METHODS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.worker: return worker(args.worker)
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
            p = OUT/'status.tmp'
            p.write_text(json.dumps({'pid':os.getpid(), **data}, indent=2))
            p.replace(OUT/'status.json')
        caffeine = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-s', '-w', str(os.getpid())])
        try:
            for name in ['context_corrected_sweep.py', 'evaluate_context_halo.py', 'research_sweep.py', 'train.py',
                         'evaluate_d4.py', 'upsampling_sweep.py', 'mimu/context_hrnet.py', 'mimu/data.py',
                         'mimu/research_variants.py', 'mimu/resnet34_unet.py', 'mimu/metrics.py', 'mimu/loss.py']:
                dest = OUT/'source_snapshot'/name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists(): shutil.copy2(ROOT/name, dest)
            with (ROOT/'runs/contact_sampling_v1/queue.lock').open('a') as old_contact, (ROOT/'runs/context_hrnet_v1/queue.lock').open('a') as old_context:
                status(state='waiting_for_previous_queues')
                fcntl.flock(old_contact, fcntl.LOCK_EX)
                fcntl.flock(old_context, fcntl.LOCK_EX)
                for method in METHODS:
                    status(state='running', method=method)
                    subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--worker', method], cwd=ROOT, check=True)
                    report()
                status(state='completed')
        except BaseException as exc:
            status(state='failed', error=repr(exc))
            raise
        finally: caffeine.terminate()


if __name__ == '__main__': main()
