"""50-epoch Train-only SSL followed by matched 200-epoch DySample fine-tuning."""
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
OUT = ROOT/'runs/domain_ssl_200_v1'
SSL_EPOCHS = 50
STEPS = 32


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def setup():
    import torch
    torch.set_num_threads(4)
    torch.hub.set_dir(str(ROOT/'.torch_cache/hub'))
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; run outside sandbox')
    return torch


def preflight():
    torch = setup()
    from mimu.domain_ssl import ContrastiveEncoder, nt_xent
    OUT.mkdir(parents=True, exist_ok=True)
    result = {'device':'mps', 'torch':torch.__version__, 'attempts':[]}
    for batch in (32, 16, 8):
        model = optimizer = images = loss = z = None
        try:
            torch.manual_seed(42)
            model = ContrastiveEncoder(pretrained=False).to('mps').train()
            optimizer = torch.optim.SGD(model.parameters(), lr=.1*batch/128, momentum=.9)
            images = torch.rand(2*batch, 1, 224, 224, device='mps')
            start = time.monotonic()
            z = model(images)
            loss = nt_xent(z[:batch], z[batch:])
            loss.backward()
            if not bool(torch.isfinite(loss)) or not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):
                raise FloatingPointError('Nonfinite SSL preflight')
            optimizer.step()
            torch.mps.synchronize()
            result['attempts'].append({'batch':batch, 'seconds':time.monotonic()-start, 'loss':loss.item(),
                'driver_memory_bytes':torch.mps.driver_allocated_memory(), 'state':'passed'})
            result['batch_size'] = batch
            break
        except RuntimeError as exc:
            if not any(s in str(exc).lower() for s in ('out of memory', 'allocation', 'allocat')):
                raise
            result['attempts'].append({'batch':batch, 'state':'memory_failed', 'error':str(exc)})
        finally:
            del model, optimizer, images, loss, z
            torch.mps.empty_cache()
    if 'batch_size' not in result:
        raise RuntimeError('No feasible SSL batch >=8')
    from train import write_json
    write_json(OUT/'preflight.json', result)
    print(json.dumps(result, indent=2), flush=True)


def ssl_worker():
    torch = setup()
    from mimu.domain_ssl import ContrastiveEncoder, TrainViews, nt_xent
    from train import cpu_tree, save_checkpoint, write_json
    folder = OUT/'ssl'
    folder.mkdir(parents=True, exist_ok=True)
    if read(folder/'status.json').get('state') == 'completed':
        return
    batch = read(OUT/'preflight.json')['batch_size']
    torch.manual_seed(42)
    ds = TrainViews(ROOT/'Data')
    model = ContrastiveEncoder(pretrained=True).to('mps')
    lr = .1*batch/128
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SSL_EPOCHS, eta_min=1e-4*batch/128)
    config = {'epochs':SSL_EPOCHS, 'seed':42, 'batch_size':batch, 'steps_per_epoch':STEPS,
        'samples_per_epoch':batch*STEPS, 'input':224, 'feature':'layer4 GAP, 512 channels',
        'projection':[512,512,128], 'temperature':.07, 'optimizer':'SGD', 'lr':lr, 'momentum':.9,
        'weight_decay':1e-6, 'cosine_T_max':SSL_EPOCHS, 'eta_min':1e-4*batch/128,
        'normalization':{'mean':.674325, 'std':.229455},
        'augmentation':'same 224 anchor: independent resized crop(scale .5..1, ratio .9..1.1), D4, brightness/contrast .1; no blur',
        'views_forward':'both views concatenated in one encoder forward',
        'sampling':'uniform distinct Train image sources per batch, uniform tile per source; no masks read',
        'selection':'fixed final SSL epoch50; no validation selection',
        'train_manifest':{p.relative_to(ROOT).as_posix():sha(p) for p in ds.paths}}
    start, history = 0, []
    if (folder/'last.pt').exists():
        ck = torch.load(folder/'last.pt', map_location='cpu', weights_only=True)
        if ck['config'] != config:
            raise ValueError('SSL resume config/data mismatch')
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        torch.set_rng_state(ck['torch_rng'])
        torch.mps.set_rng_state(ck['mps_rng'])
        start, history = ck['epoch'], ck['history']
    write_json(folder/'config.json', config)
    try:
        for epoch in range(start+1, SSL_EPOCHS+1):
            begun, total = time.monotonic(), 0.
            model.train()
            epoch_lr = optimizer.param_groups[0]['lr']
            for _ in range(STEPS):
                a, b, _ = ds.batch(batch)
                optimizer.zero_grad(set_to_none=True)
                z = model(torch.cat((a,b)).to('mps'))
                loss = nt_xent(z[:batch], z[batch:])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite SSL loss')
                loss.backward()
                optimizer.step()
                total += loss.item()
            scheduler.step()
            row = {'epoch':epoch, 'loss':total/STEPS, 'lr':epoch_lr, 'seconds':time.monotonic()-begun}
            history.append(row)
            save_checkpoint(folder/'last.pt', {'epoch':epoch, 'config':config, 'model':cpu_tree(model.state_dict()),
                'optimizer':cpu_tree(optimizer.state_dict()), 'scheduler':scheduler.state_dict(),
                'history':history, 'torch_rng':torch.get_rng_state(), 'mps_rng':torch.mps.get_rng_state()})
            write_json(folder/'history.json', history)
            write_json(folder/'status.json', {'state':'ssl_training', 'pid':os.getpid(),
                'completed_epochs':epoch, 'target_epochs':SSL_EPOCHS, 'batch_size':batch, 'latest':row})
            print(f'SSL {epoch}/{SSL_EPOCHS} loss={row["loss"]:.4f} seconds={row["seconds"]:.1f}', flush=True)
        save_checkpoint(folder/'encoder.pt', {'epoch':SSL_EPOCHS, 'config':config, 'encoder':cpu_tree(model.encoder.state_dict())})
        write_json(folder/'status.json', {'state':'completed', 'completed_epochs':SSL_EPOCHS, 'target_epochs':SSL_EPOCHS, 'batch_size':batch})
    except BaseException as exc:
        write_json(folder/'status.json', {'state':'failed', 'error':repr(exc), 'completed_epochs':history[-1]['epoch'] if history else start})
        raise


def finetune_worker():
    torch = setup()
    import research_sweep as sweep
    from mimu.domain_ssl import transfer_encoder, ENCODER_PREFIXES
    from mimu.upsampling_variants import UpsamplingUNet
    from upsampling_sweep import common_digest
    from train import write_json
    path = OUT/'ssl/encoder.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['epoch'] != SSL_EPOCHS:
        raise ValueError('Incomplete SSL pretraining')
    def factory(method, pretrained):
        model = UpsamplingUNet(method, pretrained)
        before = common_digest(model)
        decoder = {k:t.clone() for k,t in model.state_dict().items() if not k.startswith(ENCODER_PREFIXES)}
        rng = torch.get_rng_state().clone()
        transfer_encoder(model, checkpoint['encoder'])
        assert torch.equal(rng, torch.get_rng_state())
        assert all(torch.equal(t, model.state_dict()[k]) for k,t in decoder.items())
        reference = read(ROOT/'runs/dysample_200_v1/dysample/config.json')
        if before != reference['initial_common_weights_sha256']:
            raise ValueError('Initial decoder/common initialization differs from 200-epoch baseline')
        write_json(OUT/'transfer_check.json', {'decoder_unchanged':True, 'rng_unchanged':True,
            'pre_transfer_common_sha256':before, 'ssl_encoder_checkpoint_sha256':sha(path)})
        return model
    sweep.ResearchUNet = factory
    sweep.model_digest = common_digest
    sweep.WEIGHTS['dysample'] = 0.
    sweep.run_method('dysample', SimpleNamespace(seed=42, device='mps', epochs=200, output=OUT,
        architecture='resnet34-unet-scse-last-stage-dysample',
        extra_config={'experiment':'single-stage Train-only SSL encoder initialization; matched DySample200 downstream',
                      'ssl_checkpoint_sha256':sha(path), 'ssl_epochs':SSL_EPOCHS,
                      'ssl_batch':checkpoint['config']['batch_size'], 'ssl_scope':'Train images only; masks ignored'}))
    report()


def report():
    import numpy as np
    from upsampling_sweep import diagnostics
    from train import write_json
    root = ROOT/'runs/dysample_200_v1'
    before = read(root/'dysample/status.json')
    after = read(OUT/'dysample/status.json')
    diagnostic = diagnostics(OUT/'dysample/d4_tta')
    write_json(OUT/'dysample/morphology.json', diagnostic)
    baseline_diagnostic = read(root/'dysample/morphology.json')
    if before['d4']['filenames'] != after['d4']['filenames']:
        raise ValueError('Validation image order differs')
    lines = ['# Train-only SSL + DySample 200 epoch', '',
             'Seed42. Train-only single-stage SSL50 후 기존 DySample과 같은 분할 학습200. Gate/U-Net++ 없음.', '',
             '| 초기화 | Best epoch | D4 mIoU % | Eut IoU % | Eut precision % | Eut recall % | Thin recall % | Al3Ni→Eut % | Eut→Al3Ni % |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, state, diag in [('ImageNet',before,baseline_diagnostic), ('ImageNet+SSL',after,diagnostic)]:
        cm = np.array(state['d4']['confusion_matrix_sum'])
        vals = [state['d4']['miou'],state['d4']['class_iou'][2],diag['precision'],diag['recall'],diag['thin_skeleton_recall'],
                cm[1,2]/cm[1].sum(),cm[2,1]/cm[2].sum()]
        lines.append(f'| {name} | {state["best_epoch"]} | '+' | '.join(f'{v*100:.4f}' for v in vals)+' |')
    delta = np.array(after['d4']['per_image_miou'])-np.array(before['d4']['per_image_miou'])
    write_json(OUT/'comparison.json', {'delta_miou_pp':float(delta.mean()*100), 'improved_images':int((delta>0).sum()),
        'total_images':len(delta), 'per_image_delta_pp':(delta*100).tolist()})
    lines += ['', '단일 seed 탐색. mIoU/클래스IoU는 이미지 평균, 나머지는 합산 진단. 추가 SSL 연산량은 대조군과 다르다.',
              'SSL loss 감소만으로 효과를 확정하지 않는다. Valid D4 성능·양방향 혼동·얇은 구조를 함께 비교한다.']
    (OUT/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--worker', choices=['ssl','finetune'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        preflight()
        if not args.launch:
            return
    if args.worker:
        return ssl_worker() if args.worker == 'ssl' else finetune_worker()
    if args.launch:
        setup()
        if not read(OUT/'preflight.json'):
            raise RuntimeError('Run --preflight before launch')
        with (OUT/'queue.log').open('a') as log:
            p = subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve())],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print('Queue PID',p.pid)
        return
    with (OUT/'queue.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return
        from train import write_json
        def status(**kw):
            write_json(OUT/'status.json', {'pid':os.getpid(), **kw})
        caffeine = subprocess.Popen(['/usr/bin/caffeinate','-i','-s','-w',str(os.getpid())])
        try:
            for name in ['domain_ssl_200.py','mimu/domain_ssl.py','research_sweep.py','upsampling_sweep.py',
                         'train.py','evaluate_d4.py','mimu/data.py','mimu/loss.py','mimu/metrics.py',
                         'mimu/upsampling_variants.py','mimu/resnet34_unet.py','mimu/research_variants.py','mimu/model.py']:
                dest = OUT/'source_snapshot'/name
                dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():shutil.copy2(ROOT/name,dest)
            with (ROOT/'runs/dysample_200_v1/queue.lock').open('a') as previous:
                status(state='waiting_for_dysample200')
                fcntl.flock(previous,fcntl.LOCK_EX)
                for stage in ['ssl','finetune']:
                    status(state='running',stage=stage,ssl_epochs=SSL_EPOCHS,finetune_epochs=200)
                    subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',stage],cwd=ROOT,check=True)
                status(state='completed')
        except BaseException as exc:
            status(state='failed',error=repr(exc))
            raise
        finally:
            caffeine.terminate()


if __name__ == '__main__':
    main()
