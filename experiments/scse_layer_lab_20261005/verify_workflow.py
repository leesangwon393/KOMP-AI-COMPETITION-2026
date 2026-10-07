#!/usr/bin/env python3
"""Exercise staged selection/resume using fixture workers, and real RNG resume."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import torch
import numpy as np
from PIL import Image
from types import SimpleNamespace

import run_suite
from komap_phase.config import ROOT
from komap_phase.data import PALETTE
from komap_phase.runtime import write_json
from komap_phase import training
from plan import config_for, unit
from summarize import summarize


def workflow(data, temp):
    args=SimpleNamespace(output=temp/'queue',data=data,device='cpu',threads=4,profile='full',
                         resume=True,stop_after='all',skip_spatial=False)
    args.output.mkdir()
    for name in ('logs','resolved_configs'):(args.output/name).mkdir()
    suite=run_suite.Suite(args)
    workers=[]
    class FixtureWorker:
        def __init__(self,command,**_kwargs):
            self.pid=999999;self.code=0
            c=json.loads(Path(command[command.index('--config')+1]).read_text())
            output=Path(command[command.index('--output')+1]);output.mkdir(parents=True,exist_ok=True)
            workers.append((c['train']['seed'],c['id']))
            score=.77 if c['id']=='B' else .78+sum(u['stage'] for u in c['model']['units'])*.001
            if c['id'].startswith('CS_'):score+=.01
            metrics=dict(miou=score,filenames=[r['image'] for r in suite.identity['valid']],
                         per_image_miou=[score]*20,class_iou=[score]*4,
                         eutectic=dict(precision=.8,recall=.8,thin_skeleton_recall=.7),
                         contact=dict(error_rate=.2),seconds_per_image=.1)
            for name,value in [('config.json',c),('data_manifest.json',suite.identity),('environment.json',suite.env),
                               ('history.json',[dict(epoch=i,valid_single_miou=score) for i in range(1,151)]),
                               ('summary.json',dict(id=c['id'],best_epoch=150,single=metrics,d4=metrics,
                                     parameters_total=1,peak_cuda_memory_bytes=None)),
                               ('status.json',dict(state='completed',epochs=150))]:write_json(output/name,value)
            # Fixture tokens satisfy existence checks; these are never loaded as models.
            for name in ('best.pt','last.pt'):(output/name).write_bytes(b'FIXTURE_NOT_A_CHECKPOINT')
        def wait(self):return self.code
        def poll(self):return self.code
    original=run_suite.subprocess.Popen
    run_suite.subprocess.Popen=FixtureWorker
    try:
        suite.execute()
        assert len(workers)==32 and len(set(workers))==32
        count=len(workers)
        suite.rows=[];suite.execute()
        assert len(workers)==count,'Resume reran a completed experiment'
        report=summarize(args.output)
        assert len(report)==64
        selected=json.loads((args.output/'position_selection.json').read_text())
        assert selected['top2']==['C4','C3']
        c=config_for('C3',42,[unit(3)])
        history=args.output/'seed42/C3/history.json'
        saved=history.read_bytes();write_json(history,[dict(epoch=1)])
        try:run_suite.completed(history.parent,c,suite.identity,suite.env)
        except ValueError:pass
        else:raise AssertionError('Incomplete completed history was accepted')
        history.write_bytes(saved)
        altered=copy.deepcopy(c);altered['model']['units'][0]['steps']=1
        try:run_suite.completed(history.parent,altered,suite.identity,suite.env)
        except ValueError:pass
        else:raise AssertionError('Different completed config was accepted')
        run_suite.guard_equal(args.output/'immutable_test.json',dict(a=1))
        try:run_suite.guard_equal(args.output/'immutable_test.json',dict(a=2))
        except ValueError:pass
        else:raise AssertionError('A persisted selection was silently overwritten')
    finally:run_suite.subprocess.Popen=original
    return dict(fixture_runs=32,completed_resume_skipped=True,selection_and_config_guards=True,
                note='Orchestration fixtures only; no quality scores')


def real_resume(temp):
    data=temp/'synthetic_data'
    generator=np.random.default_rng(123)
    for split in ('train','valid'):
        for kind in ('images','masks'):(data/split/kind).mkdir(parents=True,exist_ok=True)
        for index in range(2):
            pixels=generator.integers(0,256,size=(64,64),dtype=np.uint8)
            labels=np.zeros((64,64),dtype=np.uint8)
            labels[:32,32:]=1;labels[32:,:32]=3;labels[32:,32:]=2
            Image.fromarray(pixels).save(data/split/'images'/f'{index}_image.png')
            Image.fromarray(PALETTE[labels]).save(data/split/'masks'/f'{index}_mask.png')
    c=config_for('SMOKE_RESUME',42,[unit(3,mode='channel_cbam_gate')],profile='smoke')
    device=torch.device('cpu');torch.set_num_threads(4)
    training.train(c,data,temp/'continuous',device)
    original=training.evaluate
    calls=0
    def interrupted(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==2:raise KeyboardInterrupt('Synthetic interruption before epoch2 checkpoint')
        return original(*args,**kwargs)
    training.evaluate=interrupted
    try:
        try:training.train(c,data,temp/'resumed',device)
        except KeyboardInterrupt:pass
        else:raise AssertionError('Interruption was not exercised')
    finally:training.evaluate=original
    assert torch.load(temp/'resumed/last.pt',weights_only=True)['epoch']==1
    training.train(c,data,temp/'resumed',device,resume=True)
    a=torch.load(temp/'continuous/last.pt',weights_only=True)
    b=torch.load(temp/'resumed/last.pt',weights_only=True)
    assert a['epoch']==b['epoch']==2
    assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items()),'Resumed model differs'
    for x,y in zip(a['history'],b['history']):
        for key in ('train_total_loss','valid_single_miou','lr'):assert x[key]==y[key]
    return dict(real_cpu_epochs=5,backbone='resnet34',synthetic_size=64,
                interruption_before_epoch2_checkpoint=True,final_model_bitwise_equal=True,
                history_loss_scores_equal=True,note='Synthetic CPU training only; not alloy model performance')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'Data')
    parser.add_argument('--report',type=Path,default=ROOT/'checks/workflow.json')
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='komap_layer_verify_') as directory:
        root=Path(directory)
        result=dict(passed=True,workflow=workflow(args.data,root),resume=real_resume(root))
    args.report.parent.mkdir(parents=True,exist_ok=True);write_json(args.report,result)
    print(json.dumps(result))
