"""Snapshot tested source and launch a persistent one-GPU follow-up supervisor."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from komap_phase.config import ROOT
from komap_phase.data import manifest
from komap_phase.runtime import code_hash, device_for, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--prerequisite', type=Path, required=True)
    p.add_argument('--preflight', type=Path, required=True)
    p.add_argument('--device', choices=('mps','cuda'), default='mps')
    a = p.parse_args()
    device_for(a.device)
    report = json.loads(a.preflight.read_text())
    if not report.get('passed') or report['code_sha256'] != code_hash() or report['data_manifest'] != manifest(a.data):
        raise ValueError('Require current passing preflight and unchanged data')
    output = a.output.resolve()
    if output.exists(): raise ValueError('Output exists; preserve previous launch')
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'; snapshot.mkdir()
    for folder in ('komap_phase','configs'):
        shutil.copytree(ROOT/folder,snapshot/folder,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('run.py','run_suite.py','make_configs.py','summarize_seeds.py','experiment_queue.py',
                 'README_KO.md','requirements.txt','requirements-cuda.txt','NOTICE.md'):
        shutil.copy2(ROOT/name,snapshot/name)
    env = os.environ.copy(); env['TORCH_HOME'] = str(a.cache.resolve())
    command = [sys.executable,'-u',str(snapshot/'experiment_queue.py'),'--output',str(output/'trials'),
               '--data',str(a.data.resolve()),'--prerequisite',str(a.prerequisite.resolve()),'--device',a.device]
    with (output/'supervisor.log').open('a') as log:
        process = subprocess.Popen(command,cwd=snapshot,env=env,stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record = {'supervisor_pid':process.pid,'started_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              'command':command,'source_snapshot':str(snapshot),'preflight':str(a.preflight.resolve()),
              'new_runs':13,'epochs_each':150,'device':a.device}
    write_json(output/'launch.json',record)
    if sys.platform=='darwin' and shutil.which('caffeinate'):
        awake = subprocess.Popen(['caffeinate','-i','-w',str(process.pid)],start_new_session=True,
                                 stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        (output/'caffeinate.pid').write_text(str(awake.pid))
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
