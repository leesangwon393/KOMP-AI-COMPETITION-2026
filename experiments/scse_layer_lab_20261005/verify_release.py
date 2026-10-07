#!/usr/bin/env python3
"""Verify the immutable code and bundled source/data by SHA256 before training."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def verify():
    release=json.loads((ROOT/'release_manifest.json').read_text())
    for item in release['files']:
        path=ROOT/item['path']
        if path.stat().st_size!=item['size'] or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Release identity differs: '+item['path'])
    print(json.dumps(dict(passed=True,verified_files=len(release['files']),
                         data_counts=release['data_counts'])))


if __name__=='__main__':verify()
