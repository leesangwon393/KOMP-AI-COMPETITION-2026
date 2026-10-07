#!/usr/bin/env python3
"""Check every distributed file against the release manifest."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    files = json.loads((root/'RELEASE_MANIFEST.json').read_text())['files']
    for name, expected in files.items():
        actual = hashlib.sha256((root/name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Release file changed: ' + name)
    print(json.dumps(dict(release_verified=True, files=len(files))))


if __name__ == '__main__':
    main()
