#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare the locked upstream Silero asset without installing Whisper/CT2."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    entry=json.loads((root/"03_models/speech.json").read_text())['vad']
    folder=Path(entry['directory']).expanduser()
    missing=[name for name,value in entry['files'].items()
        if not (folder/name).is_file() or hashlib.sha256((folder/name).read_bytes()).hexdigest()!=value['sha256']]
    if missing:
        if args.check_only:raise RuntimeError('Missing/mismatched VAD artifacts: '+', '.join(missing))
        with urllib.request.urlopen(entry['source']['wheel_url'],timeout=60) as response:data=response.read()
        if hashlib.sha256(data).hexdigest()!=entry['source']['wheel_sha256']:raise ValueError('Wheel hash mismatch')
        folder.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for name,value in entry['files'].items():
                content=archive.read(value['wheel_member'])
                if hashlib.sha256(content).hexdigest()!=value['sha256']:raise ValueError('VAD artifact hash mismatch')
                destination=folder/name;temporary=destination.with_suffix('.partial')
                temporary.write_bytes(content);temporary.replace(destination)
    print(json.dumps({'verified':True,'directory':str(folder),'provider':entry['provider']}))


if __name__=='__main__':main()
