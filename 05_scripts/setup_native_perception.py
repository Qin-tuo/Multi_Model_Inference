#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Expose only the system TensorRT binding to this application environment."""
import json
from pathlib import Path
import sys
import sysconfig


def main():
    if sys.prefix==sys.base_prefix: raise RuntimeError('Run with the isolated application Python')
    root=Path(__file__).resolve().parents[1]
    entry=json.loads((root/"03_models/perception.json").read_text())['body']
    source=Path(f'/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages/tensorrt')
    if not (source/'__init__.py').is_file(): raise RuntimeError('Matching host TensorRT Python package is missing')
    target=Path(sysconfig.get_paths()['purelib'])/'tensorrt'
    if target.exists():
        if target.resolve()!=source.resolve():raise RuntimeError('Existing TensorRT package has another owner')
    else:target.symlink_to(source,target_is_directory=True)
    import tensorrt
    if tensorrt.__version__!=entry['tensorrt_version']:raise RuntimeError('Host TensorRT version differs from model lock')
    print(json.dumps({'version':tensorrt.__version__,'source':str(source),'target':str(target)}))


if __name__=='__main__':main()
