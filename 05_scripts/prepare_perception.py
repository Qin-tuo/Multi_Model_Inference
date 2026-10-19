#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare locked hand/source dependencies; verify retained native body artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(path, sha, url, check_only):
    path=Path(path)
    if path.is_file() and digest(path)==sha:return
    if check_only or not url:raise RuntimeError('Missing/mismatched retained artifact: '+str(path))
    with urllib.request.urlopen(url,timeout=90) as response:data=response.read()
    if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Download hash mismatch: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial');tmp.write_bytes(data);tmp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/"03_models/perception.json").read_text())
    body=manifest['body']
    for kind in ['model','engine']:prepare(body[kind+'_path'],body[kind+'_sha256'],None,args.check_only)
    helper=body['runtime_source']['files']['samples/python/common_runtime.py']
    prepare(body['runtime_helper'],body['runtime_helper_sha256'],helper['url'],args.check_only)
    for role in ['palm','hand']:
        entry=manifest[role]
        for name,info in entry['files'].items():
            if 'url' in info:prepare(info['path'],info['sha256'],info['url'],args.check_only)
        if entry.get('runtime_source_file'):
            target=entry['files'][entry['runtime_source_file']]
            path=Path(target['path'])
            if not path.is_file() or digest(path)!=target['sha256']:
                if args.check_only:raise RuntimeError('Patched palm source is not prepared')
                # Apply the single audited coordinate-format correction.
                original=Path(entry['files'][target['derived_from']]['path']).read_text()
                before='cv.dnn.NMSBoxes(boxes, score, self.score_threshold, self.nms_threshold, top_k=self.topK)'
                after='cv.dnn.NMSBoxes(np.c_[boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], score, self.score_threshold, self.nms_threshold, top_k=self.topK)'
                if original.count(before)!=1:raise ValueError('Palm patch context differs')
                data=original.replace(before,after).encode()
                if hashlib.sha256(data).hexdigest()!=target['sha256']:raise ValueError('Patched palm hash differs')
                path.write_bytes(data)
    print(json.dumps({'verified':True,'body':'retained TensorRT engine','hands':'pinned OpenCV source/models'}))


if __name__=='__main__':main()
