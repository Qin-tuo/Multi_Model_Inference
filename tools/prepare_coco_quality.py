"""Acquire and seal the full public COCO 2017 validation regression corpus.

This public benchmark does not replace independent local camera recordings.
S3 path-style HTTPS addresses the official bucket without bypassing TLS.
Existing sequential .part prefixes are preserved; ranged suffixes use four
bounded workers with If-Match and explicit Content-Range verification.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.project import sha256,stamp,write_json

SOURCES=[('val2017.zip','zips',815585330,'"d366be60d3dc737327160d62453e3973-98"'),
         ('annotations_trainval2017.zip','annotations',252907541,'"f4bbac642086de4f52a3fdda2de5fa2c"')]


def download(root,name,folder,size,etag):
    url=f'https://s3.amazonaws.com/images.cocodataset.org/{folder}/{name}'
    target=root/name
    if target.exists():
        if target.stat().st_size!=size:
            raise RuntimeError('Existing archive has wrong size')
        return target
    prefix=root/(name+'.part')
    start=prefix.stat().st_size if prefix.exists() else 0
    if start>size:
        raise RuntimeError('Existing prefix exceeds archive size')
    chunks=root/(name+'.chunks')
    chunks.mkdir(exist_ok=True)
    admission=chunks/'ranges.json'
    descriptor={'url':url,'size':size,'etag':etag,'prefix_bytes':start,'chunk_bytes':8*1024**2}
    if admission.exists() and json.loads(admission.read_text())!=descriptor:
        raise RuntimeError('Resume descriptor changed')
    write_json(admission,descriptor)
    ranges=[(i,min(i+8*1024**2,size)-1) for i in range(start,size,8*1024**2)]

    def fetch(pair):
        first,last=pair
        path=chunks/f'{first}-{last}'
        if path.exists() and path.stat().st_size==last-first+1:
            return path
        for attempt in range(4):
            try:
                request=urllib.request.Request(url,headers={'Range':f'bytes={first}-{last}','If-Match':etag})
                with urllib.request.urlopen(request,timeout=60) as response:
                    if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {first}-{last}/{size}':
                        raise RuntimeError('Server did not honor exact range')
                    if response.headers.get('ETag')!=etag:
                        raise RuntimeError('Source object changed')
                    with path.with_suffix('.tmp').open('wb') as out:
                        shutil.copyfileobj(response,out,1024**2)
                if path.with_suffix('.tmp').stat().st_size!=last-first+1:
                    raise RuntimeError('Incomplete range')
                path.with_suffix('.tmp').replace(path)
                return path
            except Exception:
                if attempt==3:
                    raise
                time.sleep(1+attempt)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(fetch,pair):pair for pair in ranges}
        for count,future in enumerate(as_completed(futures),1):
            future.result()
            print(json.dumps({'archive':name,'ranges_complete':count,'total_ranges':len(ranges)}),flush=True)
    merged=root/(name+'.assembling')
    with merged.open('wb') as out:
        if prefix.exists():
            with prefix.open('rb') as stream:
                shutil.copyfileobj(stream,out,1024**2)
        for first,last in ranges:
            with (chunks/f'{first}-{last}').open('rb') as stream:
                shutil.copyfileobj(stream,out,1024**2)
    if merged.stat().st_size!=size:
        raise RuntimeError('Assembled size differs')
    if '-' not in etag:
        digest=hashlib.md5(usedforsecurity=False)
        with merged.open('rb') as stream:
            while block:=stream.read(1024**2):digest.update(block)
        if digest.hexdigest()!=etag.strip('"'):
            raise RuntimeError('Archive differs from official S3 object ETag')
    with zipfile.ZipFile(merged) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('ZIP CRC failed')
    merged.replace(target)
    return target


def main(args):
    if args.output.exists():raise RuntimeError('Preserve existing sealed manifest')
    downloads=args.root/'downloads';downloads.mkdir(parents=True,exist_ok=True)
    archives=[]
    for name,folder,size,etag in SOURCES:
        path=download(downloads,name,folder,size,etag)
        archives.append({'name':name,'url':f'https://s3.amazonaws.com/images.cocodataset.org/{folder}/{name}',
            'bytes':size,'etag':etag,'sha256':sha256(path),'sha256_source':'observed after verified TLS/ranges and ZIP CRC'})
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.filename.startswith('val2017/') or member.filename in (
                    'annotations/instances_val2017.json','annotations/person_keypoints_val2017.json'):
                    p=Path(member.filename)
                    if p.is_absolute() or '..' in p.parts:raise RuntimeError('Unsafe archive path')
                    archive.extract(member,args.root)
    annotations={}
    for task,name in [('detection','instances_val2017.json'),('pose','person_keypoints_val2017.json')]:
        path=args.root/'annotations'/name;data=json.loads(path.read_text())
        annotations[task]={'path':str(path.resolve()),'sha256':sha256(path),'images':len(data['images']),
            'annotations':len(data['annotations']),'categories':len(data['categories'])}
        if task=='detection':images=sorted(data['images'],key=lambda x:x['id']);licenses=data['licenses']
    if len(images)!=5000 or len({r['id'] for r in images})!=5000:
        raise RuntimeError('Expected all 5000 official validation images')
    index=args.root/'image-hashes.jsonl'
    inputs=args.root/'inputs.tsv'
    with index.open('w') as hashes,inputs.open('w') as ordered:
        for image in images:
            path=args.root/'val2017'/image['file_name']
            row={'id':image['id'],'file_name':image['file_name'],'width':image['width'],'height':image['height'],
                'license':image['license'],'source_url':image['coco_url'],'sha256':sha256(path)}
            hashes.write(json.dumps(row,ensure_ascii=False)+'\n')
            ordered.write(str(image['id'])+'\t'+str(path.resolve())+'\n')
    report={'schema_version':1,'created_at':stamp(),'status':'SEALED','source_sha256':sha256(Path(__file__)),
        'dataset':'COCO 2017 validation','root':str(args.root.resolve()),'archives':archives,'annotations':annotations,
        'images':5000,'image_index':str(index.resolve()),'image_index_sha256':sha256(index),
        'ordered_inputs':str(inputs.resolve()),'ordered_inputs_sha256':sha256(inputs),
        'image_license_records':licenses,'official_dataset':'https://cocodataset.org/#download',
        'scope':'Full public validation regression corpus; not independent local recording, unseen pretraining data or final JT-07 proof.'}
    write_json(args.output,report)
    print(json.dumps({'status':'SEALED','images':5000,'manifest':str(args.output)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
