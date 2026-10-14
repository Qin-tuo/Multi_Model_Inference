"""Seal Chinese visual QA partitions before any model inference.

Selection is based on task categories, image identity and a fixed hash order.
References remain in separate files, never in runtime inputs. Only aggregate
metadata is printed, including for held-out/reserve questions.
"""
import argparse
import base64
from collections import Counter,defaultdict
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.project import sha256,stamp,write_json

ALLOWED=('physical_property_reasoning','function_reasoning','structuralized_imagetext_understanding',
    'attribute_recognition','object_localization','spatial_relationship','image_scene','image_quality',
    'nature_relation','physical_relation','action_recognition','identity_reasoning','attribute_comparison',
    'image_topic','image_style','ocr')
EXCLUDED_TERMS=('种族','民族','宗教','国籍','性取向','性别','年龄','几岁','男人','女人','男性','女性',
    '男孩','女孩','男子','女子','男士','女士','肤色','残疾','疾病','病症','医学','癌症','政治','党派',
    '姓名','叫什么名字','哪位名人','谁是','他是谁','她是谁','职业','收入','穷人','富人','诊断')


def rank(value):return hashlib.sha256(('jarvis-vqa-20261014-v1:'+str(value)).encode()).hexdigest()


def prepare(args):
    import pandas as pd
    from PIL import Image,ImageOps
    if args.output.exists() or (args.root/'partitions').exists():raise RuntimeError('Preserve existing frozen corpus')
    path=args.root/'MMBench_DEV_CN_V11.tsv'
    if hashlib.md5(path.read_bytes(),usedforsecurity=False).hexdigest()!='593f9b5f6bea453d870a798b34ae4f37':
        raise RuntimeError('Dataset differs from pinned upstream MD5')
    frame=pd.read_csv(path,sep='\t',dtype={'image':str},keep_default_na=False,na_values=[''])
    rows={int(row['index']):row for _,row in frame.iterrows()}
    groups=defaultdict(list)
    for index,row in rows.items():groups[index%1_000_000].append(index)
    eligible=[];excluded=Counter();image_cache={}
    for index in sorted(k for k in rows if k<1_000_000):
        row=rows[index];category=row['category']
        if category not in ALLOWED:excluded['outside_project_categories']+=1;continue
        text='\n'.join(str(row[k]) for k in ('question','hint','A','B','C','D') if k in row and not pd.isna(row[k]))
        if any(term in text for term in EXCLUDED_TERMS):excluded['outside_anonymous_visual_scope']+=1;continue
        choices={k:str(row[k]) for k in 'ABCD' if k in row and not pd.isna(row[k])}
        variants=sorted(groups[index])
        if not 2<=len(variants)<=len(choices)<=4:raise RuntimeError('Invalid supplied circular question group')
        if variants!=[index+i*1_000_000 for i in range(len(variants))]:raise RuntimeError('Non-contiguous supplied rotations')
        expected=str(row[str(row['answer'])])
        for variant in variants:
            item=rows[variant]
            if str(item[str(item['answer'])])!=expected:raise RuntimeError('Circular answers inconsistent')
            other=[str(item[k]) for k in 'ABCD' if k in item and not pd.isna(item[k])]
            if sorted(other)!=sorted(choices.values()):raise RuntimeError('Circular option sets inconsistent')
        encoded=str(row['image']);seen=set()
        while len(encoded)<=64:
            if encoded in seen:raise RuntimeError('Image reference cycle')
            seen.add(encoded);encoded=str(rows[int(encoded)]['image'])
        raw=base64.b64decode(encoded,validate=True)
        with Image.open(io.BytesIO(raw)) as opened:
            image=ImageOps.exif_transpose(opened).convert('RGB')
            digest=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
            size=list(image.size)
        image_cache[index]=raw
        eligible.append({'base_id':index,'category':category,'image_group':digest,'original_size':size,'variant_ids':variants})
    buckets={category:sorted([r for r in eligible if r['category']==category],key=lambda r:rank(r['base_id'])) for category in ALLOWED}
    used=set();partitions={}
    for split in ('development','heldout','reserve'):
        selected=[]
        while len(selected)<100:
            progressed=False
            for category in ALLOWED:
                candidates=[r for r in buckets[category] if r['image_group'] not in used]
                if candidates:
                    item=candidates[0];selected.append(item);used.add(item['image_group']);progressed=True
                if len(selected)==100:break
            if not progressed:raise RuntimeError('Insufficient distinct image groups for three100-question splits')
        partitions[split]=selected
    directory=args.root/'partitions';directory.mkdir()
    images=args.root/'images';images.mkdir(exist_ok=True)
    outputs={}
    for split,items in partitions.items():
        inputs=[];gold=[]
        for item in items:
            image_path=images/(item['image_group']+'.png')
            with Image.open(io.BytesIO(image_cache[item['base_id']])) as source:
                image=ImageOps.pad(ImageOps.exif_transpose(source).convert('RGB'),(640,480),
                    method=Image.Resampling.BICUBIC,color=(114,114,114),centering=(.5,.5))
                image.save(image_path,format='PNG')
            for index in item['variant_ids']:
                row=rows[index];choices={k:str(row[k]) for k in 'ABCD' if k in row and not pd.isna(row[k])}
                inputs.append({'id':str(index),'base_id':str(item['base_id']),'category':item['category'],
                    'image_group':item['image_group'],'image_path':str(image_path.resolve()),'image_sha256':sha256(image_path),
                    'original_size':item['original_size'],'question':str(row['question']),
                    'hint':None if pd.isna(row.get('hint')) else str(row['hint']),'choices':choices})
                gold.append({'id':str(index),'answer':str(row['answer'])})
        public=directory/(split+'-inputs.jsonl');reference=directory/(split+'-references.jsonl')
        public.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in inputs))
        reference.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in gold))
        outputs[split]={'base_questions':100,'variants':len(inputs),'ids':[r['id'] for r in inputs],
            'base_ids':[str(r['base_id']) for r in items],'image_groups':[r['image_group'] for r in items],
            'categories':dict(Counter(r['category'] for r in items)),
            'inputs':str(public.resolve()),'inputs_sha256':sha256(public),
            'references':str(reference.resolve()),'references_sha256':sha256(reference)}
    report={'created_at':stamp(),'status':'SEALED','source_sha256':sha256(Path(__file__)),
        'dataset':'MMBench_DEV_CN_V11','upstream_rows':len(frame),'upstream_questions':len(groups),
        'source_tsv':str(path.resolve()),'source_sha256_tsv':sha256(path),'upstream_md5':'593f9b5f6bea453d870a798b34ae4f37',
        'download_url':'https://opencompass.oss-cn-shanghai.aliyuncs.com/utils/benchmarks/MMBench/MMBench_DEV_CN_V11.tsv',
        'eligible_questions':len(eligible),'excluded_counts':dict(excluded),'allowed_categories':list(ALLOWED),
        'excluded_scope_terms':list(EXCLUDED_TERMS),'partitions':outputs,
        'selection':'FixedSHA256order,category round-robin;onecanonicalquestion perdecodedRGBimage hash;alloption rotations retained. No model outputs used.',
        'rotations':'Use all rotations supplied by the locked upstream TSV; some four-option questions supply onlythree rotations. Do not invent missing variants or labels.',
        'partition_scope':'Projectdev/heldout/reserve drawnfromupstreampublicdev. No claim thatpretrainedmodelsnever sawthisbenchmark;exactpixelduplicatesexcluded,near-duplicatevisualsources notcertified.',
        'image_processing':'PillowEXIFtranspose/RGB,aspect-preservingpad640x480,BICUBIC,RGB114padding;application later encodesJPEG75 asusual.',
        'scope':'ChineseMCQpubliccapabilityscreening;notnaturalfreeform,imagecapturequality,physicalvoiceorwholeJT07acceptance.'}
    write_json(args.output,report)
    print(json.dumps({'status':'SEALED','eligible':len(eligible),'excluded':dict(excluded),
        'partitions':{s:{k:r[k] for k in ('base_questions','variants','categories')} for s,r in outputs.items()}},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args())
