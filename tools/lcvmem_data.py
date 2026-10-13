"""Deterministic source-only parsing and conversation-disjoint benchmark splits."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from tools.lcvmem_prepare import CODE,DATA,DATA_SHA,COMMIT,REVISION,URL,sha

SUFFIX='请记住以上全部对话记录，回答问题。\n问题：'
SEEN='91a0135e-07a0-49aa-bafa-a3e87174cb11'


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def load():
    if sha(DATA)!=DATA_SHA:raise RuntimeError('Dataset checksum differs')
    with DATA.open() as stream:data=[json.loads(line) for line in stream]
    if len(data)!=358 or len({row['id'] for row in data})!=358:raise RuntimeError('Dataset size/IDs differ')
    for row in data:
        if not all(isinstance(row.get(key),str) for key in ('id','context','query','answer')):raise RuntimeError('Unsupported benchmark schema')
    return data


def messages(context):
    """Keep each dated speaker turn, excluding only the benchmark task footer."""
    if not context.rstrip().endswith(SUFFIX):raise RuntimeError('Unexpected benchmark footer')
    body=context.rstrip()[:-len(SUFFIX)].strip()
    date=None;role=None;lines=[];result=[]
    def flush():
        if role and lines:
            text='\n'.join(lines).strip().rstrip('”').rstrip()
            content=f'{date}的历史对话，{role}说：{text}'
            if not date or not text or len(content.encode())>8192:raise RuntimeError('Source turn is missing a date or exceeds the existing memory bound')
            result.append({'date':date,'role':role,'text':text,'content':content})
    for line in body.splitlines():
        header=re.fullmatch(r'以下是(\d{4}年\d{2}月\d{2}日)的对话记录：',line.strip())
        speaker=re.match(r'^[“" ]*(用户|AI)[:：]\s*(.*)$',line)
        if header:flush();date=header[1];role=None;lines=[]
        elif speaker:flush();role=speaker[1];lines=[speaker[2]]
        elif line.strip() not in {'','”'}:
            if role:lines.append(line)
            else:raise RuntimeError('Unparsed source text before a speaker')
    flush()
    if not result:raise RuntimeError('Empty source conversation')
    return result


def frozen_split():
    data=load();groups=defaultdict(list);ownership={}
    for row in data:
        group=digest(row['context']);groups[group].append(row['id'])
        for item in messages(row['context']):
            if len(item['text'])<32:continue
            key=digest(item['date']+'\n'+item['role']+'\n'+item['text'])
            if key in ownership and ownership[key]!=group:raise RuntimeError('Conversation overlap requires regrouping before freezing the split')
            ownership[key]=group
    seen=next(digest(row['context']) for row in data if row['id']==SEEN)
    ordered=sorted(groups,key=lambda key:digest('lcvmem-v1/'+key))
    development=[seen];count=len(groups[seen]);remaining=[key for key in ordered if key!=seen]
    while count<80:
        key=remaining.pop(0);development.append(key);count+=len(groups[key])
    heldout=[];count=0
    while count<200:
        key=remaining.pop(0);heldout.append(key);count+=len(groups[key])
    partitions={'development':development,'heldout':heldout,'reserve':remaining}
    return {'schema_version':1,'dataset':'CLongEval/LCvMem/small','data_url':URL,'data_revision':REVISION,'data_sha256':DATA_SHA,
        'code_url':'https://github.com/zexuanqiu/CLongEval','code_commit':COMMIT,
        'metric_files':{name:sha(CODE/name) for name in ('metrics.py','post_processing_for_stacked_tasks.py')},
        'upstream_code_license':'MIT; retained in the external checkout',
        'dataset_license_note':'The authors publish the dataset through their linked Hugging Face repository; it has no separate dataset card/license field. Raw contexts/questions/references are kept outside the project repository.',
        'split_policy':'Whole identical conversations, with a cross-group exact dated-turn overlap check for turns>=32 characters; SHA256(lcvmem-v1/group) order. The initially inspected example is confined to development.',
        'inspected_example_ids':[SEEN],
        'protocol':{'source_unit':'One original dated speaker message, no question/answer-dependent selection or summarization',
            'indexing':'Benchmark-authorized history events via pinned Mem0 providers, not automatic extraction of ordinary user dialogue',
            'runtime':'Unmodified AssistantSession, BGE ONNX, managed authenticated Qdrant and llama.cpp; native top5 and512-token evidence budget',
            'isolation':'One owner per question, fresh records/reminders/session, seed process exits before query process',
            'scoring':'Pinned upstream qa_f1_zh_score plus normalized exact match; errors score zero and stay in denominator',
            'acceptance':'External Chinese language-memory diagnostic, not the complete JT-10 lifecycle/95% correctness gate'},
        'partitions':{name:{'groups':len(keys),'ids':sorted([qid for key in keys for qid in groups[key]]),'group_ids':sorted(keys)} for name,keys in partitions.items()}}
