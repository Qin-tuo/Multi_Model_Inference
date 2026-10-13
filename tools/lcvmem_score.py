"""Run the pinned upstream metric in the separate evaluation environment."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.lcvmem_data import load
from tools.lcvmem_prepare import CODE,sha
from app.context_memory import presentation


def reference_in_visible_body(item,normalized_reference,normalize):
    return bool(normalized_reference) and normalized_reference in normalize(presentation(item)[0])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((ROOT/'benchmarks/corpora/lcvmem-v1.json').read_text())
    for name,value in manifest['metric_files'].items():
        if sha(CODE/name)!=value:raise RuntimeError('Upstream scoring file has changed')
    sys.path.insert(0,str(CODE))
    from metrics import qa_f1_zh_score,normalize_zh_answer
    run=json.loads((args.run_dir/'lcvmem-run.json').read_text())
    reference={row['id']:row['answer'] for row in load()}
    cases=[]
    for qid in run['ids']:
        path=args.run_dir/'answers'/(qid+'.json')
        if path.exists():row=json.loads(path.read_text())
        else:row={'id':qid,'response':'','error':{'type':'MISSING_RESULT'},'retrieval':{},'retrieved':[],'calls':[],'side_effect':False,'seconds':None}
        prediction=row.get('response','') if not row.get('error') else ''
        expected=reference[qid];normalized=normalize_zh_answer(expected)
        injected=set(row.get('retrieval',{}).get('injected_memory_ids',[]))
        hits=[item for item in row.get('retrieved',[]) if normalized and normalized in normalize_zh_answer(item['content'])]
        item={'id':qid,'f1':qa_f1_zh_score(prediction,expected),'normalized_exact':bool(prediction) and normalize_zh_answer(prediction)==normalized,
            'reference_verbatim_in_retrieved':bool(hits),'reference_verbatim_in_injected':any(hit['memory_id'] in injected and reference_in_visible_body(hit,normalized,normalize_zh_answer) for hit in hits),
            'injected_count':len(injected),'retrieved_count':len(row.get('retrieved',[])),'response_sha256':hashlib.sha256(prediction.encode()).hexdigest(),
            'error_type':row.get('error',{}).get('type') if row.get('error') else None,'side_effect':row.get('side_effect',False),
            'cross_owner_evidence':row.get('cross_owner_evidence',False),'seconds':row.get('seconds'),
            'prompt_tokens':row.get('prompt_tokens',[]),'model_request_skipped':row.get('model_request_skipped',False)}
        cases.append(item)
    total=len(cases)
    report={'schema_version':1,'status':'COMPLETE' if all((args.run_dir/'answers'/(qid+'.json')).is_file() for qid in run['ids']) else 'INCOMPLETE',
        'split':run['split'],'source_metadata':run.get('source_metadata','body-only-v1'),
        'selected':total,'partition_total':len(manifest['partitions'][run['split']]['ids']),
        'dataset_sha256':manifest['data_sha256'],'code_commit':manifest['code_commit'],'metric_files':manifest['metric_files'],
        'source_commit':run.get('source_commit'),'reader_fingerprint':run.get('reader_fingerprint'),'run_id':run['run_id'],
        'source_state':'Working tree identified by reader_fingerprint; see checkpoint commit containing the report',
        'seed_method':'Pinned Mem0 provider batch fixture import; actual BGE, no question/reference-dependent extraction',
        'scoring_packages':{name:importlib.metadata.version(name) for name in ('jieba','rouge','six','jsonlines','attrs')},
        'upstream_qa_f1_percent':100*sum(item['f1'] for item in cases)/total,
        'normalized_exact_percent':100*sum(item['normalized_exact'] for item in cases)/total,
        'errors':sum(bool(item['error_type']) for item in cases),'unexpected_mutations':sum(item['side_effect'] for item in cases),
        'cross_owner_evidence':sum(item['cross_owner_evidence'] for item in cases),
        'zero_injected_cases':sum(item['injected_count']==0 for item in cases),
        'verbatim_reference_retrieved_cases':sum(item['reference_verbatim_in_retrieved'] for item in cases),
        'verbatim_reference_injected_cases':sum(item['reference_verbatim_in_injected'] for item in cases),'cases':cases,
        'scope':'External Chinese LCvMem source questions and reference scoring, adapted to current bounded retrieval plus real AssistantSession. F1/EM and literal-reference containment are not the JT-10 95% correctness/lifecycle gate or official full-context leaderboard reproduction. No scoring failures are dropped from the denominator.',
        'raw_results_directory':str(args.run_dir/'answers'),'jt10':'NOT_VERIFIED'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('status','selected','upstream_qa_f1_percent','normalized_exact_percent','errors','zero_injected_cases','unexpected_mutations')},ensure_ascii=False))
    return 0 if report['status']=='COMPLETE' else 1


if __name__=='__main__':raise SystemExit(main())
