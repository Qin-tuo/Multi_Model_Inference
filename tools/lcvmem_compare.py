"""Paired development diagnostics; never replace the pinned official score."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.lcvmem_data import load
from tools.lcvmem_prepare import CODE,sha
from app.context_memory import presentation


def main(args):
    if args.output.exists():raise RuntimeError('Use a new comparison report')
    old=json.loads(args.baseline.read_text());new=json.loads(args.candidate.read_text())
    for report in (old,new):
        if report['status']!='COMPLETE' or report['split']!='development':raise RuntimeError('Only complete development runs can be diagnosed')
        for name,digest in report['metric_files'].items():
            if sha(CODE/name)!=digest:raise RuntimeError('Pinned metric changed')
    if any(old[key]!=new[key] for key in ('dataset_sha256','selected','metric_files','source_metadata')):
        raise RuntimeError('Comparison data/metric configuration differs')
    a={r['id']:r for r in old['cases']};b={r['id']:r for r in new['cases']}
    if set(a)!=set(b):raise RuntimeError('Question partitions differ')
    sys.path.insert(0,str(CODE));from metrics import normalize_zh_answer
    references={r['id']:r['answer'] for r in load() if r['id'] in a};rows=[]
    def evidence(row,visible):
        admitted=set(row.get('retrieval',{}).get('injected_memory_ids',[]));result=[]
        for item in row.get('retrieved',[]):
            if visible and item['memory_id'] not in admitted:continue
            text=presentation(item)[0] if visible else item['content']
            result.append([item['sources'],hashlib.sha256(text.encode()).hexdigest()])
        return result
    for qid in a:
        before=json.loads((Path(old['raw_results_directory'])/(qid+'.json')).read_text())
        after=json.loads((Path(new['raw_results_directory'])/(qid+'.json')).read_text())
        reference=normalize_zh_answer(references[qid])
        pair={}
        for label,data in [('baseline',before),('candidate',after)]:
            text=data.get('response','') if not data.get('error') else ''
            full=bool(reference) and reference in normalize_zh_answer(text)
            first=bool(reference) and reference in normalize_zh_answer(text.split('\n',1)[0])
            pair[label]={'literal_reference_in_full_response':full,'literal_reference_in_first_line':first}
        rows.append({'id':qid,'baseline_f1':a[qid]['f1'],'candidate_f1':b[qid]['f1'],
            'same_retrieved_evidence_order':evidence(before,False)==evidence(after,False),
            'same_visible_evidence_order':evidence(before,True)==evidence(after,True),
            'same_prompt_token_counts':before.get('prompt_tokens')==after.get('prompt_tokens'),
            'same_response':before.get('response')==after.get('response'),
            'baseline_visible_reference':a[qid]['reference_verbatim_in_injected'],
            'candidate_visible_reference':b[qid]['reference_verbatim_in_injected'],**pair})
    summary={'questions':len(rows),'baseline_f1_percent':old['upstream_qa_f1_percent'],'candidate_f1_percent':new['upstream_qa_f1_percent'],
        **{key:sum(row[key] for row in rows) for key in ('same_retrieved_evidence_order','same_visible_evidence_order','same_prompt_token_counts','same_response')},
        'lost_visible_reference':sum(row['baseline_visible_reference'] and not row['candidate_visible_reference'] for row in rows),
        'gained_visible_reference':sum(not row['baseline_visible_reference'] and row['candidate_visible_reference'] for row in rows)}
    for label in ('baseline','candidate'):
        summary[label+'_full_response_literal_hits']=sum(row[label]['literal_reference_in_full_response'] for row in rows)
        summary[label+'_first_line_literal_hits']=sum(row[label]['literal_reference_in_first_line'] for row in rows)
    report={'created_at':datetime.now(timezone.utc).isoformat(),'status':'DEVELOPMENT_DIAGNOSTIC','summary':summary,'cases':rows,
        'baseline_report':str(args.baseline),'candidate_report':str(args.candidate),
        'reader_fingerprints':[old['reader_fingerprint'],new['reader_fingerprint']],
        'scope':'All selected development questions retained. Official lexical F1 unchanged;full-response literal presence diagnoses first-line formatting effects,not semantic correctness. Matching evidence/token counts do not prove identical full prompts. Comparison is offline;reference answers are never passed to runtime. No heldout or JT accuracy claim.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
