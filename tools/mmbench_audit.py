"""Audit image/token/state evidence without opening any reference answers."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.project import sha256,stamp,write_json
from tools.mmbench_eval import inputs


def main(args):
    if args.output.exists():raise RuntimeError('Use a new audit report')
    run=json.loads(args.run_report.read_text())
    corpus,part,rows=inputs(run['split'])
    directory=Path(run['artifacts'])/'answers'
    files={p.stem for p in directory.glob('*.json')}
    checks={'full_run_answered':run['status']=='ANSWERED' and run['completed']==part['variants'],
        'exact_answer_membership':files==set(part['ids']),
        'runtime_inputs_still_match':part['inputs_sha256']==run['input_sha256'],
        'input_groups_disjoint':all(not set(corpus['partitions'][a]['image_groups'])&set(corpus['partitions'][b]['image_groups'])
            for a,b in (('development','heldout'),('development','reserve'),('heldout','reserve')))}
    image_missing=[];usage_mismatch=[];bad_binding=[];mutations=[];leaks=[];execution_errors=[]
    references=[]
    for row in rows:
        path=directory/(row['id']+'.json')
        if not path.exists():continue
        answer=json.loads(path.read_text())
        references.append({'id':row['id'],'sha256':sha256(path)})
        if answer['id']!=row['id'] or answer['image_sha256']!=row['image_sha256']:bad_binding.append(row['id'])
        if answer['unexpected_mutation']:mutations.append(row['id'])
        if answer['image_leak']:leaks.append(row['id'])
        if answer['error']:
            execution_errors.append({'id':row['id'],'type':answer['error']['type']})
            continue
        image=answer.get('image') or {}
        if image.get('size')!=[640,480] or image.get('source_kind')!='synthetic_fixture':image_missing.append(row['id'])
        usage=answer.get('model_usage') or []
        if not usage or answer['prompt_tokens']!=[r['prompt_tokens'] for r in usage]:usage_mismatch.append(row['id'])
    checks.update(no_image_binding_errors=not bad_binding,successful_turns_have_images=not image_missing,
        native_input_counts_match_usage=not usage_mismatch,no_unexpected_mutations=not mutations,no_image_leaks=not leaks,
        evaluation_backends_ready=run.get('backends_after',{}).get('evaluation_ready',
            run.get('backends_after',{}).get('backend_status')=='READY'))
    if run.get('backends_after'):
        checks['managed_processes_unchanged']=all(v['main_pid']==run['backends_after']['components'][k]['main_pid']
            for k,v in run['backends_before']['components'].items())
        if run['backends_before'].get('candidate'):
            checks['candidate_process_unchanged']=(run['backends_before']['candidate']==run['backends_after'].get('candidate')
                and run['backends_after']['candidate']['owns_vlm_port'])
    report={'created_at':stamp(),'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
        'source_sha256':sha256(Path(__file__)),'inference_report':str(args.run_report),'inference_report_sha256':sha256(args.run_report),
        'image_missing':image_missing,'usage_mismatch':usage_mismatch,'bad_binding':bad_binding,
        'unexpected_mutations':mutations,'image_leaks':leaks,'execution_errors':execution_errors,'answers':references,
        'scope':'Complete-input/image/native-token/state audit; execution errors remain wrong in score denominator. No reference labels opened and no accuracy claim.'}
    write_json(args.output,report);print(json.dumps({'status':report['status'],'checks':checks,'execution_errors':len(execution_errors)}),flush=True)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(main(parser.parse_args()))
