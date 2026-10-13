"""Compare all fixed-arrival development turns without dropping slow replies."""
import argparse
import json
import math
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.project import stamp,write_json


def p95(values):return sorted(values)[math.ceil(.95*len(values))-1] if values else None
def normalize(text):return re.sub(r'[\s，。！？,.!?：:；;"“”]','',text)


def main(args):
    if args.output.exists():raise RuntimeError('Preserve the existing comparison')
    before=json.loads(args.baseline.read_text());after=json.loads(args.candidate.read_text())
    checks={'same_reader_code':before['reader_fingerprint']==after['reader_fingerprint'],
        'same_memory_recipe':before['config']==after['config'],'same_retrieval_limits':before['retrieval_config']==after['retrieval_config'],
        'same_observation_mode':(before.get('stage_trace') is not None,before.get('debug_loop',False),before.get('gc_observation') is not None)==
            (after.get('stage_trace') is not None,after.get('debug_loop',False),after.get('gc_observation') is not None),
        'same_fixed_interval':before['turn_interval_seconds']==after['turn_interval_seconds']==10,
        'exactly20_attempts':len(before['turns'])==len(after['turns'])==20,
        'same_voice_inputs':[(r['label'],r.get('input_sha256')) for r in before['turns']]==[(r['label'],r.get('input_sha256')) for r in after['turns']],
        'only_candidate_routes_arithmetic':before['skip_literal_arithmetic'] is False and after['skip_literal_arithmetic'] is True}
    snapshots=[json.loads((Path(r['run_directory'])/'clone-receipt.json').read_text()) for r in (before,after)]
    checks['same_pristine_snapshot']=snapshots[0]['backup_manifest_sha256']==snapshots[1]['backup_manifest_sha256']
    rows=[]
    for name,report in [('baseline',before),('selective',after)]:
        turns=report['turns'];math_rows=[r for r in turns if r['label']=='arithmetic']
        short=[r['scheduled_end_to_first_pcm_seconds']*1000 for r in turns if r['label']!='image' and 'scheduled_end_to_first_pcm_seconds' in r]
        row={'variant':name,'status':report['status'],'voice_latency':report['voice_latency'],'all_short_p95_ms':p95(short),
            'query_p95_ms':report['query_summary']['p95_ms'],'write_p95_ms':report['write_summary']['p95_ms'],
            'background_counts':{'queries':len(report['queries']),'writes':len(report['writes'])},
            'maximum_release_lateness_ms':max((r.get('release_lateness_ms',0) for r in turns),default=None),
            'arithmetic_prompt_tokens':[r.get('prompt_tokens') for r in math_rows],
            'arithmetic_evidence_tokens':[r.get('retrieval',{}).get('memory_tokens') for r in math_rows],
            'arithmetic_skipped_reasons':[r.get('retrieval',{}).get('retrieval_skipped_reason') for r in math_rows]}
        accepted={'一加一等于二','一加一等于2','1+1=2','2','二','答案是二','答案是2','等于二','等于2'}
        row['arithmetic_text_sanity']=len(math_rows)==5 and all(normalize(r.get('response','')) in accepted for r in math_rows)
        row['arithmetic_audio_readback_sanity']=len(math_rows)==5 and all(normalize(r.get('audio_readback','')) in accepted for r in math_rows)
        row['personal_recall_still_retrieves']=all(r.get('retrieval',{}).get('retrieval_skipped_reason') is None and r.get('retrieval',{}).get('injected_count',0)>0 for r in turns if r['label']=='recall')
        row['application_checks']={k:report['checks'][k] for k in ('workload_completed','all_voice_outputs','no_voice_errors','no_query_errors','no_write_errors','no_owner_leaks','durable_writes','visual_rates','visual_age','recall_text_and_audio')}
        rows.append(row)
    math_before=rows[0]['voice_latency']['arithmetic']['p95_ms'];math_after=rows[1]['voice_latency']['arithmetic']['p95_ms']
    complete=all(checks.values()) and all(r['application_checks']['workload_completed'] for r in rows)
    report={'created_at':stamp(),'status':'DEVELOPMENT_COMPARISON' if complete else 'INCOMPLETE','checks':checks,'variants':rows,
        'arithmetic_p95_reduction_percent':100*(math_before-math_after)/math_before if complete else None,
        'sources':[str(args.baseline),str(args.candidate)],
        'scope':'One development pair,20 fixed releases at10s intervals,five repetitions pertype. Same snapshot andcode;onlyrouting flag differs. Background tail counts,livecamera,time responses,warmth andload may differ. Allslow/cold attempts retained;not5-run/100independent events,B0-B3,acoustic orfinalJT qualification. Arithmetic-subset reduction isnot the overallS2 metric.'}
    write_json(args.output,report);print(json.dumps({'checks':checks,'arithmetic_p95_reduction_percent':report['arithmetic_p95_reduction_percent'],'variants':rows},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
