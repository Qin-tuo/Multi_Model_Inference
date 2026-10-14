"""Audit JT-02 evidence and CUDA API/device correlation from a native trace.

Queue intervals follow the installed NVIDIA cuda_kern_exec_trace report:
device start minus launch API return; negative values mean overlapping API
execution and are reported separately. Whole graphs include internal gaps.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.project import sha256,stamp,write_json


def stats(values):
    values=sorted(values)
    return {'count':len(values),'p50_ms':values[math.ceil(.5*len(values))-1] if values else None,
        'p95_ms':values[math.ceil(.95*len(values))-1] if values else None,
        'max_ms':max(values) if values else None}


def launch_coverage(connection,models):
    """Account for graph capture calls that intentionally execute no kernel."""
    device_ids=set()
    for table in ('CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_GRAPH_TRACE'):
        device_ids.update(tuple(r) for r in connection.execute('SELECT globalPid,correlationId FROM '+table))
    captures={}
    opened={}
    balanced=True
    for row in connection.execute('''SELECT r.start,r.end,r.globalTid,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN StringIds s ON s.id=r.nameId WHERE r.returnValue=0
            AND (s.value LIKE '%StreamBeginCapture%' OR s.value LIKE '%StreamEndCapture%') ORDER BY r.start'''):
        if 'BeginCapture' in row['value']:
            if row['globalTid'] in opened:
                balanced=False
            opened[row['globalTid']]=row['end']
        elif row['globalTid'] in opened:
            captures.setdefault(row['globalTid'],[]).append((opened.pop(row['globalTid']),row['start']))
        else:
            balanced=False
    counts={}
    for row in connection.execute('''SELECT r.start,r.end,r.globalTid,r.correlationId,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN StringIds s ON s.id=r.nameId WHERE r.returnValue=0 AND r.callchainId IS NULL
            AND s.value IN ('cudaLaunchKernel_v7000','cuLaunchKernelEx','cuLaunchKernel','cudaGraphLaunch_v10000')'''):
        pid=(row['globalTid']>>24)&0xFFFFFF
        key=(models.get(pid),row['value'])
        target=counts.setdefault(key,Counter())
        target['calls']+=1
        if (row['globalTid']&0xFFFFFFFFFF000000,row['correlationId']) in device_ids:
            target['matched_device_execution']+=1
        elif any(start<=row['start'] and row['end']<=end for start,end in captures.get(row['globalTid'],[])):
            target['captured_into_graph_without_immediate_execution']+=1
        else:
            target['unexplained_missing_device_execution']+=1
    return {'capture_ranges_balanced':balanced and not opened,
        'capture_ranges':sum(len(x) for x in captures.values()),
        'calls':[dict(component=key[0],api=key[1],**value) for key,value in sorted(counts.items())]}


def main(args):
    if args.output.exists():
        raise RuntimeError('Preserve previous audit reports')
    trace=json.loads(args.trace.read_text())
    inventory=json.loads(args.inventory.read_text())
    database=Path(trace['artifacts'])/'timeline.sqlite'
    entry=next(r for r in trace['raw_artifacts'] if r['path']==str(database.resolve()))
    if sha256(database)!=entry['sha256']:
        raise RuntimeError('Native trace changed after collection')
    if trace['source_fingerprints']['configs/runtime.yaml']!=inventory['runtime_sha256']:
        raise RuntimeError('Inventory and trace use different runtime profiles')
    if trace['source_fingerprints']['models/manifest.json']!=inventory['model_manifest_sha256']:
        raise RuntimeError('Inventory and trace use different model manifests')
    models={r['pid']:name for name,row in trace['worker']['processes'].items() for r in row['members']}
    requests=[r for r in trace['trace']['phases'] if r['label'].startswith('assistant:request:')]
    visual=next(r for r in trace['trace']['phases'] if r['label']=='assistant:visual_only')
    measurement=next(r for r in trace['trace']['phases'] if r['label']=='assistant:measurement')
    rows=[]
    with sqlite3.connect(f'file:{database}?mode=ro',uri=True) as connection:
        connection.row_factory=sqlite3.Row
        for table,kind in (('CUPTI_ACTIVITY_KIND_KERNEL','kernel'),('CUPTI_ACTIVITY_KIND_GRAPH_TRACE','graph')):
            for value in connection.execute(f'''SELECT k.rowid AS event_id,k.start,k.end,k.deviceId,
                    p.pid,r.start AS api_start,r.end AS api_end,r.returnValue,s.value AS api_name
                FROM {table} k LEFT JOIN PROCESSES p ON p.globalPid=k.globalPid
                LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
                    ON r.correlationId=k.correlationId AND k.globalPid=(r.globalTid & 0xFFFFFFFFFF000000)
                    AND r.callchainId IS NULL
                LEFT JOIN StringIds s ON s.id=r.nameId'''):
                row=dict(value,kind=kind)
                row['component']=models.get(row['pid'])
                rows.append(row)
        diagnostics=[dict(r) for r in connection.execute('SELECT * FROM DIAGNOSTIC_EVENT')]
        process_pids={r['globalPid']:r['pid'] for r in connection.execute('SELECT * FROM PROCESSES')}
        coverage=launch_coverage(connection,models)
    selected=[r for r in rows if measurement['start_ns']<=r['start']<measurement['end_ns']]
    by_phase={}
    for name,windows in [('visual_only',[visual]),('vlm_requests',requests)]:
        phase=[r for r in selected if any(w['start_ns']<=r['start']<w['end_ns'] for w in windows)]
        by_phase[name]={}
        for model in ('llama','vision','pose'):
            values=[r for r in phase if r['component']==model]
            by_phase[name][model]={
                'activities':len(values),'kinds':dict(Counter(r['kind'] for r in values)),
                'device_lifetime':stats([(r['end']-r['start'])/1e6 for r in values]),
                'queue_after_api_return':stats([(r['start']-r['api_end'])/1e6 for r in values if r['api_end'] is not None and r['start']>=r['api_end']]),
                'device_started_before_api_return':sum(r['api_end'] is not None and r['start']<r['api_end'] for r in values),
                'api_functions':dict(Counter(r['api_name'] for r in values))}
    expected_labels={f'assistant:request:{r["id"]}:{r["kind"]}' for r in trace['worker']['requests']}
    declared_files={r['path']:r['sha256'] for model in inventory['models'].values() for r in model['artifacts']}
    same_artifacts=all(declared_files.get(r['path'])==r['sha256'] for r in trace['verified_models_engines'])
    # All relevant no-CUDA warnings must be confined to non-model helpers.
    model_pids={r['pid'] for r in rows}
    missing_cuda=[r for r in diagnostics if process_pids.get(r['globalPid']) in model_pids
        and ('No CUDA events collected' in r['text'] or 'CUDA profiling might have not been started' in r['text'])]
    request_matches=[]
    for request in trace['worker']['requests']:
        label=f'assistant:request:{request["id"]}:{request["kind"]}'
        window=next((r for r in requests if r['label']==label),None)
        request_matches.append(window is not None and abs(window['duration_ms']-request['elapsed_ms'])<5)
    checks={
        'collector_passed_declared_checks':trace['status']=='PASS' and all(trace['checks'].values()),
        'inventory_passed_declared_checks':inventory['status']=='PASS' and all(inventory['checks'].values()),
        'seven_required_functional_model_roles':set(inventory['models'])=={'vlm','asr','tts','detection','pose','embedding','vad'},
        'all_selected_tensorrt_engines_registered':inventory['active_engine_count']==6,
        'trace_artifacts_match_inventory':same_artifacts,
        'all_device_activities_have_unique_successful_launch':len({(r['kind'],r['event_id']) for r in rows})==len(rows)
            and all(r['api_start'] is not None and r['returnValue']==0 for r in rows),
        'successful_launches_have_execution_or_graph_capture':coverage['capture_ranges_balanced'] and
            all(not r.get('unexplained_missing_device_execution',0) for r in coverage['calls']),
        'three_native_gpu_processes_on_one_device':len({r['pid'] for r in selected})==3 and len({r['deviceId'] for r in selected})==1,
        'complete_request_labels':len(requests)==6 and {r['label'] for r in requests}==expected_labels,
        'nine_application_ranges':len(trace['trace']['phases'])==9,
        'request_timestamps_agree':len(request_matches)==6 and all(request_matches),
        'required_visual_and_vlm_work_during_each_request':all(all(r['components'][name]['kernel_count']+
            r['components'][name]['graph_count']>0 for name in ('llama','vision','pose')) for r in requests),
        'vlm_and_visual_device_intervals_overlap':all(
            r['pairwise_device_activity_interval_overlap_ms']['llama+vision']>0 and
            r['pairwise_device_activity_interval_overlap_ms']['llama+pose']>0 for r in requests),
        'no_missing_cuda_warning_for_gpu_workers':not missing_cuda,
        'model_execution_evidence_passed':all(row['execution_evidence']['status']=='PASS' for row in inventory['models'].values()),
        'runtime_still_matches':sha256(ROOT/'configs/runtime.yaml')==inventory['runtime_sha256'],
        'models_still_match':sha256(ROOT/'models/manifest.json')==inventory['model_manifest_sha256']}
    query=Path('/opt/nvidia/nsight-systems/2026.3.1/target-linux-sbsa-armv8/reports/cuda_kern_exec_trace.py')
    report={'created_at':stamp(),'gate':'JT-02','status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
        'requirement':'约4–7个功能模型；视觉/VLM至少两个必需GPU负载重叠请求；trace说明实际执行/争用；登记所有engine。',
        'source_sha256':sha256(Path(__file__)),'native_queue_definition_source':str(query),'native_queue_definition_sha256':sha256(query),
        'trace_report':str(args.trace),'trace_report_sha256':sha256(args.trace),
        'inventory_report':str(args.inventory),'inventory_report_sha256':sha256(args.inventory),
        'database_sha256':entry['sha256'],'whole_capture_correlated_activities':len(rows),'measurement_activities':len(selected),
        'reverse_launch_coverage':coverage,
        'phases':by_phase,'scope':'JT-02 model/engine identity and native device execution/queue evidence only. Six real text/live-camera requests.'
            ' CUDA graphs are whole-device-graph lifetime records, not individual inner kernels; internal gaps and preemption are possible.'
            ' All launches are correlated by process plus native correlationId. Within-trace phase differences are diagnostic, not causal speedup or uninstrumented performance.'
            ' No SM occupancy, physical audio, task accuracy, JT-03/04/06/07/08 or complete assistant acceptance.',
        'remaining_limitations':['Node-granularity graph attribution errors remain in both installed Nsight versions; original failed evidence preserved.',
            'Orin hardware tracing and Unified Memory fault tracing unavailable; no such metrics claimed.',
            'NVTX shutdown completeness warnings retained; all nine application ranges collected and six request durations independently cross-checked.']}
    write_json(args.output,report)
    print(json.dumps({'status':report['status'],'checks':checks,'activities':len(rows)},ensure_ascii=False),flush=True)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(main(parser.parse_args()))
