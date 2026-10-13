"""Mixed short/long record capacity on the actual authenticated window profile.

Preparation and measurement are separate commands. Timed operations use the
complete production adapter, including binding checks, journal and source spans.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.memory_runtime import open_memory
from app.memory_profiles import WINDOWS,model_path
from app.memory_store import embedding_fingerprint
from tools.long_memory_probe import documents
from tools.memory_capacity import MemoryMonitor,backend_state,statistics,timed
from tools.memory_capacity_seed import populate
from tools.project import stamp,write_json,sha256


def workload(tier, *, personal=False):
    corpus_path=ROOT/'benchmarks/corpora/memory-capacity-v1.json'
    corpus=json.loads(corpus_path.read_text())
    # Prefix deliberately excludes the project identity: long-record queries
    # need the tail, beyond the original encoder's single-window limit.
    prefix=documents()[0]['content'].split('本档案归属',1)[0]
    config={'kind':'managed-window-capacity-v1','schema_version':1,'dataset_sha256':sha256(corpus_path),
        'tier':tier,'seed_method':'sdk-batch','profile':WINDOWS,'long_every':10,
        'long_prefix_sha256':hashlib.sha256(prefix.encode()).hexdigest(),
        'query_indices':'floor(floor((j+0.5)*tier/100)/10)*10+j%2, j=0..99',
        'embedding_sha256':embedding_fingerprint(model_path(WINDOWS))}
    if personal:
        corpus['namespace']='personal'
        config.update(namespace='personal',preference_record=1,preference_content='我的饮品偏好是茉莉花茶。')
    def fact(index):
        if personal and index==1:return config['preference_content']
        k=index%len(corpus['topics']);short=corpus['record_template'].format(index=index,topic=corpus['topics'][k],value=corpus['values'][k])
        return prefix+short if index%10==0 else short
    return corpus,config,prefix,fact


def main(args):
    corpus,config,prefix,fact=workload(args.tier,personal=args.personal)
    folder=args.run_dir.resolve();marker=folder/'capacity-run.json'
    if folder.exists():
        if not marker.exists() or json.loads(marker.read_text())!=config:raise RuntimeError('Capacity directory ownership/configuration differs')
    else:folder.mkdir(parents=True,mode=0o700);write_json(marker,config)
    progress_path=folder/'population.json';progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'owner':0,'guest':0}
    settings={'memory':{'vector_store':'qdrant_server','embedding_profile':WINDOWS}}
    with MemoryMonitor() as monitor:
        if args.phase=='prepare':
            with open_memory(folder/'memory',settings=settings) as memory:
                for owner,total in [('owner',args.tier),('guest',corpus['guest_records'])]:
                    def checkpoint(count):
                        progress[owner]=count;write_json(progress_path,progress)
                        print(json.dumps({'phase':'prepare','owner':owner,'count':count,'total':total}),flush=True)
                    populate(memory,corpus,owner,total,args.tier,progress[owner],fact,checkpoint)
            write_json(folder/'prepared.json',{'created_at':stamp(),'status':'PREPARED','population':progress,'config':config,'resources':monitor.report()})
            return 0
        if args.output.exists():raise RuntimeError('Measurement report already exists; use a fresh prepared directory')
        if progress!={'owner':args.tier,'guest':corpus['guest_records']}:raise RuntimeError('Complete preparation before measurement')
        from qdrant_client import models
        from tools.lcvmem_eval import fingerprint
        reader=fingerprint();state=backend_state();queries=[];writes=[]
        with open_memory(folder/'memory',settings=settings) as memory:
            def count(user):return memory.sdk.vector_store.client.count(memory.sdk.collection_name,count_filter=models.Filter(must=[
                models.FieldCondition(key='user_id',match=models.MatchValue(value=user)),
                models.FieldCondition(key='namespace',match=models.MatchValue(value=corpus['namespace']))]),exact=True).count
            before={key:count(corpus[key]) for key in ('owner','guest')}
            if before!=progress:raise RuntimeError('Measured writes or unexpected data are already present')
            indices=[int((j+.5)*args.tier/100)//10*10+(j%2) for j in range(100)]
            for index in indices[:5]:memory.search(corpus['owner'],corpus['query_template'].format(index=index),namespace=corpus['namespace'])
            for index in indices:
                row,result=timed(lambda:memory.search(corpus['owner'],corpus['query_template'].format(index=index),namespace=corpus['namespace']))
                row.update(index=index,long_record=index%10==0)
                if result is not None:
                    row.update(expected_key_in_top5=any(x['key']==f'project-{index:05d}' for x in result['items']),returned=len(result['items']),
                        owner_leaks=sum(x['user_id']!=corpus['owner'] for x in result['items']),
                        windows=sum('evidence_window' in x for x in result['items']))
                queries.append(row)
            for index in range(50):
                short=f'临时测量T{index:03d}要求离开前关闭台灯。';text=prefix+short if index%2==0 else short;key=f'measured-{index}'
                row,result=timed(lambda:memory.remember(corpus['owner'],'fact',key,text,source_id=key,request_id=key,namespace=corpus['namespace']))
                row.update(index=index,long_record=index%2==0,utf8_bytes=len(text.encode()))
                if result is not None:
                    stored=memory.search(corpus['owner'],key=key,namespace=corpus['namespace'])['items']
                    row['verified_after_timer']=bool(result['durable'] and len(stored)==1 and stored[0]['content']==text and memory.sdk.history(result['memory_id'])
                        and memory.operation_status(corpus['owner'],key)['status']=='COMPLETED')
                writes.append(row)
            binding=memory._binding.binding
        qstat,wstat=statistics(queries),statistics(writes)
        checks={'correct_population':before==progress,'no_query_errors':qstat['errors']==0,'no_write_errors':wstat['errors']==0,
            'no_owner_leaks':all(r.get('owner_leaks')==0 for r in queries),'all_writes_durable':all(r.get('verified_after_timer') for r in writes),
            'query_p95_under200ms':qstat['p95_ms']<=200,'write_p95_under500ms':wstat['p95_ms']<=500,'app_unchanged':reader==fingerprint()}
        report={'created_at':stamp(),'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'tier':args.tier,'profile':WINDOWS,
            'config':config,'binding':{k:binding[k] for k in ('schema_version','embedding_profile','dimensions','embedding_sha256')},
            'query_summary':qstat,'write_summary':wstat,'queries':queries,'writes':writes,
            'query_long':statistics([r for r in queries if r['long_record']]),'query_short':statistics([r for r in queries if not r['long_record']]),
            'write_long':statistics([r for r in writes if r['long_record']]),'write_short':statistics([r for r in writes if not r['long_record']]),
            'numbered_recall_at5':sum(r.get('expected_key_in_top5',False) for r in queries)/len(queries),
            'guest_long_records':sum(int((j+.5)*args.tier/corpus['queries'])%10==0 for j in range(corpus['guest_records'])),
            'reader_fingerprint':reader,'backends_before':state,'backends_after':backend_state(),'resources':monitor.report(),
            'scope':'Managed authenticated production adapter with10% long owner records and100 guest distractors;100 queries split50/50 short/long,and50 complete writes split25/25. Backend residency is recorded separately; no foreground conversational load or per-frame visual timing. Not independent language quality or complete JT-11 coexistence acceptance.'}
        write_json(args.output,report)
        print(json.dumps({k:report[k] for k in ('status','checks','query_summary','write_summary','numbered_recall_at5')},ensure_ascii=False),flush=True)
        return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['prepare','measure'],required=True)
    parser.add_argument('--personal',action='store_true',help='Dedicated synthetic personal namespace for subsequent real voice co-load; one frozen preference replaces record1')
    parser.add_argument('--tier',type=int,choices=[1000,10000],default=10000);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(main(parser.parse_args()))
