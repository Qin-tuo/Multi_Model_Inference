"""Real BGE/Mem0/native-Qdrant restart checks on synthetic isolated profiles."""
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['MEM0_TELEMETRY']='false';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['NO_PROXY']='127.0.0.1,::1,localhost'
from app.memory import MemoryAdapterError
from deploy.memory.server import QdrantProcess
from tools.qdrant_capacity import adapter
from tools.project import CACHE,DATA,write_json,stamp


def main():
    root=DATA/'benchmarks'/('qdrant-recovery-'+uuid.uuid4().hex)
    model=CACHE/'models/bge-small-zh-v1.5-onnx'
    collection='capacity_recovery'
    results=[];crashes=[]
    report={'created_at':stamp(),'status':'RUNNING','scope':'Actual BGE/Mem0/SQLite and native Qdrant SIGKILL/restart; synthetic isolated data. Partial-history failure is explicitly injected. No power-loss, full backup restore, acoustic or final JT acceptance.',
        'artifacts':str(root),'iterations':results,'server_crashes':crashes}
    output=ROOT/'benchmarks/results/qdrant-recovery-smoke.json'
    def restart(server,phase,iteration):
        old=server.process.pid;start=time.monotonic_ns()
        server.process.kill();server.process.wait(timeout=10)
        code=server.process.returncode
        server.close();server.start()
        crashes.append({'iteration':iteration,'phase':phase,'old_pid':old,'new_pid':server.process.pid,
            'exit_code':code,'restart_to_ready_ms':(time.monotonic_ns()-start)/1e6})
    with QdrantProcess(root/'server',port=16334) as server:
        memory=adapter(root/'memory',model,server.url,collection)
        try:
            for i in range(10):
                user=f'recovery-owner-{i}';args=dict(source_id=f'save-{i}',request_id=f'save-{i}')
                text=f'恢复样例{i}：我偏好无糖绿茶。'
                saved=memory.remember(user,'preference','drink',text,**args)
                summary=memory.remember(user,'summary','drink-summary',text,source_id=f'summary-{i}',
                    request_id=f'summary-{i}',derived_from=(saved['memory_id'],))
                restart(server,'acknowledged_save',i)
                raw=memory.sdk.get(saved['memory_id'])
                checks={'acknowledged_save_survives':raw is not None and raw['memory']==text,
                    'owner_isolation':not memory.search('different-owner',key='drink')['items']}
                replay=memory.remember(user,'preference','drink',text,**args)
                checks['retry_same_id']=replay['memory_id']==saved['memory_id'] and replay['replayed']
                correction=f'恢复样例{i}：我现在偏好无糖乌龙茶。'
                memory.correct(user,saved['memory_id'],correction,expected_version=1,source_id=f'correct-{i}',request_id=f'correct-{i}')
                restart(server,'acknowledged_correction',i)
                row=memory.sdk.get(saved['memory_id'])
                checks['correction_survives']=row is not None and row['memory']==correction and row['metadata']['version']==2
                checks['derived_summary_erased']=memory.sdk.get(summary['memory_id']) is None and memory.sdk.history(summary['memory_id'])==[]
                memory.forget(user,saved['memory_id'],expected_version=2,request_id=f'forget-{i}')
                restart(server,'acknowledged_forget',i)
                checks['forgotten_not_revived']=memory.sdk.get(saved['memory_id']) is None and memory.sdk.history(saved['memory_id'])==[]
                checks['forgotten_query_empty']=not memory.search(user,'我偏好什么茶？')['items']
                # Commit the real server vector, then fail the actual SDK history
                # callback. Retrying after server restart must reconcile one ID.
                original=memory.sdk.db.add_history
                def fault(*_a,**_kw):raise RuntimeError('LABELLED history callback failure after vector write')
                memory.sdk.db.add_history=fault
                pending=False
                try:
                    memory.remember(user,'fact','partial','部分提交的合成记录。',source_id=f'partial-{i}',request_id=f'partial-{i}')
                except MemoryAdapterError as error:pending=error.code=='STORAGE_ERROR'
                finally:memory.sdk.db.add_history=original
                checks['partial_not_acknowledged']=pending and memory.operation_status(user,f'partial-{i}')['status']=='PENDING'
                restart(server,'partial_vector_before_history',i)
                recovered=memory.remember(user,'fact','partial','部分提交的合成记录。',source_id=f'partial-{i}',request_id=f'partial-{i}')
                copies=memory.search(user,key='partial')['items']
                checks['partial_reconciled_once']=recovered['durable'] and recovered['replayed'] and len(copies)==1 and len(memory.sdk.history(recovered['memory_id']))==1
                memory.forget(user,recovered['memory_id'],request_id=f'forget-partial-{i}')
                results.append({'iteration':i,'checks':checks,'status':'PASS' if all(checks.values()) else 'FAIL'})
                write_json(output,report);print(json.dumps(results[-1]),flush=True)
            memory.close()
            memory=adapter(root/'memory',model,server.url,collection)
            report['after_adapter_reopen_empty']=all(not memory.list_memories(f'recovery-owner-{i}') for i in range(10))
            report['server_kind']=memory.capabilities()['vector_store']
            report['status']='PASS' if report['after_adapter_reopen_empty'] and all(r['status']=='PASS' for r in results) and len(crashes)==40 else 'FAIL'
        finally:memory.close()
    report['server_stopped']=True;report['updated_at']=stamp();write_json(output,report)
    print(json.dumps({'status':report['status'],'iterations':len(results),'server_crashes':len(crashes)}))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
