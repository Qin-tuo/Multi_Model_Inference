"""Only benchmark-authorized dialogue events; no reference answers enter storage."""
from datetime import datetime,timezone
import hashlib
import json
import uuid

from tools.lcvmem_data import messages


def populate(memory,run_id,row,*,source_metadata='body-only-v1'):
    marker=memory.data_dir.parent.parent/'lcvmem-run.json'
    if not marker.is_file() or json.loads(marker.read_text()).get('run_id')!=run_id:
        raise RuntimeError('Dialogue import is restricted to a marked benchmark directory')
    from mem0.utils.lemmatization import lemmatize_for_bm25
    from app.memory_sources import provenance
    if source_metadata not in {'body-only-v1','calendar-speaker-v1'}:raise ValueError('Unknown source metadata profile')
    user='lcvmem-'+row['id'];events=messages(row['context'])
    if memory.generation(user)!=0:raise RuntimeError('Benchmark owner already has maintenance history')
    with memory._gate:
        for start in range(0,len(events),32):
            batch=[]
            for index,item in enumerate(events[start:start+32],start):
                content=item['content'];source=f'lcv:{row["id"]}:{index}';request=f'seed-{index}';key=f'history-{index}'
                args={'type':'event','key':key,'content':content,'source_id':source,'source_excerpt':content,
                    'namespace':'personal','expected_generation':None,'derived_from':[]}
                source_fields=(provenance(datetime.strptime(item['date'],'%Y年%m月%d日').date().isoformat(),
                    'user' if item['role']=='用户' else 'assistant') if source_metadata=='calendar-speaker-v1' else {})
                args.update(source_fields)
                _,prior=memory._start(user,request,'remember',args)
                if prior:continue
                mid=str(uuid.uuid5(uuid.NAMESPACE_URL,f'lcvmem-v1/{run_id}/{row["id"]}/{index}'))
                created=datetime.now(timezone.utc).isoformat()
                payload={'namespace':'personal','memory_type':'event','key':key,'version':1,'generation':0,
                    'source_id':source,'source_excerpt':content,'operation_key':request,'derived_from':[],'user_id':user,
                    'role':'user','data':content,'hash':hashlib.md5(content.encode()).hexdigest(),'created_at':created,'updated_at':created,
                    'text_lemmatized':lemmatize_for_bm25(content)}
                payload.update(source_fields)
                batch.append((mid,payload,request))
            if not batch:continue
            vectors=memory.sdk.embedding_model.embed_batch([payload['data'] for _,payload,_ in batch],'add')
            memory.sdk.vector_store.insert(vectors=vectors,ids=[mid for mid,_,_ in batch],payloads=[payload for _,payload,_ in batch])
            for mid,payload,request in batch:
                if not memory.sdk.history(mid):memory.sdk.db.add_history(mid,None,payload['data'],'ADD',created_at=payload['created_at'],updated_at=payload['updated_at'],role='user')
                memory._finish(user,request,{'memory_id':mid,'version':1,'generation':0,'durable':True,'replayed':False,'atomicity':'reconciled_non_atomic'})
    return len(events)
