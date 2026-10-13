"""Read-only diagnosis of the frozen numbered near-duplicate capacity corpus."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.memory_runtime import open_memory
from tools.managed_memory_capacity import workload
from tools.lcvmem_eval import fingerprint
from tools.project import stamp,write_json


def main(args):
    corpus,config,_,fact=workload(10000)
    if json.loads((args.run_dir/'capacity-run.json').read_text())!=config:raise RuntimeError('Expected frozen synthetic10k window fixture')
    if args.output.exists():raise RuntimeError('Refusing to overwrite a diagnostic report')
    rows=[];reader=fingerprint()
    with open_memory(args.run_dir/'memory') as memory:
        store=memory.sdk.vector_store
        for j in range(100):
            index=int((j+.5)*10000/100)//10*10+j%2
            key=f'project-{index:05d}';query=corpus['query_template'].format(index=index)
            exact=memory.search(corpus['owner'],key=key,namespace=corpus['namespace'])['items']
            if len(exact)!=1 or exact[0]['content']!=fact(index):raise RuntimeError('Stored synthetic source is missing or changed')
            expected=exact[0]
            result=memory.search(corpus['owner'],query,namespace=corpus['namespace'])
            vector=memory.sdk.embedding_model.embed(query,'search')
            candidates=store.search(query,vector,top_k=60,filters={'user_id':corpus['owner'],'namespace':corpus['namespace']})
            point=store.client.retrieve(store.collection_name,ids=[expected['memory_id']],with_vectors=True,with_payload=False)[0]
            values=point.vector.get('') if isinstance(point.vector,dict) else point.vector
            score=float(np.max(np.asarray(values,dtype=np.float32)@np.asarray(vector[0],dtype=np.float32)))
            query_with_topic=f"合成项目P{index:05d}的{corpus['topics'][index%len(corpus['topics'])]}要求是什么？"
            topic_result=memory.search(corpus['owner'],query_with_topic,namespace=corpus['namespace'])
            row={'index':index,'long_record':index%10==0,'source_exact':True,'method':result['retrieval_method'],
                'target_rank_in_dense60':next((i+1 for i,p in enumerate(candidates) if str(p.id)==expected['memory_id']),None),
                'target_cosine_score':score,'dense60_min_score':candidates[-1].score if candidates else None,
                'target_in_adapter_top5':any(x['memory_id']==expected['memory_id'] for x in result['items']),
                'topic_in_query_target_in_top5':any(x['memory_id']==expected['memory_id'] for x in topic_result['items']),
                'returned':[{'key':x['key'],'score':x['score'],'long_record':len(memory.sdk.embedding_model.spans(x['content']))>1} for x in result['items']]}
            if row['long_record']:
                row['selected_target_window_tail']=memory.sdk.embedding_model.best_window(expected['content'],values,query)['text'][-140:]
            rows.append(row)
        summary={}
        for kind,length in [('long',True),('short',False)]:
            values=[r for r in rows if r['long_record']==length]
            summary[kind]={'count':len(values),'adapter_top5_hits':sum(r['target_in_adapter_top5'] for r in values),
                'target_present_in_dense60':sum(r['target_rank_in_dense60'] is not None for r in values),
                'with_topic_top5_hits':sum(r['topic_in_query_target_in_top5'] for r in values)}
    report={'created_at':stamp(),'status':'DIAGNOSTIC','summary':summary,'rows':rows,'reader_fingerprint':reader,
        'app_unchanged':reader==fingerprint(),
        'scope':'Read-only analysis of exact sources,native dense60 candidates and actual adapter top5 in frozen synthetic numbered near-duplicates. The topic-query counterfactual is development-only,not a production fix or independent quality. Other fixture preparation may run concurrently;no latency claims.'}
    write_json(args.output,report);print(json.dumps(summary,ensure_ascii=False),flush=True)
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(main(parser.parse_args()))
