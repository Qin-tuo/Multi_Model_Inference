"""Isolated native Qdrant BM25/MaxSim fusion on the frozen10k synthetic data.

Copies canonical seed points into an owned temporary collection; never changes
the source binding or application retrieval configuration. BM25 runs locally in
the existing Qdrant server, without FastEmbed or a cloud inference request.
"""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sys
import uuid

import httpx
from qdrant_client import models

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.memory_runtime import open_memory
from deploy.memory.runtime import load,private_key
from tools.managed_memory_capacity import workload
from tools.project import DATA,stamp,write_json

OPTIONS={'tokenizer':'multilingual','stemmer':{'type':'none'},'stopwords':{},'lowercase':True}


def main(args):
    corpus,config,_,_=workload(10000)
    if json.loads((args.run_dir/'capacity-run.json').read_text())!=config:raise RuntimeError('Expected frozen synthetic capacity store')
    if args.output.exists():raise RuntimeError('Refusing to overwrite lexical feasibility results')
    service=load(DATA/'qdrant');owner=uuid.uuid4().hex;name='lexical_probe_'+owner
    rows=[];counts={'owner':0,'guest':0};checks={};error=None
    condition={'must':[{'key':'user_id','match':{'value':corpus['owner']}},{'key':'namespace','match':{'value':corpus['namespace']}}]}
    with closing(service.client(timeout=30)) as client,httpx.Client(base_url=service.url,headers={'api-key':private_key(service.key_path)},trust_env=False,timeout=60) as http:
        server_version=client.info().version
        def post(path,body):
            response=http.post(path,json=body);response.raise_for_status();return response.json()['result']
        try:
            client.create_collection(name,vectors_config=models.VectorParams(size=512,distance=models.Distance.COSINE,
                on_disk=True,multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
                hnsw_config=models.HnswConfigDiff(m=0)),
                sparse_vectors_config={'lexical':models.SparseVectorParams(modifier=models.Modifier.IDF)},metadata={'probe_owner':owner})
            with open_memory(args.run_dir/'memory') as memory:
                offset=None
                while True:
                    values,offset=memory.sdk.vector_store.client.scroll(memory.sdk.collection_name,offset=offset,limit=64,with_payload=True,with_vectors=True)
                    points=[]
                    for value in values:
                        payload=value.payload
                        if not payload.get('operation_key','').startswith('seed-'):continue
                        label=next((label for label in ('owner','guest') if payload.get('user_id')==corpus[label]),None)
                        if label is None or payload.get('namespace')!=corpus['namespace']:raise RuntimeError('Unexpected seed ownership')
                        counts[label]+=1
                        dense=value.vector.get('') if isinstance(value.vector,dict) else value.vector
                        points.append({'id':str(value.id),'payload':payload,'vector':{'':dense,'lexical':{
                            'text':payload['data'],'model':'qdrant/bm25','options':OPTIONS}}})
                    if points:
                        response=http.put(f'/collections/{name}/points',params={'wait':'true'},json={'points':points})
                        response.raise_for_status()
                    if offset is None:break
                checks['frozen_seed_counts']=counts=={'owner':10000,'guest':100}
                for j in range(100):
                    index=int((j+.5)*10000/100)//10*10+j%2;query=corpus['query_template'].format(index=index)
                    vector=memory.sdk.embedding_model.embed(query,'search')
                    lexical={'text':query,'model':'qdrant/bm25','options':OPTIONS}
                    row={'index':index,'long_record':index%10==0}
                    for mode in ('bm25','rrf','dbsf'):
                        body={'limit':5,'with_payload':True,'filter':condition}
                        if mode=='bm25':body.update(query=lexical,using='lexical')
                        else:body.update(prefetch=[{'query':vector,'limit':60,'filter':condition},
                            {'query':lexical,'using':'lexical','limit':60,'filter':condition}],query={'fusion':mode})
                        hits=post(f'/collections/{name}/points/query',body)['points']
                        row[mode]={'target_in_top5':any(x['payload']['key']==f'project-{index:05d}' for x in hits),
                            'owner_leaks':sum(x['payload']['user_id']!=corpus['owner'] for x in hits),
                            'keys':[x['payload']['key'] for x in hits]}
                    rows.append(row)
                checks['no_owner_leaks']=all(r[mode]['owner_leaks']==0 for r in rows for mode in ('bm25','rrf','dbsf'))
        except Exception as failure:error={'type':type(failure).__name__,'message':str(failure)}
        finally:
            if client.collection_exists(name):
                if (client.get_collection(name).config.metadata or {}).get('probe_owner')!=owner:raise RuntimeError('Cannot clean an unrelated collection')
                client.delete_collection(name)
    summary={mode:{'top5_hits':sum(r[mode]['target_in_top5'] for r in rows),
        'long_hits':sum(r[mode]['target_in_top5'] for r in rows if r['long_record']),
        'short_hits':sum(r[mode]['target_in_top5'] for r in rows if not r['long_record'])} for mode in ('bm25','rrf','dbsf')}
    report={'created_at':stamp(),'status':'ERROR' if error else 'FEASIBILITY','error':error,'checks':checks,'counts':counts,
        'server_version':server_version,
        'bm25_options':OPTIONS,'summary':summary,'rows':rows,
        'sources':['https://qdrant.tech/documentation/inference/inference-bm25/','https://qdrant.tech/documentation/search/text-search/full-text-search/'],
        'scope':'Native local Qdrant BM25 and native RRF/DBSF,with unchanged seed MaxSim vectors and owner/namespace filters. Synthetic numbered stress only;source and production configuration unchanged. Other fixture preparation may overlap,no latency claim. Not Mem0 integration,forget/restore lifecycle,independent quality or JT acceptance.'}
    write_json(args.output,report);print(json.dumps({k:report[k] for k in ('status','error','checks','summary')},ensure_ascii=False),flush=True)
    return 1 if error else 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(main(parser.parse_args()))
