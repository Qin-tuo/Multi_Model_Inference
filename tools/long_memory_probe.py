"""Audit accepted long records with real Mem0/BGE/Qdrant and native token counts.

No model generation is used. Synthetic documents share a long prefix and differ
at their tails, exposing embedding truncation separately from prompt packing.
"""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.context_memory import memory_message
from app.memory_runtime import open_memory
from app.tools import RememberArgs
from deploy.memory.runtime import load
from tools.project import DATA,stamp,write_json


def documents():
    prefix='以下为合成项目档案。'+''.join(f'第{i}项记录：设备安装前需核对清单，标记连接方向，检查供电并保存测试结果，后续维护时逐项确认。' for i in range(1,23))
    projects=['海岚','青竹','云杉','星河','远帆','松林','银沙','南岭']
    return [{'project':project,'key':'synthetic-project-'+str(index),
        'content':prefix+f'本档案归属{project}项目。最终交付箱编号为{project}七号。'} for index,project in enumerate(projects)]


def main():
    service=load(DATA/'qdrant');binding={};rows=[];owner='synthetic-long-record'
    with tempfile.TemporaryDirectory(prefix='long-memory-boundary-') as tmp:
        directory=Path(tmp)/'memory'
        try:
            with open_memory(directory) as memory:
                binding=json.loads((directory/'store.json').read_text())
                for index,document in enumerate(documents()):
                    project,content,key=document['project'],document['content'],document['key']
                    RememberArgs(key=key,content=content)
                    saved=memory.remember(owner,'document',key,content,source_id=f'synthetic-turn-{index}',
                        source_excerpt='请记住，'+content,request_id=f'save-{index}',source_date='2026-10-13',source_speaker='user')
                    rows.append({'project':project,'key':key,'memory_id':saved['memory_id'],'durable':saved['durable'],
                        'utf8_bytes':len(content.encode()),'characters':len(content),'content_sha256':hashlib.sha256(content.encode()).hexdigest()})
                model=memory.sdk.embedding_model.model
                vectors=memory.sdk.vector_store.client.retrieve(memory.sdk.collection_name,ids=[r['memory_id'] for r in rows],with_vectors=True,with_payload=False)
                hashes={str(point.id):hashlib.sha256(json.dumps(point.vector,sort_keys=True).encode()).hexdigest() for point in vectors}
                for row in rows:
                    found=memory.search(owner,f'{row["project"]}项目的最终交付箱编号是什么？')['items']
                    exact=memory.search(owner,key=row['key'])['items']
                    item=exact[0]
                    _,bounded=memory_message([item],'http://127.0.0.1:8088/v1',compact_references=True)
                    _,full=memory_message([item],'http://127.0.0.1:8088/v1',max_tokens=8192,compact_references=True)
                    row.update(exact_content_preserved=hashlib.sha256(item['content'].encode()).hexdigest()==row['content_sha256'],
                        queried_target_in_top5=any(x['memory_id']==row['memory_id'] for x in found),
                        selected_count=len(found),embedding_sha256=hashes[row['memory_id']],
                        embedding_input_tokens=int(model.tokenize([item['content']])['input_ids'].shape[1]),
                        evidence_tokens_needed=full['memory_tokens'],injected_under512=bounded['injected_count'])
                checks={'accepted_records_durable':all(r['durable'] for r in rows),
                    'exact_storage_roundtrip':all(r['exact_content_preserved'] for r in rows),
                    'distinct_tail_documents_have_distinct_vectors':len(set(hashes.values()))==len(rows),
                    'each_tail_query_finds_its_record':all(r['queried_target_in_top5'] for r in rows),
                    'accepted_records_fit_recall_budget':all(r['injected_under512']>0 for r in rows)}
        finally:
            if binding.get('collection'):
                with closing(service.client()) as client:client.delete_collection(binding['collection'])
    report={'created_at':stamp(),'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'cases':rows,
        'scope':'Eight synthetic long records validated by RememberArgs and saved through real MemoryAdapter/BGE/Mem0/authenticated native Qdrant. Native llama tokenizer only,no generation or public conversational save claim. Identical prefixes with project names/answers at the tail separate vector truncation and512-token packing from durable storage.'}
    write_json(ROOT/'benchmarks/results/long-memory-boundary-baseline.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2));return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
