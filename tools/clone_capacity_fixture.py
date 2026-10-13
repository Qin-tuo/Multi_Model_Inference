"""Clone only an unmeasured synthetic capacity seed using native snapshots.

Retains exact point/history/journal state for repeated workloads. Not a user
data copy or restore command; ordinary data must use the fenced restore path.
"""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.memory_profiles import profile_of,entity_metadata
from app.memory_store import new_binding,collection_metadata,StoreBinding
from deploy.data.backup import create,validate,sqlite_copy,sqlite_state,upload_snapshot_to_new_collection,collection_state
from deploy.memory.runtime import load,atomic_json
from tools.managed_memory_capacity import workload
from tools.project import stamp,sha256


def main(args):
    source=args.source.resolve();target=args.target.resolve()
    if source==target or source.is_relative_to(target) or target.is_relative_to(source):raise RuntimeError('Fixture directories must be separate')
    config=json.loads((source/'capacity-run.json').read_text())
    corpus,expected,_,fact=workload(config['tier'],personal=config.get('namespace')=='personal',profile=config['profile'])
    if config!=expected or json.loads((source/'population.json').read_text())!={'owner':config['tier'],'guest':100}:
        raise RuntimeError('Source is not a completed frozen capacity seed')
    if any((source/name).exists() for name in ('coexistence-started.json','records.sqlite','reminders.sqlite')):
        raise RuntimeError('Only unused seed fixtures may be cloned')
    if target.exists():raise RuntimeError('Target must be new; preserve any incomplete clone for inspection')
    target.mkdir(parents=True,mode=0o700);receipt=target/'clone-receipt.json'
    state={'kind':'synthetic-capacity-clone-v1','status':'PREPARING','created_at':stamp(),'source':str(source),'target':str(target),'config':config}
    atomic_json(receipt,state)
    backup=target/'seed-backup';create(source,backup);manifest=validate(backup)
    if any(name in manifest['files'] for name in ('records.sqlite','reminders.sqlite')):raise RuntimeError('Source contains application state beyond the seed')
    original=manifest['source_binding'];service=load(original['service_directory'])
    if profile_of(original)!=config['profile'] or original['embedding_sha256']!=config['embedding_sha256']:
        raise RuntimeError('Source model/index binding differs from the frozen recipe')
    if manifest['collections']['memory']['points']!=config['tier']+100 or manifest['collections']['entities']['points']!=0:
        raise RuntimeError('Source has measured writes or non-seed entities')
    with sqlite3.connect(backup/'memory/operations.sqlite') as db:
        rows=db.execute('SELECT user_id,request_id,operation,status FROM operations').fetchall()
        wanted={(corpus[label],f'seed-{label}-{i}','remember','COMPLETED') for label,total in [('owner',config['tier']),('guest',100)] for i in range(total)}
        if len(rows)!=len(wanted) or set(rows)!=wanted:raise RuntimeError('Source has non-seed or unfinished operations')
        if db.execute('SELECT count(*) FROM deletion_barriers').fetchone()[0] or db.execute('SELECT count(*) FROM cancelled_sources').fetchone()[0]:
            raise RuntimeError('Source has maintenance history')
        if set(db.execute('SELECT user_id,generation FROM owner_generations').fetchall())!={(corpus['owner'],0),(corpus['guest'],0)}:
            raise RuntimeError('Source generations differ from the seed')
    memory=target/'memory';memory.mkdir()
    binding=new_binding(memory,service,original['embedding_sha256'],original['dimensions'],config['profile'])
    state.update(binding=binding,backup_manifest_sha256=sha256(backup/'manifest.json'));atomic_json(receipt,state)
    with closing(service.client(timeout=60)) as client:
        for role,description in manifest['collections'].items():
            name=binding['collection']+('_entities' if role=='entities' else '')
            upload_snapshot_to_new_collection(client,name,backup/description['file'],manifest['files'][description['file']]['sha256'])
            if collection_state(client,name)!={key:description[key] for key in ('sha256','points')}:raise RuntimeError('Cloned point state differs from snapshot')
            client.update_collection(name,metadata=entity_metadata(binding) if role=='entities' else collection_metadata(binding))
        # Check canonical seed content,not only a forgeable directory marker.
        offset=None;seen=set()
        while True:
            points,offset=client.scroll(binding['collection'],limit=128,offset=offset,with_payload=True,with_vectors=False)
            for point in points:
                p=point.payload;label=next((label for label in ('owner','guest') if p.get('user_id')==corpus[label]),None)
                if label is None:raise RuntimeError('Non-synthetic owner in source')
                index=int(p['key'].removeprefix('project-'))
                total=config['tier'] if label=='owner' else 100
                content=fact(index if label=='owner' else int((index+.5)*config['tier']/corpus['queries']))
                if (not 0<=index<total or p.get('namespace')!=corpus['namespace'] or p.get('version')!=1
                        or p.get('data')!=content or p.get('source_excerpt')!=content or p.get('derived_from')!=[]
                        or p.get('source_id')!=f'synthetic-{label}-{index}' or p.get('operation_key')!=f'seed-{label}-{index}'):
                    raise RuntimeError('Source is not the frozen synthetic body/provenance')
                seen.add((label,index))
            if offset is None:break
        if len(seen)!=config['tier']+100:raise RuntimeError('Seed key inventory differs')
    for name in ('history.sqlite','operations.sqlite'):
        sqlite_copy(backup/'memory'/name,memory/name)
        if sqlite_state(memory/name)!=manifest['databases']['memory/'+name]:raise RuntimeError('Cloned SQLite logical state differs')
    atomic_json(memory/'store.json',binding)
    owned=StoreBinding(memory,service,embedding_sha256=binding['embedding_sha256'],dimensions=binding['dimensions'],profile=config['profile'])
    try:owned.ready()
    finally:owned.close()
    state.update(status='COMPLETE',completed_at=stamp(),binding=json.loads((memory/'store.json').read_text()),
        verified_points=config['tier']+100,point_states=manifest['collections'])
    atomic_json(receipt,state)
    atomic_json(target/'capacity-run.json',config)
    atomic_json(target/'population.json',{'owner':config['tier'],'guest':100})
    atomic_json(target/'prepared.json',{'created_at':stamp(),'status':'PREPARED','config':config,
        'population':{'owner':config['tier'],'guest':100},'method':'verified-native-snapshot-clone',
        'clone_receipt_sha256':sha256(receipt),'scope':'Exact snapshot point checksums and canonical synthetic seed/history/journal verified;BGE vectors reused,not re-encoded. Timed application operations still execute actual BGE/Mem0.'})
    print(json.dumps({'status':'PREPARED','points':config['tier']+100,'target':str(target),'method':'verified-native-snapshot-clone'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--target',type=Path,required=True)
    main(parser.parse_args())
