"""Quiescent application backup via native Qdrant snapshots and SQLite backup.

No model, audio, image or credential files are copied. A backup is usable only
after all declared files and source consistency checks have been confirmed.
Restoration must additionally apply current deletion and delivery records.
"""
from contextlib import ExitStack,closing
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sqlite3
import urllib.parse
import urllib.request
import uuid

from app.memory_store import check_transition,collection_metadata
from deploy.memory.migrate import canonical,lock
from deploy.memory.runtime import VERSION,atomic_json,load,private_key

DATABASES=('memory/operations.sqlite','memory/history.sqlite','records.sqlite','reminders.sqlite')
REQUIRED={'memory/operations.sqlite':{'operations','owner_generations','deletion_barriers','cancelled_sources'},
    'memory/history.sqlite':{'history','messages'},'records.sqlite':{'turns','turn_memory_refs','deleted_memory_refs','notes'},
    'reminders.sqlite':{'assistant_reminders','assistant_notifications','assistant_reminder_operations','assistant_reminder_policy','apscheduler_jobs'}}
FILES=set(DATABASES)|{'memory.snapshot','entities.snapshot','store-binding.json'}


def now():return datetime.now(timezone.utc).isoformat()


def sync_directory(path):
    descriptor=os.open(path,os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def file_info(path):
    digest=hashlib.sha256();size=0
    with Path(path).open('rb') as stream:
        while block:=stream.read(1024*1024):digest.update(block);size+=len(block)
    return {'sha256':digest.hexdigest(),'bytes':size}


def quoted(name):return '"'+name.replace('"','""')+'"'


def sqlite_state(path,required=()):
    """Logical checksum, including scheduler pickle blobs without unpickling."""
    digest=hashlib.sha256();counts={}
    with closing(sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)) as db:
        if db.execute('PRAGMA quick_check').fetchall()!=[('ok',)]:raise RuntimeError('SQLite integrity check failed')
        names=[row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        if not set(required)<=set(names):raise RuntimeError('Application database schema is incomplete')
        for row in db.execute('SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name'):digest.update(canonical(row))
        for name in names:
            digest.update(canonical(name));count=0
            columns=[row[1] for row in db.execute(f'PRAGMA table_info({quoted(name)})')]
            order=','.join(quoted(name) for name in columns)
            for row in db.execute(f'SELECT * FROM {quoted(name)} ORDER BY {order}'):
                digest.update(canonical([{'blob_sha256':hashlib.sha256(value).hexdigest()} if isinstance(value,bytes) else value for value in row]))
                count+=1
            counts[name]=count
    return {'sha256':digest.hexdigest(),'tables':counts}


def sqlite_copy(source,destination):
    destination=Path(destination);temporary=destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.tmp')
    destination.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    descriptor=os.open(temporary,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(descriptor)
    try:
        with closing(sqlite3.connect(Path(source).resolve().as_uri()+'?mode=ro',uri=True)) as src,closing(sqlite3.connect(temporary)) as dst:
            src.backup(dst,pages=256)
            dst.execute('PRAGMA journal_mode=DELETE');dst.execute('PRAGMA synchronous=FULL')
            if dst.execute('PRAGMA quick_check').fetchall()!=[('ok',)]:raise RuntimeError('Copied SQLite database is invalid')
        with temporary.open('rb') as file:os.fsync(file.fileno())
        os.replace(temporary,destination);sync_directory(destination.parent)
    finally:
        if temporary.exists():temporary.unlink()


def collection_state(client,name):
    digest=hashlib.sha256();count=0;offset=None
    while True:
        rows,offset=client.scroll(name,limit=256,offset=offset,with_payload=True,with_vectors=True)
        for row in rows:digest.update(canonical(row.model_dump(mode='json')));count+=1
        if offset is None:break
    if client.count(name,exact=True).count!=count:raise RuntimeError('Collection changed while being read')
    return {'sha256':digest.hexdigest(),'points':count}


def upload_snapshot_to_new_collection(client,name,path,checksum):
    """Use the upstream upload endpoint; file URLs outside its root are denied."""
    from qdrant_client import models
    if client.collection_exists(name):raise RuntimeError('Snapshot upload requires a new collection; refusing to overwrite existing data')
    with Path(path).open('rb') as stream:
        result=client.http.snapshots_api.recover_from_uploaded_snapshot(collection_name=name,wait=True,
            priority=models.SnapshotPriority.SNAPSHOT,checksum=checksum,snapshot=stream)
    if result.status!='ok' or result.result is not True:raise RuntimeError('Qdrant did not confirm snapshot upload recovery')


def download_snapshot(service,client,collection,destination,manifest,publish):
    # Record the returned server filename before transfer so an interrupted
    # transfer can clean up only its own temporary snapshot on the next run.
    snapshot=client.create_snapshot(collection,wait=True)
    if not snapshot or not re.fullmatch(r'[A-Za-z0-9_.-]+',snapshot.name):raise RuntimeError('Unexpected snapshot descriptor')
    manifest['server_snapshots'].append({'collection':collection,'name':snapshot.name});publish()
    destination=Path(destination);temporary=destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        request=urllib.request.Request(service.url+'/collections/'+urllib.parse.quote(collection,safe='')+'/snapshots/'+urllib.parse.quote(snapshot.name,safe=''),
            headers={'api-key':private_key(service.key_path),'Accept-Encoding':'identity'})
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        descriptor=os.open(temporary,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        with os.fdopen(descriptor,'wb') as output,opener.open(request,timeout=60) as response:
            while block:=response.read(1024*1024):output.write(block)
            output.flush();os.fsync(output.fileno())
        evidence=file_info(temporary)
        if evidence['bytes']!=snapshot.size or (snapshot.checksum and evidence['sha256']!=snapshot.checksum):
            raise RuntimeError('Downloaded Qdrant snapshot checksum or size differs')
        os.replace(temporary,destination);sync_directory(destination.parent)
        return evidence
    finally:
        if temporary.exists():temporary.unlink()


def validate(directory):
    directory=Path(directory).resolve();manifest_path=directory/'manifest.json'
    if manifest_path.is_symlink():raise RuntimeError('Backup manifest must be a regular file')
    manifest=json.loads(manifest_path.read_text())
    if (manifest.get('schema_version')!=1 or manifest.get('kind')!='mmassistant_application_backup'
            or manifest.get('status')!='COMPLETE' or not re.fullmatch('[0-9a-f]{32}',manifest.get('backup_id',''))):
        raise RuntimeError('Backup is incomplete or unsupported')
    files=manifest.get('files',{})
    if not {'memory/operations.sqlite','memory/history.sqlite','memory.snapshot','store-binding.json'}<=files.keys() or set(files)-FILES:
        raise RuntimeError('Backup file inventory is incomplete or unsupported')
    binding=manifest.get('source_binding',{})
    roles=manifest.get('collections',{})
    if (binding.get('backend')!='qdrant_server' or binding.get('status')!='READY'
            or not re.fullmatch('[0-9a-f]{32}',binding.get('store_id',''))
            or binding.get('collection')!='assistant_'+binding['store_id']
            or not isinstance(binding.get('dimensions'),int) or binding['dimensions']<=0
            or not re.fullmatch('[0-9a-f]{64}',binding.get('embedding_sha256',''))
            or not {'memory'}<=roles.keys() or set(roles)-{'memory','entities'}):
        raise RuntimeError('Backup collection inventory or binding is invalid')
    if {name for name in files if name.endswith('.snapshot')}!={role+'.snapshot' for role in roles}:
        raise RuntimeError('Backup snapshot inventory differs')
    for role,item in roles.items():
        if (item.get('file')!=role+'.snapshot' or item.get('collection')!=binding['collection']+('_entities' if role=='entities' else '')
                or not isinstance(item.get('points'),int) or item['points']<0 or not re.fullmatch('[0-9a-f]{64}',item.get('sha256',''))):
            raise RuntimeError('Backup collection descriptor is invalid')
    for name,expected in files.items():
        path=directory/name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(directory):
            raise RuntimeError('Backup file is missing or escapes the backup directory')
        if file_info(path)!=expected:raise RuntimeError('Backup file checksum differs: '+name)
    for name,expected in manifest.get('databases',{}).items():
        if name not in DATABASES or name not in files or sqlite_state(directory/name,REQUIRED[name])!=expected:
            raise RuntimeError('Backup database verification failed')
    if set(manifest.get('databases',{}))!=set(files)&set(DATABASES):raise RuntimeError('Backup database inventory differs')
    if json.loads((directory/'store-binding.json').read_text())!=manifest.get('source_binding'):
        raise RuntimeError('Backup source binding differs')
    return manifest


def create(data_directory,output,*,checkpoint=lambda phase:None):
    source=Path(data_directory).expanduser().resolve();output=Path(output).expanduser().resolve()
    if (source==output or source.is_relative_to(output) or output.is_relative_to(source)):
        raise RuntimeError('Backup and application data directories must be separate')
    memory=source/'memory';binding_path=memory/'store.json'
    if not binding_path.is_file():raise RuntimeError('Backup requires a registered native memory store; migrate legacy memory first')
    manifest_path=output/'manifest.json'
    with ExitStack() as stack:
        lock(stack,memory/'adapter.writer.lock')
        # Acquire all owners even if a database has never been created.
        lock(stack,source/'records.sqlite.owner.lock');lock(stack,source/'reminders.sqlite.owner.lock')
        check_transition(memory)
        binding=json.loads(binding_path.read_text())
        service=load(binding['service_directory'])
        if (binding.get('backend')!='qdrant_server' or binding.get('directory')!=str(memory)
                or binding.get('instance_id')!=service.instance_id or binding.get('status')!='READY'
                or not re.fullmatch('[0-9a-f]{32}',binding.get('store_id',''))
                or binding.get('collection')!='assistant_'+binding['store_id']):raise RuntimeError('Memory store binding is invalid')
        lock(stack,service.directory/'leases'/(binding['collection']+'.lock'))
        client=stack.enter_context(closing(service.client(timeout=60)))
        info=client.get_collection(binding['collection'])
        if info.config.metadata!=collection_metadata(binding) or info.config.params.vectors.size!=binding['dimensions']:
            raise RuntimeError('Native collection metadata differs from source binding')
        roles={'memory':binding['collection']}
        entity=binding['collection']+'_entities'
        if client.collection_exists(entity):
            info=client.get_collection(entity)
            if info.config.metadata not in (None,{},dict(collection_metadata(binding),role='entities')):
                raise RuntimeError('Entity collection has unrelated ownership metadata')
            roles['entities']=entity
        databases=[name for name in DATABASES if (source/name).is_file()]
        if not set(DATABASES[:2])<=set(databases):raise RuntimeError('Memory journal or history is missing')
        before={name:sqlite_state(source/name,REQUIRED[name]) for name in databases}
        collections={role:collection_state(client,name) for role,name in roles.items()}
        output.mkdir(parents=True,exist_ok=True,mode=0o700)
        lock(stack,output/'backup.owner.lock')
        if manifest_path.exists():
            manifest=json.loads(manifest_path.read_text())
            if (manifest.get('schema_version')!=1 or manifest.get('kind')!='mmassistant_application_backup'
                    or manifest.get('source_binding')!=binding or manifest.get('source_directory')!=str(source)
                    or manifest.get('directory')!=str(output) or not re.fullmatch('[0-9a-f]{32}',manifest.get('backup_id',''))):
                raise RuntimeError('Refusing to overwrite an unrelated backup directory')
            if manifest.get('status')=='COMPLETE':raise RuntimeError('Completed backups are immutable; choose a new output directory')
            if manifest.get('status')!='WRITING':raise RuntimeError('Unsupported backup state')
        else:
            if set(p.name for p in output.iterdir())-{'backup.owner.lock'}:raise RuntimeError('Backup output is not empty')
            manifest={'schema_version':1,'kind':'mmassistant_application_backup','status':'WRITING','backup_id':uuid.uuid4().hex,
                'created_at':now(),'directory':str(output),'source_directory':str(source),'source_binding':binding,'server_snapshots':[]}
        def publish():atomic_json(manifest_path,manifest)
        # Delete only temporary server snapshots named in this owned receipt.
        for item in manifest['server_snapshots']:
            if item.get('collection') not in roles.values() or not re.fullmatch(r'[A-Za-z0-9_.-]+',item.get('name','')):
                raise RuntimeError('Invalid temporary snapshot ownership receipt')
            if item['name'] in {s.name for s in client.list_snapshots(item['collection'])}:
                client.delete_snapshot(item['collection'],item['name'],wait=True)
        manifest.update(status='WRITING',started_at=now(),files={},databases={},collections={},server_snapshots=[],
            versions={name:importlib.metadata.version(name) for name in ('mem0ai','qdrant-client','SQLAlchemy','APScheduler')},
            qdrant_version=VERSION,embedding_sha256=binding['embedding_sha256'])
        publish();checkpoint('marked')
        try:
            atomic_json(output/'store-binding.json',binding)
            manifest['files']['store-binding.json']=file_info(output/'store-binding.json')
            for name in databases:
                sqlite_copy(source/name,output/name)
                copied=sqlite_state(output/name,REQUIRED[name])
                if copied!=before[name]:raise RuntimeError('SQLite source changed while copying')
                manifest['files'][name]=file_info(output/name);manifest['databases'][name]=copied
                publish();checkpoint('database')
            for role,name in roles.items():
                filename=role+'.snapshot'
                manifest['files'][filename]=download_snapshot(service,client,name,output/filename,manifest,publish)
                manifest['collections'][role]=dict(collections[role],collection=name,file=filename)
                publish();checkpoint('snapshot')
            if any(sqlite_state(source/name,REQUIRED[name])!=before[name] for name in databases):
                raise RuntimeError('Application databases changed during backup')
            if any(collection_state(client,name)!=collections[role] for role,name in roles.items()):
                raise RuntimeError('Memory collection changed during backup')
            if (json.loads(binding_path.read_text())!=binding
                    or {name for name in DATABASES if (source/name).is_file()}!=set(databases)
                    or client.collection_exists(entity)!=('entities' in roles)):
                raise RuntimeError('Application store inventory changed during backup')
            checkpoint('verified')
            # Check bytes/logical DBs before publishing the completion marker.
            for name,evidence in manifest['files'].items():
                if file_info(output/name)!=evidence:raise RuntimeError('Backup file changed before publication')
            manifest.update(status='COMPLETE',completed_at=now())
            publish()
        finally:
            cleanup=[]
            for item in manifest['server_snapshots']:
                try:client.delete_snapshot(item['collection'],item['name'],wait=True)
                except Exception:cleanup.append(item)
            manifest['server_snapshots']=cleanup;publish()
        return validate(output)
