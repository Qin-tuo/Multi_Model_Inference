"""Private local Qdrant configuration; systemd owns the actual server process."""
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import stat
import uuid
import urllib.request
import urllib.error

from deploy.memory.prepare import BINARY,VERSION,validate_binary


def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as out:
            json.dump(value,out,ensure_ascii=False,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
        os.replace(temporary,path)
        descriptor=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    finally:
        if temporary.exists():temporary.unlink()


def private_key(path):
    path=Path(path)
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o077:
            raise RuntimeError('Memory service credential must be a private file owned by the current user')
        value=os.read(fd,256).decode().strip()
    finally:os.close(fd)
    if not re.fullmatch('[0-9a-f]{64}',value):raise RuntimeError('Invalid memory service credential format')
    return value


@dataclass(frozen=True)
class MemoryService:
    directory:Path
    instance_id:str
    port:int
    @property
    def url(self):return f'http://127.0.0.1:{self.port}'
    @property
    def key_path(self):return self.directory/'api-key'
    def client(self):
        from qdrant_client import QdrantClient
        # Check synchronously through the same proxy-free authenticated client;
        # the SDK's optional background compatibility helper has its own HTTP path.
        client=QdrantClient(url=self.url,api_key=private_key(self.key_path),timeout=5,trust_env=False,check_compatibility=False)
        try:
            if client.info().version!=VERSION:raise RuntimeError('Memory server version differs from the qualified release')
            return client
        except BaseException:client.close();raise


def load(directory):
    directory=Path(directory).resolve()
    data=json.loads((directory/'deployment.json').read_text())
    if (data.get('schema_version')!=1 or data.get('version')!=VERSION
            or data.get('directory')!=str(directory) or not re.fullmatch('[0-9a-f]{32}',data.get('instance_id',''))
            or not isinstance(data.get('port'),int) or not 1024<=data['port']<=65535):
        raise RuntimeError('Memory service deployment binding differs; explicit migration is required')
    private_key(directory/'api-key')
    return MemoryService(directory,data['instance_id'],data['port'])


def prepare(directory,*,port=6333):
    directory=Path(directory).resolve()
    if not 1024<=port<=65535:raise ValueError('Invalid service port')
    if directory.exists() and not (directory/'deployment.json').exists():
        if set(p.name for p in directory.iterdir())-{'setup.lock','api-key'}:
            raise RuntimeError('Refusing to take over an unregistered memory service directory')
    directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    with (directory/'setup.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        key_path=directory/'api-key'
        if not key_path.exists():
            fd=os.open(key_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'w') as out:out.write(secrets.token_hex(32)+'\n');out.flush();os.fsync(out.fileno())
        key=private_key(key_path)
        marker=directory/'deployment.json'
        if not marker.exists():atomic_json(marker,{'schema_version':1,'version':VERSION,'directory':str(directory),'instance_id':uuid.uuid4().hex,'port':port})
        service=load(directory)
        if service.port!=port:raise RuntimeError('Existing memory service port differs')
        config={'log_level':'WARN','telemetry_disabled':True,
            'service':{'host':'127.0.0.1','http_port':port,'grpc_port':None,'max_workers':2,'enable_cors':False,'api_key':key},
            'storage':{'storage_path':str(directory/'storage'),'snapshots_path':str(directory/'snapshots'),
                'on_disk_payload':True,'performance':{'max_search_threads':2,'optimizer_cpu_budget':1},
                'optimizers':{'default_segment_number':2,'max_optimization_threads':1},'hnsw_index':{'m':0}},
            'cluster':{'enabled':False}}
        target=directory/'config.yaml'
        if target.exists() and json.loads(target.read_text())!=config:
            raise RuntimeError('Memory service configuration has local edits; refusing to overwrite')
        if not target.exists():atomic_json(target,config)
        (directory/'leases').mkdir(mode=0o700,exist_ok=True)
        return service


def health(directory):
    try:
        service=load(directory)
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request=urllib.request.Request(service.url+'/collections',headers={'api-key':private_key(service.key_path)})
        with opener.open(request,timeout=3) as response:result=json.load(response)
        unauthorized=False
        try:
            with opener.open(service.url+'/collections',timeout=3):pass
        except urllib.error.HTTPError as error:unauthorized=error.code in {401,403}
        return {'ready':result.get('status')=='ok' and unauthorized,'authentication_required':unauthorized,
                'instance_id':service.instance_id,'scope':'Authenticated loopback storage readiness; not memory quality acceptance'}
    except (OSError,ValueError,RuntimeError,KeyError):return {'ready':False,'reason':'memory_service_unavailable_or_binding_invalid'}


def serve(directory):
    validate_binary()
    port=load(directory).port if (Path(directory)/'deployment.json').exists() else 6333
    service=prepare(directory,port=port)
    if not BINARY.is_file():raise RuntimeError('Prepare the pinned Qdrant binary first')
    env={k:v for k,v in os.environ.items() if not k.startswith('QDRANT_')}
    env.update(QDRANT__SERVICE__HOST='127.0.0.1',QDRANT__SERVICE__HTTP_PORT=str(service.port))
    os.chdir(service.directory)
    os.execve(str(BINARY),[str(BINARY),'--config-path',str(service.directory/'config.yaml'),'--disable-telemetry'],env)
