"""Native Qdrant process for isolated evaluations; the application's store is unchanged."""
import fcntl
import json
from pathlib import Path
import os
import socket
import subprocess
import time
import urllib.request

from deploy.memory.prepare import BINARY,VERSION


class QdrantProcess:
    def __init__(self,folder,*,port=16333):
        self.folder=Path(folder).resolve();self.port=port
        if not 1024<=port<=65535:raise ValueError('Invalid Qdrant port')
        self.url=f'http://127.0.0.1:{port}'
        self.process=self.owner=self.log=None
    def start(self):
        if self.process is not None:raise RuntimeError('Qdrant process already started')
        marker=self.folder/'evaluation.json'
        expected={'purpose':'isolated_qdrant_evaluation','version':VERSION,'port':self.port}
        if self.folder.exists() and (not marker.is_file() or json.loads(marker.read_text())!=expected):
            raise RuntimeError('Qdrant evaluation directory is not owned by this configuration')
        self.folder.mkdir(parents=True,exist_ok=True,mode=0o700)
        if not marker.exists():marker.write_text(json.dumps(expected)+'\n')
        self.owner=(self.folder/'evaluation.owner.lock').open('a+b')
        try:
            fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1',self.port))==0:raise RuntimeError('Qdrant port is already occupied; refusing to adopt it')
            config={'log_level':'WARN','telemetry_disabled':True,
                'service':{'host':'127.0.0.1','http_port':self.port,'grpc_port':None,'max_workers':2,'enable_cors':False},
                'storage':{'storage_path':str(self.folder/'storage'),'snapshots_path':str(self.folder/'snapshots'),
                    'on_disk_payload':True,'performance':{'max_search_threads':2,'optimizer_cpu_budget':1},
                    'optimizers':{'default_segment_number':2,'max_optimization_threads':1},
                    'hnsw_index':{'m':0}},'cluster':{'enabled':False}}
            # Qdrant's config crate in this release enables YAML, not JSON.
            # JSON syntax is valid YAML; use its supported file extension.
            config_path=self.folder/'config.yaml';config_path.write_text(json.dumps(config,indent=2)+'\n')
            env={k:v for k,v in os.environ.items() if not k.startswith('QDRANT_')}
            env.update(QDRANT__SERVICE__HOST='127.0.0.1',QDRANT__SERVICE__HTTP_PORT=str(self.port))
            self.log=(self.folder/'server.log').open('ab')
            self.process=subprocess.Popen([str(BINARY),'--config-path',str(config_path),'--disable-telemetry'],
                cwd=self.folder,env=env,stdin=subprocess.DEVNULL,stdout=self.log,stderr=subprocess.STDOUT)
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                if self.process.poll() is not None:raise RuntimeError('Qdrant exited during startup; inspect its local log')
                try:
                    with opener.open(self.url+'/readyz',timeout=1) as response:
                        if response.status==200:
                            listeners=subprocess.check_output(['ss','-H','-ltnp'],text=True,timeout=3)
                            owned=[line.split()[3] for line in listeners.splitlines() if f'pid={self.process.pid},' in line]
                            if owned!=[f'127.0.0.1:{self.port}']:
                                raise RuntimeError('Qdrant must own exactly its declared loopback HTTP listener')
                            return self
                except OSError:pass
                time.sleep(.1)
            raise TimeoutError('Qdrant readiness deadline exceeded')
        except BaseException:
            self.close();raise
    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=5)
        if self.log:self.log.close();self.log=None
        if self.owner:self.owner.close();self.owner=None
        self.process=None
    def __enter__(self):return self.start()
    def __exit__(self,*_):self.close()
