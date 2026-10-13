"""Choose the configured Mem0 storage connection; no alternate memory engine."""
import json
from pathlib import Path
from app.memory import Mem0Adapter
from app.memory_profiles import LEGACY,WINDOWS,model_path,profile_of
from deploy.memory.runtime import load
from tools.project import ROOT,CACHE,DATA


def open_memory(directory,*,settings=None):
    settings=settings or json.loads((ROOT/'configs/runtime.yaml').read_text())
    directory=Path(directory).resolve()
    backend=settings['memory'].get('vector_store','qdrant_local')
    binding=directory/'store.json'
    service=None
    profile=settings['memory'].get('embedding_profile',LEGACY)
    if binding.exists():
        recorded=json.loads(binding.read_text())
        profile=profile_of(recorded)
        service=load(Path(recorded['service_directory']))
    elif backend=='qdrant_server':
        service=load(DATA/'qdrant')
    elif backend!='qdrant_local':raise ValueError('Unknown configured memory store')
    if profile==WINDOWS and service is None:raise ValueError('Token windows require the managed native Qdrant service')
    return Mem0Adapter(directory,bge_path=model_path(profile),embedding_backend='onnx',server_config=service,embedding_profile=profile)
