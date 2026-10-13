"""Choose the configured Mem0 storage connection; no alternate memory engine."""
import json
from pathlib import Path
from app.memory import Mem0Adapter
from deploy.memory.runtime import load
from tools.project import ROOT,CACHE,DATA


def open_memory(directory,*,settings=None):
    settings=settings or json.loads((ROOT/'configs/runtime.yaml').read_text())
    directory=Path(directory).resolve()
    backend=settings['memory'].get('vector_store','qdrant_local')
    binding=directory/'store.json'
    service=None
    if binding.exists():
        recorded=json.loads(binding.read_text())
        service=load(Path(recorded['service_directory']))
    elif backend=='qdrant_server':
        service=load(DATA/'qdrant')
    elif backend!='qdrant_local':raise ValueError('Unknown configured memory store')
    return Mem0Adapter(directory,bge_path=CACHE/'models/bge-small-zh-v1.5-onnx',embedding_backend='onnx',server_config=service)
