"""Application directory cutover fence, checked while each store owner is held."""
import json
from pathlib import Path

MARKER='data-transition.json'


def read_state(directory):
    directory=Path(directory).resolve();path=directory/MARKER
    if not path.exists():return None
    state=json.loads(path.read_text())
    if state.get('schema_version')!=1 or state.get('directory')!=str(directory):
        raise RuntimeError('Application data transition binding is invalid')
    return state


def check_available(directory):
    state=read_state(directory)
    if state is None:return
    if state.get('status')=='RETIRED':
        raise RuntimeError('Application data directory is retired; use '+state['target'])
    if state.get('status')!='ACTIVE':
        raise RuntimeError('Application data restoration is incomplete; resume restore-data')
    path=Path(directory)/'memory/store.json'
    if not path.is_file() or json.loads(path.read_text())!=dict(state['binding'],status='READY'):
        raise RuntimeError('Restored application memory binding differs')
