"""Use pinned VLMEvalKit exact-matching routines without package side effects.

Only the six named pure scoring functions and their regex are compiled. This
avoids the upstream package initializer's global TLS override/.env loading and
does not load cloud judges, model registries or optional GPU backends.
"""
import ast
import copy
import json
import logging
from pathlib import Path
import string
import re
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]


def load_scorer():
    import pandas as pd
    from tools.project import sha256
    lock=json.loads((ROOT/'deploy/evaluation/mmbench.lock.json').read_text())
    source=Path(lock['toolkit_directory'])
    namespace={'pd':pd,'cp':copy,'re':re,'string':string,'logger':logging.getLogger('mmbench-local'),
        'os':SimpleNamespace(environ={})}
    selections={
        'vlmeval/utils/matching_util.py':{'can_infer_option','can_infer_text','can_infer'},
        'vlmeval/dataset/utils/multiple_choice.py':{'build_choices','prefetch_answer','prefetch_circular_group'}}
    for relative,names in selections.items():
        path=source/relative
        if sha256(path)!=lock['source_sha256'][relative]:raise RuntimeError('Upstream scorer changed')
        tree=ast.parse(path.read_text());nodes=[];found=set()
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in names:
                nodes.append(node);found.add(node.name)
            elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_VERBOSE_ANSWER_RE' for t in node.targets):
                nodes.append(node)
        if found!=names:raise RuntimeError('Missing pinned scorer function')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def evaluate_group(rows):
    import pandas as pd
    scorer=load_scorer()
    data=pd.DataFrame(rows)
    parsed=[scorer['prefetch_answer'](row) or 'Z' for _,row in data.iterrows()]
    result=scorer['prefetch_circular_group'](data)
    return {'parsed':parsed,'circular_correct':bool(result and result['hit']),
        'variant_correct':[p==r['GT'] for p,r in zip(parsed,rows)]}
