from __future__ import annotations
import json
from dataclasses import asdict, fields, is_dataclass
from compiler.ir import *

def _serialize(node):
    if node is None:
        return None
    if isinstance(node, (int, float, str, bool)):
        return node
    if isinstance(node, list):
        return [_serialize(x) for x in node]
    if isinstance(node, tuple):
        return [_serialize(x) for x in node]
    if isinstance(node, dict):
        return {k: _serialize(v) for k, v in node.items()}
    if is_dataclass(node):
        d = {'_type': type(node).__name__}
        for f in fields(node):
            d[f.name] = _serialize(getattr(node, f.name))
        return d
    return str(node)

def emit_json(ir_module: IRModule, indent: int=2) -> str:
    return json.dumps(_serialize(ir_module), indent=indent)

def emit_json_file(ir_module: IRModule, path: str, indent: int=2):
    with open(path, 'w') as f:
        f.write(emit_json(ir_module, indent))