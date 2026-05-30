from __future__ import annotations
import json
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from frontend.parser import parse

def serialize(node):
    if node is None:
        return None
    if isinstance(node, bool):
        return node
    if isinstance(node, (int, float, str)):
        return node
    if isinstance(node, list):
        return [serialize(x) for x in node]
    if isinstance(node, tuple):
        return [serialize(x) for x in node]
    if isinstance(node, dict):
        return {k: serialize(v) for k, v in node.items()}
    if is_dataclass(node):
        out = {'_type': type(node).__name__}
        for f in fields(node):
            out[f.name] = serialize(getattr(node, f.name))
        return out
    return str(node)

def dump(path: str) -> str:
    src = Path(path).read_text()
    prog = parse(src)
    return json.dumps(serialize(prog), indent=2)

def main(argv):
    if len(argv) < 2:
        print('usage: ast_to_json_legacy.py <file.tri>', file=sys.stderr)
        return 2
    print(dump(argv[1]))
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv))