from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from frontend.parser_universal import parse
from compiler.lower import lower_module
from compiler.emit_json import emit_json

def dump(path: str) -> str:
    src = Path(path).read_text()
    mod = parse(src, path)
    ir = lower_module(mod)
    return emit_json(ir)

def main(argv):
    if len(argv) < 2:
        print('usage: ir_to_json.py <file.tri>', file=sys.stderr)
        return 2
    print(dump(argv[1]))
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv))