from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from frontend.parser import parse
from compiler.typecheck import typecheck, TypeCheckError
from stdlib.regimes import list_regimes

def main(argv):
    if len(argv) < 2:
        print('usage: typecheck_legacy_to_text.py <file.tri>', file=sys.stderr)
        return 2
    path = argv[1]
    src = Path(path).read_text()
    prog = parse(src)
    registry = set(list_regimes())
    try:
        typecheck(prog, registry)
    except TypeCheckError as exc:
        for e in exc.errors:
            print(e)
        return 1
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv))