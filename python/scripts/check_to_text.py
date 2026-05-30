from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from frontend.parser_universal import parse
from compiler.typecheck_universal import typecheck, TypeCheckError

def main(argv):
    if len(argv) < 2:
        print('usage: check_to_text.py <file.tri>', file=sys.stderr)
        return 2
    path = argv[1]
    src = Path(path).read_text()
    mod = parse(src, path)
    try:
        typecheck(mod)
    except TypeCheckError as exc:
        for e in exc.errors:
            print(e)
        return 1
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv))