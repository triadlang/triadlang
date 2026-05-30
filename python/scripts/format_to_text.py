from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from frontend.parser_universal import parse
from compiler.formatter import format_universal

def main(argv):
    if len(argv) < 2:
        print('usage: format_to_text.py <file.tri>', file=sys.stderr)
        return 2
    path = argv[1]
    src = Path(path).read_text()
    mod = parse(src, path)
    sys.stdout.write(format_universal(mod))
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv))