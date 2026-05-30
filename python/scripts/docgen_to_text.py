from __future__ import annotations
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from compiler.docgen import parse_tri_docs, to_markdown, to_html

def main(argv):
    if len(argv) < 2:
        print('usage: docgen_to_text.py <file.tri> [--html]', file=sys.stderr)
        return 2
    path = argv[1]
    as_html = '--html' in argv[2:]
    src = Path(path).read_text()
    doc = parse_tri_docs(src)
    if as_html:
        out = to_html(doc, os.path.basename(path))
    else:
        out = to_markdown(doc, os.path.basename(path))
    sys.stdout.write(out)
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv))