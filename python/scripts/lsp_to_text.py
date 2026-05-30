from __future__ import annotations
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cli.lsp import TriadLSP

def main(argv):
    if len(argv) < 2:
        print('usage: lsp_to_text.py <command> [args...]', file=sys.stderr)
        return 2
    cmd = argv[1]
    s = TriadLSP()
    out = None
    if cmd == 'initialize':
        out = {'capabilities': {'completionProvider': {'triggerCharacters': ['.', ' ']}, 'definitionProvider': True, 'hoverProvider': True, 'documentSymbolProvider': True, 'textDocumentSync': {'openClose': True, 'change': 1}}}
    elif cmd == 'documentSymbol' and len(argv) >= 4:
        path, uri = (argv[2], argv[3])
        s._docs[uri] = Path(path).read_text()
        out = s._document_symbols({'textDocument': {'uri': uri}})
    elif cmd == 'completion' and len(argv) >= 5:
        path, line, ch = (argv[2], int(argv[3]), int(argv[4]))
        uri = 'file://' + str(Path(path).resolve())
        s._docs[uri] = Path(path).read_text()
        out = s._completions({'textDocument': {'uri': uri}, 'position': {'line': line, 'character': ch}})
    elif cmd == 'definition' and len(argv) >= 6:
        path, line, ch, uri = (argv[2], int(argv[3]), int(argv[4]), argv[5])
        s._docs[uri] = Path(path).read_text()
        out = s._definition({'textDocument': {'uri': uri}, 'position': {'line': line, 'character': ch}})
    elif cmd == 'hover' and len(argv) >= 5:
        path, line, ch = (argv[2], int(argv[3]), int(argv[4]))
        uri = 'file://' + str(Path(path).resolve())
        s._docs[uri] = Path(path).read_text()
        out = s._hover({'textDocument': {'uri': uri}, 'position': {'line': line, 'character': ch}})
    else:
        print(f'unknown command or wrong args: {cmd}', file=sys.stderr)
        return 2
    print(json.dumps(out))
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv))