from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

def _typecheck_diagnostics(src: str):
    try:
        from frontend.parser import parse, ParseError
        from compiler.typecheck import typecheck, TypeCheckError
        from stdlib.regimes import list_regimes
    except Exception as e:
        return [(0, 0, f'server init error: {e}')]
    try:
        prog = parse(src)
    except ParseError as e:
        return [(0, 0, f'parse error: {e}')]
    try:
        typecheck(prog, set(list_regimes()))
        return []
    except TypeCheckError as e:
        out = []
        for line in e.errors:
            ln = 0
            if line.startswith('L') and ':' in line:
                try:
                    ln = int(line.split(':')[0][1:])
                except Exception:
                    pass
            out.append((max(0, ln - 1), 0, line))
        return out

def main():
    try:
        from pygls.server import LanguageServer
        from lsprotocol.types import TEXT_DOCUMENT_DID_OPEN, TEXT_DOCUMENT_DID_CHANGE, DidOpenTextDocumentParams, DidChangeTextDocumentParams, Diagnostic, DiagnosticSeverity, Range, Position, COMPLETION, CompletionParams, CompletionList, CompletionItem, CompletionItemKind
    except ImportError:
        print('pygls / lsprotocol not installed. Run: pip install pygls', file=sys.stderr)
        sys.exit(1)
    server = LanguageServer('triadlang-lsp', 'v0.1')

    def publish(ls, uri, src):
        diags = []
        for ln, col, msg in _typecheck_diagnostics(src):
            diags.append(Diagnostic(range=Range(start=Position(line=ln, character=col), end=Position(line=ln, character=col + 80)), message=msg, severity=DiagnosticSeverity.Error, source='triadlang'))
        ls.publish_diagnostics(uri, diags)

    @server.feature(TEXT_DOCUMENT_DID_OPEN)
    def on_open(ls, params: 'DidOpenTextDocumentParams'):
        publish(ls, params.text_document.uri, params.text_document.text)

    @server.feature(TEXT_DOCUMENT_DID_CHANGE)
    def on_change(ls, params: 'DidChangeTextDocumentParams'):
        doc = ls.workspace.get_document(params.text_document.uri)
        publish(ls, params.text_document.uri, doc.source)

    @server.feature(COMPLETION)
    def on_completion(ls, params: 'CompletionParams'):
        from stdlib.regimes import list_regimes
        items = []
        for r in list_regimes():
            items.append(CompletionItem(label=r, kind=CompletionItemKind.Class, detail='TriadLang regime'))
        for kw in ('evolve', 'couple', 'pair', 'ring', 'sequence', 'OBSERVE', 'assert', 'substrate', 'composed_of'):
            items.append(CompletionItem(label=kw, kind=CompletionItemKind.Keyword))
        return CompletionList(is_incomplete=False, items=items)
    server.start_io()
if __name__ == '__main__':
    main()