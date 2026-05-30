from __future__ import annotations
import json
import sys
import os
import re
from typing import Optional
_KEYWORDS = ['let', 'const', 'fn', 'return', 'if', 'elif', 'else', 'for', 'while', 'break', 'continue', 'class', 'type', 'import', 'from', 'as', 'match', 'with', 'and', 'or', 'not', 'true', 'false', 'none', 'yield', 'async', 'await', 'try', 'catch', 'finally', 'throw', 'reg', 'observe', 'run', 'couple', 'pair', 'ring', 'entity', 'world', 'in']
_STDLIB_MODULES = {'math': ['sqrt', 'sin', 'cos', 'tan', 'log', 'log10', 'exp', 'floor', 'ceil', 'abs', 'pi', 'e', 'min', 'max', 'clamp', 'pow'], 'random': ['random', 'randint', 'choice', 'seed', 'shuffle', 'uniform'], 'io': ['print', 'input'], 'string': ['split', 'join', 'replace', 'lower', 'upper', 'strip', 'starts_with', 'ends_with', 'contains'], 'json': ['parse', 'stringify'], 'fs': ['read_text', 'write_text', 'exists', 'listdir'], 'time': ['now', 'sleep'], 'collections': ['len', 'range', 'enumerate', 'sorted', 'reversed', 'zip', 'map', 'filter']}
_BUILTINS = ['len', 'range', 'str', 'int', 'float', 'abs', 'min', 'max', 'sorted', 'print', 'type', 'list', 'dict', 'enumerate']
_TRIAD_KEYWORDS = ['reg', 'observe', 'run', 'couple', 'pair', 'ring', 'entity', 'world']

class TriadLSP:

    def __init__(self):
        self._docs: dict[str, str] = {}
        self._root: Optional[str] = None

    def run(self):
        while True:
            headers = {}
            while True:
                line = sys.stdin.readline()
                if not line:
                    return
                line = line.strip()
                if not line:
                    break
                if ':' in line:
                    k, v = line.split(':', 1)
                    headers[k.strip().lower()] = v.strip()
            content_length = int(headers.get('content-length', 0))
            if content_length == 0:
                continue
            body = sys.stdin.read(content_length)
            try:
                msg = json.loads(body)
            except json.JSONDecodeError:
                continue
            self._handle(msg)

    def _handle(self, msg: dict):
        method = msg.get('method', '')
        params = msg.get('params', {})
        msg_id = msg.get('id')
        result = None
        if method == 'initialize':
            result = {'capabilities': {'completionProvider': {'triggerCharacters': ['.', ' ']}, 'definitionProvider': True, 'hoverProvider': True, 'documentSymbolProvider': True, 'textDocumentSync': {'openClose': True, 'change': 1}}}
        elif method == 'initialized':
            self._send_response(msg_id, None)
            return
        elif method == 'textDocument/didOpen':
            uri = params.get('textDocument', {}).get('uri', '')
            text = params.get('textDocument', {}).get('text', '')
            self._docs[uri] = text
            self._send_response(msg_id, None)
            return
        elif method == 'textDocument/didChange':
            uri = params.get('textDocument', {}).get('uri', '')
            changes = params.get('contentChanges', [])
            if changes:
                self._docs[uri] = changes[0].get('text', '')
            self._send_response(msg_id, None)
            return
        elif method == 'textDocument/didClose':
            uri = params.get('textDocument', {}).get('uri', '')
            self._docs.pop(uri, None)
            self._send_response(msg_id, None)
            return
        elif method == 'textDocument/completion':
            result = self._completions(params)
        elif method == 'textDocument/definition':
            result = self._definition(params)
        elif method == 'textDocument/hover':
            result = self._hover(params)
        elif method == 'textDocument/documentSymbol':
            result = self._document_symbols(params)
        elif method == 'textDocument/publishDiagnostics' or method == 'shutdown' or method == 'exit':
            self._send_response(msg_id, None)
            return
        self._send_response(msg_id, result)

    def _send_response(self, msg_id, result):
        if msg_id is None:
            return
        body = json.dumps({'jsonrpc': '2.0', 'id': msg_id, 'result': result})
        sys.stdout.write(f'Content-Length: {len(body)}\r\n\r\n{body}')
        sys.stdout.flush()

    def _send_notification(self, method: str, params: dict):
        body = json.dumps({'jsonrpc': '2.0', 'method': method, 'params': params})
        sys.stdout.write(f'Content-Length: {len(body)}\r\n\r\n{body}')
        sys.stdout.flush()

    def _word_at(self, line: str, char: int) -> Optional[str]:
        if char >= len(line):
            char = len(line) - 1
        if char < 0:
            return None
        if not line[char].isalnum() and line[char] != '_':
            if char > 0 and (line[char - 1].isalnum() or line[char - 1] == '_'):
                char -= 1
            else:
                return None
        start = char
        while start > 0 and (line[start - 1].isalnum() or line[start - 1] == '_'):
            start -= 1
        end = char + 1
        while end < len(line) and (line[end].isalnum() or line[end] == '_'):
            end += 1
        return line[start:end]

    def _parse_symbols(self, text: str) -> list[dict]:
        symbols = []
        for i, line in enumerate(text.split('\n')):
            stripped = line.lstrip()
            m_kw = re.match('^(let|const)\\s+(\\w+)', stripped)
            if m_kw:
                symbols.append({'name': m_kw.group(2), 'kind': 'variable', 'line': i, 'col': len(line) - len(stripped)})
                continue
            m_fn = re.match('^fn\\s+(\\w+)', stripped)
            if m_fn:
                symbols.append({'name': m_fn.group(1), 'kind': 'function', 'line': i, 'col': len(line) - len(stripped)})
                continue
            m_cls = re.match('^class\\s+(\\w+)', stripped)
            if m_cls:
                symbols.append({'name': m_cls.group(1), 'kind': 'class', 'line': i, 'col': len(line) - len(stripped)})
                continue
            m_type = re.match('^type\\s+(\\w+)', stripped)
            if m_type:
                symbols.append({'name': m_type.group(1), 'kind': 'class', 'line': i, 'col': len(line) - len(stripped)})
                continue
        return symbols

    def _completions(self, params: dict) -> dict:
        uri = params.get('textDocument', {}).get('uri', '')
        pos = params.get('position', {})
        line_num = pos.get('line', 0)
        char = pos.get('character', 0)
        text = self._docs.get(uri, '')
        lines = text.split('\n')
        if line_num >= len(lines):
            return {'isIncomplete': False, 'items': []}
        current_line = lines[line_num]
        prefix = current_line[:char]
        items = []
        dot_match = re.search('(\\w+)\\.(\\w*)$', prefix)
        if dot_match:
            module_name = dot_match.group(1)
            partial = dot_match.group(2)
            if module_name in _STDLIB_MODULES:
                for fn_name in _STDLIB_MODULES[module_name]:
                    if fn_name.startswith(partial):
                        items.append({'label': fn_name, 'kind': 3, 'detail': f'{module_name}.{fn_name}'})
                return {'isIncomplete': False, 'items': items}
        word_match = re.search('(\\w+)$', prefix)
        partial = word_match.group(1) if word_match else ''
        if not partial and (not prefix.endswith(' ')):
            return {'isIncomplete': False, 'items': []}
        symbols = self._parse_symbols(text)
        local_names = set()
        for sym in symbols:
            if sym['name'].startswith(partial):
                items.append({'label': sym['name'], 'kind': 13 if sym['kind'] == 'variable' else 12 if sym['kind'] == 'function' else 7, 'detail': sym['kind']})
            local_names.add(sym['name'])
        if not prefix.endswith('.'):
            for kw in _KEYWORDS:
                if kw.startswith(partial) and kw not in local_names:
                    items.append({'label': kw, 'kind': 14, 'detail': 'keyword'})
            for b in _BUILTINS:
                if b.startswith(partial) and b not in local_names:
                    items.append({'label': b, 'kind': 3, 'detail': 'builtin'})
            for mod in _STDLIB_MODULES:
                if mod.startswith(partial):
                    items.append({'label': mod, 'kind': 9, 'detail': 'module'})
            for tk in _TRIAD_KEYWORDS:
                if tk.startswith(partial) and tk not in local_names:
                    items.append({'label': tk, 'kind': 14, 'detail': 'triad'})
        return {'isIncomplete': False, 'items': items}

    def _definition(self, params: dict) -> Optional[dict]:
        uri = params.get('textDocument', {}).get('uri', '')
        pos = params.get('position', {})
        line_num = pos.get('line', 0)
        char = pos.get('character', 0)
        text = self._docs.get(uri, '')
        lines = text.split('\n')
        if line_num >= len(lines):
            return None
        current_line = lines[line_num]
        word = self._word_at(current_line, char)
        if not word:
            return None
        for i, line in enumerate(text.split('\n')):
            stripped = line.lstrip()
            col = len(line) - len(stripped)
            if re.match(f'^(let|const)\\s+{re.escape(word)}\\b', stripped):
                return {'uri': uri, 'range': {'start': {'line': i, 'character': col}, 'end': {'line': i, 'character': col + len(stripped)}}}
            if re.match(f'^fn\\s+{re.escape(word)}\\b', stripped):
                return {'uri': uri, 'range': {'start': {'line': i, 'character': col}, 'end': {'line': i, 'character': col + len(stripped)}}}
            if re.match(f'^class\\s+{re.escape(word)}\\b', stripped):
                return {'uri': uri, 'range': {'start': {'line': i, 'character': col}, 'end': {'line': i, 'character': col + len(stripped)}}}
            if re.match(f'^type\\s+{re.escape(word)}\\b', stripped):
                return {'uri': uri, 'range': {'start': {'line': i, 'character': col}, 'end': {'line': i, 'character': col + len(stripped)}}}
        return None

    def _hover(self, params: dict) -> Optional[dict]:
        uri = params.get('textDocument', {}).get('uri', '')
        pos = params.get('position', {})
        line_num = pos.get('line', 0)
        char = pos.get('character', 0)
        text = self._docs.get(uri, '')
        lines = text.split('\n')
        if line_num >= len(lines):
            return None
        current_line = lines[line_num]
        word = self._word_at(current_line, char)
        if not word:
            return None
        if word in _STDLIB_MODULES:
            fns = ', '.join(_STDLIB_MODULES[word])
            return {'contents': {'kind': 'markdown', 'value': f'**module {word}**\n\nExports: `{fns}`'}}
        for mod, fns in _STDLIB_MODULES.items():
            if word in fns:
                return {'contents': {'kind': 'markdown', 'value': f'**{mod}.{word}**\n\nStdlib function from `{mod}`'}}
        symbols = self._parse_symbols(text)
        for sym in symbols:
            if sym['name'] == word:
                return {'contents': {'kind': 'markdown', 'value': f"**{word}** ({sym['kind']})\n\nDefined at line {sym['line'] + 1}"}}
        if word in _KEYWORDS:
            return {'contents': {'kind': 'markdown', 'value': f'**{word}** — keyword'}}
        return None

    def _document_symbols(self, params: dict) -> list:
        uri = params.get('textDocument', {}).get('uri', '')
        text = self._docs.get(uri, '')
        symbols = self._parse_symbols(text)
        result = []
        kind_map = {'variable': 6, 'function': 12, 'class': 5}
        for sym in symbols:
            result.append({'name': sym['name'], 'kind': kind_map.get(sym['kind'], 6), 'location': {'uri': uri, 'range': {'start': {'line': sym['line'], 'character': sym['col']}, 'end': {'line': sym['line'], 'character': sym['col'] + len(sym['name'])}}}})
        return result

def main():
    server = TriadLSP()
    server.run()
if __name__ == '__main__':
    main()