from __future__ import annotations


class TriadLangError(Exception):
    pass

class LexError(TriadLangError):
    def __init__(self, msg, line=0, col=0, file=''):
        self.msg = msg
        self.line = line
        self.col = col
        self.file = file
        super().__init__(self._format())

    def _format(self):
        parts = ['error[LEX]:']
        parts.append(f' {self.msg}')
        if self.file:
            parts.append(f'\n  file: {self.file}')
        if self.line:
            parts.append(f'\n  line: {self.line}')
        if self.col:
            parts.append(f'\n  col: {self.col}')
        return ''.join(parts)

class ParseError(TriadLangError):
    def __init__(self, msg, line=0, col=0, file='', source='', hint=''):
        self.msg = msg
        self.line = line
        self.col = col
        self.file = file
        self.source = source
        self.hint = hint
        super().__init__(self._format())

    def _format(self):
        if self.source:
            from frontend.diagnostics import format_code_error
            return format_code_error(self.file or '<triad>', self.source, self.line, self.col, self.msg, hint=self.hint, code='PARSE')
        parts = [f'error[PARSE]: {self.msg}']
        if self.file:
            parts.append(f'\n  file: {self.file}')
        if self.line:
            parts.append(f'\n  line: {self.line}')
        if self.col:
            parts.append(f'\n  col: {self.col}')
        return ''.join(parts)

class TypeCheckError(TriadLangError):
    pass

class CCompileError(TriadLangError):
    pass

