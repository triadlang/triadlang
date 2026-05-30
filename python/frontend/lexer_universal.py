from __future__ import annotations
from dataclasses import dataclass
KEYWORDS = {'let', 'const', 'fn', 'return', 'if', 'else', 'elif', 'for', 'in', 'while', 'break', 'continue', 'type', 'class', 'super', 'import', 'from', 'as', 'true', 'false', 'none', 'and', 'or', 'not', 'try', 'catch', 'finally', 'throw', 'self', 'match', 'case', 'yield', 'async', 'await', 'reg', 'entity', 'world', 'couple', 'pair', 'ring', 'observe', 'OBSERVE', 'run', 'evolve', 'sequence', 'via', 'each_for', 'substrate', 'composed_of', 'assert', 'persistent', 'extended', 'structurally_open', 'non_trivial_memory', 'atomic', 'anti_collapsed', 'over_seeds'}

@dataclass
class Token:
    kind: str
    value: str
    line: int
    col: int

    def __repr__(self):
        return f'Token({self.kind}, {self.value!r}, L{self.line}:C{self.col})'

class LexError(Exception):

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

def tokenize(src: str, file: str='') -> list[Token]:
    tokens: list[Token] = []
    i = 0
    n = len(src)
    line = 1
    col = 1

    def peek(off=0):
        p = i + off
        return src[p] if p < n else '\x00'

    def adv():
        nonlocal i, col
        c = src[i]
        i += 1
        col += 1
        return c
    while i < n:
        c = src[i]
        if c == '\n':
            adv()
            line += 1
            col = 1
            continue
        if c in ' \t\r':
            adv()
            continue
        if c == '/' and peek(1) == '/':
            while i < n and src[i] != '\n':
                adv()
            continue
        if c == '/' and peek(1) == '*':
            sl, sc = (line, col)
            adv()
            adv()
            while i < n:
                if src[i] == '*' and peek(1) == '/':
                    adv()
                    adv()
                    break
                if src[i] == '\n':
                    adv()
                    line += 1
                    col = 1
                else:
                    adv()
            else:
                raise LexError('unterminated block comment', sl, sc, file)
            continue
        if c == '#':
            while i < n and src[i] != '\n':
                adv()
            continue
        if c == 'f' and peek(1) == '"':
            sl, sc = (line, col)
            adv()
            adv()
            parts = []
            buf = []
            while i < n and src[i] != '"':
                if src[i] == '{':
                    if buf:
                        parts.append(('str', ''.join(buf)))
                        buf = []
                    adv()
                    expr_buf = []
                    depth = 1
                    while i < n and depth > 0:
                        if src[i] == '{':
                            depth += 1
                        elif src[i] == '}':
                            depth -= 1
                        if depth > 0:
                            expr_buf.append(adv())
                        else:
                            adv()
                    parts.append(('expr', ''.join(expr_buf)))
                    continue
                if src[i] == '\\':
                    adv()
                    if i < n:
                        esc = adv()
                        buf.append({'n': '\n', 't': '\t', '\\': '\\', '"': '"'}.get(esc, esc))
                    continue
                if src[i] == '\n':
                    line += 1
                    col = 1
                buf.append(adv())
            if i >= n:
                raise LexError('unterminated f-string', sl, sc, file)
            adv()
            if buf:
                parts.append(('str', ''.join(buf)))
            tokens.append(Token('FSTRING', parts, sl, sc))
            continue
        if c == '"':
            sl, sc = (line, col)
            adv()
            buf = []
            while i < n and src[i] != '"':
                if src[i] == '\\':
                    adv()
                    if i < n:
                        esc = adv()
                        buf.append({'n': '\n', 't': '\t', '\\': '\\', '"': '"'}.get(esc, esc))
                    continue
                if src[i] == '\n':
                    line += 1
                    col = 1
                buf.append(adv())
            if i >= n:
                raise LexError('unterminated string', sl, sc, file)
            adv()
            tokens.append(Token('STRING', ''.join(buf), sl, sc))
            continue
        if c.isdigit() or (c == '.' and peek(1).isdigit()):
            sl, sc = (line, col)
            start = i
            if c == '0' and peek(1) in 'xXbB':
                adv()
                adv()
                while i < n and (src[i].isalnum() or src[i] == '_'):
                    adv()
            else:
                while i < n and src[i].isdigit():
                    adv()
                if i < n and src[i] == '.' and (peek(1) != '.'):
                    adv()
                    while i < n and src[i].isdigit():
                        adv()
                if i < n and src[i] in 'eE':
                    adv()
                    if i < n and src[i] in '+-':
                        adv()
                    while i < n and src[i].isdigit():
                        adv()
            tokens.append(Token('NUMBER', src[start:i], sl, sc))
            continue
        if c == '-' and peek(1).isdigit():
            prev = tokens[-1] if tokens else None
            if prev is None or (prev.kind in ('SYMBOL', 'KEYWORD') and prev.value not in (')', ']')):
                sl, sc = (line, col)
                start = i
                adv()
                while i < n and src[i].isdigit():
                    adv()
                if i < n and src[i] == '.':
                    adv()
                    while i < n and src[i].isdigit():
                        adv()
                tokens.append(Token('NUMBER', src[start:i], sl, sc))
                continue
        if c.isalpha() or c == '_':
            sl, sc = (line, col)
            start = i
            while i < n and (src[i].isalnum() or src[i] == '_'):
                adv()
            val = src[start:i]
            kind = 'KEYWORD' if val in KEYWORDS else 'IDENT'
            tokens.append(Token(kind, val, sl, sc))
            continue
        two = src[i:i + 2]
        if two in ('==', '!=', '<=', '>=', '->', '**', '+=', '-=', '*=', '/=', '=>'):
            tokens.append(Token('SYMBOL', two, line, col))
            adv()
            adv()
            continue
        if c in '+-*/%=<>(){}[];:,.!@&|^~':
            tokens.append(Token('SYMBOL', c, line, col))
            adv()
            continue
        raise LexError(f'unexpected character {c!r}', line, col, file)
    tokens.append(Token('EOF', '', line, col))
    return tokens