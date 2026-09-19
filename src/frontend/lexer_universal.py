from __future__ import annotations

from dataclasses import dataclass

from frontend.errors import LexError as _BaseLexError

KEYWORDS = {'let', 'const', 'fn', 'return', 'if', 'else', 'elif', 'for', 'in', 'while', 'break', 'continue', 'type', 'class', 'super', 'import', 'from', 'as', 'true', 'false', 'none', 'and', 'or', 'not', 'try', 'catch', 'finally', 'throw', 'self', 'match', 'case', 'yield', 'async', 'await', 'with', 'reg', 'entity', 'world', 'couple', 'pair', 'ring', 'observe', 'OBSERVE', 'run', 'evolve', 'sequence', 'via', 'each_for', 'substrate', 'composed_of', 'assert', 'persistent', 'extended', 'structurally_open', 'mem_memory', 'atomic', 'anti_collapsed', 'over_seeds', 'is', 'pass', 'del', 'inherits'}

@dataclass
class Token:
    kind: str
    value: str
    line: int
    col: int

    def __repr__(self):
        return f'Token({self.kind}, {self.value!r}, L{self.line}:C{self.col})'

class LexError(_BaseLexError):
    pass

_REGEX_PREVENT_TOKENS = {'IDENT', 'NUMBER', 'STRING', 'BYTES', 'REGEX'}

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

    def _try_regex():
        nonlocal i, line, col, tokens
        sl, sc = line, col
        adv()
        pattern_buf = []
        escaped = False
        in_char_class = False
        while i < n:
            ch = src[i]
            if escaped:
                pattern_buf.append(adv())
                escaped = False
                continue
            if ch == '\\':
                escaped = True
                pattern_buf.append(adv())
                continue
            if ch == '[':
                in_char_class = True
                pattern_buf.append(adv())
                continue
            if ch == ']':
                in_char_class = False
                pattern_buf.append(adv())
                continue
            if ch == '/' and not in_char_class:
                adv()
                flags_buf = []
                while i < n and src[i].isalpha():
                    flags_buf.append(adv())
                tokens.append(Token('REGEX', (''.join(pattern_buf), ''.join(flags_buf)), sl, sc))
                return True
            if ch == '\n':
                return False
            pattern_buf.append(adv())
        return False

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
            adv(); adv()
            while i < n:
                if src[i] == '*' and peek(1) == '/':
                    adv(); adv()
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
        if c == 'r' and peek(1) == '"' or c == 'R' and peek(1) == '"':
            sl, sc = (line, col)
            adv(); adv()
            buf = []
            while i < n and src[i] != '"':
                if src[i] == '\n':
                    line += 1
                    col = 1
                buf.append(adv())
            if i >= n:
                raise LexError('unterminated raw string', sl, sc, file)
            adv()
            tokens.append(Token('STRING', ''.join(buf), sl, sc))
            continue
        if c == 'r' and peek(1) == 'f' and peek(2) == '"' or c == 'R' and peek(1) == 'F' and peek(2) == '"':
            sl, sc = (line, col)
            adv(); adv(); adv()
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
                if src[i] == '\n':
                    line += 1
                    col = 1
                buf.append(adv())
            if i >= n:
                raise LexError('unterminated raw f-string', sl, sc, file)
            adv()
            if buf:
                parts.append(('str', ''.join(buf)))
            tokens.append(Token('FSTRING', parts, sl, sc))
            continue
        if c == 'b' and peek(1) == '"':
            sl, sc = (line, col)
            adv(); adv()
            buf = []
            while i < n and src[i] != '"':
                if src[i] == '\\':
                    adv()
                    if i < n:
                        esc = adv()
                        buf.append({'n': '\n', 't': '\t', '\\': '\\', '"': '"', 'x': '\\x'}.get(esc, esc))
                    continue
                buf.append(adv())
            if i >= n:
                raise LexError('unterminated bytes literal', sl, sc, file)
            adv()
            tokens.append(Token('BYTES', ''.join(buf), sl, sc))
            continue
        if c == 'f' and peek(1) == '"':
            sl, sc = (line, col)
            adv(); adv()
            parts = []
            buf = []
            while i < n and src[i] != '"':
                if src[i] == '{':
                    if buf:
                        parts.append(('str', ''.join(buf)))
                        buf = []
                    adv()
                    expr_buf = []
                    fmt_buf = []
                    depth = 1
                    in_expr = True
                    bracket_depth = 0
                    while i < n and depth > 0:
                        ch = src[i]
                        if ch == '{':
                            depth += 1
                        elif ch == '}':
                            depth -= 1
                        if depth > 0:
                            if in_expr and ch == '\\' and src[i+1:i+2] in '\"\'\'':
                                adv()
                                expr_buf.append(adv())
                            elif in_expr and ch in '([{':
                                bracket_depth += 1
                            elif in_expr and ch in ')]}':
                                bracket_depth -= 1
                            if in_expr and ch == ':' and bracket_depth == 0:
                                in_expr = False
                                adv()
                                continue
                            elif in_expr and ch == '!' and bracket_depth == 0:
                                in_expr = False
                            if in_expr:
                                expr_buf.append(adv())
                            else:
                                fmt_buf.append(adv())
                        else:
                            adv()
                    expr_str = ''.join(expr_buf)
                    fmt_str = ''.join(fmt_buf).strip()
                    if fmt_str:
                        parts.append(('expr', expr_str, fmt_str))
                    else:
                        parts.append(('expr', expr_str))
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
            if peek(1) == '"' and peek(2) == '"':
                sl, sc = (line, col)
                adv(); adv(); adv()
                buf = []
                while i < n:
                    if src[i] == '"' and peek(1) == '"' and peek(2) == '"':
                        adv(); adv(); adv()
                        break
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
                tokens.append(Token('STRING', ''.join(buf), sl, sc))
                continue
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
                adv(); adv()
                while i < n and (src[i].isalnum() or src[i] == '_'):
                    adv()
            else:
                while i < n and (src[i].isdigit() or src[i] == '_'):
                    adv()
                if i < n and src[i] == '.' and (peek(1) != '.'):
                    adv()
                    while i < n and (src[i].isdigit() or src[i] == '_'):
                        adv()
                if i < n and src[i] in 'eE':
                    adv()
                    if i < n and src[i] in '+-':
                        adv()
                    while i < n and (src[i].isdigit() or src[i] == '_'):
                        adv()
            raw = src[start:i]
            clean = raw.replace('_', '')
            if i < n and src[i] in 'jJ' and clean:
                adv()
                clean = clean + src[i - 1]
            tokens.append(Token('NUMBER', clean, sl, sc))
            continue
        if c == '-' and (peek(1).isdigit() or peek(1) == '.'):
            prev = tokens[-1] if tokens else None
            if prev is None or (prev.kind in ('SYMBOL', 'KEYWORD') and prev.value not in (')', ']')):
                sl, sc = (line, col)
                start = i
                adv()
                while i < n and (src[i].isdigit() or src[i] == '.'):
                    adv()
                if i < n and src[i] in 'eE':
                    adv()
                    if i < n and src[i] in '+-':
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
        if c == '/':
            prev = tokens[-1] if tokens else None
            is_division = (prev is not None and
                           (prev.kind in _REGEX_PREVENT_TOKENS or
                            prev.value in (')', ']', '}', '++', '--')))
            if not is_division and peek(1) not in ('/', '*', '\x00', ' ', '\t', '\n'):
                save_i = i
                save_line = line
                save_col = col
                if _try_regex():
                    continue
                i = save_i
                line = save_line
                col = save_col
        three = src[i:i + 3]
        if three in ('<<=', '>>=', '**=', '//=', '??=', '||=', '&&=', '...', '..='):
            tokens.append(Token('SYMBOL', three, line, col))
            adv(); adv(); adv()
            continue
        two = src[i:i + 2]
        if two in ('==', '!=', '<=', '>=', '->', '**', '+=', '-=', '*=', '/=', '=>', '<<', '>>', '&=', '|=', '^=', '%=', '@=', '??', '?.', '|>', '//', '?:'):
            tokens.append(Token('SYMBOL', two, line, col))
            adv(); adv()
            continue
        if c == '.' and peek(1) == '.' and peek(2) != '.':
            tokens.append(Token('SYMBOL', '..', line, col))
            adv(); adv()
            continue
        if c in '+-*/%=<>(){}[];:,.!@&|^~?':
            tokens.append(Token('SYMBOL', c, line, col))
            adv()
            continue
        raise LexError(f'unexpected character {c!r}', line, col, file)
    tokens.append(Token('EOF', '', line, col))
    return tokens

