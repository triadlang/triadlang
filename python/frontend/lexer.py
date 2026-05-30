from __future__ import annotations
from dataclasses import dataclass
KEYWORDS = {'reg', 'loop', 'segment', 'if', 'else', 'true', 'false', 'OUT', 'HALT', 'evolve', 'for', 'couple', 'pair', 'ring', 'sequence', 'via', 'to', 'kappa', 'each_for', 'OBSERVE', 'over_seeds', 'assert', 'persistent', 'extended', 'structurally_open', 'non_trivial_memory', 'atomic', 'anti_collapsed', 'substrate', 'composed_of'}
OPCODES = {'MOV', 'LOAD', 'ADD', 'SUB', 'MUL', 'DIV', 'AND', 'OR', 'NOT', 'XOR', 'CMP', 'INC', 'DEC', 'ZERO', 'SHIFT_LEFT', 'SHIFT_RIGHT', 'PROBE', 'CHECKPOINT'}
LEGACY_KEYWORDS = {'regime', 'register', 'init', 'evolve', 'mov', 'for', 'with', 'kappa', 'read', 'let', 'print', 'none', 'harmonic', 'mode'}

@dataclass
class Token:
    kind: str
    value: str
    line: int
    col: int

    def __repr__(self):
        return f'Token({self.kind}, {self.value!r}, L{self.line}:C{self.col})'

class LexError(Exception):
    pass

def tokenize(src: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    n = len(src)
    line = 1
    col = 1

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
        if c == '/' and i + 1 < n and (src[i + 1] == '/'):
            while i < n and src[i] != '\n':
                adv()
            continue
        if c == '#':
            while i < n and src[i] != '\n':
                adv()
            continue
        if c == '@':
            sl, sc = (line, col)
            adv()
            start = i
            while i < n and (src[i].isalnum() or src[i] == '_'):
                adv()
            ident = src[start:i]
            arg = ''
            if i < n and src[i] == '(':
                adv()
                depth = 1
                buf_start = i
                while i < n and depth > 0:
                    if src[i] == '(':
                        depth += 1
                    elif src[i] == ')':
                        depth -= 1
                        if depth == 0:
                            break
                    if src[i] == '\n':
                        line += 1
                        col = 1
                    adv()
                arg = src[buf_start:i]
                if i < n and src[i] == ')':
                    adv()
            tokens.append(Token('ANNOT', f'{ident}({arg})', sl, sc))
            continue
        if c.isdigit() or (c == '-' and i + 1 < n and src[i + 1].isdigit()):
            sl, sc = (line, col)
            start = i
            is_neg = c == '-'
            if c == '-':
                adv()
            if i + 1 < n and src[i] == '0' and (src[i + 1] in 'bBxX'):
                base_char = src[i + 1].lower()
                adv()
                adv()
                hex_or_bin_start = i
                allowed = '01' if base_char == 'b' else '0123456789abcdefABCDEF'
                while i < n and src[i] in allowed:
                    adv()
                num_str = src[hex_or_bin_start:i]
                if base_char == 'b':
                    val = int(num_str, 2)
                else:
                    val = int(num_str, 16)
                if is_neg:
                    val = -val
                tokens.append(Token('NUMBER', str(val), sl, sc))
                continue
            while i < n and src[i].isdigit():
                adv()
            if i < n and src[i] == '.':
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
        if c.isalpha() or c == '_':
            sl, sc = (line, col)
            start = i
            while i < n and (src[i].isalnum() or src[i] == '_'):
                adv()
            value = src[start:i]
            if value in OPCODES:
                kind = 'OPCODE'
            elif value in KEYWORDS or value in LEGACY_KEYWORDS:
                kind = 'KEYWORD'
            else:
                kind = 'IDENT'
            tokens.append(Token(kind, value, sl, sc))
            continue
        if c == '"':
            sl, sc = (line, col)
            adv()
            start = i
            while i < n and src[i] != '"':
                if src[i] == '\n':
                    line += 1
                    col = 1
                adv()
            if i >= n:
                raise LexError(f'unterminated string at L{sl}:C{sc}')
            val = src[start:i]
            adv()
            tokens.append(Token('STRING', val, sl, sc))
            continue
        if c == ':' and i + 1 < n and (src[i + 1] == '='):
            tokens.append(Token('SYMBOL', ':=', line, col))
            adv()
            adv()
            continue
        if c == '-' and i + 1 < n and (src[i + 1] == '>'):
            tokens.append(Token('SYMBOL', '->', line, col))
            adv()
            adv()
            continue
        if c == '=' and i + 1 < n and (src[i + 1] == '='):
            tokens.append(Token('SYMBOL', '==', line, col))
            adv()
            adv()
            continue
        if c == '<' and i + 1 < n and (src[i + 1] == '='):
            tokens.append(Token('SYMBOL', '<=', line, col))
            adv()
            adv()
            continue
        if c == '>' and i + 1 < n and (src[i + 1] == '='):
            tokens.append(Token('SYMBOL', '>=', line, col))
            adv()
            adv()
            continue
        if c in ';,(){}[]:=<>+-*/':
            tokens.append(Token('SYMBOL', c, line, col))
            adv()
            continue
        raise LexError(f'unexpected character {c!r} at L{line}:C{col}')
    tokens.append(Token('EOF', '', line, col))
    return tokens
if __name__ == '__main__':
    import sys
    src = open(sys.argv[1]).read() if len(sys.argv) > 1 else '\n@T(10.0)\nreg n = 5;\nreg acc = 1;\n\nloop count {\n    CMP cond, n, ONE;\n    if cond {\n        MUL tmp, acc, n;\n        MOV acc, tmp;\n        SUB n, n, ONE;\n    } else {\n        HALT;\n    }\n}\nOUT acc;\n'
    for t in tokenize(src):
        print(t)