from __future__ import annotations
import re
from typing import Optional

def optimize_source(source: str, level: int=2) -> str:
    lines = source.split('\n')
    lines = _fold_constants(lines)
    if level >= 2:
        lines = _dead_code_elimination(lines)
        lines = _fold_constants(lines)
    return '\n'.join(lines)
_CONST_ASSIGN_RE = re.compile('^(\\s*)(\\w[\\w.]*\\s*=\\s*)(.+)\\s*$')
_NUM_LIT = re.compile('^-?\\d+\\.?\\d*(?:[eE][+-]?\\d+)?$')
_BIN_OP_RE = re.compile('\\s*(\\*\\*|[+\\-*/%])\\s*')

def _try_fold_expr(expr: str) -> Optional[str]:
    expr = expr.strip().strip('()')
    for m in _BIN_OP_RE.finditer(expr):
        left_s = expr[:m.start()].strip()
        op = m.group(1)
        right_s = expr[m.end():].strip()
        if not left_s or not right_s:
            continue
        if not _NUM_LIT.match(left_s) or not _NUM_LIT.match(right_s):
            continue
        try:
            if '.' in left_s or '.' in right_s or 'e' in left_s.lower() or ('e' in right_s.lower()):
                left, right = (float(left_s), float(right_s))
            else:
                left, right = (int(left_s), int(right_s))
        except ValueError:
            continue
        try:
            if op == '+':
                result = left + right
            elif op == '-':
                result = left - right
            elif op == '*':
                result = left * right
            elif op == '/':
                if right == 0:
                    return None
                result = left / right
            elif op == '%':
                if right == 0:
                    return None
                result = left % right
            elif op == '**':
                if abs(right) > 100:
                    return None
                result = left ** right
            else:
                continue
        except (OverflowError, ZeroDivisionError):
            return None
        if isinstance(result, float):
            if result == int(result) and abs(result) < 1000000000000000.0 and isinstance(left, int) and isinstance(right, int) and (op != '/'):
                return str(int(result))
            return repr(result)
        return repr(result)
    return None

def _fold_constants(lines: list[str]) -> list[str]:
    result = []
    for line in lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith('#') or stripped.startswith('def ') or stripped.startswith('class '):
            result.append(line)
            continue
        if stripped.startswith('if ') or stripped.startswith('elif ') or stripped.startswith('while '):
            result.append(line)
            continue
        m = _CONST_ASSIGN_RE.match(line)
        if m:
            indent = m.group(1)
            prefix = m.group(2)
            expr_part = m.group(3)
            folded = _try_fold_expr(expr_part)
            if folded is not None:
                result.append(f'{indent}{prefix}{folded}')
                continue
        result.append(line)
    return result

def _dead_code_elimination(lines: list[str]) -> list[str]:
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if stripped == 'return' or stripped.startswith(('return ', 'break', 'continue', 'raise ')):
            result.append(line)
            i += 1
            indent = len(line) - len(stripped)
            while i < len(lines):
                next_stripped = lines[i].lstrip()
                if not next_stripped:
                    result.append(lines[i])
                    i += 1
                    continue
                next_indent = len(lines[i]) - len(next_stripped)
                if next_indent >= indent and next_stripped:
                    i += 1
                else:
                    break
            continue
        if_re = re.match('^(\\s*)if\\s+(False|0)\\s*:', line)
        if if_re:
            indent = len(if_re.group(1))
            i += 1
            while i < len(lines):
                next_stripped = lines[i].lstrip()
                if not next_stripped:
                    i += 1
                    continue
                next_indent = len(lines[i]) - len(next_stripped)
                if next_indent > indent:
                    i += 1
                else:
                    break
            continue
        if_true = re.match('^(\\s*)if\\s+(True|1)\\s*:', line)
        if if_true:
            block_indent = len(if_true.group(1)) + 4
            i += 1
            while i < len(lines):
                next_stripped = lines[i].lstrip()
                if not next_stripped:
                    result.append(lines[i])
                    i += 1
                    continue
                next_indent = len(lines[i]) - len(next_stripped)
                if next_indent == block_indent:
                    result.append(lines[i][4:] if next_indent >= 4 else lines[i])
                    i += 1
                elif next_indent > block_indent:
                    result.append(lines[i][4:] if len(lines[i]) >= 4 else lines[i])
                    i += 1
                else:
                    nstripped = lines[i].lstrip()
                    if nstripped.startswith('elif ') or nstripped.startswith('else:'):
                        pindent = len(lines[i]) - len(nstripped)
                        i += 1
                        while i < len(lines):
                            ns2 = lines[i].lstrip()
                            if not ns2:
                                i += 1
                                continue
                            ni2 = len(lines[i]) - len(ns2)
                            if ni2 > pindent:
                                i += 1
                            else:
                                break
                    else:
                        result.append(lines[i])
                        i += 1
                    break
            continue
        if_not_false = re.match('^(\\s*)if\\s+not\\s+False\\s*:', line)
        if if_not_false:
            block_indent = len(if_not_false.group(1)) + 4
            i += 1
            while i < len(lines):
                next_stripped = lines[i].lstrip()
                if not next_stripped:
                    result.append(lines[i])
                    i += 1
                    continue
                next_indent = len(lines[i]) - len(next_stripped)
                if next_indent >= block_indent:
                    dedented = lines[i][4:] if len(lines[i]) >= 4 else lines[i]
                    result.append(dedented)
                    i += 1
                else:
                    nstripped = lines[i].lstrip()
                    if nstripped.startswith('elif ') or nstripped.startswith('else:'):
                        pindent = len(lines[i]) - len(nstripped)
                        i += 1
                        while i < len(lines):
                            ns2 = lines[i].lstrip()
                            if not ns2:
                                i += 1
                                continue
                            ni2 = len(lines[i]) - len(ns2)
                            if ni2 > pindent:
                                i += 1
                            else:
                                break
                    else:
                        result.append(lines[i])
                        i += 1
                    break
            continue
        result.append(line)
        i += 1
    return result