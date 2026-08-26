from __future__ import annotations

import copy

from compiler.ir import (
    IRAssign,
    IRBinOp,
    IRBool,
    IRCall,
    IRConst,
    IRExprStmt,
    IRField,
    IRFloat,
    IRFor,
    IRFunction,
    IRIf,
    IRIndex,
    IRInt,
    IRLet,
    IRList,
    IRModule,
    IRReturn,
    IRString,
    IRUnaryOp,
    IRWhile,
)


def _is_int(n):
    return isinstance(n, IRInt)

def _is_float(n):
    return isinstance(n, IRFloat)

def _num(n):
    return n.value

def _fold_binop(node: IRBinOp):
    left = fold_expr(node.left)
    right = fold_expr(node.right)
    folded = IRBinOp(op=node.op, left=left, right=right)

    if (_is_int(left) or _is_float(left)) and (_is_int(right) or _is_float(right)):
        a, b, op = _num(left), _num(right), node.op
        both_int = _is_int(left) and _is_int(right)
        try:
            if op == '+':
                r = a + b
            elif op == '-':
                r = a - b
            elif op == '*':
                r = a * b
            elif op == '/':
                if b == 0:
                    return folded
                r = a / b
                return IRFloat(value=float(r))
            elif op == '%':
                if b == 0:
                    return folded
                r = a % b
            elif op == '**':
                if isinstance(b, int) and abs(b) > 64:
                    return folded
                r = a ** b
                if both_int and isinstance(b, int) and b >= 0:
                    return IRInt(value=int(r))
                return IRFloat(value=float(r))
            else:
                return folded
        except (ZeroDivisionError, OverflowError, ValueError):
            return folded
        if both_int and op in ('+', '-', '*', '%'):
            return IRInt(value=int(r))
        return IRFloat(value=float(r))

    if node.op == '+' and isinstance(left, IRString) and isinstance(right, IRString):
        return IRString(value=left.value + right.value)

    return folded

def _fold_unaryop(node: IRUnaryOp):
    operand = fold_expr(node.operand)
    if node.op == '-':
        if _is_int(operand):
            return IRInt(value=-operand.value)
        if _is_float(operand):
            return IRFloat(value=-operand.value)
    if node.op == 'not' and isinstance(operand, IRBool):
        return IRBool(value=not operand.value)
    return IRUnaryOp(op=node.op, operand=operand)

def fold_expr(node):
    if isinstance(node, IRBinOp):
        return _fold_binop(node)
    if isinstance(node, IRUnaryOp):
        return _fold_unaryop(node)
    if isinstance(node, IRCall):
        return IRCall(func=fold_expr(node.func),
                      args=[fold_expr(a) for a in node.args],
                      kwargs={k: fold_expr(v) for k, v in node.kwargs.items()})
    if isinstance(node, IRList):
        return IRList(elements=[fold_expr(e) for e in node.elements])
    if isinstance(node, IRIndex):
        return IRIndex(obj=fold_expr(node.obj), index=fold_expr(node.index))
    if isinstance(node, IRField):
        return IRField(obj=fold_expr(node.obj), field=node.field)
    return node

def _fold_stmt(stmt):
    if isinstance(stmt, IRLet):
        return IRLet(name=stmt.name, type_ann=stmt.type_ann,
                     value=fold_expr(stmt.value) if stmt.value is not None else None)
    if isinstance(stmt, IRConst):
        return IRConst(name=stmt.name, value=fold_expr(stmt.value))
    if isinstance(stmt, IRAssign):
        return IRAssign(target=fold_expr(stmt.target), value=fold_expr(stmt.value))
    if isinstance(stmt, IRExprStmt):
        return IRExprStmt(expr=fold_expr(stmt.expr))
    if isinstance(stmt, IRReturn):
        return IRReturn(value=fold_expr(stmt.value) if stmt.value is not None else None)
    if isinstance(stmt, IRIf):
        return IRIf(condition=fold_expr(stmt.condition),
                    then_body=[_fold_stmt(s) for s in stmt.then_body],
                    elif_clauses=[(fold_expr(c), [_fold_stmt(s) for s in b]) for c, b in stmt.elif_clauses],
                    else_body=[_fold_stmt(s) for s in stmt.else_body] if stmt.else_body else None)
    if isinstance(stmt, IRFor):
        return IRFor(var=stmt.var, iter=fold_expr(stmt.iter),
                     body=[_fold_stmt(s) for s in stmt.body])
    if isinstance(stmt, IRWhile):
        return IRWhile(condition=fold_expr(stmt.condition),
                       body=[_fold_stmt(s) for s in stmt.body])
    if isinstance(stmt, IRFunction):
        return IRFunction(name=stmt.name, params=list(stmt.params),
                          body=[_fold_stmt(s) for s in stmt.body],
                          star_idx=stmt.star_idx, kw_idx=stmt.kw_idx)
    return stmt

def constant_fold(module: IRModule) -> IRModule:
    out = copy.copy(module)
    out.body = [_fold_stmt(s) for s in module.body]
    return out

_TERMINATORS = (IRReturn,)

def _y_const(node):
    if isinstance(node, IRBool):
        return node.value
    if isinstance(node, IRInt):
        return node.value != 0
    if isinstance(node, IRFloat):
        return node.value != 0.0
    if isinstance(node, IRString):
        return len(node.value) > 0
    return None

def _dce_body(body):
    result = []
    for stmt in body:
        stmt = _dce_stmt(stmt)
        if stmt is None:
            continue
        result.append(stmt)
        if isinstance(stmt, _TERMINATORS):
            break
    return result

def _dce_stmt(stmt):
    if isinstance(stmt, IRIf):
        cond_val = _y_const(stmt.condition)
        if cond_val is True:
            return IRIf(condition=stmt.condition,
                        then_body=_dce_body(stmt.then_body),
                        elif_clauses=[], else_body=None)
        if cond_val is False:
            if stmt.elif_clauses:
                first_cond, first_body = stmt.elif_clauses[0]
                rest = stmt.elif_clauses[1:]
                return _dce_stmt(IRIf(
                    condition=first_cond,
                    then_body=first_body,
                    elif_clauses=rest,
                    else_body=stmt.else_body))
            if stmt.else_body:
                return IRIf(condition=IRBool(value=True),
                            then_body=_dce_body(stmt.else_body),
                            elif_clauses=[], else_body=None)
            return None
        return IRIf(condition=stmt.condition,
                    then_body=_dce_body(stmt.then_body),
                    elif_clauses=[(c, _dce_body(b)) for c, b in stmt.elif_clauses],
                    else_body=_dce_body(stmt.else_body) if stmt.else_body else None)
    if isinstance(stmt, IRFor):
        return IRFor(var=stmt.var, iter=stmt.iter, body=_dce_body(stmt.body))
    if isinstance(stmt, IRWhile):
        if _y_const(stmt.condition) is False:
            return None
        return IRWhile(condition=stmt.condition, body=_dce_body(stmt.body))
    if isinstance(stmt, IRFunction):
        return IRFunction(name=stmt.name, params=list(stmt.params),
                          body=_dce_body(stmt.body),
                          star_idx=stmt.star_idx, kw_idx=stmt.kw_idx)
    return stmt

def dead_code_elimination(module: IRModule) -> IRModule:
    out = copy.copy(module)
    out.body = _dce_body(module.body)
    return out

def optimize(module: IRModule, level: int = 2) -> IRModule:
    module = constant_fold(module)
    if level >= 2:
        module = dead_code_elimination(module)
        module = constant_fold(module)
    return module

