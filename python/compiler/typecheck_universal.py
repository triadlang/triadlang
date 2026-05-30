from __future__ import annotations
from frontend.ast_nodes import *

class TypeCheckError(Exception):

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__('\n'.join(errors))

def typecheck(mod: Module) -> None:
    errors: list[str] = []
    declared: set[str] = set()
    functions: dict[str, int] = {}
    types_declared: set[str] = set()
    builtins = {'print', 'input', 'len', 'range', 'enumerate', 'str', 'int', 'float', 'type', 'abs', 'min', 'max', 'append', 'sorted', 'reversed', 'true', 'false', 'none'}
    declared.update(builtins)
    builtin_arities = {'len': 1, 'abs': 1, 'min': -1, 'max': -1, 'int': 1, 'float': 1, 'str': 1, 'type': 1, 'range': -1, 'enumerate': 1, 'sorted': 1, 'reversed': 1}

    def _pos(p: Pos) -> str:
        parts = []
        if p.file:
            parts.append(f'file: {p.file}')
        if p.line:
            parts.append(f'line: {p.line}')
        if p.col:
            parts.append(f'col: {p.col}')
        return ', '.join(parts)

    def check_expr(e: Expr, scope: set[str]):
        if isinstance(e, Ident):
            if e.name not in scope and e.name != 'self':
                errors.append(f"error[E2001]: undefined variable '{e.name}' ({_pos(e.pos)})")
        elif isinstance(e, BinOp):
            check_expr(e.left, scope)
            check_expr(e.right, scope)
        elif isinstance(e, UnaryOp):
            check_expr(e.operand, scope)
        elif isinstance(e, CallExpr):
            check_expr(e.func, scope)
            for a in e.args:
                check_expr(a, scope)
            for v in e.kwargs.values():
                check_expr(v, scope)
            if isinstance(e.func, Ident):
                fn_name = e.func.name
                n_args = len(e.args)
                if fn_name in builtin_arities:
                    arity = builtin_arities[fn_name]
                    if arity >= 0 and n_args != arity:
                        errors.append(f"error[E2002]: '{fn_name}' expects {arity} argument(s), got {n_args} ({_pos(e.pos)})")
                elif fn_name in functions:
                    arity = functions[fn_name]
                    if arity >= 0 and n_args != arity:
                        errors.append(f"error[E2002]: '{fn_name}' expects {arity} argument(s), got {n_args} ({_pos(e.pos)})")
        elif isinstance(e, MethodCallExpr):
            check_expr(e.obj, scope)
            for a in e.args:
                check_expr(a, scope)
            for v in e.kwargs.values():
                check_expr(v, scope)
        elif isinstance(e, IndexExpr):
            check_expr(e.obj, scope)
            check_expr(e.index, scope)
        elif isinstance(e, FieldExpr):
            check_expr(e.obj, scope)
        elif isinstance(e, ListExpr):
            for el in e.elements:
                check_expr(el, scope)
        elif isinstance(e, TupleExpr):
            for el in e.elements:
                check_expr(el, scope)
        elif isinstance(e, ListCompExpr):
            check_expr(e.iter, scope)
            inner = scope.copy()
            inner.add(e.var)
            check_expr(e.expr, inner)
            if e.condition:
                check_expr(e.condition, inner)
        elif isinstance(e, MapExpr):
            for k, v in e.pairs:
                check_expr(k, scope)
                check_expr(v, scope)
        elif isinstance(e, FStringExpr):
            for ptype, pval in e.parts:
                if ptype != 'str':
                    check_expr(pval, scope)
        elif isinstance(e, LambdaExpr):
            inner = scope.copy()
            for p in e.params:
                inner.add(p.name if isinstance(p, Param) else p)
            check_body(e.body, inner)
        elif isinstance(e, AssignExpr):
            check_expr(e.target, scope)
            check_expr(e.value, scope)
        elif isinstance(e, YieldExpr):
            if e.value:
                check_expr(e.value, scope)
        elif isinstance(e, AwaitExpr):
            check_expr(e.value, scope)

    def check_body(stmts: list[Stmt], scope: set[str]):
        for s in stmts:
            check_stmt(s, scope)

    def check_stmt(s: Stmt, scope: set[str]):
        if isinstance(s, LetStmt):
            if s.value:
                check_expr(s.value, scope)
            if s.name in scope and s.name not in builtins:
                pass
            scope.add(s.name)
        elif isinstance(s, ConstStmt):
            check_expr(s.value, scope)
            if s.name in scope and s.name not in builtins:
                pass
            scope.add(s.name)
        elif isinstance(s, DestructLetStmt):
            check_expr(s.value, scope)
            for n in s.names:
                scope.add(n)
        elif isinstance(s, MapDestructStmt):
            check_expr(s.value, scope)
            for n in s.names:
                scope.add(n)
        elif isinstance(s, AssignStmt):
            check_expr(s.value, scope)
            check_expr(s.target, scope)
        elif isinstance(s, ExprStmt):
            check_expr(s.expr, scope)
        elif isinstance(s, ReturnStmt):
            if s.value:
                check_expr(s.value, scope)
        elif isinstance(s, IfStmt):
            check_expr(s.condition, scope)
            check_body(s.then_body, scope.copy())
            for c, b in s.elif_clauses:
                check_expr(c, scope)
                check_body(b, scope.copy())
            if s.else_body:
                check_body(s.else_body, scope.copy())
        elif isinstance(s, ForStmt):
            check_expr(s.iter, scope)
            inner = scope.copy()
            inner.add(s.var)
            check_body(s.body, inner)
        elif isinstance(s, WhileStmt):
            check_expr(s.condition, scope)
            check_body(s.body, scope.copy())
        elif isinstance(s, FnDecl):
            n_params = len(s.params)
            functions[s.name] = n_params
            scope.add(s.name)
            inner = scope.copy()
            for p in s.params:
                pname = p.name if isinstance(p, Param) else p
                inner.add(pname)
            check_body(s.body, inner)
        elif isinstance(s, TypeDecl):
            scope.add(s.name)
            types_declared.add(s.name)
            for m in s.methods:
                mname = m.name
                inner = scope.copy()
                inner.add('self')
                for p in m.params:
                    pname = p.name if isinstance(p, Param) else p
                    inner.add(pname)
                check_body(m.body, inner)
        elif isinstance(s, ImportStmt):
            name = s.alias or s.path[-1]
            scope.add(name)
        elif isinstance(s, FromImportStmt):
            for n in s.names:
                scope.add(n)
        elif isinstance(s, RegStmt):
            scope.add(s.name)
        elif isinstance(s, EntityDecl):
            scope.add(s.name)
        elif isinstance(s, WorldDecl):
            scope.add(s.name)
        elif isinstance(s, ObserveStmt):
            if s.target not in scope:
                errors.append(f"error[E2010]: undefined observe target '{s.target}' ({_pos(s.pos)})")
        elif isinstance(s, RunStmt):
            if s.duration:
                check_expr(s.duration, scope)
        elif isinstance(s, TryCatchStmt):
            check_body(s.body, scope.copy())
            if s.catch_body:
                inner = scope.copy()
                if s.catch_var:
                    inner.add(s.catch_var)
                check_body(s.catch_body, inner)
            if s.finally_body:
                check_body(s.finally_body, scope.copy())
        elif isinstance(s, ThrowStmt):
            if s.value:
                check_expr(s.value, scope)
        elif isinstance(s, YieldStmt):
            if s.value:
                check_expr(s.value, scope)
        elif isinstance(s, CoupleStmt):
            for name in [s.src, s.dst]:
                if name not in scope:
                    errors.append(f"error[E2011]: undefined substrate '{name}' ({_pos(s.pos)})")
        elif isinstance(s, PairStmt):
            for name in [s.a, s.b]:
                if name not in scope:
                    errors.append(f"error[E2011]: undefined substrate '{name}' ({_pos(s.pos)})")
        elif isinstance(s, RingStmt):
            for name in s.members:
                if name not in scope:
                    errors.append(f"error[E2011]: undefined substrate '{name}' ({_pos(s.pos)})")
    scope = declared.copy()
    check_body(mod.body, scope)
    if errors:
        raise TypeCheckError(errors)