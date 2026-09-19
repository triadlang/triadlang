from __future__ import annotations

from frontend.ast_nodes import *
from frontend.errors import TypeCheckError

_KNOWN_METRICS = {'k_star', 'IPR', 'crystallinity', 'peak', 'FWHM', 'norm', 'stabilization',
    'memory_persistence', 'bravais_family', 'cluster_count', 'cluster_centroids',
    'slow_state', 'slow_state_late_mean', 'LZc', 'metastability', 'phi_id', 'pcist',
    'causal_density', 'kuramoto', 'atom_count', 'atom_centroids', 'atom_separation',
    'atom_persistence_late', 'atoms_per_region', 'atomicity_ratio'}
_KNOWN_PREDICATES = {'persistent', 'extended', 'structurally_open', 'mem_memory',
    'atomic', 'anti_collapsed'}
_KNOWN_PROJECTIONS = {'centroid_density', 'integrated_density', 'dominant_k_star',
    'fourier_band', 'atoms_of_atoms', 'none'}

_RUNTIME_BUILTINS_CACHE = None

def _runtime_builtin_names() -> set:
    global _RUNTIME_BUILTINS_CACHE
    if _RUNTIME_BUILTINS_CACHE is not None:
        return _RUNTIME_BUILTINS_CACHE
    names: set = set()
    try:
        from runtime.compiler_runtime import TriadCompiler
        env = TriadCompiler()._make_globals('<typecheck>')
        names.update(k for k in env if not k.startswith('_'))
    except Exception as exc:
        import logging as _logging
        _logging.getLogger(__name__).debug('typecheck: failed to load runtime builtins: %s', exc)
    try:
        import builtins as _py_builtins
        names.update(k for k in vars(_py_builtins) if not k.startswith('_'))
    except Exception as exc:
        import logging as _logging
        _logging.getLogger(__name__).debug('typecheck: failed to load python builtins: %s', exc)
    try:
        from compiler.c_codegen import _CCALL_NAMES, _ML_NAMES, _PYTHON_NAMES, _SOLVER_NAMES
        names.update(_ML_NAMES)
        names.update(_SOLVER_NAMES)
        names.update(_CCALL_NAMES)
        names.update(_PYTHON_NAMES)
    except Exception as exc:
        import logging as _logging
        _logging.getLogger(__name__).debug('typecheck: failed to load c_codegen names: %s', exc)
    _RUNTIME_BUILTINS_CACHE = names
    return names

def typecheck(mod: Module) -> None:

    errors: list[str] = []
    declared: set[str] = set()
    functions: dict[str, int] = {}
    types_declared: set[str] = set()
    builtins = {'print', 'input', 'len', 'range', 'enumerate', 'str', 'int', 'float', 'type', 'abs', 'min', 'max', 'append', 'sorted', 'reversed', 'true', 'false', 'none',
                 'ccall', 'py_call', 'py_eval', 'py_exec', 'run_async', 'sleep', 'gather', 'create_task', 'async_read', 'async_write', 'async_fetch',
                 'set_device', 'device', 'cuda_available'}
    builtins |= _runtime_builtin_names()
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

    def _suggest(name: str, scope: set[str], max_dist: int = 3) -> str | None:

        best_name = None
        best_dist = max_dist + 1
        for candidate in sorted(scope):
            if abs(len(candidate) - len(name)) > max_dist:
                continue
            d = 0
            for a, b in zip(name, candidate):
                d += (a != b)
                if d > max_dist:
                    break
            d += abs(len(name) - len(candidate))
            if d < best_dist and d <= max_dist:
                best_dist = d
                best_name = candidate
        return best_name

    var_types: dict[str, str | None] = {}
    _NUMERIC = {'int', 'float'}

    def _infer(e: Expr) -> str | None:
        if isinstance(e, IntLit):
            return 'int'
        if isinstance(e, FloatLit):
            return 'float'
        if isinstance(e, StringLit):
            return 'str'
        if isinstance(e, BoolLit):
            return 'bool'
        if isinstance(e, ListExpr):
            return 'list'
        if isinstance(e, Ident):
            return var_types.get(e.name)
        if isinstance(e, FStringExpr):
            return 'str'
        if isinstance(e, BinOp):
            lt = _infer(e.left)
            rt = _infer(e.right)
            if e.op == '+':
                if lt == 'str' and rt == 'str':
                    return 'str'
                if lt in _NUMERIC and rt in _NUMERIC:
                    return 'float' if 'float' in (lt, rt) else 'int'
            elif e.op in ('-', '*', '/', '%', '**'):
                if lt in _NUMERIC and rt in _NUMERIC:
                    return 'float' if (e.op in ('/', '**') or 'float' in (lt, rt)) else 'int'
            return None
        return None

    def _check_binop_types(e: BinOp):
        lt = _infer(e.left)
        rt = _infer(e.right)
        if lt is None or rt is None:
            return
        op = e.op

        if op == '+':
            str_side = ('str' in (lt, rt))
            num_side = (lt in _NUMERIC or rt in _NUMERIC)
            if str_side and num_side and lt != rt:
                errors.append(
                    f"error[E2003]: cannot apply '+' to {lt} and {rt}; "
                    f"a string and a number do not combine ({_pos(e.pos)})\n"
                    f"  help: wrap the number with str(...) to concatenate")
        elif op in ('-', '*', '/', '%', '**'):

            if 'str' in (lt, rt):
                other = rt if lt == 'str' else lt
                errors.append(
                    f"error[E2003]: cannot apply '{op}' to {lt} and {rt}; "
                    f"'{op}' is not defined on strings ({_pos(e.pos)})")

    def check_expr(e: Expr, scope: set[str]):
        if isinstance(e, Ident):
            if e.name not in scope and e.name != 'self':
                msg = f"error[E2001]: undefined variable '{e.name}' ({_pos(e.pos)})"
                suggestion = _suggest(e.name, scope)
                if suggestion:
                    msg += f"\n  help: did you mean '{suggestion}'?"
                errors.append(msg)
        elif isinstance(e, BinOp):
            check_expr(e.left, scope)
            check_expr(e.right, scope)
            _check_binop_types(e)
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
                    min_a, max_a = functions[fn_name]
                    n_total = n_args + len(e.kwargs)
                    if n_total < min_a or (max_a >= 0 and n_total > max_a):
                        if max_a < 0:
                            expected = f'at least {min_a}'
                        elif min_a == max_a:
                            expected = str(min_a)
                        else:
                            expected = f'{min_a} to {max_a}'
                        errors.append(f"error[E2002]: '{fn_name}' expects {expected} argument(s), got {n_total} ({_pos(e.pos)})")
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
            for part in e.parts:
                if part[0] != 'str':
                    check_expr(part[1], scope)
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
                var_types[s.name] = _infer(s.value)
            else:
                var_types[s.name] = None
            scope.add(s.name)
        elif isinstance(s, ConstStmt):
            check_expr(s.value, scope)
            var_types[s.name] = _infer(s.value)
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

            if isinstance(s.target, Ident):
                var_types[s.target.name] = _infer(s.value)
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
            min_a, max_a, variadic = 0, 0, False
            for p in s.params:
                if isinstance(p, Param):
                    if p.is_args or p.is_kwargs:
                        variadic = True
                        continue
                    max_a += 1
                    if p.default is None:
                        min_a += 1
                else:
                    max_a += 1
                    min_a += 1
            functions[s.name] = (min_a, -1 if variadic else max_a)
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
                inner = scope.copy()
                inner.add('self')
                for p in m.params:
                    pname = p.name if isinstance(p, Param) else p
                    inner.add(pname)
                check_body(m.body, inner)
        elif isinstance(s, ClassDecl):
            scope.add(s.name)
            types_declared.add(s.name)
            for m in s.methods:
                inner = scope.copy()
                inner.add('self')
                for p in m.params:
                    pname = p.name if isinstance(p, Param) else p
                    inner.add(pname)
                check_body(m.body, inner)
        elif isinstance(s, ImportStmt):

            name = s.alias or s.path[0]
            scope.add(name)
        elif isinstance(s, FromImportStmt):
            aliases = list(s.aliases) if s.aliases else [None] * len(s.names)
            if len(aliases) < len(s.names):
                aliases = aliases + [None] * (len(s.names) - len(aliases))
            for n, a in zip(s.names, aliases):
                scope.add(a or n)
        elif isinstance(s, RegStmt):
            scope.add(s.name)
        elif isinstance(s, SubstrateDecl):
            scope.add(s.name)
            for _m in s.members:
                if _m not in scope:
                    errors.append(f"error[E2012]: substrate '{s.name}': composed_of '{_m}' not declared ({_pos(s.pos)})")
            for _v in (s.properties or {}).values():
                check_expr(_v, scope)
            for _v in (s.overrides or {}).values():
                check_expr(_v, scope)
        elif isinstance(s, EntityDecl):
            scope.add(s.name)
        elif isinstance(s, WorldDecl):
            scope.add(s.name)
        elif isinstance(s, ObserveStmt):
            if s.target not in scope:
                errors.append(f"error[E2010]: undefined observe target '{s.target}' ({_pos(s.pos)})")
        elif isinstance(s, RunStmt):
            if s.target and s.target not in scope:
                errors.append(f"error[E2011]: undefined evolve target '{s.target}' ({_pos(s.pos)})")
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

