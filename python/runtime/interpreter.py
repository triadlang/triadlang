from __future__ import annotations
import os
import sys
from frontend.ast_nodes import *

class TriadError(Exception):

    def __init__(self, msg, pos=None):
        self.msg = msg
        self.pos = pos
        super().__init__(self._fmt())

    def _fmt(self):
        parts = [f'runtime error: {self.msg}']
        if self.pos and self.pos.file:
            parts.append(f'\n  file: {self.pos.file}')
        if self.pos and self.pos.line:
            parts.append(f'\n  line: {self.pos.line}')
        return ''.join(parts)

class ReturnSignal(Exception):

    def __init__(self, value):
        self.value = value

class BreakSignal(Exception):
    pass

class ContinueSignal(Exception):
    pass

class TriadObject:

    def __init__(self, type_name, fields):
        self.type_name = type_name
        self.fields = dict(fields)

    def __repr__(self):
        pairs = ', '.join((f'{k}={v!r}' for k, v in self.fields.items()))
        return f'{self.type_name}({pairs})'

class TriadFunction:

    def __init__(self, name, params, body, closure, is_generator=False, is_async=False):
        self.name = name
        self.params = params
        self.body = body
        self.closure = closure
        self.is_generator = is_generator
        self.is_async = is_async

    def __repr__(self):
        tags = []
        if self.is_generator:
            tags.append('generator')
        if self.is_async:
            tags.append('async')
        tag = ' '.join(tags) + ' ' if tags else ''
        return f'<{tag}fn {self.name}>'

class TriadModule:

    def __init__(self, name, env):
        self.name = name
        self.env = env

    def __repr__(self):
        return f'<module {self.name}>'

class Environment:

    def __init__(self, parent=None):
        self.vars: dict = {}
        self.parent = parent

    def get(self, name):
        if name in self.vars:
            return self.vars[name]
        if self.parent:
            return self.parent.get(name)
        return None

    def has(self, name):
        if name in self.vars:
            return True
        if self.parent:
            return self.parent.has(name)
        return False

    def set(self, name, val):
        self.vars[name] = val

    def update(self, name, val):
        if name in self.vars:
            self.vars[name] = val
            return True
        if self.parent:
            return self.parent.update(name, val)
        return False

class Interpreter:

    def _has_yield(self, stmts):
        for s in stmts:
            if isinstance(s, YieldStmt):
                return True
            if isinstance(s, IfStmt):
                if self._has_yield(s.then_body):
                    return True
                for _, body in s.elif_clauses:
                    if self._has_yield(body):
                        return True
                if s.else_body and self._has_yield(s.else_body):
                    return True
            if isinstance(s, ForStmt) and self._has_yield(s.body):
                return True
            if isinstance(s, WhileStmt) and self._has_yield(s.body):
                return True
            if isinstance(s, FnDecl) and self._has_yield(s.body):
                return True
            if isinstance(s, (TryCatchStmt,)):
                if self._has_yield(s.body) or self._has_yield(s.catch_body) or self._has_yield(s.finally_body):
                    return True
        return False

    def __init__(self):
        self.globals = Environment()
        self._module_cache: dict[str, TriadModule] = {}
        self._search_paths: list[str] = ['.']
        self._triad_rt = None
        self._triad_subs: dict = {}
        self._triad_T: float = 10.0
        self._triad_ran: bool = False
        self._setup_builtins()

    def _setup_builtins(self):
        g = self.globals

        def _print(*args, **kwargs):
            parts = []
            for a in args:
                if a is None:
                    parts.append('none')
                elif isinstance(a, bool):
                    parts.append('true' if a else 'false')
                else:
                    parts.append(str(a))
            print(' '.join(parts))
            return None

        def _input(*args):
            prompt = args[0] if args else ''
            return input(prompt)

        def _len(x):
            return len(x)

        def _range(*args):
            return list(range(*[int(a) for a in args]))

        def _enumerate(xs):
            return [list(p) for p in enumerate(xs)]

        def _str(x):
            if x is None:
                return 'none'
            if isinstance(x, bool):
                return 'true' if x else 'false'
            return str(x)

        def _int(x):
            return int(x)

        def _float(x):
            return float(x)

        def _type_of(x):
            if x is None:
                return 'None'
            if isinstance(x, bool):
                return 'Bool'
            if isinstance(x, int):
                return 'Int'
            if isinstance(x, float):
                return 'Float'
            if isinstance(x, str):
                return 'String'
            if isinstance(x, list):
                return 'List'
            if isinstance(x, dict):
                return 'Map'
            if isinstance(x, TriadObject):
                return x.type_name
            if isinstance(x, TriadFunction):
                return 'Function'
            return type(x).__name__

        def _abs(x):
            return abs(x)

        def _min(*args):
            if len(args) == 1 and isinstance(args[0], list):
                return min(args[0])
            return min(args)

        def _max(*args):
            if len(args) == 1 and isinstance(args[0], list):
                return max(args[0])
            return max(args)
        import math as _math
        import random as _random
        import json as _json
        for name, fn in [('print', _print), ('input', _input), ('len', _len), ('range', _range), ('enumerate', _enumerate), ('str', _str), ('int', _int), ('float', _float), ('type', _type_of), ('abs', _abs), ('min', _min), ('max', _max), ('append', lambda lst, x: lst.append(x) or lst)]:
            g.set(name, fn)
        math_env = {'sqrt': _math.sqrt, 'sin': _math.sin, 'cos': _math.cos, 'tan': _math.tan, 'log': _math.log, 'log10': _math.log10, 'exp': _math.exp, 'floor': _math.floor, 'ceil': _math.ceil, 'abs': abs, 'pi': _math.pi, 'e': _math.e, 'min': _min, 'max': _max, 'clamp': lambda x, lo, hi: max(lo, min(x, hi)), 'pow': pow}
        self._module_cache['math'] = TriadModule('math', math_env)
        random_env = {'random': _random.random, 'randint': _random.randint, 'choice': _random.choice, 'seed': _random.seed, 'shuffle': lambda x: _random.shuffle(x) or x, 'uniform': _random.uniform}
        self._module_cache['random'] = TriadModule('random', random_env)
        io_env = {'print': _print, 'input': _input}
        self._module_cache['io'] = TriadModule('io', io_env)
        string_env = {'split': lambda s, sep=None: s.split(sep), 'join': lambda sep, lst: sep.join((str(x) for x in lst)), 'replace': lambda s, old, new: s.replace(old, new), 'lower': lambda s: s.lower(), 'upper': lambda s: s.upper(), 'strip': lambda s: s.strip(), 'starts_with': lambda s, p: s.startswith(p), 'ends_with': lambda s, p: s.endswith(p), 'contains': lambda s, p: p in s}
        self._module_cache['string'] = TriadModule('string', string_env)
        json_env = {'parse': _json.loads, 'stringify': lambda x, indent=None: _json.dumps(x, indent=indent, default=str)}
        self._module_cache['json'] = TriadModule('json', json_env)

        def _read_text(p):
            with open(p) as f:
                return f.read()

        def _write_text(p, content):
            with open(p, 'w') as f:
                f.write(content)

        def _exists(p):
            return os.path.exists(p)

        def _listdir(p='.'):
            return os.listdir(p)
        fs_env = {'read_text': _read_text, 'write_text': _write_text, 'exists': _exists, 'listdir': _listdir}
        self._module_cache['fs'] = TriadModule('fs', fs_env)
        import time as _time
        time_env = {'now': _time.time, 'sleep': _time.sleep}
        self._module_cache['time'] = TriadModule('time', time_env)
        collections_env = {'len': _len, 'range': _range, 'enumerate': _enumerate, 'sorted': lambda x, **kw: sorted(x, **kw), 'reversed': lambda x: list(reversed(x)), 'zip': lambda *args: [list(t) for t in zip(*args)], 'map': lambda f, xs: [f(x) for x in xs], 'filter': lambda f, xs: [x for x in xs if f(x)], 'reduce': lambda f, xs, init=None: __import__('functools').reduce(f, xs) if init is None else __import__('functools').reduce(f, xs, init)}
        self._module_cache['collections'] = TriadModule('collections', collections_env)

    def run(self, mod: Module):
        self._search_paths = [os.path.dirname(os.path.abspath(mod.file)) if mod.file else '.']
        env = Environment(self.globals)
        self._exec_body(mod.body, env)

    def run_repl_line(self, mod: Module, env: Environment):
        for stmt in mod.body:
            result = self._exec_stmt(stmt, env)
            if isinstance(stmt, ExprStmt) and result is not None:
                return result
        return None

    def _exec_body(self, stmts: list[Stmt], env: Environment):
        for s in stmts:
            self._exec_stmt(s, env)

    def _exec_stmt(self, s: Stmt, env: Environment):
        if isinstance(s, LetStmt):
            val = self._eval(s.value, env) if s.value else None
            env.set(s.name, val)
            return val
        if isinstance(s, DestructLetStmt):
            val = self._eval(s.value, env)
            if isinstance(val, (list, tuple)):
                for i, name in enumerate(s.names):
                    env.set(name, val[i] if i < len(val) else None)
            return None
        if isinstance(s, MapDestructStmt):
            val = self._eval(s.value, env)
            if isinstance(val, dict):
                for name in s.names:
                    env.set(name, val.get(name))
            return None
        if isinstance(s, ConstStmt):
            env.set(s.name, self._eval(s.value, env))
            return None
        if isinstance(s, AssignStmt):
            val = self._eval(s.value, env)
            self._assign(s.target, val, env, s.pos)
            return None
        if isinstance(s, ExprStmt):
            return self._eval(s.expr, env)
        if isinstance(s, ReturnStmt):
            raise ReturnSignal(self._eval(s.value, env) if s.value else None)
        if isinstance(s, BreakStmt):
            raise BreakSignal()
        if isinstance(s, ContinueStmt):
            raise ContinueSignal()
        if isinstance(s, IfStmt):
            if self._truthy(self._eval(s.condition, env)):
                self._exec_body(s.then_body, env)
            else:
                matched = False
                for cond, body in s.elif_clauses:
                    if self._truthy(self._eval(cond, env)):
                        self._exec_body(body, env)
                        matched = True
                        break
                if not matched and s.else_body:
                    self._exec_body(s.else_body, env)
            return None
        if isinstance(s, ForStmt):
            iterable = self._eval(s.iter, env)
            if not hasattr(iterable, '__iter__'):
                raise TriadError(f'cannot iterate over {type(iterable).__name__}', s.pos)
            for item in iterable:
                env.set(s.var, item)
                try:
                    self._exec_body(s.body, env)
                except BreakSignal:
                    break
                except ContinueSignal:
                    continue
            return None
        if isinstance(s, WhileStmt):
            while self._truthy(self._eval(s.condition, env)):
                try:
                    self._exec_body(s.body, env)
                except BreakSignal:
                    break
                except ContinueSignal:
                    continue
            return None
        if isinstance(s, FnDecl):
            is_gen = self._has_yield(s.body)
            fn = TriadFunction(s.name, s.params, s.body, env, is_generator=is_gen, is_async=s.is_async)
            env.set(s.name, fn)
            return None
        if isinstance(s, TypeDecl):
            type_meta = {'name': s.name, 'fields': s.fields, 'methods': {}}
            for m in s.methods:
                type_meta['methods'][m.name] = TriadFunction(m.name, m.params, m.body, env)

            def constructor(*args, **kwargs):
                fields_dict = {}
                for i, f in enumerate(s.fields):
                    if f.name in kwargs:
                        fields_dict[f.name] = kwargs[f.name]
                    elif i < len(args):
                        fields_dict[f.name] = args[i]
                    elif f.default is not None:
                        fields_dict[f.name] = self._eval(f.default, env)
                    else:
                        fields_dict[f.name] = None
                obj = TriadObject(s.name, fields_dict)
                for mname, mfn in type_meta['methods'].items():
                    obj.fields[mname] = mfn
                return obj
            env.set(s.name, constructor)
            return None
        if isinstance(s, ImportStmt):
            mod = self._resolve_import(s.path, s.pos)
            name = s.alias or s.path[-1]
            env.set(name, mod)
            return None
        if isinstance(s, FromImportStmt):
            mod = self._resolve_import(s.path, s.pos)
            for name in s.names:
                if isinstance(mod, TriadModule):
                    val = mod.env.get(name) if isinstance(mod.env, dict) else None
                    if val is None and isinstance(mod.env, dict):
                        val = mod.env.get(name)
                else:
                    val = getattr(mod, name, None)
                env.set(name, val)
            return None
        if isinstance(s, EntityDecl):
            fields = {k: self._eval(v, env) for k, v in s.fields.items()}
            entity = TriadObject(s.base or 'Entity', fields)
            entity.fields['_name'] = s.name
            for m in s.methods:
                entity.fields[m.name] = TriadFunction(m.name, m.params, m.body, env)
            env.set(s.name, entity)
            return None
        if isinstance(s, WorldDecl):
            world = {'_triad_world': True, 'name': s.name}
            world.update({k: self._eval(v, env) for k, v in s.fields.items()})
            env.set(s.name, world)
            return None
        if isinstance(s, (CoupleStmt, PairStmt, RingStmt)):
            self._exec_coupling(s, env)
            return None
        if isinstance(s, ObserveStmt):
            self._exec_observe(s, env)
            return None
        if isinstance(s, RunStmt):
            self._exec_run(s, env)
            return None
        if isinstance(s, RegStmt):
            self._exec_reg(s, env)
            return None
        if isinstance(s, ClassDecl):
            self._exec_class(s, env)
            return None
        if isinstance(s, MatchStmt):
            self._exec_match(s, env)
            return None
        if isinstance(s, TryCatchStmt):
            try:
                self._exec_body(s.body, env)
            except Exception as ex:
                if s.catch_body:
                    child = Environment(env)
                    child.set(s.catch_var or '_err', ex)
                    self._exec_body(s.catch_body, child)
            finally:
                if s.finally_body:
                    self._exec_body(s.finally_body, env)
            return None
        if isinstance(s, ThrowStmt):
            val = self._eval(s.value, env) if s.value else 'error'
            raise Exception(val)
        if isinstance(s, AnnotationStmt):
            if s.key == 'T':
                self._triad_T = float(s.args)
            return None
        raise TriadError(f'unknown statement type: {type(s).__name__}')

    def _ensure_runtime(self):
        if self._triad_rt is None:
            from runtime.multi_runtime import MultiRuntime
            self._triad_rt = MultiRuntime(dt=0.005, record_every=4)

    def _exec_reg(self, s, env):
        self._ensure_runtime()
        from stdlib.regimes import resolve_regime
        regime = s.regime or 'B0'
        p = resolve_regime(regime, seed=0, L=32.0, N=128, dt=0.005)
        sub = self._triad_rt.add_substrate(s.name, p)
        self._triad_subs[s.name] = sub
        env.set(s.name, sub)

    def _exec_coupling(self, s, env):
        self._ensure_runtime()
        from runtime.multi_runtime import CouplingEdge, Segment
        kappa = self._eval(s.kappa, env) if s.kappa else -3.0
        dur = self._eval(s.duration, env) if hasattr(s, 'duration') and s.duration else self._triad_T
        if isinstance(s, CoupleStmt):
            edges = [CouplingEdge(src_id=self._triad_subs[s.src].id, dst_id=self._triad_subs[s.dst].id, kappa=kappa)]
        elif isinstance(s, PairStmt):
            edges = [CouplingEdge(src_id=self._triad_subs[s.a].id, dst_id=self._triad_subs[s.b].id, kappa=kappa), CouplingEdge(src_id=self._triad_subs[s.b].id, dst_id=self._triad_subs[s.a].id, kappa=kappa)]
        elif isinstance(s, RingStmt):
            members = s.members
            ids = [self._triad_subs[m].id for m in members]
            edges = [CouplingEdge(src_id=ids[i], dst_id=ids[(i + 1) % len(ids)], kappa=kappa) for i in range(len(ids))]
        else:
            return
        self._triad_rt.add_segment(Segment(t_start=self._triad_rt.global_t, t_end=self._triad_rt.global_t + dur, edges=edges))
        self._triad_rt.global_t += dur
        self._triad_ran = False

    def _exec_observe(self, s, env):
        self._ensure_runtime()
        if not self._triad_ran:
            self._triad_rt.run(verbose=False)
            self._triad_ran = True
        sub = self._triad_subs.get(s.target)
        if sub is None:
            raise TriadError(f"unknown substrate '{s.target}'", s.pos)
        import numpy as _np
        from runtime.observables import dominant_wavenumber as _obs_kstar, crystallinity as _obs_C, peak_density as _obs_peak, ipr as _obs_ipr, fwhm as _obs_fwhm
        L = sub.params.L
        dx = sub.dx
        kmin = 2.0 * _np.pi / L
        row = {}
        for m in s.metrics:
            if m == 'k_star':
                row['k_star'] = float(_obs_kstar(sub.psi, dx, k_min=kmin))
            elif m == 'crystallinity':
                row['crystallinity'] = float(_obs_C(sub.psi, dx))
            elif m == 'peak':
                row['peak'] = float((_np.abs(sub.psi) ** 2).max())
            elif m == 'IPR':
                row['IPR'] = float(_obs_ipr(sub.psi, dx))
            elif m == 'FWHM':
                row['FWHM'] = float(_obs_fwhm(sub.psi, dx))
            elif m == 'norm':
                row['norm'] = float((_np.abs(sub.psi) ** 2).sum() * dx)
            elif m == 'atom_count':
                row['atom_count'] = float((_np.abs(sub.psi) ** 2).sum() * dx)
            else:
                row[m] = 'unknown_metric'
        parts = [f'{k}={v:.4f}' for k, v in row.items()]
        print(f"{s.target} = {{ {', '.join(parts)} }}")

    def _exec_run(self, s, env):
        self._ensure_runtime()
        if not self._triad_ran:
            self._triad_rt.run(verbose=False)
            self._triad_ran = True

    def _exec_class(self, s: ClassDecl, env: Environment):
        parent_class = env.get(s.parent) if s.parent else None
        type_meta = {'name': s.name, 'parent': parent_class, 'fields': s.fields, 'methods': {}}
        for m in s.methods:
            type_meta['methods'][m.name] = TriadFunction(m.name, m.params, m.body, env)

        def constructor(*args, **kwargs):
            fields_dict = {}
            if parent_class and callable(parent_class):
                parent_obj = parent_class(*args, **kwargs)
                if isinstance(parent_obj, TriadObject):
                    fields_dict.update(parent_obj.fields)
            for i, f in enumerate(s.fields):
                if f.name in kwargs:
                    fields_dict[f.name] = kwargs[f.name]
                elif i < len(args):
                    fields_dict[f.name] = args[i]
                elif f.default is not None:
                    fields_dict[f.name] = self._eval(f.default, env)
                else:
                    fields_dict[f.name] = None
            obj = TriadObject(s.name, fields_dict)
            all_methods = {}
            if parent_class and isinstance(parent_class, dict) and ('methods' in parent_class):
                all_methods.update(parent_class['methods'])
            all_methods.update(type_meta['methods'])
            for mname, mfn in all_methods.items():
                obj.fields[mname] = mfn
            return obj
        constructor._class_meta = type_meta
        env.set(s.name, constructor)

    def _exec_match(self, s: MatchStmt, env: Environment):
        subject = self._eval(s.subject, env)
        for case in s.cases:
            bindings = self._match_pattern(subject, case.pattern)
            if bindings is not None:
                child = Environment(env)
                for k, v in bindings.items():
                    child.set(k, v)
                if case.guard:
                    guard_val = self._eval(case.guard, child)
                    if not self._truthy(guard_val):
                        continue
                try:
                    self._exec_body(case.body, child)
                except ReturnSignal:
                    raise
                return None
        if s.else_body:
            try:
                self._exec_body(s.else_body, env)
            except ReturnSignal:
                raise
        return None

    def _match_pattern(self, subject, pattern_expr):
        if isinstance(pattern_expr, Ident):
            if pattern_expr.name == '_':
                return {}
            return {pattern_expr.name: subject}
        if isinstance(pattern_expr, IntLit) or isinstance(pattern_expr, FloatLit):
            return {} if subject == pattern_expr.value else None
        if isinstance(pattern_expr, StringLit):
            return {} if subject == pattern_expr.value else None
        if isinstance(pattern_expr, BoolLit):
            return {} if subject == pattern_expr.value else None
        if isinstance(pattern_expr, ListExpr):
            if not isinstance(subject, list):
                return None
            if len(subject) != len(pattern_expr.elements):
                return None
            bindings = {}
            for sub, pat in zip(subject, pattern_expr.elements):
                inner = self._match_pattern(sub, pat)
                if inner is None:
                    return None
                bindings.update(inner)
            return bindings
        if isinstance(pattern_expr, NoneLit):
            return {} if subject is None else None
        return {} if subject == pattern_expr else None

    def _eval(self, e: Expr, env: Environment):
        if e is None:
            return None
        if isinstance(e, IntLit):
            return e.value
        if isinstance(e, FloatLit):
            return e.value
        if isinstance(e, BoolLit):
            return e.value
        if isinstance(e, StringLit):
            return e.value
        if isinstance(e, NoneLit):
            return None
        if isinstance(e, Ident):
            if e.name == 'self':
                return env.get('self')
            val = env.get(e.name)
            if val is None and (not env.has(e.name)):
                raise TriadError(f"undefined variable '{e.name}'", e.pos)
            return val
        if isinstance(e, BinOp):
            left = self._eval(e.left, env)
            if e.op == 'and':
                return left if not self._truthy(left) else self._eval(e.right, env)
            if e.op == 'or':
                return left if self._truthy(left) else self._eval(e.right, env)
            right = self._eval(e.right, env)
            return self._binop(e.op, left, right, e.pos)
        if isinstance(e, UnaryOp):
            val = self._eval(e.operand, env)
            if e.op == '-':
                return -val
            if e.op == 'not':
                return not self._truthy(val)
            raise TriadError(f"unknown unary op '{e.op}'", e.pos)
        if isinstance(e, CallExpr):
            func = self._eval(e.func, env)
            args = [self._eval(a, env) for a in e.args]
            kwargs = {k: self._eval(v, env) for k, v in e.kwargs.items()}
            return self._call(func, args, kwargs, e.pos)
        if isinstance(e, MethodCallExpr):
            obj = self._eval(e.obj, env)
            args = [self._eval(a, env) for a in e.args]
            kwargs = {k: self._eval(v, env) for k, v in e.kwargs.items()}
            return self._method_call(obj, e.method, args, kwargs, e.pos)
        if isinstance(e, IndexExpr):
            obj = self._eval(e.obj, env)
            idx = self._eval(e.index, env)
            try:
                return obj[idx]
            except (KeyError, IndexError, TypeError) as ex:
                raise TriadError(str(ex), e.pos)
        if isinstance(e, FieldExpr):
            obj = self._eval(e.obj, env)
            return self._field_access(obj, e.field, e.pos)
        if isinstance(e, ListExpr):
            return [self._eval(el, env) for el in e.elements]
        if isinstance(e, TupleExpr):
            return tuple((self._eval(el, env) for el in e.elements))
        if isinstance(e, MapExpr):
            return {self._eval(k, env): self._eval(v, env) for k, v in e.pairs}
        if isinstance(e, LambdaExpr):
            return TriadFunction('<lambda>', e.params, e.body, env)
        if isinstance(e, AssignExpr):
            val = self._eval(e.value, env)
            self._assign(e.target, val, env)
            return val
        if isinstance(e, AwaitExpr):
            import asyncio
            val = self._eval(e.value, env)
            if asyncio.iscoroutine(val):
                return asyncio.get_event_loop().run_until_complete(val)
            return val
        if isinstance(e, YieldExpr):
            return self._eval(e.value, env) if e.value else None
        raise TriadError(f'cannot eval {type(e).__name__}')

    def _assign(self, target, val, env, pos=None):
        if isinstance(target, Ident):
            if not env.update(target.name, val):
                env.set(target.name, val)
        elif isinstance(target, FieldExpr):
            obj = self._eval(target.obj, env)
            if isinstance(obj, TriadObject):
                obj.fields[target.field] = val
            elif isinstance(obj, dict):
                obj[target.field] = val
            else:
                raise TriadError(f'cannot set field on {type(obj).__name__}', pos)
        elif isinstance(target, IndexExpr):
            obj = self._eval(target.obj, env)
            idx = self._eval(target.index, env)
            obj[idx] = val
        else:
            raise TriadError(f'invalid assignment target', pos)

    def _call(self, func, args, kwargs, pos=None):
        if func is None:
            raise TriadError('calling None', pos)
        if callable(func) and (not isinstance(func, TriadFunction)):
            try:
                if kwargs:
                    return func(*args, **kwargs)
                return func(*args)
            except Exception as ex:
                raise TriadError(str(ex), pos)
        if isinstance(func, TriadFunction):
            if func.is_generator:
                return self._make_generator(func, args, kwargs)
            child = Environment(func.closure)
            for i, p in enumerate(func.params):
                name = p.name if isinstance(p, Param) else p
                if name in kwargs:
                    child.set(name, kwargs[name])
                elif i < len(args):
                    child.set(name, args[i])
                elif isinstance(p, Param) and p.default is not None:
                    child.set(name, self._eval(p.default, func.closure))
                else:
                    child.set(name, None)
            try:
                self._exec_body(func.body, child)
            except ReturnSignal as r:
                return r.value
            return None
        raise TriadError(f'not callable: {func!r}', pos)

    def _make_generator(self, func, args, kwargs):
        child = Environment(func.closure)
        for i, p in enumerate(func.params):
            name = p.name if isinstance(p, Param) else p
            if name in kwargs:
                child.set(name, kwargs[name])
            elif i < len(args):
                child.set(name, args[i])
            elif isinstance(p, Param) and p.default is not None:
                child.set(name, self._eval(p.default, func.closure))
            else:
                child.set(name, None)

        def _flatten(stmts):
            for s in stmts:
                if isinstance(s, YieldStmt):
                    val = self._eval(s.value, child) if s.value else None
                    yield val
                elif isinstance(s, ForStmt):
                    iterable = self._eval(s.iter, child)
                    for item in iterable:
                        child.set(s.var, item)
                        try:
                            yield from _flatten(s.body)
                        except BreakSignal:
                            break
                        except ContinueSignal:
                            continue
                elif isinstance(s, WhileStmt):
                    while self._truthy(self._eval(s.condition, child)):
                        try:
                            yield from _flatten(s.body)
                        except BreakSignal:
                            break
                        except ContinueSignal:
                            continue
                elif isinstance(s, IfStmt):
                    if self._truthy(self._eval(s.condition, child)):
                        yield from _flatten(s.then_body)
                    else:
                        matched = False
                        for cond, body in s.elif_clauses:
                            if self._truthy(self._eval(cond, child)):
                                yield from _flatten(body)
                                matched = True
                                break
                        if not matched and s.else_body:
                            yield from _flatten(s.else_body)
                elif isinstance(s, TryCatchStmt):
                    try:
                        yield from _flatten(s.body)
                    except Exception as ex:
                        if s.catch_body:
                            catch_env = Environment(child)
                            catch_env.set(s.catch_var or '_err', ex)
                            for cs in s.catch_body:
                                self._exec_stmt(cs, catch_env)
                    finally:
                        if s.finally_body:
                            for fs in s.finally_body:
                                self._exec_stmt(fs, child)
                else:
                    self._exec_stmt(s, child)
        return _flatten(func.body)

    def _method_call(self, obj, method, args, kwargs, pos=None):
        if isinstance(obj, str):
            m = {'split': lambda: obj.split(*args) if args else obj.split(), 'join': lambda: obj.join((str(x) for x in args[0])), 'replace': lambda: obj.replace(args[0], args[1]), 'lower': lambda: obj.lower(), 'upper': lambda: obj.upper(), 'strip': lambda: obj.strip(), 'starts_with': lambda: obj.startswith(args[0]), 'ends_with': lambda: obj.endswith(args[0]), 'contains': lambda: args[0] in obj, 'len': lambda: len(obj), 'find': lambda: obj.find(args[0])}.get(method)
            if m:
                return m()
        if isinstance(obj, list):
            m = {'push': lambda: obj.append(args[0]) or None, 'append': lambda: obj.append(args[0]) or None, 'pop': lambda: obj.pop(*args), 'len': lambda: len(obj), 'contains': lambda: args[0] in obj, 'map': lambda: [self._call(args[0], [x], {}, pos) for x in obj], 'filter': lambda: [x for x in obj if self._truthy(self._call(args[0], [x], {}, pos))], 'sort': lambda: (obj.sort(), obj)[-1], 'reverse': lambda: (obj.reverse(), obj)[-1], 'join': lambda: args[0].join((str(x) for x in obj)) if args else ''.join((str(x) for x in obj)), 'insert': lambda: obj.insert(int(args[0]), args[1]), 'remove': lambda: obj.remove(args[0]), 'index': lambda: obj.index(args[0]), 'slice': lambda: obj[int(args[0]):int(args[1])] if len(args) >= 2 else obj[int(args[0]):]}.get(method)
            if m:
                return m()
        if isinstance(obj, dict):
            m = {'keys': lambda: list(obj.keys()), 'values': lambda: list(obj.values()), 'items': lambda: [[k, v] for k, v in obj.items()], 'get': lambda: obj.get(args[0], args[1] if len(args) > 1 else None), 'set': lambda: obj.__setitem__(args[0], args[1]) or None, 'has': lambda: args[0] in obj, 'remove': lambda: obj.pop(args[0], None), 'len': lambda: len(obj)}.get(method)
            if m:
                return m()
        if isinstance(obj, TriadObject):
            fn = obj.fields.get(method)
            if fn is not None:
                if isinstance(fn, TriadFunction):
                    child = Environment(fn.closure)
                    child.set('self', obj)
                    params = fn.params
                    for i, p in enumerate(params):
                        pname = p.name if isinstance(p, Param) else p
                        if pname == 'self':
                            continue
                        idx = i - (1 if params and (params[0].name if isinstance(params[0], Param) else params[0]) == 'self' else 0)
                        if idx >= 0 and idx < len(args):
                            child.set(pname, args[idx])
                    try:
                        self._exec_body(fn.body, child)
                    except ReturnSignal as r:
                        return r.value
                    return None
                if callable(fn):
                    return fn(*args, **kwargs)
        if isinstance(obj, TriadModule):
            val = obj.env.get(method) if isinstance(obj.env, dict) else None
            if callable(val):
                return val(*args, **kwargs)
            return val
        raise TriadError(f"no method '{method}' on {type(obj).__name__}", pos)

    def _field_access(self, obj, field, pos=None):
        if isinstance(obj, TriadObject):
            if field in obj.fields:
                return obj.fields[field]
            raise TriadError(f"no field '{field}' on {obj.type_name}", pos)
        if isinstance(obj, dict):
            if field in obj:
                return obj[field]
            raise TriadError(f"no key '{field}' in map", pos)
        if isinstance(obj, TriadModule):
            if isinstance(obj.env, dict):
                if field in obj.env:
                    return obj.env[field]
            raise TriadError(f"no '{field}' in module {obj.name}", pos)
        if isinstance(obj, list):
            if field == 'length':
                return len(obj)
        if isinstance(obj, str):
            if field == 'length':
                return len(obj)
        raise TriadError(f"cannot access '{field}' on {type(obj).__name__}", pos)

    def _binop(self, op, l, r, pos=None):
        try:
            if op == '+':
                if isinstance(l, str) or isinstance(r, str):
                    return str(l) + str(r)
                return l + r
            if op == '-':
                return l - r
            if op == '*':
                if isinstance(l, str) and isinstance(r, int):
                    return l * r
                return l * r
            if op == '/':
                return l / r
            if op == '%':
                return l % r
            if op == '**':
                return l ** r
            if op == '==':
                return l == r
            if op == '!=':
                return l != r
            if op == '<':
                return l < r
            if op == '<=':
                return l <= r
            if op == '>':
                return l > r
            if op == '>=':
                return l >= r
        except Exception as ex:
            raise TriadError(f'{type(l).__name__} {op} {type(r).__name__}: {ex}', pos)
        raise TriadError(f"unknown op '{op}'", pos)

    def _truthy(self, v):
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return v != 0
        if isinstance(v, float):
            return v != 0.0
        if isinstance(v, str):
            return len(v) > 0
        if isinstance(v, list):
            return len(v) > 0
        if isinstance(v, dict):
            return len(v) > 0
        return True

    def _resolve_import(self, path: list[str], pos=None) -> TriadModule:
        key = '.'.join(path)
        if key in self._module_cache:
            return self._module_cache[key]
        for sp in self._search_paths:
            rel = os.path.join(sp, *path) + '.tri'
            if os.path.exists(rel):
                return self._load_module(rel, key)
            rel_dir = os.path.join(sp, *path, '__init__.tri')
            if os.path.exists(rel_dir):
                return self._load_module(rel_dir, key)
        modules_dir = os.path.join(os.getcwd(), 'triad_modules')
        if os.path.isdir(modules_dir) and len(path) >= 1:
            pkg_rest = path[1:] if len(path) > 1 else []
            rel = os.path.join(modules_dir, *path)
            if pkg_rest:
                rel = os.path.join(modules_dir, path[0], *pkg_rest) + '.tri'
            else:
                rel = os.path.join(modules_dir, path[0], '__init__.tri')
            if os.path.exists(rel):
                return self._load_module(rel, key)
            rel_dir = os.path.join(modules_dir, *path, '__init__.tri')
            if os.path.exists(rel_dir):
                return self._load_module(rel_dir, key)
            if not pkg_rest:
                main_rel = os.path.join(modules_dir, path[0], 'main.tri')
                if os.path.exists(main_rel):
                    return self._load_module(main_rel, key)
        stdlib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stdlib')
        for sp in [stdlib_dir]:
            rel = os.path.join(sp, *path) + '.tri'
            if os.path.exists(rel):
                return self._load_module(rel, key)
            rel = os.path.join(sp, *path, '__init__.tri')
            if os.path.exists(rel):
                return self._load_module(rel, key)
        raise TriadError(f"cannot resolve import '{key}'", pos)

    def _load_module(self, filepath: str, key: str) -> TriadModule:
        from frontend.parser_universal import parse
        with open(filepath) as f:
            src = f.read()
        mod_ast = parse(src, filepath)
        env = Environment(self.globals)
        old_paths = self._search_paths[:]
        self._search_paths.insert(0, os.path.dirname(os.path.abspath(filepath)))
        self._exec_body(mod_ast.body, env)
        self._search_paths = old_paths
        mod = TriadModule(key, env.vars)
        self._module_cache[key] = mod
        return mod