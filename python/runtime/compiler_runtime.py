from __future__ import annotations
import os
import sys
import math
import random
import json
import time as _time_mod
import hashlib
import marshal
from frontend.ast_nodes import *

class CompileError(Exception):
    pass
_COMPILER_VERSION = 8

class TriadCompiler:

    def __init__(self):
        self._indent = 0
        self._lines: list[str] = []
        self._module_cache: dict[str, dict] = {}
        self._search_paths: list[str] = ['.']
        self._prelude_emitted = False
        self._scope_depth = 0
        self._var_types: dict[str, str] = {}
        self._needs_triad_native = False
        self._backend = 'solver'

    def compile_and_run(self, mod: Module):
        self._search_paths = [os.path.dirname(os.path.abspath(mod.file)) if mod.file else '.']
        filepath = mod.file or '<triad>'
        self._needs_triad_native = self._detect_triad_native(mod.body)
        cached = self._load_cache(filepath)
        if cached is not None:
            env = self._make_globals(filepath)
            exec(cached, env)
            return
        code = self._compile_module(mod)
        env = self._make_globals(filepath)
        compiled = compile(code, filepath, 'exec')
        self._save_cache(filepath, code, compiled)
        exec(compiled, env)

    def _cache_path(self, filepath: str) -> str | None:
        if filepath == '<triad>' or not os.path.isfile(filepath):
            return None
        d = os.path.join(os.path.dirname(os.path.abspath(filepath)), '__triadcache__')
        name = os.path.basename(filepath)
        return os.path.join(d, name + 'c')

    def _source_hash(self, filepath: str) -> bytes:
        with open(filepath, 'rb') as f:
            h = hashlib.sha256(f.read())
        h.update(f'v{_COMPILER_VERSION}'.encode())
        return h.digest()

    def _load_cache(self, filepath: str):
        cp = self._cache_path(filepath)
        if cp is None or not os.path.isfile(cp):
            return None
        try:
            with open(cp, 'rb') as f:
                stored_hash = f.read(32)
                if stored_hash != self._source_hash(filepath):
                    return None
                return marshal.loads(f.read())
        except Exception:
            return None

    def _save_cache(self, filepath: str, source: str, compiled):
        cp = self._cache_path(filepath)
        if cp is None:
            return
        try:
            os.makedirs(os.path.dirname(cp), exist_ok=True)
            with open(cp, 'wb') as f:
                f.write(self._source_hash(filepath))
                f.write(marshal.dumps(compiled))
        except Exception:
            pass

    def compile_to_source(self, mod: Module) -> str:
        return self._compile_module(mod)

    def _compile_module(self, mod: Module) -> str:
        self._indent = 0
        self._lines = []
        self._needs_triad_native = self._detect_triad_native(mod.body)
        self._emit_prelude()
        for stmt in mod.body:
            self._compile_stmt(stmt)
        raw = '\n'.join(self._lines)
        from compiler.source_optimizer import optimize_source
        return optimize_source(raw)
    _TRIAD_NATIVE_TYPES = (RegStmt, ObserveStmt, RunStmt, CoupleStmt, PairStmt, RingStmt)

    def _detect_triad_native(self, body: list) -> bool:
        for s in body:
            if isinstance(s, self._TRIAD_NATIVE_TYPES):
                return True
            if isinstance(s, (FnDecl, LambdaExpr)):
                inner = s.body if isinstance(s, FnDecl) else []
                if isinstance(inner, list) and self._detect_triad_native(inner):
                    return True
            if isinstance(s, IfStmt):
                if self._detect_triad_native(s.then_body):
                    return True
                for _, b in s.elif_clauses:
                    if self._detect_triad_native(b):
                        return True
                if s.else_body and self._detect_triad_native(s.else_body):
                    return True
            if isinstance(s, WhileStmt) and self._detect_triad_native(s.body):
                return True
            if isinstance(s, ForStmt) and self._detect_triad_native(s.body):
                return True
        return False

    def _emit(self, line: str):
        self._lines.append('    ' * self._indent + line)

    def _emit_prelude(self):
        if self._needs_triad_native:
            if self._backend == 'vm':
                self._emit('from runtime.vm_runtime import VMRuntime as _MRT, CouplingEdge as _CE, Segment as _Seg')
            else:
                self._emit('from runtime.multi_runtime import MultiRuntime as _MRT, CouplingEdge as _CE, Segment as _Seg')
            self._emit('from stdlib.regimes import resolve_regime as _resolve_regime')
            self._emit('from runtime.observables import dominant_wavenumber as _obs_kstar, crystallinity as _obs_C, peak_density as _obs_peak, ipr as _obs_ipr, fwhm as _obs_fwhm')
            self._emit('_triad_rt = _MRT(dt=0.005, record_every=4)')
            self._emit('_triad_subs = {}')
            self._emit('_triad_T = 10.0')
            self._emit('_triad_pending_edges = []')
            self._emit('_triad_pending_duration = None')
            self._emit('')
        self._prelude_emitted = True

    def _infer_type(self, expr) -> str | None:
        if isinstance(expr, IntLit):
            return 'int'
        if isinstance(expr, FloatLit):
            return 'float'
        if isinstance(expr, BoolLit):
            return 'bool'
        if isinstance(expr, StringLit):
            return 'str'
        if isinstance(expr, FStringExpr):
            return 'str'
        if isinstance(expr, NoneLit):
            return 'none'
        if isinstance(expr, ListExpr):
            return 'list'
        if isinstance(expr, ListCompExpr):
            return 'list'
        if isinstance(expr, TupleExpr):
            return 'tuple'
        if isinstance(expr, MapExpr):
            return 'dict'
        if isinstance(expr, BinOp):
            lt = self._infer_type(expr.left)
            rt = self._infer_type(expr.right)
            if lt == 'ndarray' or rt == 'ndarray':
                return 'ndarray'
            if lt == 'float' or rt == 'float':
                return 'float'
            if lt == 'int' and rt == 'int':
                return 'float' if expr.op == '/' else 'int'
            if lt == 'str' and expr.op == '+':
                return 'str'
            if lt == 'list' and expr.op == '+':
                return 'list'
        if isinstance(expr, UnaryOp):
            if expr.op == 'not':
                return 'bool'
            if expr.op == '-':
                inner = self._infer_type(expr.operand)
                if inner == 'int':
                    return 'int'
                if inner == 'float':
                    return 'float'
                if inner == 'ndarray':
                    return 'ndarray'
            return self._infer_type(expr.operand)
        if isinstance(expr, CallExpr):
            if isinstance(expr.func, FieldExpr):
                if isinstance(expr.func.obj, Ident) and expr.func.obj.name in ('np', 'numpy'):
                    return 'ndarray'
                obj_t = self._infer_type(expr.func.obj)
                if obj_t == 'ndarray':
                    if expr.func.field in ('shape', 'size', 'ndim', 'dtype'):
                        return 'int'
                    if expr.func.field in ('sum', 'mean', 'std', 'var', 'min', 'max', 'dot', 'T'):
                        return 'ndarray'
                    return 'ndarray'
            if isinstance(expr.func, Ident):
                fn = expr.func.name
                if fn in ('int', 'len', 'abs'):
                    return 'int'
                if fn in ('float',):
                    return 'float'
                if fn in ('str',):
                    return 'str'
                if fn in ('bool',):
                    return 'bool'
                if fn in ('list', 'sorted', 'reversed'):
                    return 'list'
                if fn in ('dict',):
                    return 'dict'
                if fn in ('sum',):
                    if expr.args:
                        arg_t = self._infer_type(expr.args[0])
                        if arg_t == 'ndarray':
                            return 'ndarray'
                    return 'int'
        if isinstance(expr, MethodCallExpr):
            if isinstance(expr.obj, Ident) and expr.obj.name in ('np', 'numpy'):
                return 'ndarray'
            obj_t = self._infer_type(expr.obj)
            if obj_t == 'ndarray':
                if expr.method in ('sum', 'mean', 'std', 'var', 'min', 'max'):
                    return 'ndarray'
                return 'ndarray'
            if obj_t == 'str':
                if expr.method in ('split',):
                    return 'list'
                if expr.method in ('upper', 'lower', 'strip', 'replace'):
                    return 'str'
                if expr.method in ('starts_with', 'ends_with', 'contains'):
                    return 'bool'
                if expr.method == 'len':
                    return 'int'
        if isinstance(expr, IndexExpr):
            obj_t = self._infer_type(expr.obj)
            if obj_t == 'list':
                return None
            if obj_t == 'dict':
                return None
            if obj_t == 'ndarray':
                return 'ndarray'
            if obj_t == 'str':
                return 'str'
        if isinstance(expr, FieldExpr):
            obj_t = self._infer_type(expr.obj)
            if obj_t == 'ndarray' and expr.field in ('shape', 'size'):
                return 'int'
        if isinstance(expr, Ident):
            return self._var_types.get(expr.name)
        return None

    def _compile_stmt(self, s: Stmt):
        if isinstance(s, LetStmt):
            if s.value is not None:
                inferred = s.type_ann or self._infer_type(s.value)
                if inferred:
                    self._var_types[s.name] = inferred
                self._emit(f'{self._safe_name(s.name)} = {self._compile_expr(s.value)}')
            else:
                self._emit(f'{self._safe_name(s.name)} = None')
        elif isinstance(s, DestructLetStmt):
            names = ', '.join((self._safe_name(n) for n in s.names))
            self._emit(f'{names} = {self._compile_expr(s.value)}')
        elif isinstance(s, MapDestructStmt):
            val_var = f'_map_{len(self._lines)}'
            self._emit(f'{val_var} = {self._compile_expr(s.value)}')
            for name in s.names:
                self._emit(f'{self._safe_name(name)} = {val_var}.get({name!r})')
        elif isinstance(s, ConstStmt):
            inferred = self._infer_type(s.value)
            if inferred:
                self._var_types[s.name] = inferred
            self._emit(f'{self._safe_name(s.name)} = {self._compile_expr(s.value)}')
        elif isinstance(s, AssignStmt):
            if isinstance(s.target, Ident):
                inferred = self._infer_type(s.value)
                if inferred:
                    self._var_types[s.target.name] = inferred
            self._emit(f'{self._compile_expr(s.target)} = {self._compile_expr(s.value)}')
        elif isinstance(s, ExprStmt):
            self._emit(self._compile_expr(s.expr))
        elif isinstance(s, ReturnStmt):
            if s.value:
                self._emit(f'return {self._compile_expr(s.value)}')
            else:
                self._emit('return None')
        elif isinstance(s, BreakStmt):
            self._emit('break')
        elif isinstance(s, ContinueStmt):
            self._emit('continue')
        elif isinstance(s, IfStmt):
            self._emit(f'if {self._compile_expr(s.condition)}:')
            self._indent += 1
            self._compile_body(s.then_body)
            self._indent -= 1
            for cond, body in s.elif_clauses:
                self._emit(f'elif {self._compile_expr(cond)}:')
                self._indent += 1
                self._compile_body(body)
                self._indent -= 1
            if s.else_body:
                self._emit('else:')
                self._indent += 1
                self._compile_body(s.else_body)
                self._indent -= 1
        elif isinstance(s, ForStmt):
            if not self._try_vectorize_for(s):
                self._emit(f'for {self._safe_name(s.var)} in {self._compile_expr(s.iter)}:')
                self._indent += 1
                self._compile_body(s.body)
                self._indent -= 1
        elif isinstance(s, WhileStmt):
            self._emit(f'while {self._compile_expr(s.condition)}:')
            self._indent += 1
            self._compile_body(s.body)
            self._indent -= 1
        elif isinstance(s, FnDecl):
            params = ', '.join((self._compile_param(p) for p in s.params))
            prefix = 'async ' if s.is_async else ''
            self._emit(f'{prefix}def {self._safe_name(s.name)}({params}):')
            self._indent += 1
            self._scope_depth += 1
            if not s.body:
                self._emit('pass')
            else:
                outer_vars = self._find_nonlocals(s.body, {p.name if isinstance(p, Param) else p for p in s.params})
                if outer_vars:
                    if self._scope_depth == 1:
                        self._emit(f"global {', '.join((self._safe_name(n) for n in sorted(outer_vars)))}")
                    else:
                        self._emit(f"nonlocal {', '.join((self._safe_name(n) for n in sorted(outer_vars)))}")
                self._compile_body(s.body)
            self._scope_depth -= 1
            self._indent -= 1
            self._emit('')
        elif isinstance(s, TypeDecl):
            self._compile_type_decl(s)
        elif isinstance(s, ClassDecl):
            self._compile_class_decl(s)
        elif isinstance(s, MatchStmt):
            self._compile_match(s)
        elif isinstance(s, ImportStmt):
            self._compile_import(s)
        elif isinstance(s, FromImportStmt):
            self._compile_from_import(s)
        elif isinstance(s, RegStmt):
            regime = s.regime or 'B0'
            self._emit(f'_triad_p_{s.name} = _resolve_regime({regime!r}, seed=0, L=32.0, N=128, dt=0.005)')
            self._emit(f'_triad_subs[{s.name!r}] = _triad_rt.add_substrate({s.name!r}, _triad_p_{s.name})')
            self._emit(f'{self._safe_name(s.name)} = _triad_subs[{s.name!r}]')
        elif isinstance(s, EntityDecl):
            fields = ', '.join((f'{k!r}: {self._compile_expr(v)}' for k, v in s.fields.items()))
            self._emit(f"{self._safe_name(s.name)} = _TriObj({s.base or 'Entity'!r}, _name={s.name!r}, **{{{fields}}})")
            for m in s.methods:
                params = ', '.join(['self'] + [self._compile_param(p) for p in m.params if (p.name if isinstance(p, Param) else p) != 'self'])
                self._emit(f'def _{s.name}_{m.name}({params}):')
                self._indent += 1
                self._compile_body(m.body)
                self._indent -= 1
                self._emit(f'{self._safe_name(s.name)}.{m.name} = _{s.name}_{m.name}')
        elif isinstance(s, WorldDecl):
            fields = ', '.join((f'{k!r}: {self._compile_expr(v)}' for k, v in s.fields.items()))
            self._emit(f"{self._safe_name(s.name)} = {{'_triad_world': True, 'name': {s.name!r}, {fields}}}")
        elif isinstance(s, ObserveStmt):
            target = s.target
            metrics = s.metrics
            self._emit(f"if not hasattr(_triad_rt, '_ran'):")
            self._indent += 1
            self._emit(f'_triad_rt.run(verbose=False)')
            self._emit(f'_triad_rt._ran = True')
            self._indent -= 1
            self._emit(f'_obs_sub = _triad_subs[{target!r}]')
            self._emit(f'_obs_row = {{}}')
            self._emit(f'_obs_L = _obs_sub.params.L')
            self._emit(f'_obs_kmin = 2.0 * _np.pi / _obs_L')
            for m in metrics:
                if m == 'k_star':
                    self._emit(f"_obs_row['k_star'] = float(_obs_kstar(_obs_sub.psi, _obs_sub.dx, k_min=_obs_kmin))")
                elif m == 'crystallinity':
                    self._emit(f"_obs_row['crystallinity'] = float(_obs_C(_obs_sub.psi, _obs_sub.dx))")
                elif m == 'peak':
                    self._emit(f"_obs_row['peak'] = float((_np.abs(_obs_sub.psi)**2).max())")
                elif m == 'IPR':
                    self._emit(f"_obs_row['IPR'] = float(_obs_ipr(_obs_sub.psi, _obs_sub.dx))")
                elif m == 'FWHM':
                    self._emit(f"_obs_row['FWHM'] = float(_obs_fwhm(_obs_sub.psi, _obs_sub.dx))")
                elif m == 'norm':
                    self._emit(f"_obs_row['norm'] = float((_np.abs(_obs_sub.psi)**2).sum() * _obs_sub.dx)")
                elif m == 'atom_count':
                    self._emit(f"_obs_row['atom_count'] = float((_np.abs(_obs_sub.psi)**2).sum() * _obs_sub.dx)")
                else:
                    self._emit(f"_obs_row[{m!r}] = 'unknown_metric'")
            self._emit(f"_obs_parts = [f'{{k}}={{v:.4f}}' for k, v in _obs_row.items()]")
            self._emit(f"_tri_print({target!r}, '=', '{{', ', '.join(_obs_parts), '}}')")
        elif isinstance(s, RunStmt):
            dur = self._compile_expr(s.duration) if s.duration else '_triad_T'
            self._emit(f"if not hasattr(_triad_rt, '_ran'):")
            self._indent += 1
            self._emit(f'_triad_rt.run(verbose=False)')
            self._emit(f'_triad_rt._ran = True')
            self._indent -= 1
        elif isinstance(s, CoupleStmt):
            kappa = self._compile_expr(s.kappa) if s.kappa else '-3.0'
            dur = self._compile_expr(s.duration) if s.duration else '_triad_T'
            self._emit(f'_triad_rt.add_segment(_Seg(t_start=_triad_rt.global_t, t_end=_triad_rt.global_t + {dur}, edges=[_CE(src_id=_triad_subs[{s.src!r}].id, dst_id=_triad_subs[{s.dst!r}].id, kappa={kappa})]))')
            self._emit(f'_triad_rt.global_t += {dur}')
        elif isinstance(s, PairStmt):
            kappa = self._compile_expr(s.kappa) if s.kappa else '-3.0'
            dur = self._compile_expr(s.duration) if s.duration else '_triad_T'
            self._emit(f'_triad_rt.add_segment(_Seg(t_start=_triad_rt.global_t, t_end=_triad_rt.global_t + {dur}, edges=[_CE(src_id=_triad_subs[{s.a!r}].id, dst_id=_triad_subs[{s.b!r}].id, kappa={kappa}), _CE(src_id=_triad_subs[{s.b!r}].id, dst_id=_triad_subs[{s.a!r}].id, kappa={kappa})]))')
            self._emit(f'_triad_rt.global_t += {dur}')
        elif isinstance(s, RingStmt):
            members = s.members
            kappa = self._compile_expr(s.kappa) if s.kappa else '-3.0'
            dur = self._compile_expr(s.duration) if s.duration else '_triad_T'
            self._emit(f'_ring_names = {members!r}')
            self._emit(f'_ring_ids = [_triad_subs[n].id for n in _ring_names]')
            self._emit(f'_ring_edges = [_CE(src_id=_ring_ids[i], dst_id=_ring_ids[(i+1) % len(_ring_ids)], kappa={kappa}) for i in range(len(_ring_ids))]')
            self._emit(f'_triad_rt.add_segment(_Seg(t_start=_triad_rt.global_t, t_end=_triad_rt.global_t + {dur}, edges=_ring_edges))')
            self._emit(f'_triad_rt.global_t += {dur}')
        elif isinstance(s, SequenceStmt):
            inputs = self._compile_expr(s.inputs)
            each = self._compile_expr(s.each_for) if s.each_for else '_triad_T'
            self._emit(f'_seq_target_id = _triad_subs[{s.target!r}].id')
            self._emit(f'for _seq_e in {inputs}:')
            self._indent += 1
            self._emit('_seq_pot = (lambda _x, _e=_seq_e: _np.broadcast_to(_np.asarray(_e, dtype=float), _x.shape).astype(float))')
            self._emit(f'_triad_rt.add_segment(_Seg(t_start=_triad_rt.global_t, t_end=_triad_rt.global_t + {each}, v_ext_override={{_seq_target_id: _seq_pot}}))')
            self._emit(f'_triad_rt.global_t += {each}')
            self._indent -= 1
        elif isinstance(s, TryCatchStmt):
            self._emit('try:')
            self._indent += 1
            self._compile_body(s.body)
            self._indent -= 1
            if s.catch_body:
                var = self._safe_name(s.catch_var) if s.catch_var else '_tri_err'
                self._emit(f'except Exception as {var}:')
                self._indent += 1
                self._compile_body(s.catch_body)
                self._indent -= 1
            if s.finally_body:
                self._emit('finally:')
                self._indent += 1
                self._compile_body(s.finally_body)
                self._indent -= 1
        elif isinstance(s, ThrowStmt):
            if s.value:
                self._emit(f'raise Exception({self._compile_expr(s.value)})')
            else:
                self._emit('raise')
        elif isinstance(s, YieldStmt):
            if s.value:
                self._emit(f'yield {self._compile_expr(s.value)}')
            else:
                self._emit('yield')
        elif isinstance(s, AnnotationStmt):
            if s.key == 'T':
                self._emit(f'_triad_T = {s.args}')
            elif s.key == 'D':
                self._emit(f'_triad_D = {s.args}')
            elif s.key == 'N':
                self._emit(f'_triad_N = {s.args}')
            elif s.key == 'dt':
                self._emit(f'_triad_dt = {s.args}')
        else:
            try:
                from frontend.parser import Annotation as _PAnn
                from frontend.parser import RegDecl as _PRegDecl
                from frontend.parser import RingStmt as _PRing
                from frontend.parser import ObserveStmt as _PObs
            except ImportError:
                _PAnn = _PRegDecl = _PRing = _PObs = type(None)
            if isinstance(s, _PAnn):
                if s.key == 'T':
                    self._emit(f'_triad_T = {s.raw_args}')
                elif s.key == 'D':
                    self._emit(f'_triad_D = {s.raw_args}')
                elif s.key == 'N':
                    self._emit(f'_triad_N = {s.raw_args}')
                elif s.key == 'dt':
                    self._emit(f'_triad_dt = {s.raw_args}')
            elif isinstance(s, _PRegDecl):
                regime = s.regime_name or 'B0'
                self._emit(f'_triad_p_{s.name} = _resolve_regime({regime!r}, seed=0, L=32.0, N=128, dt=0.005)')
                self._emit(f'_triad_subs[{s.name!r}] = _triad_rt.add_substrate({s.name!r}, _triad_p_{s.name})')
                self._emit(f'{self._safe_name(s.name)} = _triad_subs[{s.name!r}]')
            elif isinstance(s, _PRing):
                members = [m.name for m in s.members]
                self._emit(f'_triad_pending_edges = [(_CE({members!r}, [], {s.kappa}), {s.duration!r})]')
            elif isinstance(s, _PObs):
                metrics_map = {'k_star': '_obs_kstar', 'crystallinity': '_obs_C', 'peak': '_obs_peak', 'ipr': '_obs_ipr', 'fwhm': '_obs_fwhm', 'atom_count': '_obs_peak'}
                obs_fields = []
                for m in s.metrics:
                    fn = metrics_map.get(m, '_obs_peak')
                    obs_fields.append(f'{m!r}: {fn}')
                self._emit(f"_triad_obs_{s.target.name} = {{{', '.join(obs_fields)}}}")
            else:
                self._emit(f'pass  # unhandled: {type(s).__name__}')

    def _compile_body(self, stmts: list[Stmt]):
        if not stmts:
            self._emit('pass')
            return
        for s in stmts:
            self._compile_stmt(s)

    def _compile_class_decl(self, s: ClassDecl):
        name = self._safe_name(s.name)
        parent = self._safe_name(s.parent) if s.parent else 'object'
        field_names = [f.name for f in s.fields]
        defaults = {}
        for f in s.fields:
            if f.default is not None:
                defaults[f.name] = self._compile_expr(f.default)
        user_has_init = any((m.name == 'init' for m in s.methods))
        self._emit(f'class {name}({parent}):')
        self._indent += 1
        if s.fields and (not user_has_init):
            init_params = []
            for f in s.fields:
                if f.name in defaults:
                    init_params.append(f'{f.name}={defaults[f.name]}')
                else:
                    init_params.append(f'{f.name}=None')
            self._emit(f"def __init__(self, {', '.join(init_params)}):")
            self._indent += 1
            if s.parent and s.parent != 'object':
                self._emit(f'super({name}, self).__init__()')
            for f in s.fields:
                self._emit(f'self.{f.name} = {f.name}')
            self._indent -= 1
        elif not user_has_init:
            self._emit('pass')
            self._indent -= 1
            self._indent += 1
        if field_names:
            self._emit(f"def __repr__(self): return f'{name}({', '.join((f'{fn}={{self.{fn}!r}}' for fn in field_names))})'")
        for m in s.methods:
            raw_params = [self._compile_param(p) for p in m.params if (p.name if isinstance(p, Param) else p) != 'self']
            params = ', '.join(['self'] + raw_params)
            method_name = '__init__' if m.name == 'init' else m.name
            self._emit(f'def {method_name}({params}):')
            self._indent += 1
            self._compile_body(m.body)
            self._indent -= 1
        self._indent -= 1
        self._emit('')

    def _compile_match(self, s: MatchStmt):
        subject = self._compile_expr(s.subject)
        var = f'_match_{len(self._lines)}'
        self._emit(f'{var} = {subject}')
        first = True
        for case in s.cases:
            kw = 'if' if first else 'elif'
            first = False
            cond = self._compile_pattern_cond(var, case.pattern)
            if case.guard:
                guard = self._compile_expr(case.guard)
                cond = f'({cond}) and {guard}'
            self._emit(f'{kw} {cond}:')
            self._indent += 1
            self._compile_pattern_bindings(var, case.pattern)
            self._compile_body(case.body)
            self._indent -= 1
        if s.else_body:
            self._emit('else:')
            self._indent += 1
            self._compile_body(s.else_body)
            self._indent -= 1

    def _compile_pattern_cond(self, var, pat) -> str:
        if isinstance(pat, Ident):
            if pat.name == '_':
                return 'True'
            return 'True'
        if isinstance(pat, (IntLit, FloatLit, StringLit, BoolLit)):
            val = self._compile_expr(pat)
            return f'{var} == {val}'
        if isinstance(pat, ListExpr):
            conds = [f'len({var}) == {len(pat.elements)}']
            for i, el in enumerate(pat.elements):
                inner = self._compile_pattern_cond(f'{var}[{i}]', el)
                conds.append(inner)
            return ' and '.join(conds)
        return f'{var} == {self._compile_expr(pat)}'

    def _compile_pattern_bindings(self, var, pat):
        if isinstance(pat, Ident):
            if pat.name != '_':
                self._emit(f'{self._safe_name(pat.name)} = {var}')
        elif isinstance(pat, ListExpr):
            for i, el in enumerate(pat.elements):
                self._compile_pattern_bindings(f'{var}[{i}]', el)

    def _compile_expr(self, e: Expr) -> str:
        if e is None:
            return 'None'
        if isinstance(e, IntLit):
            return str(e.value)
        if isinstance(e, FloatLit):
            return repr(e.value)
        if isinstance(e, BoolLit):
            return 'True' if e.value else 'False'
        if isinstance(e, StringLit):
            return repr(e.value)
        if isinstance(e, NoneLit):
            return 'None'
        if isinstance(e, Ident):
            return self._safe_name(e.name)
        if isinstance(e, BinOp):
            l = self._compile_expr(e.left)
            r = self._compile_expr(e.right)
            op = e.op
            if op == 'and':
                return f'({l} and {r})'
            if op == 'or':
                return f'({l} or {r})'
            if op == '+':
                return f'({l} + {r})'
            if op in ('==', '!=', '<', '>', '<=', '>='):
                return f'({l} {op} {r})'
            lt = self._infer_type(e.left)
            rt = self._infer_type(e.right)
            if (lt == 'ndarray' or rt == 'ndarray') and self._needs_triad_native:
                return f'({l} {op} {r})'
            return f'({l} {op} {r})'
        if isinstance(e, UnaryOp):
            val = self._compile_expr(e.operand)
            if e.op == 'not':
                return f'(not {val})'
            if e.op in ('*', '**'):
                return f'{e.op}{val}'
            return f'({e.op}{val})'
        if isinstance(e, CallExpr):
            func = self._compile_expr(e.func)
            args = [self._compile_expr(a) for a in e.args]
            kwargs = [f'{k}={self._compile_expr(v)}' for k, v in e.kwargs.items()]
            all_args = ', '.join(args + kwargs)
            return f'{func}({all_args})'
        if isinstance(e, MethodCallExpr):
            obj = self._compile_expr(e.obj)
            method = e.method
            if method == 'push':
                method = 'append'
            elif method == 'contains':
                args = [self._compile_expr(a) for a in e.args]
                return f'({args[0]} in {obj})'
            elif method == 'len':
                return f'len({obj})'
            args = [self._compile_expr(a) for a in e.args]
            kwargs = [f'{k}={self._compile_expr(v)}' for k, v in e.kwargs.items()]
            all_args = ', '.join(args + kwargs)
            return f'{obj}.{method}({all_args})'
        if isinstance(e, IndexExpr):
            if isinstance(e.index, SliceExpr):
                s = e.index
                start = self._compile_expr(s.start) if s.start else ''
                end = self._compile_expr(s.end) if s.end else ''
                if s.step:
                    step = self._compile_expr(s.step)
                    return f'{self._compile_expr(e.obj)}[{start}:{end}:{step}]'
                return f'{self._compile_expr(e.obj)}[{start}:{end}]'
            return f'{self._compile_expr(e.obj)}[{self._compile_expr(e.index)}]'
        if isinstance(e, FieldExpr):
            return f'{self._compile_expr(e.obj)}.{e.field}'
        if isinstance(e, ListExpr):
            elems = ', '.join((self._compile_expr(el) for el in e.elements))
            return f'[{elems}]'
        if isinstance(e, TupleExpr):
            if len(e.elements) == 0:
                return '()'
            if len(e.elements) == 1:
                return f'({self._compile_expr(e.elements[0])},)'
            elems = ', '.join((self._compile_expr(el) for el in e.elements))
            return f'({elems})'
        if isinstance(e, FStringExpr):
            parts = []
            for ptype, pval in e.parts:
                if ptype == 'str':
                    escaped = pval.replace('\\', '\\\\').replace("'", "\\'").replace('{', '{{').replace('}', '}}')
                    parts.append(escaped)
                else:
                    parts.append('{' + self._compile_expr(pval) + '}')
            return "f'" + ''.join(parts) + "'"
        if isinstance(e, ListCompExpr):
            vec = self._try_vectorize_listcomp(e)
            if vec is not None:
                return vec
            expr = self._compile_expr(e.expr)
            var = self._safe_name(e.var)
            iter_expr = self._compile_expr(e.iter)
            if e.condition:
                cond = self._compile_expr(e.condition)
                return f'[{expr} for {var} in {iter_expr} if {cond}]'
            return f'[{expr} for {var} in {iter_expr}]'
        if isinstance(e, MapExpr):
            pairs = ', '.join((f'{self._compile_expr(k)}: {self._compile_expr(v)}' for k, v in e.pairs))
            return f'{{{pairs}}}'
        if isinstance(e, LambdaExpr):
            params = ', '.join((self._compile_param(p) for p in e.params))
            return f'(lambda {params}: None)'
        if isinstance(e, AssignExpr):
            return f'({self._compile_expr(e.target)} := {self._compile_expr(e.value)})'
        if isinstance(e, YieldExpr):
            if e.value:
                return f'(yield {self._compile_expr(e.value)})'
            return '(yield)'
        if isinstance(e, AwaitExpr):
            return f'(await {self._compile_expr(e.value)})'
        return 'None'

    def _compile_param(self, p) -> str:
        if isinstance(p, Param):
            prefix = ''
            if p.is_kwargs:
                prefix = '**'
            elif p.is_args:
                prefix = '*'
            if p.default is not None:
                return f'{prefix}{self._safe_name(p.name)}={self._compile_expr(p.default)}'
            return f'{prefix}{self._safe_name(p.name)}'
        return str(p)

    def _iter_type(self, iter_expr) -> str | None:
        if isinstance(iter_expr, Ident):
            return self._var_types.get(iter_expr.name)
        return self._infer_type(iter_expr)

    def _try_vectorize_for(self, s) -> bool:
        if len(s.body) != 1:
            return False
        stmt = s.body[0]
        var = self._safe_name(s.var)
        iter_code = self._compile_expr(s.iter)
        iter_type = self._iter_type(s.iter)
        if isinstance(stmt, ExprStmt) and isinstance(stmt.expr, MethodCallExpr):
            mc = stmt.expr
            if mc.method == 'push' and len(mc.args) == 1:
                target = self._compile_expr(mc.obj)
                expr = self._compile_expr(mc.args[0])
                self._emit(f'{target}.extend([{expr} for {var} in {iter_code}])')
                return True
        if isinstance(stmt, AssignStmt) and isinstance(stmt.target, Ident) and isinstance(stmt.value, BinOp):
            target_code = self._compile_expr(stmt.target)
            left_code = self._compile_expr(stmt.value.left)
            if target_code == left_code and target_code != var:
                right_code = self._compile_expr(stmt.value.right)
                op = stmt.value.op
                if op == '+' and right_code == var and (iter_type == 'ndarray'):
                    self._emit(f'{target_code} += float(_np.sum({iter_code}))')
                    return True
                if op == '+':
                    target_type = self._var_types.get(stmt.target.name if isinstance(stmt.target, Ident) else None)
                    if target_type not in ('str', 'list'):
                        self._emit(f'{target_code} += sum({right_code} for {var} in {iter_code})')
                        return True
                if op == '*':
                    self._emit(f'import functools as _ft; {target_code} *= _ft.reduce(lambda _a, _b: _a * _b, ({right_code} for {var} in {iter_code}), 1)')
                    return True
        return False

    def _try_vectorize_listcomp(self, e: ListCompExpr) -> str | None:
        if not self._needs_triad_native:
            return None
        iter_type = self._infer_type(e.iter)
        if iter_type != 'ndarray':
            return None
        iter_code = self._compile_expr(e.iter)
        var = self._safe_name(e.var)
        expr_type = self._infer_type(e.expr)
        if expr_type not in ('int', 'float', 'ndarray'):
            return None
        compiled_expr = self._compile_expr(e.expr)
        if compiled_expr == var:
            return f'_np.array({iter_code})'
        if e.condition:
            cond = self._compile_expr(e.condition)
            return f'_np.array([{compiled_expr} for {var} in {iter_code} if {cond}])'
        simple_binop = isinstance(e.expr, BinOp) and (not e.condition)
        if simple_binop:
            return f'_np.array({compiled_expr})'
        return None
        stmt = s.body[0]
        var = self._safe_name(s.var)
        iter_code = self._compile_expr(s.iter)
        iter_type = self._iter_type(s.iter)
        if isinstance(stmt, ExprStmt) and isinstance(stmt.expr, MethodCallExpr):
            mc = stmt.expr
            if mc.method == 'push' and len(mc.args) == 1:
                target = self._compile_expr(mc.obj)
                expr = self._compile_expr(mc.args[0])
                self._emit(f'{target}.extend([{expr} for {var} in {iter_code}])')
                return True
        if isinstance(stmt, AssignStmt) and isinstance(stmt.target, Ident) and isinstance(stmt.value, BinOp):
            target_code = self._compile_expr(stmt.target)
            left_code = self._compile_expr(stmt.value.left)
            if target_code == left_code and target_code != var:
                right_code = self._compile_expr(stmt.value.right)
                op = stmt.value.op
                if op == '+' and right_code == var and (iter_type == 'ndarray'):
                    self._emit(f'{target_code} += float(_np.sum({iter_code}))')
                    return True
                if op == '+':
                    target_type = self._var_types.get(stmt.target.name if isinstance(stmt.target, Ident) else None)
                    if target_type not in ('str', 'list'):
                        self._emit(f'{target_code} += sum({right_code} for {var} in {iter_code})')
                        return True
                if op == '*':
                    self._emit(f'import functools as _ft; {target_code} *= _ft.reduce(lambda _a, _b: _a * _b, ({right_code} for {var} in {iter_code}), 1)')
                    return True
        return False

    def _compile_type_decl(self, s: TypeDecl):
        name = self._safe_name(s.name)
        field_names = [f.name for f in s.fields]
        defaults = {}
        for f in s.fields:
            if f.default is not None:
                defaults[f.name] = self._compile_expr(f.default)
        self._emit(f'class {name}:')
        self._indent += 1
        params = []
        for f in s.fields:
            if f.name in defaults:
                params.append(f'{f.name}={defaults[f.name]}')
            else:
                params.append(f'{f.name}=None')
        self._emit(f"def __init__(self, {', '.join(params)}):")
        self._indent += 1
        for f in s.fields:
            self._emit(f'self.{f.name} = {f.name}')
        self._indent -= 1
        repr_parts = ', '.join((f'{f.name}={{self.{f.name}!r}}' for f in s.fields))
        self._emit(f"def __repr__(self): return f'{name}({repr_parts})'")
        for m in s.methods:
            params = ', '.join(['self'] + [self._compile_param(p) for p in m.params if (p.name if isinstance(p, Param) else p) != 'self'])
            self._emit(f'def {m.name}({params}):')
            self._indent += 1
            self._compile_body(m.body)
            self._indent -= 1
        self._indent -= 1
        self._emit('')

    def _compile_import(self, s: ImportStmt):
        key = '.'.join(s.path)
        alias = self._safe_name(s.alias or s.path[-1])
        self._emit(f'{alias} = _tri_import({s.path!r}, _search_paths)')

    def _compile_from_import(self, s: FromImportStmt):
        key = '.'.join(s.path)
        tmp = f"_mod_{'_'.join(s.path)}"
        self._emit(f'{tmp} = _tri_import({s.path!r}, _search_paths)')
        for name in s.names:
            self._emit(f'{self._safe_name(name)} = getattr({tmp}, {name!r})')

    def _find_nonlocals(self, body: list[Stmt], params: set[str]) -> set[str]:
        assigned = set()
        declared = set(params)

        def scan(stmts):
            for s in stmts:
                if isinstance(s, LetStmt):
                    declared.add(s.name)
                elif isinstance(s, ConstStmt):
                    declared.add(s.name)
                elif isinstance(s, ForStmt):
                    declared.add(s.var)
                    scan(s.body)
                elif isinstance(s, AssignStmt):
                    if isinstance(s.target, Ident):
                        assigned.add(s.target.name)
                elif isinstance(s, IfStmt):
                    scan(s.then_body)
                    for _, b in s.elif_clauses:
                        scan(b)
                    if s.else_body:
                        scan(s.else_body)
                elif isinstance(s, WhileStmt):
                    scan(s.body)
                elif isinstance(s, FnDecl):
                    declared.add(s.name)
                elif isinstance(s, ImportStmt):
                    declared.add(s.alias or s.path[-1])
                elif isinstance(s, FromImportStmt):
                    for name in s.names:
                        declared.add(name)
        scan(body)
        return assigned - declared

    def _safe_name(self, name: str) -> str:
        PYTHON_KEYWORDS = {'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield'}
        if name in PYTHON_KEYWORDS:
            return f'_tri_{name}'
        if name == 'self':
            return 'self'
        if name == 'print':
            return '_tri_print'
        if name == 'input':
            return 'input'
        if name == 'type':
            return '_tri_type'
        return name

    def _make_globals(self, file_path: str='') -> dict:
        g = {}
        g['_tri_print'] = _tri_print
        g['_TriObj'] = _TriObj
        g['_TriModule'] = _TriModule
        g['_tri_import'] = _tri_import
        g['_search_paths'] = self._search_paths
        g['_math'] = math
        g['_random'] = random
        g['_json'] = json
        g['_os'] = os
        g['_time_mod'] = _time_mod
        if self._needs_triad_native:
            import numpy
            g['_np'] = numpy
        elif 'numpy' in sys.modules:
            g['_np'] = sys.modules['numpy']
        else:
            g['_np'] = None
        g['len'] = len
        g['range'] = range
        g['enumerate'] = lambda xs: [list(p) for p in enumerate(xs)]
        g['str'] = _tri_str
        g['int'] = int
        g['float'] = float
        g['_tri_type'] = lambda x: type(x).__name__
        g['abs'] = abs
        g['min'] = min
        g['max'] = max
        g['append'] = lambda lst, x: lst.append(x) or lst
        g['sorted'] = sorted
        g['reversed'] = lambda x: list(reversed(x))
        g['__builtins__'] = __builtins__
        return g

def _tri_print(*args):
    parts = []
    for a in args:
        if a is None:
            parts.append('none')
        elif isinstance(a, bool):
            parts.append('true' if a else 'false')
        else:
            parts.append(str(a))
    print(' '.join(parts))

def _tri_str(x):
    if x is None:
        return 'none'
    if isinstance(x, bool):
        return 'true' if x else 'false'
    return str(x)

class _TriObj:

    def __init__(self, _type, **kw):
        self._type = _type
        self.__dict__.update(kw)

    def __repr__(self):
        d = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return f'{self._type}({d})'

class _TriModule:

    def __init__(self, name, ns):
        self._name = name
        self.__dict__.update(ns)

    def __repr__(self):
        return f'<module {self._name}>'
_STDLIB = {}

def _init_stdlib():
    _STDLIB['math'] = _TriModule('math', {'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan, 'log': math.log, 'log10': math.log10, 'exp': math.exp, 'floor': math.floor, 'ceil': math.ceil, 'abs': abs, 'pi': math.pi, 'e': math.e, 'min': min, 'max': max, 'clamp': lambda x, lo, hi: max(lo, min(x, hi)), 'pow': pow})
    _STDLIB['random'] = _TriModule('random', {'random': random.random, 'randint': random.randint, 'choice': random.choice, 'seed': random.seed, 'shuffle': lambda x: random.shuffle(x) or x, 'uniform': random.uniform})
    _STDLIB['io'] = _TriModule('io', {'print': _tri_print, 'input': input})
    _STDLIB['string'] = _TriModule('string', {'split': lambda s, sep=None: s.split(sep), 'join': lambda sep, lst: sep.join((str(x) for x in lst)), 'replace': lambda s, old, new: s.replace(old, new), 'lower': lambda s: s.lower(), 'upper': lambda s: s.upper(), 'strip': lambda s: s.strip(), 'starts_with': lambda s, p: s.startswith(p), 'ends_with': lambda s, p: s.endswith(p), 'contains': lambda s, p: p in s})
    _STDLIB['json'] = _TriModule('json', {'parse': json.loads, 'stringify': lambda x, indent=None: json.dumps(x, indent=indent, default=str)})
    _STDLIB['fs'] = _TriModule('fs', {'read_text': lambda p: open(p).read(), 'write_text': lambda p, c: open(p, 'w').write(c), 'exists': os.path.exists, 'listdir': os.listdir})
    _STDLIB['time'] = _TriModule('time', {'now': _time_mod.time, 'sleep': _time_mod.sleep})
    _STDLIB['collections'] = _TriModule('collections', {'len': len, 'range': lambda *a: list(range(*a)), 'enumerate': lambda xs: [list(p) for p in enumerate(xs)], 'sorted': sorted, 'reversed': lambda x: list(reversed(x)), 'zip': lambda *args: [list(t) for t in zip(*args)], 'map': lambda f, xs: [f(x) for x in xs], 'filter': lambda f, xs: [x for x in xs if f(x)]})
    from stdlib import plot as _plot_mod
    _STDLIB['plot'] = _TriModule('plot', {'line': _plot_mod.line, 'scatter': _plot_mod.scatter, 'heatmap': _plot_mod.heatmap, 'histogram': _plot_mod.histogram, 'bar': _plot_mod.bar, 'save': _plot_mod.save, 'show': _plot_mod.show})
_init_stdlib()

def _lazy_triad_module():
    from stdlib import triad_module as _triad_mod
    return _TriModule('triad', {'regime': _triad_mod.regime, 'solve': _triad_mod.solve, 'fast_solve': _triad_mod.fast_solve, 'solve_coupled': _triad_mod.solve_coupled, 'k_star': _triad_mod.k_star, 'crystal': _triad_mod.crystal, 'peak': _triad_mod.peak, 'norm': _triad_mod.norm, 'ipr': _triad_mod.get_ipr, 'fwhm': _triad_mod.get_fwhm, 'spectrum': _triad_mod.spectrum, 'pr': _triad_mod.pr, 'TriadParams': _triad_mod.TriadParams})

def _lazy_tensor_module():
    from runtime import tensor as _t
    from runtime import ml_device as _md
    return _TriModule('triad.tensor', {'tensor': _t.tensor, 'zeros': _t.zeros, 'ones': _t.ones, 'randn': _t.randn, 'rand': _t.rand, 'arange': _t.arange, 'linspace': _t.linspace, 'eye': _t.eye, 'exp': _t.exp, 'log': _t.log, 'sqrt': _t.sqrt, 'sin': _t.sin, 'cos': _t.cos, 'tanh': _t.tanh, 'sigmoid': _t.sigmoid, 'relu': _t.relu, 'softmax': _t.softmax, 'cross_entropy': _t.cross_entropy, 'mse_loss': _t.mse_loss, 'cat': _t.cat, 'stack': _t.stack, 'bmm': _t.bmm, 'transpose': _t.transpose, 'layer_norm': _t.layer_norm, 'no_grad': _t.no_grad, 'TriadTensor': _t.TriadTensor, 'set_device': _md.set_device, 'device': _md.device, 'cuda_available': _md.cuda_available, 'is_gpu': _md.is_gpu, 'sync': _md.sync, 'asnumpy': _md.asnumpy, 'mem_info': _md.mem_info})

def _lazy_nn_module():
    from runtime import nn as _nn
    return _TriModule('triad.nn', {'Module': _nn.Module, 'Linear': _nn.Linear, 'Conv1d': _nn.Conv1d, 'Conv2d': _nn.Conv2d, 'ReLU': _nn.ReLU, 'Tanh': _nn.Tanh, 'Sigmoid': _nn.Sigmoid, 'Softmax': _nn.Softmax, 'Sequential': _nn.Sequential, 'Flatten': _nn.Flatten, 'Dropout': _nn.Dropout, 'BatchNorm1d': _nn.BatchNorm1d, 'Parameter': _nn.Parameter, 'Embedding': _nn.Embedding, 'LayerNorm': _nn.LayerNorm, 'MultiHeadAttention': _nn.MultiHeadAttention, 'FeedForward': _nn.FeedForward, 'TransformerBlock': _nn.TransformerBlock, 'Transformer': _nn.Transformer, 'TriadSSM': _nn.TriadSSM, 'TriadSSMBlock': _nn.TriadSSMBlock, 'SGD': _nn.SGD, 'Adam': _nn.Adam})

def _lazy_data_module():
    from runtime import data as _data
    return _TriModule('triad.data', {'Dataset': _data.Dataset, 'DataLoader': _data.DataLoader})

def _lazy_losses_module():
    from runtime import losses as _losses
    return _TriModule('triad.losses', {'bce_loss': _losses.bce_loss, 'binary_cross_entropy_with_logits': _losses.binary_cross_entropy_with_logits, 'huber_loss': _losses.huber_loss, 'kl_div': _losses.kl_div, 'cosine_similarity_loss': _losses.cosine_similarity_loss, 'l1_loss': _losses.l1_loss, 'smooth_l1_loss': _losses.smooth_l1_loss, 'nll_loss': _losses.nll_loss})

def _lazy_metrics_module():
    from runtime import metrics as _metrics
    return _TriModule('triad.metrics', {'accuracy': _metrics.accuracy, 'top_k_accuracy': _metrics.top_k_accuracy, 'perplexity': _metrics.perplexity, 'confusion_matrix': _metrics.confusion_matrix, 'precision_recall_f1': _metrics.precision_recall_f1, 'r2_score': _metrics.r2_score, 'mean_absolute_error': _metrics.mean_absolute_error, 'root_mean_squared_error': _metrics.root_mean_squared_error})

def _lazy_trainer_module():
    from runtime import trainer as _trainer
    from runtime import serialization as _serialization
    return _TriModule('triad.train', {'Trainer': _trainer.Trainer, 'EarlyStopping': _trainer.EarlyStopping, 'LRScheduler': _trainer.LRScheduler, 'LossHistory': _trainer.LossHistory, 'save_weights': _serialization.save_weights, 'load_weights': _serialization.load_weights, 'save_checkpoint': _serialization.save_checkpoint, 'load_checkpoint': _serialization.load_checkpoint})

def _tri_import(path: list[str], search_paths: list[str]):
    key = '.'.join(path)
    if key == 'triad':
        if key not in _STDLIB:
            _STDLIB[key] = _lazy_triad_module()
        return _STDLIB[key]
    if key == 'triad.tensor':
        if key not in _STDLIB:
            _STDLIB[key] = _lazy_tensor_module()
        return _STDLIB[key]
    if key == 'triad.nn':
        if key not in _STDLIB:
            _STDLIB[key] = _lazy_nn_module()
        return _STDLIB[key]
    if key == 'triad.data':
        if key not in _STDLIB:
            _STDLIB[key] = _lazy_data_module()
        return _STDLIB[key]
    if key == 'triad.losses':
        if key not in _STDLIB:
            _STDLIB[key] = _lazy_losses_module()
        return _STDLIB[key]
    if key == 'triad.metrics':
        if key not in _STDLIB:
            _STDLIB[key] = _lazy_metrics_module()
        return _STDLIB[key]
    if key == 'triad.train':
        if key not in _STDLIB:
            _STDLIB[key] = _lazy_trainer_module()
        return _STDLIB[key]
    if key in _STDLIB:
        return _STDLIB[key]
    for sp in search_paths:
        rel = os.path.join(sp, *path) + '.tri'
        if os.path.exists(rel):
            return _load_tri_module(rel, key, search_paths)
        rel_dir = os.path.join(sp, *path, '__init__.tri')
        if os.path.exists(rel_dir):
            return _load_tri_module(rel_dir, key, search_paths)
    modules_dir = os.path.join(os.getcwd(), 'triad_modules')
    if os.path.isdir(modules_dir) and len(path) >= 1:
        pkg_rest = path[1:] if len(path) > 1 else []
        if pkg_rest:
            rel = os.path.join(modules_dir, path[0], *pkg_rest) + '.tri'
        else:
            rel = os.path.join(modules_dir, path[0], '__init__.tri')
        if os.path.exists(rel):
            return _load_tri_module(rel, key, search_paths)
        if not pkg_rest:
            main_rel = os.path.join(modules_dir, path[0], 'main.tri')
            if os.path.exists(main_rel):
                return _load_tri_module(main_rel, key, search_paths)
    stdlib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stdlib')
    for sp in [stdlib_dir]:
        rel = os.path.join(sp, *path) + '.tri'
        if os.path.exists(rel):
            return _load_tri_module(rel, key, search_paths)
    try:
        return __import__(key)
    except ImportError:
        pass
    raise ImportError(f"cannot resolve import '{key}'")
_MODULE_CACHE: dict[str, _TriModule] = {}

def _load_tri_module(filepath: str, key: str, search_paths: list[str]) -> _TriModule:
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    from frontend.parser_universal import parse
    with open(filepath) as f:
        src = f.read()
    mod_ast = parse(src, filepath)
    compiler = TriadCompiler()
    compiler._search_paths = [os.path.dirname(os.path.abspath(filepath))] + search_paths
    code = compiler.compile_to_source(mod_ast)
    env = compiler._make_globals(filepath)
    env['_search_paths'] = compiler._search_paths
    compiled = compile(code, filepath, 'exec')
    exec(compiled, env)
    ns = {k: v for k, v in env.items() if not k.startswith('_') and k not in ('math', 'random', 'json', 'os', 'time')}
    mod = _TriModule(key, ns)
    _MODULE_CACHE[key] = mod
    return mod