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
from runtime.source_map import SourceMap

class CompileError(Exception):
    pass
_COMPILER_VERSION = 9

class TriadCompiler:

    def __init__(self):
        self._indent = 0
        self._lines: list[str] = []
        self._module_cache: dict[str, dict] = {}
        self._search_paths: list[str] = ['.']
        self._scope_depth = 0
        self._var_types: dict[str, str] = {}
        self._sourcemap = None
        self._needs_triad_native = False
        self._par_next = False
        self._gpu_next = False
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
        
        if self._sourcemap is not None:
            env['_triad_sourcemap'] = self._sourcemap
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
        source_file = mod.file or '<triad>'
        self._sourcemap = SourceMap(source_file)
        self._needs_triad_native = self._detect_triad_native(mod.body)
        self._emit_prelude()
        for stmt in mod.body:
            self._sourcemap.mark_stmt(stmt)
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
        if self._sourcemap is not None:
            self._sourcemap.record_line()

    def _emit_prelude(self):
        if self._needs_triad_native:
            if self._backend == 'vm':
                self._emit('from runtime.vm_runtime import VMRuntime as _MRT, CouplingEdge as _CE, Segment as _Seg')
            else:
                self._emit('from runtime.core.multi_runtime import MultiRuntime as _MRT, CouplingEdge as _CE, Segment as _Seg')
            self._emit('from stdlib.regimes import resolve_regime as _resolve_regime')
            self._emit('from runtime.physics.observables import dominant_wavenumber as _obs_kstar, crystallinity as _obs_C, peak_density as _obs_peak, ipr as _obs_ipr, fwhm as _obs_fwhm')
            self._emit('from runtime.physics.observables_atoms import atom_count_nd as _obs_atom_count')
            self._emit('_triad_rt = _MRT(dt=0.005, record_every=4)')
            self._emit('_triad_subs = {}')
            self._emit('_triad_T = 10.0')
            self._emit('_triad_pending_edges = []')
            self._emit('_triad_pending_duration = None')
            self._emit('')

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
            loop_id = f'for_{s.pos.line}_{s.pos.col}'
            if self._par_next:
                self._par_next = False
                self._compile_par_for(s)
            elif not self._try_vectorize_for(s):
                self._emit(f'_triad_jit_hit({loop_id!r})')
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
            fn_name = self._safe_name(s.name)
            is_gpu = self._gpu_next
            self._gpu_next = False
            if is_gpu:
                
                self._emit(f'{prefix}def _gpu_inner_{fn_name}({params}):')
            else:
                self._emit(f'{prefix}def {fn_name}({params}):')
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
            if is_gpu:
                
                inner_name = f'_gpu_inner_{fn_name}'
                param_names = [p.name if isinstance(p, Param) else p for p in s.params]
                args_str = ', '.join(param_names)
                self._emit(f'def {fn_name}({args_str}):')
                self._indent += 1
                self._emit(f'return gpu_map({inner_name}, {param_names[0]})' if len(param_names) == 1 else f'return gpu_fused({inner_name}, {args_str})')
                self._indent -= 1
            if s.decorators:
                for dec in s.decorators:
                    self._emit(f'{fn_name} = {self._compile_expr(dec)}({fn_name})')
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
                    self._emit("from runtime.physics.observables_atoms import atom_count_nd as _atom_count_nd")
                    self._emit(f"_obs_row['atom_count'] = float(_atom_count_nd(_obs_sub.psi, _obs_sub.dx))")
                else:
                    self._emit(f"_obs_row[{m!r}] = 'unknown_metric'")
            self._emit(f"_obs_parts = [f'{{k}}={{v:.4f}}' for k, v in _obs_row.items()]")
            self._emit(f"_tri_print({target!r}, '=', '{{', ', '.join(_obs_parts), '}}')")
        elif isinstance(s, RunStmt):
            dur = self._compile_expr(s.duration) if s.duration else '_triad_T'
            if s.target and s.duration:
                self._emit(f'_triad_rt.add_segment(_Seg(t_start=_triad_rt.global_t, t_end=_triad_rt.global_t + {dur}, edges=[], active_ids={{_triad_subs[{s.target!r}].id}}))')
                self._emit(f'_triad_rt.global_t += {dur}')
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
        elif isinstance(s, WithStmt):
            ctx_expr = self._compile_expr(s.expr)
            if s.var:
                var_name = self._safe_name(s.var)
                self._emit(f'with {ctx_expr} as {var_name}:')
            else:
                self._emit(f'with {ctx_expr}:')
            self._indent += 1
            self._compile_body(s.body)
            self._indent -= 1
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
            elif s.key == 'par':
                self._par_next = True
            elif s.key == 'gpu':
                self._gpu_next = True
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
                metrics_map = {'k_star': '_obs_kstar', 'crystallinity': '_obs_C', 'peak': '_obs_peak', 'ipr': '_obs_ipr', 'fwhm': '_obs_fwhm', 'atom_count': '_obs_atom_count'}
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
        if isinstance(e, BytesLit):
            return 'b' + repr(e.value)
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
            if isinstance(e.func, Ident) and e.func.name == 'str':
                func = '_tri_str'
            else:
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
                if not e.args and not e.kwargs:
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
            for part in e.parts:
                if part[0] == 'str':
                    escaped = part[1].replace('\\', '\\\\').replace("'", "\\'").replace('{', '{{').replace('}', '}}')
                    parts.append(escaped)
                else:
                    fmt_spec = part[2] if len(part) > 2 and part[2] else None
                    compiled_expr = self._compile_expr(part[1])
                    if fmt_spec:
                        parts.append('{' + compiled_expr + ':' + fmt_spec + '}')
                    else:
                        parts.append('{' + compiled_expr + '}')
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
            if e.body and len(e.body) > 0:
                fn_id = f'_tri_lambda_{len(self._lines)}'
                self._emit(f'def {fn_id}({params}):')
                self._indent += 1
                self._compile_body(e.body)
                self._indent -= 1
                return fn_id
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

    def _compile_par_for(self, s):
        """Compile a @par for-loop that runs iterations concurrently."""
        var = self._safe_name(s.var)
        iter_code = self._compile_expr(s.iter)
        
        saved_lines = self._lines
        saved_indent = self._indent
        self._lines = []
        self._indent = 0
        self._compile_body(s.body)
        body_lines = list(self._lines)
        self._lines = saved_lines
        self._indent = saved_indent
        
        if not hasattr(self, '_par_counter'):
            self._par_counter = 0
        self._par_counter += 1
        helper_name = f'_tri_par_body_{self._par_counter}'
        self._emit(f'def {helper_name}({var}):')
        self._indent += 1
        for line in body_lines:
            self._emit(line)
        self._indent -= 1
        self._emit(f'list(_tri_par_map({helper_name}, {iter_code}))')

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
                        self._emit(f'{target_code} += _tri_sum({right_code} for {var} in {iter_code})')
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
        if s.alias:
            alias = self._safe_name(s.alias)
        elif len(s.path) > 1:
            alias = self._safe_name(s.path[0])
        else:
            alias = self._safe_name(s.path[-1])
        self._emit(f'{alias} = _tri_import({s.path!r}, _search_paths)')

    def _compile_from_import(self, s: FromImportStmt):
        key = '.'.join(s.path)
        tmp = f"_mod_{'_'.join(s.path)}"
        self._emit(f'{tmp} = _tri_import({s.path!r}, _search_paths)')
        aliases = s.aliases if s.aliases else [None] * len(s.names)
        for i, name in enumerate(s.names):
            target = self._safe_name(aliases[i]) if aliases[i] else self._safe_name(name)
            self._emit(f'{target} = getattr({tmp}, {name!r})')

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
        PYTHON_KEYWORDS = {'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'case', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'match', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield'}
        if name in PYTHON_KEYWORDS:
            return f'_tri_{name}'
        if name == 'self':
            return 'self'
        if name == 'print':
            return '_tri_print'
        if name == 'input':
            return 'input'
        if name == 'type':
            return 'type'
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
        try:
            import numpy as _numpy_mod
            g['_np'] = _numpy_mod
        except ImportError:
            g['_np'] = None
        g['len'] = len
        g['_tri_sum'] = sum
        g['sqrt'] = math.sqrt
        g['sin'] = math.sin
        g['cos'] = math.cos
        g['exp'] = math.exp
        g['log'] = math.log
        g['range'] = range
        g['enumerate'] = lambda xs: [list(p) for p in enumerate(xs)]
        g['str'] = str
        g['_tri_str'] = _tri_str
        g['_tri_None'] = None
        g['_tri_True'] = True
        g['_tri_False'] = False
        g['int'] = int
        g['float'] = float
        g['type'] = type
        g['abs'] = abs
        g['min'] = min
        g['max'] = max
        g['append'] = lambda lst, x: lst.append(x) or lst
        g['sorted'] = sorted
        g['reversed'] = lambda x: list(reversed(x))
        g['ccall'] = _tri_ccall
        g['solver_solve'] = _tri_solver_solve
        g['py_call'] = _tri_py_call
        g['py_eval'] = _tri_py_eval
        g['py_exec'] = _tri_py_exec
        g['run_async'] = _tri_async_run
        g['sleep'] = _tri_sleep
        g['gather'] = _tri_gather
        g['create_task'] = _tri_create_task
        g['async_read'] = _tri_async_read_file
        g['async_write'] = _tri_async_write_file
        g['async_fetch'] = _tri_async_fetch
        g['_tri_par_map'] = _tri_par_map
        g['set_device'] = _tri_set_device
        g['device'] = _tri_device
        g['cuda_available'] = _tri_cuda_available
        from runtime import jit_tiered as _jit_mod
        g['_triad_jit_hit'] = _jit_mod._triad_jit_hit
        g['_triad_jit_is_hot'] = _jit_mod._triad_jit_is_hot
        from runtime.ml import gpu_kernel as _gk
        g['gpu_map'] = _gk.gpu_map
        g['gpu_fused'] = _gk.gpu_fused_map
        g['gpu_available'] = _gk.gpu_available
        g['gpu_array'] = _gk.gpu_array
        g['gpu_to_cpu'] = _gk.gpu_to_cpu
        g['gpu_info'] = _gk.gpu_info
        g['__builtins__'] = __builtins__
        g['__name__'] = '__triad__'
        g['__file__'] = file_path or '__triad__'
        return g

def _tri_solver_solve(cfg=None):
    """run the triad p1 plus p2 plus p3 solver from interpreted .tri code."""
    from dataclasses import fields as _dc_fields
    import numpy as _np_local
    from runtime.core.solver import TriadParams as _TriadParams, integrate as _triad_integrate

    raw = dict(cfg or {})
    
    raw.pop('mode', None)
    raw.setdefault('backend', 'cpu')

    allowed = {f.name for f in _dc_fields(_TriadParams)}
    params = {k: v for k, v in raw.items() if k in allowed}
    p = _TriadParams(**params)
    r = _triad_integrate(p)

    density = _np_local.asarray(r.get('density'))
    if density.ndim > 1:
        density_final = _np_local.asarray(density[-1], dtype=float)
    else:
        density_final = _np_local.asarray(density, dtype=float)
    x = _np_local.asarray(r.get('x'), dtype=float)
    dx = float(r.get('dx', p.L / p.N))
    norm = float(density_final.sum() * dx)
    peak = float(density_final.max()) if density_final.size else 0.0

    psi_final = _np_local.asarray(r.get('psi_final'))
    if psi_final.size:
        from runtime.physics.observables import crystallinity as _cryst
        cryst = float(_cryst(psi_final, dx))
    else:
        cryst = 0.0

    return {
        'N': int(p.N),
        'norm': norm,
        'peak': peak,
        'dx': dx,
        'crystallinity': cryst,
        'density': density_final.tolist(),
        'x': x.tolist(),
    }

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

import ctypes
import ctypes.util as _ctypes_util
import threading as _threading

_TRI_CCALL_CACHE: dict[str, ctypes.CDLL] = {}
_TRI_CCALL_LOCK = _threading.Lock()

_C_TYPE_MAP = {
    'int': ctypes.c_int,
    'cint': ctypes.c_int,
    'uint': ctypes.c_uint,
    'long': ctypes.c_long,
    'ulong': ctypes.c_ulong,
    'longlong': ctypes.c_longlong,
    'float': ctypes.c_float,
    'cfloat': ctypes.c_float,
    'double': ctypes.c_double,
    'cdouble': ctypes.c_double,
    'char': ctypes.c_char,
    'cchar': ctypes.c_char,
    'string': ctypes.c_char_p,
    'cstring': ctypes.c_char_p,
    'void': None,
    'void*': ctypes.c_void_p,
    'ptr': ctypes.c_void_p,
    'size_t': ctypes.c_size_t,
    'bool': ctypes.c_bool,
    'cbool': ctypes.c_bool,
}

def _tri_load_lib(lib_name: str) -> ctypes.CDLL:
    with _TRI_CCALL_LOCK:
        if lib_name in _TRI_CCALL_CACHE:
            return _TRI_CCALL_CACHE[lib_name]
        
        try:
            lib = ctypes.CDLL(lib_name)
        except OSError:
            resolved = _ctypes_util.find_library(lib_name.replace('lib', '').replace('.so', '').replace('.so.6', ''))
            if resolved is None:
                raise OSError(f'ccall: cannot find library "{lib_name}"')
            lib = ctypes.CDLL(resolved)
        _TRI_CCALL_CACHE[lib_name] = lib
        return lib

def _tri_ccall(lib_name: str, func_name: str, *args,
               ret_type: str = 'double', arg_types: list = None):
    """Call a C function via FFI.
    
    ccall("libm.so.6", "cos", 0.0)
    ccall("libc.so.6", "abs", -42, ret_type="int", arg_types=["int"])
    """
    _safe_guard('ccall')
    lib = _tri_load_lib(lib_name)
    func = getattr(lib, func_name)
    
    c_ret = _C_TYPE_MAP.get(ret_type, ctypes.c_double)
    func.restype = c_ret
    
    if arg_types:
        func.argtypes = [_C_TYPE_MAP.get(t, ctypes.c_double) for t in arg_types]
    else:
        inferred = []
        for a in args:
            if isinstance(a, int):
                inferred.append(ctypes.c_int)
            elif isinstance(a, float):
                inferred.append(ctypes.c_double)
            elif isinstance(a, str):
                inferred.append(ctypes.c_char_p)
            elif isinstance(a, bool):
                inferred.append(ctypes.c_bool)
            elif a is None:
                inferred.append(ctypes.c_void_p)
            else:
                inferred.append(ctypes.c_double)
        func.argtypes = inferred
    
    c_args = []
    for a, at in zip(args, func.argtypes):
        if at is ctypes.c_char_p and isinstance(a, str):
            c_args.append(a.encode('utf-8'))
        elif at is ctypes.c_void_p and a is None:
            c_args.append(None)
        else:
            c_args.append(a)
    
    result = func(*c_args)
    
    if c_ret is None:
        return None
    if isinstance(result, bytes):
        return result.decode('utf-8', errors='replace')
    return result

def _safe_guard(name: str):
    if _SAFE_MODE:
        raise PermissionError(
            f"{name} is disabled in safe mode (triad run --safe). "
            f"it can reach arbitrary python/native code, so it is blocked when "
            f"running untrusted programs.")

def _tri_py_call(module_name: str, func_name: str, *args):
    """Call a Python function by module and name."""
    _safe_guard('py_call')
    import importlib
    mod = importlib.import_module(module_name)
    fn = getattr(mod, func_name)
    return fn(*args)

def _tri_py_eval(code: str):
    """Evaluate a Python expression string."""
    _safe_guard('py_eval')
    return eval(code)

def _tri_py_exec(code: str):
    """Execute Python code string."""
    _safe_guard('py_exec')
    exec(code)

import asyncio as _asyncio

def _tri_async_run(coro):
    """Run an async function and return its result. Works from sync context."""
    try:
        loop = _asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_asyncio.run, coro)
            return future.result()
    return _asyncio.run(coro)

def _tri_sleep(seconds):
    """Async sleep. Use with: await sleep(1.0)"""
    return _asyncio.sleep(seconds)

def _tri_gather(*coros):
    """Run multiple async functions concurrently. await gather(fn1(), fn2())"""
    async def _run():
        return await _asyncio.gather(*coros)
    return _tri_async_run(_run())

def _tri_create_task(coro):
    """Schedule an async task. Returns Task."""
    return _asyncio.create_task(coro)

import concurrent.futures as _cf
_tri_par_pool = _cf.ThreadPoolExecutor(max_workers=None)  

def _tri_par_map(fn, iterable):
    """Map function over iterable in parallel using thread pool."""
    return _tri_par_pool.map(fn, iterable)

def _tri_set_device(dev='cpu', dtype='float32'):
    from runtime.ml import ml_device as _md
    _md.set_device(dev, dtype)

def _tri_device():
    from runtime.ml import ml_device as _md
    return _md.device()

def _tri_cuda_available():
    from runtime.ml import ml_device as _md
    return _md.cuda_available()

async def _tri_async_read_file(path):
    """Async file read. Returns string content."""
    try:
        import aiofiles
        async with aiofiles.open(path, 'r') as f:
            return await f.read()
    except ImportError:
        loop = _asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: open(path, 'r').read())

async def _tri_async_write_file(path, content):
    """Async file write."""
    try:
        import aiofiles
        async with aiofiles.open(path, 'w') as f:
            await f.write(content)
    except ImportError:
        loop = _asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: open(path, 'w').write(content))

async def _tri_async_fetch(url, method='GET', headers=None, body=None):
    """Async HTTP request. Returns {status, body, headers}."""
    import urllib.request
    req = urllib.request.Request(url, method=method,
                                  data=body.encode() if isinstance(body, str) else body)
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    loop = _asyncio.get_event_loop()
    resp = await loop.run_in_executor(None, lambda: urllib.request.urlopen(req))
    status = resp.status
    data = resp.read().decode('utf-8', errors='replace')
    return {'status': status, 'body': data}

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

    def __getattr__(self, attr):
        try:
            mod = __import__(self._name)
            for part in self._name.split('.')[1:]:
                mod = getattr(mod, part)
            return getattr(mod, attr)
        except (ImportError, AttributeError):
            raise AttributeError(f"module '{self._name}' has no attribute '{attr}'")
_STDLIB = {}

def _init_stdlib():
    import io as _io
    _STDLIB['math'] = _TriModule('math', {'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan, 'log': math.log, 'log10': math.log10, 'exp': math.exp, 'floor': math.floor, 'ceil': math.ceil, 'abs': abs, 'pi': math.pi, 'e': math.e, 'min': min, 'max': max, 'clamp': lambda x, lo, hi: max(lo, min(x, hi)), 'pow': pow})
    _STDLIB['random'] = _TriModule('random', {'random': random.random, 'randint': random.randint, 'choice': random.choice, 'seed': random.seed, 'shuffle': lambda x: random.shuffle(x) or x, 'uniform': random.uniform})
    _STDLIB['io'] = _TriModule('io', {'print': _tri_print, 'input': input})
    _STDLIB['string'] = _TriModule('string', {'split': lambda s, sep=None: s.split(sep), 'join': lambda sep, lst: sep.join((str(x) for x in lst)), 'replace': lambda s, old, new: s.replace(old, new), 'lower': lambda s: s.lower(), 'upper': lambda s: s.upper(), 'strip': lambda s: s.strip(), 'starts_with': lambda s, p: s.startswith(p), 'ends_with': lambda s, p: s.endswith(p), 'contains': lambda s, p: p in s})
    _STDLIB['json'] = _TriModule('json', {'parse': json.loads, 'stringify': lambda x, indent=None: json.dumps(x, indent=indent, default=str), 'decode': json.loads, 'encode': lambda x, indent=None: json.dumps(x, indent=indent, default=str)})
    _STDLIB['fs'] = _TriModule('fs', {'read_text': lambda p: open(p).read(), 'write_text': lambda p, c: open(p, 'w').write(c), 'exists': os.path.exists, 'listdir': os.listdir})
    _STDLIB['time'] = _TriModule('time', {'now': _time_mod.time, 'sleep': _time_mod.sleep})
    _STDLIB['collections'] = _TriModule('collections', {'len': len, 'range': lambda *a: list(range(*a)), 'enumerate': lambda xs: [list(p) for p in enumerate(xs)], 'sorted': sorted, 'reversed': lambda x: list(reversed(x)), 'zip': lambda *args: [list(t) for t in zip(*args)], 'map': lambda f, xs: [f(x) for x in xs], 'filter': lambda f, xs: [x for x in xs if f(x)]})
    from stdlib import plot as _plot_mod
    _STDLIB['plot'] = _TriModule('plot', {'line': _plot_mod.line, 'scatter': _plot_mod.scatter, 'heatmap': _plot_mod.heatmap, 'histogram': _plot_mod.histogram, 'bar': _plot_mod.bar, 'save': _plot_mod.save, 'show': _plot_mod.show})

    import datetime as _dt
    _STDLIB['datetime'] = _TriModule('datetime', {
        'now': lambda: _dt.datetime.now().isoformat(),
        'today': lambda: _dt.date.today().isoformat(),
        'timestamp': _time_mod.time,
        'strftime': lambda fmt, ts=None: _dt.datetime.fromtimestamp(ts or _time_mod.time()).strftime(fmt),
        'strptime': lambda s, fmt: _dt.datetime.strptime(s, fmt).isoformat(),
        'delta': lambda days=0, seconds=0: (_dt.timedelta(days=days, seconds=seconds)).total_seconds(),
        'year': lambda: _dt.datetime.now().year,
        'month': lambda: _dt.datetime.now().month,
        'day': lambda: _dt.datetime.now().day,
        'hour': lambda: _dt.datetime.now().hour,
        'weekday': lambda: _dt.datetime.now().weekday(),
    })

    import re as _re
    _STDLIB['regex'] = _TriModule('regex', {
        'match': lambda pattern, text: _re.match(pattern, text) is not None,
        'search': lambda pattern, text: (_re.search(pattern, text).group(0) if _re.search(pattern, text) else None),
        'findall': lambda pattern, text: _re.findall(pattern, text),
        'split': lambda pattern, text: _re.split(pattern, text),
        'replace': lambda pattern, repl, text: _re.sub(pattern, repl, text),
        'groups': lambda pattern, text: (list(_re.search(pattern, text).groups()) if _re.search(pattern, text) else []),
    })

    import urllib.request as _ureq
    import urllib.parse as _uparse
    import json as _json_mod
    def _tri_http_get(url, headers=None, timeout=30):
        req = _ureq.Request(url, headers=headers or {})
        with _ureq.urlopen(req, timeout=timeout) as resp:
            return {'status': resp.status, 'body': resp.read().decode(), 'headers': dict(resp.headers)}
    def _tri_http_post(url, data=None, json_data=None, headers=None, timeout=30):
        body = None
        hdrs = headers or {}
        if json_data is not None:
            body = _json_mod.dumps(json_data).encode()
            hdrs['Content-Type'] = 'application/json'
        elif data is not None:
            body = data.encode() if isinstance(data, str) else data
        req = _ureq.Request(url, data=body, headers=hdrs, method='POST')
        with _ureq.urlopen(req, timeout=timeout) as resp:
            return {'status': resp.status, 'body': resp.read().decode(), 'headers': dict(resp.headers)}
    _STDLIB['net'] = _TriModule('net', {
        'get': _tri_http_get,
        'post': _tri_http_post,
        'url_encode': lambda params: _uparse.urlencode(params),
        'url_decode': lambda s: dict(_uparse.parse_qsl(s)),
    })

    import subprocess as _sp
    _STDLIB['os'] = _TriModule('os', {
        'cwd': os.getcwd,
        'chdir': os.chdir,
        'env': lambda key, default=None: os.environ.get(key, default),
        'setenv': lambda key, val: os.environ.__setitem__(key, val),
        'listdir': os.listdir,
        'mkdir': lambda p: os.makedirs(p, exist_ok=True),
        'remove': os.remove,
        'rename': os.rename,
        'copy': lambda src, dst: open(dst, 'wb').write(open(src, 'rb').read()),
        'exists': os.path.exists,
        'is_file': os.path.isfile,
        'is_dir': os.path.isdir,
        'size': lambda p: os.path.getsize(p),
        'mtime': lambda p: os.path.getmtime(p),
        'join': lambda *parts: os.path.join(*parts),
        'split': os.path.split,
        'basename': os.path.basename,
        'dirname': os.path.dirname,
        'ext': lambda p: os.path.splitext(p)[1],
        'shell': lambda cmd, **kw: _sp.run(cmd, shell=True, capture_output=True, text=True, **kw).stdout,
        'exec_cmd': lambda cmd, **kw: _sp.run(cmd, shell=True, capture_output=True, text=True, **kw),
    })

    import hashlib as _hl
    _STDLIB['hash'] = _TriModule('hash', {
        'md5': lambda s: _hl.md5(s.encode() if isinstance(s, str) else s).hexdigest(),
        'sha256': lambda s: _hl.sha256(s.encode() if isinstance(s, str) else s).hexdigest(),
        'sha1': lambda s: _hl.sha1(s.encode() if isinstance(s, str) else s).hexdigest(),
    })

    _STDLIB['sys'] = _TriModule('sys', {
        'argv': list(sys.argv),
        'exit': lambda code=0: sys.exit(code),
        'version': 'TriadLang 0.7.0',
        'platform': sys.platform,
        'path': list(sys.path),
        'executable': sys.executable,
        'getenv': lambda k, d=None: os.environ.get(k, d),
        'setenv': lambda k, v: os.environ.__setitem__(k, v),
        'stdin': sys.stdin,
        'stdout': sys.stdout,
        'stderr': sys.stderr,
    })

    import csv as _csv
    _STDLIB['csv'] = _TriModule('csv', {
        'parse': lambda s: list(_csv.reader(s.splitlines())),
        'stringify': lambda rows: '\n'.join(_csv.writer(_io.StringIO()).writerows(rows) or _csv.writer(_io.StringIO()) and ''),
        'read': lambda path: list(_csv.reader(open(path))),
        'write': lambda path, rows: _csv.writer(open(path, 'w', newline='')).writerows(rows),
        'DictReader': _csv.DictReader,
    })

    import logging as _logging
    _STDLIB['logging'] = _TriModule('logging', {
        'basicConfig': _logging.basicConfig,
        'debug': _logging.debug,
        'info': _logging.info,
        'warning': _logging.warning,
        'error': _logging.error,
        'critical': _logging.critical,
        'getLogger': _logging.getLogger,
        'DEBUG': _logging.DEBUG,
        'INFO': _logging.INFO,
        'WARNING': _logging.WARNING,
        'ERROR': _logging.ERROR,
        'CRITICAL': _logging.CRITICAL,
    })

    import threading as _threading
    _STDLIB['threading'] = _TriModule('threading', {
        'Thread': _threading.Thread,
        'Lock': _threading.Lock,
        'Event': _threading.Event,
        'current_thread': _threading.current_thread,
        'active_count': _threading.active_count,
        'enumerate': _threading.enumerate,
    })

    import subprocess as _sp
    _STDLIB['subprocess'] = _TriModule('subprocess', {
        'run': _sp.run,
        'Popen': _sp.Popen,
        'PIPE': _sp.PIPE,
        'STDOUT': _sp.STDOUT,
        'CalledProcessError': _sp.CalledProcessError,
        'call': _sp.call,
        'check_output': _sp.check_output,
    })

    import socket as _socket
    _STDLIB['socket'] = _TriModule('socket', {
        'socket': _socket.socket,
        'AF_INET': _socket.AF_INET,
        'AF_INET6': _socket.AF_INET6,
        'SOCK_STREAM': _socket.SOCK_STREAM,
        'SOCK_DGRAM': _socket.SOCK_DGRAM,
        'gethostbyname': _socket.gethostbyname,
        'gethostname': _socket.gethostname,
        'create_connection': _socket.create_connection,
    })

_init_stdlib()

def _lazy_triad_module():
    from stdlib import triad_module as _triad_mod
    return _TriModule('triad', {'regime': _triad_mod.regime, 'solve': _triad_mod.solve, 'fast_solve': _triad_mod.fast_solve, 'solve_coupled': _triad_mod.solve_coupled, 'k_star': _triad_mod.k_star, 'crystal': _triad_mod.crystal, 'peak': _triad_mod.peak, 'norm': _triad_mod.norm, 'ipr': _triad_mod.get_ipr, 'fwhm': _triad_mod.get_fwhm, 'spectrum': _triad_mod.spectrum, 'pr': _triad_mod.pr, 'TriadParams': _triad_mod.TriadParams})

def _lazy_tensor_module():
    from runtime.ml import tensor as _t
    from runtime.ml import ml_device as _md
    return _TriModule('triad.tensor', {'tensor': _t.tensor, 'zeros': _t.zeros, 'ones': _t.ones, 'randn': _t.randn, 'rand': _t.rand, 'arange': _t.arange, 'linspace': _t.linspace, 'eye': _t.eye, 'exp': _t.exp, 'log': _t.log, 'sqrt': _t.sqrt, 'sin': _t.sin, 'cos': _t.cos, 'tanh': _t.tanh, 'sigmoid': _t.sigmoid, 'relu': _t.relu, 'softmax': _t.softmax, 'cross_entropy': _t.cross_entropy, 'mse_loss': _t.mse_loss, 'cat': _t.cat, 'stack': _t.stack, 'bmm': _t.bmm, 'transpose': _t.transpose, 'layer_norm': _t.layer_norm, 'no_grad': _t.no_grad, 'TriadTensor': _t.TriadTensor, 'set_device': _md.set_device, 'device': _md.device, 'cuda_available': _md.cuda_available, 'is_gpu': _md.is_gpu, 'sync': _md.sync, 'asnumpy': _md.asnumpy, 'mem_info': _md.mem_info})

def _lazy_nn_module():
    from runtime.ml import nn as _nn
    from runtime.ml import language as _lang
    return _TriModule('triad.nn', {'Module': _nn.Module, 'Linear': _nn.Linear, 'Conv1d': _nn.Conv1d, 'Conv2d': _nn.Conv2d, 'ReLU': _nn.ReLU, 'Tanh': _nn.Tanh, 'Sigmoid': _nn.Sigmoid, 'Softmax': _nn.Softmax, 'Sequential': _nn.Sequential, 'Flatten': _nn.Flatten, 'Dropout': _nn.Dropout, 'BatchNorm1d': _nn.BatchNorm1d, 'Parameter': _nn.Parameter, 'Embedding': _nn.Embedding, 'LayerNorm': _nn.LayerNorm, 'MultiHeadAttention': _nn.MultiHeadAttention, 'FeedForward': _nn.FeedForward, 'TransformerBlock': _nn.TransformerBlock, 'Transformer': _nn.Transformer, 'TriadSSM': _nn.TriadSSM, 'TriadSSMBlock': _nn.TriadSSMBlock, 'TriadFullBlock': _lang.TriadFullBlock, 'TriadLM': _lang.TriadLM, 'TriadFullLM': _lang.TriadFullLM, 'CharTokenizer': _lang.CharTokenizer, 'TextDataset': _lang.TextDataset, 'SGD': _nn.SGD, 'Adam': _nn.Adam})

def _lazy_data_module():
    from runtime.ml import data as _data
    return _TriModule('triad.data', {'Dataset': _data.Dataset, 'DataLoader': _data.DataLoader})

def _lazy_losses_module():
    from runtime.ml import losses as _losses
    return _TriModule('triad.losses', {'bce_loss': _losses.bce_loss, 'binary_cross_entropy_with_logits': _losses.binary_cross_entropy_with_logits, 'huber_loss': _losses.huber_loss, 'kl_div': _losses.kl_div, 'cosine_similarity_loss': _losses.cosine_similarity_loss, 'l1_loss': _losses.l1_loss, 'smooth_l1_loss': _losses.smooth_l1_loss, 'nll_loss': _losses.nll_loss})

def _lazy_metrics_module():
    from runtime.ml import metrics as _metrics
    return _TriModule('triad.metrics', {'accuracy': _metrics.accuracy, 'top_k_accuracy': _metrics.top_k_accuracy, 'perplexity': _metrics.perplexity, 'confusion_matrix': _metrics.confusion_matrix, 'precision_recall_f1': _metrics.precision_recall_f1, 'r2_score': _metrics.r2_score, 'mean_absolute_error': _metrics.mean_absolute_error, 'root_mean_squared_error': _metrics.root_mean_squared_error})

def _lazy_trainer_module():
    from runtime.ml import trainer as _trainer
    from runtime.ml import serialization as _serialization
    return _TriModule('triad.train', {'Trainer': _trainer.Trainer, 'EarlyStopping': _trainer.EarlyStopping, 'LRScheduler': _trainer.LRScheduler, 'LossHistory': _trainer.LossHistory, 'save_weights': _serialization.save_weights, 'load_weights': _serialization.load_weights, 'save_checkpoint': _serialization.save_checkpoint, 'load_checkpoint': _serialization.load_checkpoint})

def _lazy_inference_module():
    from runtime.ml import model_loader as _ml
    from runtime.ml import forward as _fwd
    from runtime.ml import generate as _gen
    return _TriModule('triad.inference', {'load_model': _ml.load_model, 'load_tokenizer': _ml.load_tokenizer, 'forward': _fwd.forward, 'generate': _gen.generate, 'chat': _gen.chat, 'stream_generate': _gen.stream_generate, 'GenerationResult': _gen.GenerationResult, 'StreamToken': _gen.StreamToken})

def _lazy_generate_module():
    from runtime.ml import generate as _gen
    return _TriModule('triad.generate', {'generate': _gen.generate, 'chat': _gen.chat, 'stream_generate': _gen.stream_generate, 'sample_token': _gen.sample_token, 'GenerationResult': _gen.GenerationResult, 'StreamToken': _gen.StreamToken})

def _lazy_kernel_module():
    from stdlib import triad_module as _tm
    def prepare(params_dict):
        from runtime.core.solver import TriadParams
        regime_name = params_dict.pop('regime', None)
        if regime_name:
            p = _tm.regime(regime_name, **params_dict)
        else:
            defaults = dict(seed=42, L=32.0, N=128, dt=0.005, T=2.0,
                            Lambda=1.0, sigma=0.5, alpha=0.3, Gamma=0.1,
                            mode=2)
            defaults.update(params_dict)
            p = TriadParams(**defaults)
        return p
    def run(params_dict):
        p = prepare(dict(params_dict))
        result = _tm.solve(p)
        return result
    def extract(result, observables=None):
        if observables is None:
            observables = ['crystallinity', 'peak', 'norm', 'k_star']
        out = {}
        for obs in observables:
            if obs == 'crystallinity':
                out['crystallinity'] = float(result.crystallinity)
            elif obs == 'peak':
                out['peak'] = float(result.peak)
            elif obs == 'norm':
                out['norm'] = float(result.norm)
            elif obs == 'k_star':
                out['k_star'] = float(result.k_star)
            elif obs == 'ipr':
                out['ipr'] = float(result.ipr)
            elif obs == 'fwhm':
                out['fwhm'] = float(result.fwhm)
            elif hasattr(result, obs):
                val = getattr(result, obs)
                out[obs] = float(val) if hasattr(val, '__float__') else val
        return out
    return _TriModule('triad.kernel', {'prepare': prepare, 'run': run, 'extract': extract})

def _lazy_calibrate_module():
    from runtime.calibrator import calibrate, GaussNewtonCalibrator, CalibrationConfig
    return _TriModule('triad.calibrate', {
        'calibrate': calibrate,
        'GaussNewtonCalibrator': GaussNewtonCalibrator,
        'CalibrationConfig': CalibrationConfig,
    })

def _lazy_curriculum_module():
    from runtime.sigma_curriculum import (run_curriculum, SigmaSchedule,
                                          pattern_metrics, turing_wavelength,
                                          pattern_entropy)
    return _TriModule('triad.curriculum', {
        'run_curriculum': run_curriculum,
        'SigmaSchedule': SigmaSchedule,
        'pattern_metrics': pattern_metrics,
        'turing_wavelength': turing_wavelength,
        'pattern_entropy': pattern_entropy,
    })

def _lazy_memory_module():
    from runtime.memory_recall import (HopfieldMemory, overlap,
                                       capacity_estimate, batch_recall_accuracy)
    return _TriModule('triad.memory', {
        'HopfieldMemory': HopfieldMemory,
        'overlap': overlap,
        'capacity_estimate': capacity_estimate,
        'batch_recall_accuracy': batch_recall_accuracy,
    })

def _lazy_sat_module():
    from runtime.sat_solver import (SATInstance, solve_sat,
                                    solve_sat_iterative, solve_sat_batch)
    return _TriModule('triad.sat', {
        'SATInstance': SATInstance,
        'solve_sat': solve_sat,
        'solve_sat_iterative': solve_sat_iterative,
        'solve_sat_batch': solve_sat_batch,
    })

def _lazy_energy_module():
    from runtime.physics.energy_readout import EnergyReadout, EnergyReadoutConfig, relax
    return _TriModule('triad.energy', {
        'EnergyReadout': EnergyReadout,
        'EnergyReadoutConfig': EnergyReadoutConfig,
        'relax': relax,
    })

def _lazy_qubits_module():
    from runtime.physics.qubits import (QubitState, TriadCircuit, Gates,
                                        QuantumHardware, substrate_gate)
    return _TriModule('triad.qubits', {
        'QubitState': QubitState,
        'TriadCircuit': TriadCircuit,
        'Gates': Gates,
        'QuantumHardware': QuantumHardware,
        'substrate_gate': substrate_gate,
    })

def _lazy_consciousness_module():
    from runtime.physics.consciousness import (
        consciousness_report, integrated_information, lempel_ziv_complexity,
        causal_density, metastability, workspace_ignition)
    return _TriModule('triad.consciousness', {
        'report': consciousness_report,
        'phi': integrated_information,
        'lzc': lempel_ziv_complexity,
        'causal_density': causal_density,
        'metastability': metastability,
        'workspace_ignition': workspace_ignition,
    })

_ALLOWED_PREFIXES: list[str] | None = None

_SAFE_MODE: bool = False
_SAFE_IMPORT_PREFIXES = [
    'triad', 'math', 'random', 'statistics', 'json', 'itertools',
    'functools', 'collections', 're', 'datetime',
]

def set_allowed_import_prefixes(prefixes: list[str] | None):
    global _ALLOWED_PREFIXES
    _ALLOWED_PREFIXES = prefixes

def set_safe_mode(enabled: bool):
    global _SAFE_MODE
    _SAFE_MODE = enabled
    if enabled:
        set_allowed_import_prefixes(_SAFE_IMPORT_PREFIXES)
    else:
        set_allowed_import_prefixes(None)

def is_safe_mode() -> bool:
    return _SAFE_MODE

def _tri_import(path: list[str], search_paths: list[str]):
    key = '.'.join(path)
    
    if _SAFE_MODE:
        tri_local = _resolve_tri_file(key, path, search_paths)
        if tri_local is not None:
            return tri_local
        top = path[0] if path else key
        allowed = any(key == p or top == p or key.startswith(p + '.')
                      for p in _SAFE_IMPORT_PREFIXES)
        if not allowed:
            raise ImportError(
                f"import '{key}' blocked in safe mode (triad run --safe). "
                f"allowed: {_SAFE_IMPORT_PREFIXES}")
    mod = _resolve_triad_stdlib(key)
    if mod is not None:
        return mod
    mod = _resolve_tri_file(key, path, search_paths)
    if mod is not None:
        return mod
    mod = _resolve_triad_modules_dir(key, path, search_paths)
    if mod is not None:
        return mod
    mod = _resolve_stdlib_dir(key, path, search_paths)
    if mod is not None:
        return mod
    mod = _resolve_python_import(key)
    if mod is not None:
        return mod
    parts = []
    parts.append(f"cannot resolve import '{key}'")
    parts.append("  tried: triad stdlib, .tri files in search paths, triad_modules/, stdlib dir, python __import__")
    try:
        __import__(key)
    except ImportError as ie:
        parts.append(f"  python also failed: {ie}")
    raise ImportError('\n'.join(parts))

def _resolve_triad_stdlib(key: str):
    _TRIAD_STDLIB_KEYS = {
        'triad': _lazy_triad_module,
        'triad.tensor': _lazy_tensor_module,
        'triad.nn': _lazy_nn_module,
        'triad.data': _lazy_data_module,
        'triad.losses': _lazy_losses_module,
        'triad.metrics': _lazy_metrics_module,
        'triad.train': _lazy_trainer_module,
        'triad.inference': _lazy_inference_module,
        'triad.generate': _lazy_generate_module,
        'triad.kernel': _lazy_kernel_module,
        'triad.calibrate': _lazy_calibrate_module,
        'triad.curriculum': _lazy_curriculum_module,
        'triad.memory': _lazy_memory_module,
        'triad.sat': _lazy_sat_module,
        'triad.energy': _lazy_energy_module,
        'triad.consciousness': _lazy_consciousness_module,
        'triad.qubits': _lazy_qubits_module,
    }
    if key in _TRIAD_STDLIB_KEYS:
        if key not in _STDLIB:
            _STDLIB[key] = _TRIAD_STDLIB_KEYS[key]()
        return _STDLIB[key]
    if key in _STDLIB:
        return _STDLIB[key]
    return None

def _resolve_tri_file(key: str, path: list[str], search_paths: list[str]):
    for sp in search_paths:
        rel = os.path.join(sp, *path) + '.tri'
        if os.path.exists(rel):
            return _load_tri_module(rel, key, search_paths)
        rel_dir = os.path.join(sp, *path, '__init__.tri')
        if os.path.exists(rel_dir):
            return _load_tri_module(rel_dir, key, search_paths)
    return None

def _resolve_triad_modules_dir(key: str, path: list[str], search_paths: list[str]):
    modules_dir = os.path.join(os.getcwd(), 'triad_modules')
    if not os.path.isdir(modules_dir) or len(path) < 1:
        return None
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
    return None

def _resolve_stdlib_dir(key: str, path: list[str], search_paths: list[str]):
    stdlib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stdlib')
    rel = os.path.join(stdlib_dir, *path) + '.tri'
    if os.path.exists(rel):
        return _load_tri_module(rel, key, search_paths)
    return None

def _resolve_python_import(key: str):
    global _ALLOWED_PREFIXES
    if _ALLOWED_PREFIXES is not None:
        allowed = any(key == p or key.startswith(p + '.') for p in _ALLOWED_PREFIXES)
        if not allowed:
            raise ImportError(
                f"import '{key}' blocked by allowed-prefix policy. "
                f"allowed prefixes: {_ALLOWED_PREFIXES}"
            )
    try:
        return __import__(key)
    except ImportError:
        return None
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
