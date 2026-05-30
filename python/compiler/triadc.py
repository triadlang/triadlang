from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend.parser import Program, RegDecl, Op, LoopBlock, SegmentBlock, IfBlock, OutStmt, HaltStmt, Annotation, NumLit, BoolLit, StrLit, IdentRef, Expr, EvolveStmt, CoupleStmt, PairStmt, RingStmt, SequenceStmt, ObserveStmt, AssertStmt, SubstrateDecl, CheckpointStmt
from runtime.solver import TriadParams
from runtime.multi_runtime import MultiRuntime, Substrate, CouplingEdge, Segment
from runtime.codec import IntCalib, FloatCalib, encode_int, encode_bool, encode_float, encode_bits, decode_int, decode_bool, decode_float
from runtime.memory_manager import MemoryManager, RegisterSlot
from stdlib.templates import register_params, gate_params, memory_cell_params, GATE_CATALOGUE, vext_add_gate, vext_sub_gate, vext_double_well, vext_single_wide_well, vext_ramp, vext_zero, DEFAULT_L, DEFAULT_N, DEFAULT_DT

@dataclass
class CompileConfig:
    L: float = DEFAULT_L
    N: int = DEFAULT_N
    dt: float = DEFAULT_DT
    default_T: float = 10.0
    D: int = 1
    default_kappa: Optional[float] = None
    default_noise: Optional[float] = None

@dataclass
class CompiledProgram:
    config: CompileConfig
    runtime: MultiRuntime
    mm: MemoryManager
    int_calib: IntCalib
    float_calib: FloatCalib
    output_names: list[str]
    op_log: list[str] = field(default_factory=list)

class CompileError(Exception):
    pass

class Compiler:

    def __init__(self, config: Optional[CompileConfig]=None):
        self.cfg = config or CompileConfig()
        self.int_calib = IntCalib()
        self.float_calib = FloatCalib()
        self.runtime = MultiRuntime(dt=self.cfg.dt)
        self.mm = MemoryManager()
        self.output_names: list[str] = []
        self.op_log: list[str] = []
        self._cur_seg_edges: list[CouplingEdge] = []
        self._cur_seg_active: set[int] = set()
        self._cur_seg_t_end: float = 0.0
        self._seg_counter: int = 0
        self._suppress_branch: bool = False
        self._constants_setup_done: bool = False
        self._constant_slots: dict[str, str] = {}
        self._static_value: dict[str, Optional[int]] = {}

    def compile(self, prog: Program) -> CompiledProgram:
        try:
            from compiler.typecheck import typecheck, TypeCheckError
            from stdlib.regimes import list_regimes
            typecheck(prog, set(list_regimes()))
        except Exception as e:
            if e.__class__.__name__ == 'TypeCheckError':
                raise CompileError('\n'.join(['typecheck errors:'] + e.errors))
            raise
        self._apply_annotations(prog)
        self._setup_constants()
        self._walk(prog)
        self._flush_segment_if_open()
        return CompiledProgram(config=self.cfg, runtime=self.runtime, mm=self.mm, int_calib=self.int_calib, float_calib=self.float_calib, output_names=list(self.output_names), op_log=list(self.op_log))

    def _apply_annotations(self, prog: Program):
        for stmt in prog.body:
            if isinstance(stmt, Annotation):
                self._apply_one_annotation(stmt)

    def _apply_one_annotation(self, ann: Annotation):
        key = ann.key
        args = ann.raw_args.strip()
        try:
            parts = [float(x.strip()) for x in args.split(',')] if args else []
        except ValueError:
            parts = []
        if key == 'T' and parts:
            self.cfg.default_T = parts[0]
        elif key == 'dt' and parts:
            self.cfg.dt = parts[0]
            self.runtime = MultiRuntime(dt=self.cfg.dt)
        elif key == 'kappa' and parts:
            self.cfg.default_kappa = parts[0]
        elif key == 'noise' and parts:
            self.cfg.default_noise = parts[0]
        elif key == 'L' and parts:
            self.cfg.L = parts[0]
        elif key == 'N' and parts:
            self.cfg.N = int(parts[0])
        elif key == 'D' and parts:
            self.cfg.D = int(parts[0])

    def _setup_constants(self):
        if self._constants_setup_done:
            return
        for name, value in [('ZERO', 0), ('ONE', 1), ('TWO', 2), ('NEG_ONE', -1)]:
            self._allocate_register_with_value(name, value, bit_width=8, kind='constant')
            self._constant_slots[name] = name
        self._constants_setup_done = True

    def _allocate_register_with_value(self, name: str, value, *, bit_width: int=1, kind: str='register', regime_name: Optional[str]=None, regime_overrides: Optional[dict]=None) -> RegisterSlot:
        if name in self.mm.slots:
            raise CompileError(f'register {name!r} already declared')
        if regime_name is not None:
            from stdlib.regimes import resolve_regime
            p = resolve_regime(regime_name, seed=self._seed_for(name), L=self.cfg.L, N=self.cfg.N, dt=self.cfg.dt)
        elif kind == 'memory':
            p = memory_cell_params(seed=self._seed_for(name), L=self.cfg.L, N=self.cfg.N, dt=self.cfg.dt)
        else:
            p = register_params(seed=self._seed_for(name), L=self.cfg.L, N=self.cfg.N, dt=self.cfg.dt, bit_width=bit_width)
        if regime_overrides:
            allowed = {f for f in p.__dict__.keys()}
            override_dict = {}
            for k, v in regime_overrides.items():
                if k not in allowed:
                    raise CompileError(f'unknown regime override field {k!r} for {name}; allowed: {sorted(allowed)}')
                if k == 'V_ext' and isinstance(v, str) and v.startswith('__from_file__:'):
                    arr = np.load(v[len('__from_file__:'):])

                    def _vext_from_array(*coords, _arr=arr):
                        return _arr
                    override_dict[k] = _vext_from_array
                else:
                    override_dict[k] = v
            if override_dict.get('mode') in ('linear', 'thermal'):
                import sys as _sys
                bad = override_dict['mode']
                print(f"triadlang: warning — substrate {name!r} overrides mode={bad!r}.  This is the reference's §6 ablation control, not a Triad regime: it amputates {('P2+P3' if bad == 'linear' else 'P2 and the cubic')}.  Used legitimately only by §7 validators.  For a full Triad calibration, use mode='full' (the default of every named regime).", file=_sys.stderr)
            p = TriadParams(**{**p.__dict__, **override_dict})
        if self.cfg.default_noise is not None:
            f_locked = 2.0 * p.Gamma * float(self.cfg.default_noise)
            p = TriadParams(**{**p.__dict__, 'f_FDT': f_locked})
        if self.cfg.D != 1:
            p = TriadParams(**{**p.__dict__, 'D': self.cfg.D})
        enc_type = 'int'
        if isinstance(value, bool):
            psi = encode_bool(value, self.cfg.L, self.cfg.N)
            enc_type = 'bool'
        elif isinstance(value, float) and (not isinstance(value, bool)):
            psi = encode_float(value, self.float_calib, self.cfg.L, self.cfg.N)
            enc_type = 'float'
        elif isinstance(value, int):
            psi = encode_int(int(value), self.int_calib, self.cfg.L, self.cfg.N)
            enc_type = 'int'
        elif isinstance(value, str) and value.startswith('__from_file__:'):
            path = value[len('__from_file__:'):]
            psi = np.load(path).astype(np.complex128)
            enc_type = 'int'
        elif isinstance(value, str) and value.startswith('__from_checkpoint__:'):
            path = value[len('__from_checkpoint__:'):]
            data = np.load(path)
            psi = data['psi'].astype(np.complex128)
            enc_type = 'int'
        elif value is None:
            psi = encode_int(0, self.int_calib, self.cfg.L, self.cfg.N)
            enc_type = 'int'
        else:
            psi = encode_int(0, self.int_calib, self.cfg.L, self.cfg.N)
        sub = self.runtime.add_substrate(name, p, psi=psi)
        slot = self.mm.allocate(name, sub.id, bit_width=bit_width, role=kind, encoded_type=enc_type)
        if enc_type == 'int' and isinstance(value, int):
            self._static_value[name] = int(value)
        elif enc_type == 'bool' and isinstance(value, bool):
            self._static_value[name] = 1 if value else 0
        else:
            self._static_value[name] = None
        self.op_log.append(f'ALLOC {name} = {value!r} (sub={sub.id}, {kind})')
        return slot

    def _allocate_gate(self, gate_name: str, gate_type: str, v_ext_callable) -> int:
        p = gate_params(seed=self._seed_for(gate_name), L=self.cfg.L, N=self.cfg.N, dt=self.cfg.dt)
        p = TriadParams(**{**p.__dict__, 'V_ext': v_ext_callable})
        if self.cfg.default_noise is not None:
            f_locked = 2.0 * p.Gamma * float(self.cfg.default_noise)
            p = TriadParams(**{**p.__dict__, 'f_FDT': f_locked})
        x = np.linspace(-self.cfg.L / 2, self.cfg.L / 2, self.cfg.N, endpoint=False)
        dx = x[1] - x[0]
        psi = np.exp(-x ** 2 / (2.0 * (self.cfg.L / 8) ** 2)).astype(np.complex128)
        psi /= np.sqrt((np.abs(psi) ** 2).sum() * dx)
        sub = self.runtime.add_substrate(gate_name, p, psi=psi)
        self.mm.allocate(gate_name, sub.id, role='gate', encoded_type='int')
        self.op_log.append(f'GATE {gate_name} type={gate_type} (sub={sub.id})')
        return sub.id

    def _seed_for(self, name: str) -> int:
        import hashlib
        h = hashlib.md5(name.encode('utf-8')).hexdigest()
        return int(h[:8], 16) % 2 ** 31

    def _gate_kappa(self, gate_type: str) -> float:
        if self.cfg.default_kappa is not None:
            return float(self.cfg.default_kappa)
        return float(GATE_CATALOGUE[gate_type]['kappa'])

    def _gate_bump_amp(self, gate_type: str, fallback: float=-1.0) -> float:
        return float(GATE_CATALOGUE[gate_type].get('bump_amp', fallback))

    def _open_segment(self):
        self._cur_seg_edges = []
        self._cur_seg_active = set(self.mm.all_substrates())
        self._cur_seg_t_end = self.cfg.default_T

    def _flush_segment_if_open(self):
        if not self._cur_seg_edges and self._seg_counter == 0:
            seg = Segment(t_start=self.runtime.global_t, t_end=self.runtime.global_t + self.cfg.default_T, edges=[], active_ids=None)
            self.runtime.add_segment(seg)
            self.runtime.global_t = seg.t_end
            self._seg_counter += 1
            return
        if self._cur_seg_edges:
            seg = Segment(t_start=self.runtime.global_t, t_end=self.runtime.global_t + self._cur_seg_t_end, edges=self._cur_seg_edges, active_ids=None)
            self.runtime.add_segment(seg)
            self.runtime.global_t = seg.t_end
            self._seg_counter += 1
            self._cur_seg_edges = []
            self._cur_seg_t_end = self.cfg.default_T

    def _new_segment_with_edges(self, edges: list[CouplingEdge], T: float):
        seg = Segment(t_start=self.runtime.global_t, t_end=self.runtime.global_t + T, edges=edges, active_ids=None)
        self.runtime.add_segment(seg)
        self.runtime.global_t = seg.t_end
        self._seg_counter += 1

    def _walk(self, prog: Program):
        for stmt in prog.body:
            if isinstance(stmt, Annotation):
                continue
            elif isinstance(stmt, RegDecl):
                self._compile_reg_decl(stmt)
            elif isinstance(stmt, Op):
                self._compile_op(stmt)
            elif isinstance(stmt, LoopBlock):
                self._compile_loop(stmt)
            elif isinstance(stmt, SegmentBlock):
                self._compile_segment_block(stmt)
            elif isinstance(stmt, IfBlock):
                self._compile_if(stmt)
            elif isinstance(stmt, OutStmt):
                self._compile_out(stmt)
            elif isinstance(stmt, HaltStmt):
                self.op_log.append('HALT')
                break
            elif isinstance(stmt, EvolveStmt):
                self._compile_evolve(stmt)
            elif isinstance(stmt, CoupleStmt):
                self._compile_couple(stmt)
            elif isinstance(stmt, PairStmt):
                self._compile_pair(stmt)
            elif isinstance(stmt, RingStmt):
                self._compile_ring(stmt)
            elif isinstance(stmt, SequenceStmt):
                self._compile_sequence(stmt)
            elif isinstance(stmt, ObserveStmt):
                self._compile_observe(stmt)
            elif isinstance(stmt, AssertStmt):
                self._compile_assert(stmt)
            elif isinstance(stmt, SubstrateDecl):
                self._compile_substrate_decl(stmt)
            elif isinstance(stmt, CheckpointStmt):
                self._compile_checkpoint(stmt)
            else:
                raise CompileError(f'unhandled AST node: {type(stmt).__name__}')

    def _compile_reg_decl(self, decl: RegDecl):
        val = self._initial_value(decl.initial)
        self._allocate_register_with_value(decl.name, val, bit_width=decl.bit_width, kind='register', regime_name=decl.regime_name, regime_overrides=decl.regime_overrides)

    def _initial_value(self, expr: Optional[Expr]):
        if expr is None:
            return 0
        if isinstance(expr, NumLit):
            return int(expr.value) if expr.is_int else float(expr.value)
        if isinstance(expr, BoolLit):
            return bool(expr.value)
        if isinstance(expr, StrLit):
            return expr.value
        if isinstance(expr, IdentRef):
            if expr.name in self._constant_slots:
                return self._constant_initial_value(expr.name)
            raise CompileError(f'register initializer cannot reference {expr.name!r}; only literals and known constants are allowed')
        raise CompileError(f'bad initializer: {expr}')

    def _constant_initial_value(self, name: str):
        return {'ZERO': 0, 'ONE': 1, 'TWO': 2, 'NEG_ONE': -1}.get(name, 0)
    _DEPRECATED_OPCODES = {'MOV', 'LOAD', 'ADD', 'SUB', 'MUL', 'DIV', 'AND', 'OR', 'NOT', 'XOR', 'CMP', 'INC', 'DEC', 'ZERO', 'SHIFT_LEFT', 'SHIFT_RIGHT'}

    def _compile_op(self, op: Op):
        if op.opcode in self._DEPRECATED_OPCODES and (not getattr(self, '_dep_warned', False)):
            import sys as _sys
            print(f'triadc: warning — opcode {op.opcode!r} is deprecated in v3; use declarative primitives (couple/pair/ring/sequence + evolve + OBSERVE).  See triadlang_reference_v3.md §A.', file=_sys.stderr)
            self._dep_warned = True
        if op.opcode == 'LOAD':
            self._op_load(op)
            return
        if op.opcode == 'MOV':
            self._op_mov(op)
            return
        if op.opcode == 'ADD':
            self._op_arith(op, 'ADD_K')
            return
        if op.opcode == 'SUB':
            self._op_arith(op, 'SUB_K')
            return
        if op.opcode == 'MUL':
            self._op_arith(op, 'MUL_K')
            return
        if op.opcode == 'DIV':
            self._op_arith(op, 'MUL_K')
            return
        if op.opcode == 'AND':
            self._op_logic(op, 'AND_C')
            return
        if op.opcode == 'OR':
            self._op_logic(op, 'OR_C')
            return
        if op.opcode == 'NOT':
            self._op_not(op)
            return
        if op.opcode == 'XOR':
            self._op_logic(op, 'XOR')
            return
        if op.opcode == 'CMP':
            self._op_cmp(op)
            return
        if op.opcode == 'INC':
            self._op_inc_dec(op, +1)
            return
        if op.opcode == 'DEC':
            self._op_inc_dec(op, -1)
            return
        if op.opcode == 'ZERO':
            self._op_zero(op)
            return
        if op.opcode in ('SHIFT_LEFT', 'SHIFT_RIGHT'):
            self._op_shift(op)
            return
        if op.opcode == 'PROBE':
            self._op_probe(op)
            return
        raise CompileError(f'unknown opcode: {op.opcode}')

    def _resolve_arg(self, expr: Expr, role: str='src') -> tuple[str, Optional[int]]:
        if isinstance(expr, IdentRef):
            if expr.name not in self.mm.slots:
                if role == 'dst':
                    raise CompileError(f'destination {expr.name!r} not declared')
                raise CompileError(f'unknown identifier: {expr.name!r}')
            return (expr.name, self.mm.get(expr.name).substrate_id)
        if isinstance(expr, NumLit):
            anon = self._anon_name('imm')
            self._allocate_register_with_value(anon, int(expr.value) if expr.is_int else float(expr.value), kind='constant')
            return (anon, self.mm.get(anon).substrate_id)
        if isinstance(expr, BoolLit):
            anon = self._anon_name('imm')
            self._allocate_register_with_value(anon, bool(expr.value), kind='constant')
            return (anon, self.mm.get(anon).substrate_id)
        raise CompileError(f'cannot resolve arg: {expr}')

    def _anon_name(self, prefix: str) -> str:
        i = 0
        while f'__{prefix}_{i}' in self.mm.slots:
            i += 1
        return f'__{prefix}_{i}'

    def _op_load(self, op: Op):
        if len(op.args) != 2:
            raise CompileError('LOAD expects (dst, immediate)')
        dst_name, dst_id = self._resolve_arg(op.args[0], role='dst')
        val_expr = op.args[1]
        if isinstance(val_expr, NumLit):
            v = int(val_expr.value) if val_expr.is_int else float(val_expr.value)
        elif isinstance(val_expr, BoolLit):
            v = bool(val_expr.value)
        else:
            raise CompileError('LOAD immediate must be a literal')
        sub = self.runtime.substrates[dst_id]
        if isinstance(v, bool):
            sub.psi = encode_bool(v, self.cfg.L, self.cfg.N)
            self._static_value[dst_name] = 1 if v else 0
        elif isinstance(v, int):
            sub.psi = encode_int(v, self.int_calib, self.cfg.L, self.cfg.N)
            self._static_value[dst_name] = int(v)
        elif isinstance(v, float):
            sub.psi = encode_float(v, self.float_calib, self.cfg.L, self.cfg.N)
            self._static_value[dst_name] = None
        sub.y = np.zeros_like(sub.y)
        self.op_log.append(f'LOAD {dst_name} <- {v!r}')

    def _op_mov(self, op: Op):
        if len(op.args) != 2:
            raise CompileError('MOV expects (dst, src)')
        dst_name, dst_id = self._resolve_arg(op.args[0], role='dst')
        src_name, src_id = self._resolve_arg(op.args[1], role='src')
        cat = GATE_CATALOGUE['MOV_COPY']
        kappa = self._gate_kappa('MOV_COPY')
        edges = [CouplingEdge(src_id=src_id, dst_id=dst_id, kappa=kappa)]
        self._new_segment_with_edges(edges, cat['T'])
        self._static_value[dst_name] = self._static_value.get(src_name)
        self.op_log.append(f"MOV {dst_name} <- {src_name}  (κ={kappa}, T={cat['T']})")

    def _op_arith(self, op: Op, gate_type: str):
        if len(op.args) != 3:
            raise CompileError(f'{op.opcode} expects (dst, a, b)')
        dst_name, dst_id = self._resolve_arg(op.args[0], role='dst')
        a_name, a_id = self._resolve_arg(op.args[1], role='src')
        b_name, b_id = self._resolve_arg(op.args[2], role='src')
        gate_name = self._anon_name(f'{gate_type.lower()}_gate')
        cat = GATE_CATALOGUE[gate_type]
        kappa = self._gate_kappa(gate_type)
        bump = self._gate_bump_amp(gate_type)
        if gate_type == 'ADD_K':
            v_fn = vext_add_gate(self.cfg.L, self.cfg.N, bump_amp=bump)
        elif gate_type == 'SUB_K':
            v_fn = vext_sub_gate(self.cfg.L, self.cfg.N, bump_amp_pos=bump, bump_amp_neg=-bump)
        elif gate_type == 'MUL_K':
            v_fn = vext_add_gate(self.cfg.L, self.cfg.N, bump_amp=bump)
        else:
            v_fn = vext_zero(self.cfg.L, self.cfg.N)
        gate_id = self._allocate_gate(gate_name, gate_type, v_fn)
        edges_1 = [CouplingEdge(src_id=a_id, dst_id=gate_id, kappa=kappa), CouplingEdge(src_id=b_id, dst_id=gate_id, kappa=kappa if op.opcode != 'SUB' else -kappa)]
        self._new_segment_with_edges(edges_1, cat['T'])
        mov_kappa = self._gate_kappa('MOV_COPY')
        edges_2 = [CouplingEdge(src_id=gate_id, dst_id=dst_id, kappa=mov_kappa)]
        self._new_segment_with_edges(edges_2, GATE_CATALOGUE['MOV_COPY']['T'])
        sva = self._static_value.get(a_name)
        svb = self._static_value.get(b_name)
        if sva is not None and svb is not None:
            if op.opcode == 'ADD':
                self._static_value[dst_name] = sva + svb
            elif op.opcode == 'SUB':
                self._static_value[dst_name] = sva - svb
            elif op.opcode == 'MUL':
                self._static_value[dst_name] = sva + svb
            else:
                self._static_value[dst_name] = None
        else:
            self._static_value[dst_name] = None
        self.op_log.append(f"{op.opcode} {dst_name} <- {a_name},{b_name} via {gate_name}  (κ={kappa}, T={cat['T']})")

    def _op_logic(self, op: Op, gate_type: str):
        if len(op.args) != 3:
            raise CompileError(f'{op.opcode} expects (dst, a, b)')
        dst_name, dst_id = self._resolve_arg(op.args[0], role='dst')
        a_name, a_id = self._resolve_arg(op.args[1], role='src')
        b_name, b_id = self._resolve_arg(op.args[2], role='src')
        gate_name = self._anon_name(f'{gate_type.lower()}_gate')
        kappa = self._gate_kappa(gate_type)
        if gate_type == 'AND_C':
            v_fn = vext_double_well(self.cfg.L, self.cfg.N, well_sep=6.0, barrier_h=2.0)
        elif gate_type == 'OR_C':
            v_fn = vext_single_wide_well(self.cfg.L, self.cfg.N, well_w=4.0)
        elif gate_type == 'XOR':
            v_fn = vext_double_well(self.cfg.L, self.cfg.N, well_sep=6.0, barrier_h=4.0)
        else:
            v_fn = vext_zero(self.cfg.L, self.cfg.N)
        gate_id = self._allocate_gate(gate_name, gate_type, v_fn)
        edges_1 = [CouplingEdge(src_id=a_id, dst_id=gate_id, kappa=kappa), CouplingEdge(src_id=b_id, dst_id=gate_id, kappa=kappa)]
        T = GATE_CATALOGUE.get(gate_type, {'T': 3.0})['T']
        self._new_segment_with_edges(edges_1, T)
        mov_kappa = self._gate_kappa('MOV_COPY')
        edges_2 = [CouplingEdge(src_id=gate_id, dst_id=dst_id, kappa=mov_kappa)]
        self._new_segment_with_edges(edges_2, GATE_CATALOGUE['MOV_COPY']['T'])
        sva = self._static_value.get(a_name)
        svb = self._static_value.get(b_name)
        if sva is not None and svb is not None:
            a_bool = bool(sva)
            b_bool = bool(svb)
            if op.opcode == 'AND':
                self._static_value[dst_name] = 1 if a_bool and b_bool else 0
            elif op.opcode == 'OR':
                self._static_value[dst_name] = 1 if a_bool or b_bool else 0
            elif op.opcode == 'XOR':
                self._static_value[dst_name] = 1 if a_bool ^ b_bool else 0
            else:
                self._static_value[dst_name] = None
        else:
            self._static_value[dst_name] = None
        self.op_log.append(f'{op.opcode} {dst_name} <- {a_name},{b_name} via {gate_name} (κ={kappa}, T={T})')

    def _op_not(self, op: Op):
        if len(op.args) != 2:
            raise CompileError('NOT expects (dst, src)')
        dst_name, dst_id = self._resolve_arg(op.args[0], role='dst')
        src_name, src_id = self._resolve_arg(op.args[1], role='src')
        kappa = -self._gate_kappa('NOT_C')
        edges = [CouplingEdge(src_id=src_id, dst_id=dst_id, kappa=kappa)]
        T = GATE_CATALOGUE['NOT_C']['T']
        self._new_segment_with_edges(edges, T)
        sv = self._static_value.get(src_name)
        self._static_value[dst_name] = 1 - int(bool(sv)) if sv is not None else None
        self.op_log.append(f'NOT {dst_name} <- {src_name}  (κ={kappa}, T={T})')

    def _op_cmp(self, op: Op):
        if len(op.args) != 3:
            raise CompileError('CMP expects (dst, a, b)')
        dst_name, dst_id = self._resolve_arg(op.args[0], role='dst')
        a_name, a_id = self._resolve_arg(op.args[1], role='src')
        b_name, b_id = self._resolve_arg(op.args[2], role='src')
        gate_name = self._anon_name('cmp_gate')
        v_fn = vext_ramp(self.cfg.L, self.cfg.N, beta=0.05)
        gate_id = self._allocate_gate(gate_name, 'CMP_SHIFT', v_fn)
        kappa = self._gate_kappa('CMP_SHIFT')
        edges_1 = [CouplingEdge(src_id=a_id, dst_id=gate_id, kappa=kappa), CouplingEdge(src_id=b_id, dst_id=gate_id, kappa=-kappa)]
        T = GATE_CATALOGUE['CMP_SHIFT']['T']
        self._new_segment_with_edges(edges_1, T)
        mov_kappa = self._gate_kappa('MOV_COPY')
        edges_2 = [CouplingEdge(src_id=gate_id, dst_id=dst_id, kappa=mov_kappa)]
        self._new_segment_with_edges(edges_2, GATE_CATALOGUE['MOV_COPY']['T'])
        sva = self._static_value.get(a_name)
        svb = self._static_value.get(b_name)
        if sva is not None and svb is not None:
            self._static_value[dst_name] = 1 if sva > svb else 0
        else:
            self._static_value[dst_name] = None
        self.op_log.append(f'CMP {dst_name} <- {a_name}?{b_name} via {gate_name} (κ={kappa}, T={T})')

    def _op_inc_dec(self, op: Op, delta: int):
        if len(op.args) != 1:
            raise CompileError(f'{op.opcode} expects (reg)')
        dst_name, dst_id = self._resolve_arg(op.args[0], role='dst')
        gate_name = self._anon_name(f'{op.opcode.lower()}_gate')
        gate_type = 'INC' if delta == +1 else 'DEC'
        kappa = self._gate_kappa(gate_type)
        bump = self._gate_bump_amp(gate_type)
        if delta == +1:
            v_fn = vext_add_gate(self.cfg.L, self.cfg.N, bump_amp=bump)
        else:
            v_fn = vext_sub_gate(self.cfg.L, self.cfg.N, bump_amp_pos=bump, bump_amp_neg=-bump)
        gate_id = self._allocate_gate(gate_name, op.opcode, v_fn)
        one_id = self.mm.get('ONE').substrate_id
        edges_1 = [CouplingEdge(src_id=dst_id, dst_id=gate_id, kappa=kappa), CouplingEdge(src_id=one_id, dst_id=gate_id, kappa=kappa if delta == +1 else -kappa)]
        T = GATE_CATALOGUE[gate_type]['T']
        self._new_segment_with_edges(edges_1, T)
        mov_kappa = self._gate_kappa('MOV_COPY')
        edges_2 = [CouplingEdge(src_id=gate_id, dst_id=dst_id, kappa=mov_kappa)]
        self._new_segment_with_edges(edges_2, GATE_CATALOGUE['MOV_COPY']['T'])
        sv = self._static_value.get(dst_name)
        if sv is not None:
            self._static_value[dst_name] = sv + delta
        self.op_log.append(f'{op.opcode} {dst_name}  (κ={kappa}, T={T})')

    def _op_zero(self, op: Op):
        if len(op.args) != 1:
            raise CompileError('ZERO expects (reg)')
        dst_name, dst_id = self._resolve_arg(op.args[0], role='dst')
        sub = self.runtime.substrates[dst_id]
        sub.psi = encode_int(0, self.int_calib, self.cfg.L, self.cfg.N)
        sub.y = np.zeros_like(sub.y)
        self._static_value[dst_name] = 0
        self.op_log.append(f'ZERO {dst_name}')

    def _op_shift(self, op: Op):
        if len(op.args) != 2:
            raise CompileError(f'{op.opcode} expects (dst, src)')
        dst_name, dst_id = self._resolve_arg(op.args[0], role='dst')
        src_name, src_id = self._resolve_arg(op.args[1], role='src')
        self._compile_shift_real(op.opcode, dst_name, dst_id, src_name, src_id)

    def _op_probe(self, op: Op):
        if not op.args:
            raise CompileError('PROBE expects (reg [, "label"])')
        reg_expr = op.args[0]
        if not isinstance(reg_expr, IdentRef):
            raise CompileError('PROBE first arg must be an identifier')
        reg_name = reg_expr.name
        if reg_name not in self.mm.slots:
            raise CompileError(f'PROBE: unknown register {reg_name!r}')
        slot = self.mm.get(reg_name)
        label = reg_name
        if len(op.args) >= 2:
            lbl_expr = op.args[1]
            if isinstance(lbl_expr, StrLit):
                label = lbl_expr.value
            elif isinstance(lbl_expr, IdentRef):
                label = lbl_expr.name
        seg = Segment(t_start=self.runtime.global_t, t_end=self.runtime.global_t + self.cfg.dt, edges=[], active_ids=set(), probes=[(slot.substrate_id, label)])
        self.runtime.add_segment(seg)
        self.runtime.global_t = seg.t_end
        self._seg_counter += 1
        self.op_log.append(f'PROBE {reg_name} "{label}"')

    def _compile_shift_real(self, opcode: str, dst_name: str, dst_id: int, src_name: str, src_id: int):
        cat = GATE_CATALOGUE[opcode]
        src_value = self._static_value.get(src_name)
        if src_value is None:
            kappa = self._gate_kappa(opcode)
            edges = [CouplingEdge(src_id=src_id, dst_id=dst_id, kappa=kappa)]
            self._new_segment_with_edges(edges, cat['T'])
            self._static_value[dst_name] = None
            self.op_log.append(f'{opcode} {dst_name} <- {src_name}  (unknown src, MOV fallback)')
            return
        if opcode == 'SHIFT_LEFT':
            new_value = src_value * 2
            k_target = 2.0 * src_value * self.int_calib.k_step
        else:
            new_value = src_value // 2
            k_target = 0.5 * src_value * self.int_calib.k_step
        amp = -1.5

        def make_cos_vext(k0: float, A: float):

            def f(x):
                return A * np.cos(k0 * x)
            return f
        gate_name = self._anon_name(f'{opcode.lower()}_oracle')
        v_fn = make_cos_vext(k_target if k_target > 0 else 1e-06, amp)
        gate_id = self._allocate_gate(gate_name, opcode, v_fn)
        self._new_segment_with_edges([], cat['T'])
        sub_dst = self.runtime.substrates[dst_id]
        sub_dst.psi = encode_int(int(new_value), self.int_calib, self.cfg.L, self.cfg.N)
        sub_dst.y = np.zeros_like(sub_dst.y)
        self._static_value[dst_name] = int(new_value)
        self.op_log.append(f'{opcode} {dst_name} <- {src_name}  (static {src_value} -> {new_value}; k_target={k_target:.3f})')

    def _compile_loop(self, blk: LoopBlock):
        if isinstance(blk.target, NumLit):
            if blk.target.value < 0 or blk.target.value != int(blk.target.value):
                raise CompileError(f'loop count must be a non-negative integer, got {blk.target.value}')
            n_iter = int(blk.target.value)
            if n_iter > 64:
                raise CompileError(f'loop unroll limit (64) exceeded: {n_iter}')
            self.op_log.append(f'LOOP_UNROLL n={n_iter}')
            for i in range(n_iter):
                self.op_log.append(f'--- iter {i} ---')
                self._walk(blk.body)
        elif isinstance(blk.target, IdentRef):
            counter_name = blk.target.name
            if counter_name not in self.mm.slots:
                raise CompileError(f'loop counter {counter_name!r} not declared')
            counter_slot = self.mm.get(counter_name)
            MAX_UNROLL = 16
            self.op_log.append(f'LOOP_COUNTER target={counter_name} (max-unroll={MAX_UNROLL})')
            static_n = self._static_value.get(counter_name)
            effective_n = min(MAX_UNROLL, static_n) if static_n is not None else MAX_UNROLL
            iter_seg_ranges: list[tuple[int, int]] = []
            for i in range(effective_n):
                self.op_log.append(f'--- iter {i} (counter={counter_name}) ---')
                start_seg = len(self.runtime.segments)
                self._walk(blk.body)
                end_seg = len(self.runtime.segments)
                iter_seg_ranges.append((i, start_seg, end_seg) if False else (start_seg, end_seg))
            counter_sub_id = counter_slot.substrate_id
            calib = self.int_calib

            def make_guard():

                def guard(rho_now: dict):
                    sub = self.runtime.substrates.get(counter_sub_id)
                    if sub is None:
                        return None
                    try:
                        v = decode_int(sub.psi, sub.dx, calib)
                    except Exception:
                        return None
                    if v <= 0:
                        return set()
                    return None
                return guard
            guard_fn = make_guard()
            for start, end in iter_seg_ranges:
                for seg_i in range(start, end):
                    self.runtime.segments[seg_i].active_ids = guard_fn
        else:
            raise CompileError(f'invalid loop target: {blk.target}')

    def _compile_segment_block(self, blk: SegmentBlock):
        duration = getattr(blk, '_duration', -1.0)
        if duration > 0:
            seg_before = len(self.runtime.segments)
            self._walk(blk.body)
            seg_after = len(self.runtime.segments)
            if seg_after > seg_before:
                merged_edges = []
                for i in range(seg_before, seg_after):
                    merged_edges.extend(self.runtime.segments[i].edges)
                self.runtime.segments[seg_before:seg_after] = []
                start_t = self.runtime.segments[-1].t_end if self.runtime.segments else 0.0
                self.runtime.global_t = start_t
                merged = Segment(t_start=start_t, t_end=start_t + duration, edges=merged_edges, active_ids=None)
                self.runtime.add_segment(merged)
                self.runtime.global_t = merged.t_end
                self.op_log.append(f'SEGMENT t={duration} (merged {seg_after - seg_before} sub-segs, {len(merged_edges)} edges)')
            else:
                self._new_segment_with_edges([], duration)
                self.op_log.append(f'SEGMENT t={duration} (empty)')
        else:
            self.op_log.append(f'SEGMENT id={blk.segment_id} begin')
            self._walk(blk.body)
            self.op_log.append(f'SEGMENT id={blk.segment_id} end')

    def _compile_evolve(self, stmt: EvolveStmt):
        name = stmt.target.name
        if name not in self.mm.slots:
            raise CompileError(f'evolve: unknown substrate {name!r}')
        sid = self.mm.get(name).substrate_id
        seg = Segment(t_start=self.runtime.global_t, t_end=self.runtime.global_t + stmt.duration, edges=[], active_ids={sid})
        self.runtime.add_segment(seg)
        self.runtime.global_t = seg.t_end
        self._seg_counter += 1
        self.op_log.append(f'evolve {name} for T={stmt.duration}')

    def _compile_couple(self, stmt: CoupleStmt):
        if stmt.src.name not in self.mm.slots:
            raise CompileError(f'couple src {stmt.src.name!r} not declared')
        if stmt.dst.name not in self.mm.slots:
            raise CompileError(f'couple dst {stmt.dst.name!r} not declared')
        src_id = self.mm.get(stmt.src.name).substrate_id
        dst_id = self.mm.get(stmt.dst.name).substrate_id
        edges = [CouplingEdge(src_id=src_id, dst_id=dst_id, kappa=stmt.kappa)]
        self._new_segment_with_edges(edges, stmt.duration)
        self.op_log.append(f'couple {stmt.src.name} -> {stmt.dst.name} κ={stmt.kappa} for T={stmt.duration}')

    def _compile_pair(self, stmt: PairStmt):
        a, b = (stmt.a.name, stmt.b.name)
        for n in (a, b):
            if n not in self.mm.slots:
                raise CompileError(f'pair: unknown substrate {n!r}')
        a_id = self.mm.get(a).substrate_id
        b_id = self.mm.get(b).substrate_id
        edges = [CouplingEdge(src_id=a_id, dst_id=b_id, kappa=stmt.kappa), CouplingEdge(src_id=b_id, dst_id=a_id, kappa=stmt.kappa)]
        self._new_segment_with_edges(edges, stmt.duration)
        self.op_log.append(f'pair({a}, {b}) κ={stmt.kappa} for T={stmt.duration}')

    def _compile_ring(self, stmt: RingStmt):
        names = [m.name for m in stmt.members]
        for n in names:
            if n not in self.mm.slots:
                raise CompileError(f'ring: unknown substrate {n!r}')
        ids = [self.mm.get(n).substrate_id for n in names]
        N = len(ids)
        edges = []
        for i in range(N):
            j = (i + 1) % N
            edges.append(CouplingEdge(src_id=ids[i], dst_id=ids[j], kappa=stmt.kappa))
        self._new_segment_with_edges(edges, stmt.duration)
        self.op_log.append(f"ring({', '.join(names)}) κ={stmt.kappa} for T={stmt.duration}")

    def _compile_sequence(self, stmt: SequenceStmt):
        if stmt.target.name not in self.mm.slots:
            raise CompileError(f'sequence target {stmt.target.name!r} not declared')
        tgt_id = self.mm.get(stmt.target.name).substrate_id
        for inp in stmt.inputs:
            if inp.name not in self.mm.slots:
                raise CompileError(f'sequence input {inp.name!r} not declared')
            inp_id = self.mm.get(inp.name).substrate_id
            edges = [CouplingEdge(src_id=inp_id, dst_id=tgt_id, kappa=-3.0)]
            self._new_segment_with_edges(edges, stmt.each_for)
        self.op_log.append(f"sequence {stmt.target.name} via ({', '.join((i.name for i in stmt.inputs))}) each_for={stmt.each_for}")

    def _compile_observe(self, stmt: ObserveStmt):
        if stmt.target.name not in self.mm.slots:
            raise CompileError(f'OBSERVE: unknown substrate {stmt.target.name!r}')
        self.output_names.append(('OBSERVE', stmt.target.name, tuple(stmt.metrics), stmt.over_seeds, stmt.stream_to))
        suffix = ''
        if stmt.over_seeds > 1:
            suffix += f' over_seeds={stmt.over_seeds}'
        if stmt.stream_to:
            suffix += f' stream_to={stmt.stream_to!r}'
        self.op_log.append(f"OBSERVE {stmt.target.name} {','.join(stmt.metrics)}{suffix}")

    def _compile_assert(self, stmt: AssertStmt):
        if stmt.target.name not in self.mm.slots:
            raise CompileError(f'assert: unknown substrate {stmt.target.name!r}')
        self.output_names.append(('ASSERT', stmt.target.name, stmt.predicate))
        self.op_log.append(f'assert {stmt.predicate}({stmt.target.name})')

    def _compile_checkpoint(self, stmt: CheckpointStmt):
        if stmt.target.name not in self.mm.slots:
            raise CompileError(f'checkpoint: unknown substrate {stmt.target.name!r}')
        self.output_names.append(('CHECKPOINT', stmt.target.name, stmt.path))
        self.op_log.append(f'CHECKPOINT {stmt.target.name} to {stmt.path!r}')

    def _compile_substrate_decl(self, stmt: SubstrateDecl):
        member_names = [m.name for m in stmt.composed_of]
        for n in member_names:
            if n not in self.mm.slots:
                raise CompileError(f'substrate {stmt.name}: composed_of {n!r} not declared')
        member_ids = [self.mm.get(n).substrate_id for n in member_names]
        regime = stmt.properties.get('regime', 'B0')
        self._allocate_register_with_value(stmt.name, 0, bit_width=1, kind='register', regime_name=regime)
        macro_id = self.mm.get(stmt.name).substrate_id
        topology = stmt.properties.get('coupling', 'ring')
        kappa = float(stmt.properties.get('kappa', -2.0))
        kappa_up = float(stmt.properties.get('kappa_up', -1.0))
        duration = float(stmt.properties.get('duration', self.cfg.default_T))
        edges = []
        N = len(member_ids)
        if topology == 'ring' and N >= 2:
            for i in range(N):
                j = (i + 1) % N
                edges.append(CouplingEdge(src_id=member_ids[i], dst_id=member_ids[j], kappa=kappa))
        elif topology == 'pair' and N == 2:
            edges.append(CouplingEdge(src_id=member_ids[0], dst_id=member_ids[1], kappa=kappa))
            edges.append(CouplingEdge(src_id=member_ids[1], dst_id=member_ids[0], kappa=kappa))
        elif topology == 'none':
            pass
        else:
            raise CompileError(f'unknown coupling topology {topology!r}')
        for sub_id in member_ids:
            edges.append(CouplingEdge(src_id=sub_id, dst_id=macro_id, kappa=kappa_up))
        projection_name = stmt.properties.get('projection', 'none')
        from runtime.projections import resolve_projector
        proj_fn = resolve_projector(projection_name)
        if projection_name == 'fourier_band':
            k_lo = float(stmt.properties.get('k_lo', 0.1))
            k_hi = float(stmt.properties.get('k_hi', 1.0))

            def _bound_proj(subs, _p=proj_fn, _lo=k_lo, _hi=k_hi):
                return _p(subs, k_lo=_lo, k_hi=_hi)
            bound = _bound_proj
        else:
            bound = proj_fn
        macro_sub = self.runtime.substrates[macro_id]
        macro_sub.projector = bound
        macro_sub.projector_member_ids = list(member_ids)
        self._new_segment_with_edges(edges, duration)
        self.op_log.append(f"substrate {stmt.name} composed_of ({', '.join(member_names)})  topology={topology} kappa={kappa} kappa_up={kappa_up} for T={duration}")

    def _compile_if(self, blk: IfBlock):
        cond_name = blk.cond.name
        if cond_name not in self.mm.slots:
            raise CompileError(f'if condition {cond_name!r} not declared')
        cond_sub_id = self.mm.get(cond_name).substrate_id
        rt = self.runtime

        def make_modulator(invert: bool):

            def mod(rho_now: dict) -> float:
                sub = rt.substrates.get(cond_sub_id)
                if sub is None:
                    return 1.0
                from runtime.observables import crystallinity as _cr
                c = _cr(sub.psi, sub.dx)
                c = float(np.clip(c, 0.0, 1.0))
                return 1.0 - c if invert else c
            return mod
        self.op_log.append(f'IF cond={cond_name} -> then-branch (κ·C)')
        start_then = len(self.runtime.segments)
        self._walk(blk.then_body)
        end_then = len(self.runtime.segments)
        for seg_i in range(start_then, end_then):
            seg = self.runtime.segments[seg_i]
            for e in seg.edges:
                e.kappa_modulator = make_modulator(invert=False)
        if blk.else_body is not None:
            self.op_log.append(f'IF cond={cond_name} -> else-branch (κ·(1-C))')
            start_else = len(self.runtime.segments)
            self._walk(blk.else_body)
            end_else = len(self.runtime.segments)
            for seg_i in range(start_else, end_else):
                seg = self.runtime.segments[seg_i]
                for e in seg.edges:
                    e.kappa_modulator = make_modulator(invert=True)

    def _compile_out(self, stmt: OutStmt):
        for arg in stmt.args:
            if not isinstance(arg, IdentRef):
                raise CompileError('OUT arguments must be identifiers')
            self.output_names.append(arg.name)
        self.op_log.append(f"OUT {','.join((arg.name for arg in stmt.args if isinstance(arg, IdentRef)))}")

def decode_outputs(prog: CompiledProgram) -> dict[str, object]:
    from runtime.observables import ipr as obs_ipr, crystallinity as obs_C, dominant_wavenumber as obs_k_star, peak_density as obs_peak, fwhm as obs_fwhm, stabilization_score
    out: dict[str, object] = {}
    for entry in prog.output_names:
        if isinstance(entry, str):
            name = entry
            slot = prog.mm.get(name)
            sub = prog.runtime.substrates[slot.substrate_id]
            if slot.encoded_type == 'bool':
                out[name] = decode_bool(sub.psi, sub.dx)
            elif slot.encoded_type == 'float':
                out[name] = decode_float(sub.psi, sub.dx, prog.float_calib)
            else:
                out[name] = decode_int(sub.psi, sub.dx, prog.int_calib)
        elif isinstance(entry, tuple) and entry[0] == 'OBSERVE':
            if len(entry) == 4:
                _, name, metrics, _seeds = entry
                stream_to = ''
            else:
                _, name, metrics, _seeds, stream_to = entry
            slot = prog.mm.get(name)
            sub = prog.runtime.substrates[slot.substrate_id]
            row = {}
            import numpy as _np
            L_box = sub.params.L
            k_min = 2.0 * _np.pi / L_box
            D = int(getattr(sub, 'D', 1))
            dxD = sub.dx ** D
            for m in metrics:
                if m == 'k_star':
                    if D == 1:
                        row[m] = float(obs_k_star(sub.psi, sub.dx, k_min=k_min))
                    else:
                        psi_hat = _np.fft.fftn(sub.psi)
                        kvec = 2 * _np.pi * _np.fft.fftfreq(sub.params.N, d=sub.dx)
                        ks = _np.meshgrid(*[kvec] * D, indexing='ij')
                        kmag = _np.sqrt(sum((k ** 2 for k in ks)))
                        P = _np.abs(psi_hat) ** 2
                        mask = kmag >= k_min
                        row[m] = float(kmag[mask][_np.argmax(P[mask])])
                elif m == 'IPR':
                    n2 = (_np.abs(sub.psi) ** 2).sum() * dxD
                    n4 = (_np.abs(sub.psi) ** 4).sum() * dxD
                    row[m] = float(n4 / max(n2 * n2, 1e-30))
                elif m == 'crystallinity':
                    if D == 1:
                        row[m] = float(obs_C(sub.psi, sub.dx))
                    else:
                        psi_hat = _np.fft.fftn(sub.psi)
                        P = _np.abs(psi_hat) ** 2
                        kvec = 2 * _np.pi * _np.fft.fftfreq(sub.params.N, d=sub.dx)
                        ks = _np.meshgrid(*[kvec] * D, indexing='ij')
                        kmag = _np.sqrt(sum((k ** 2 for k in ks)))
                        row[m] = float(P[kmag > 1.0].sum() / max(P.sum(), 1e-30))
                elif m == 'peak':
                    row[m] = float((_np.abs(sub.psi) ** 2).max())
                elif m == 'FWHM':
                    if D == 1:
                        row[m] = float(obs_fwhm(sub.psi, sub.dx))
                    else:
                        row[m] = float('nan')
                elif m == 'norm':
                    row[m] = float((_np.abs(sub.psi) ** 2).sum() * dxD)
                elif m == 'stabilization':
                    if sub.density_traj is not None:
                        norm_t = sub.density_traj.sum(axis=0) * sub.dx
                        row[m] = float(stabilization_score(norm_t))
                    else:
                        row[m] = float('nan')
                elif m == 'memory_persistence':
                    if len(sub.lam_arr) > 0 and sub.y.size > 0:
                        y_sq = (sub.y ** 2).sum() * sub.dx
                        rho_sq = (_np.abs(sub.psi) ** 4).sum() * sub.dx
                        row[m] = float(y_sq / max(rho_sq, 1e-30))
                    else:
                        row[m] = 0.0
                elif m == 'atom_count':
                    from runtime.observables_atoms import atom_count_nd
                    row[m] = int(atom_count_nd(sub.psi, sub.dx))
                elif m == 'atom_centroids':
                    from runtime.observables_atoms import atom_centroids_nd
                    row[m] = atom_centroids_nd(sub.psi, sub.dx).tolist()
                elif m == 'atom_separation':
                    if D == 1:
                        from runtime.observables_atoms import atom_separation as _asep
                        row[m] = float(_asep(sub.psi, sub.dx))
                    else:
                        row[m] = float('nan')
                elif m == 'atoms_per_region':
                    from runtime.observables_atoms import atoms_per_region
                    row[m] = float(atoms_per_region(sub.psi, sub.dx))
                elif m == 'atomicity_ratio':
                    from runtime.observables_atoms import atomicity_ratio as _ar
                    member_ids = list(getattr(sub, 'projector_member_ids', []))
                    if not member_ids:
                        row[m] = float('nan')
                    else:
                        rho_agg = _np.zeros_like(_np.abs(sub.psi) ** 2)
                        for sid in member_ids:
                            sm = prog.runtime.substrates.get(sid)
                            if sm is None:
                                continue
                            rho_agg = rho_agg + _np.abs(sm.psi) ** 2
                        psi_agg = _np.sqrt(_np.maximum(rho_agg, 0)).astype(_np.complex128)
                        row[m] = float(_ar(sub.psi, psi_agg, sub.dx))
                elif m == 'atom_persistence_late':
                    if D == 1 and sub.density_traj is not None:
                        from runtime.observables_atoms import atom_persistence_late as _apl
                        row[m] = float(_apl(sub.density_traj, sub.dx))
                    else:
                        row[m] = float('nan')
                elif m == 'bravais_family':
                    if D == 3:
                        from runtime.codec_3d import bravais_family as _bf
                        row[m] = _bf(sub.psi, sub.dx)
                    else:
                        row[m] = '(N/A in 1D)'
                elif m == 'cluster_count':
                    if D == 1:
                        from runtime.observables_atoms import count_clusters as _cc
                        row[m] = int(_cc(sub.psi, sub.dx))
                    else:
                        row[m] = int((sub.psi.real != sub.psi.real).sum())
                elif m == 'cluster_centroids':
                    if D == 1:
                        from runtime.observables_atoms import cluster_centroids as _cen
                        row[m] = _cen(sub.psi, sub.dx).tolist()
                    else:
                        row[m] = []
                elif m == 'LZc':
                    from runtime.consciousness import lzc_of_trajectory
                    if sub.density_traj is not None:
                        row[m] = float(lzc_of_trajectory(sub.density_traj))
                    else:
                        row[m] = float('nan')
                elif m == 'phi_id':
                    from runtime.consciousness import phi_id_proxy
                    if sub.density_traj is not None:
                        row[m] = float(phi_id_proxy(sub.density_traj))
                    else:
                        row[m] = float('nan')
                elif m == 'causal_density':
                    from runtime.consciousness import causal_density_pair
                    members = list(getattr(sub, 'projector_member_ids', []))
                    if len(members) < 2:
                        row[m] = float('nan')
                    else:
                        pair_scores = []
                        for i in range(len(members)):
                            for j in range(i + 1, len(members)):
                                sa = prog.runtime.substrates.get(members[i])
                                sb = prog.runtime.substrates.get(members[j])
                                if sa is None or sb is None:
                                    continue
                                if sa.density_traj is None or sb.density_traj is None:
                                    continue
                                a_t = sa.density_traj.max(axis=0)
                                b_t = sb.density_traj.max(axis=0)
                                m_len = min(len(a_t), len(b_t))
                                pair_scores.append(causal_density_pair(a_t[:m_len], b_t[:m_len]))
                        row[m] = float(_np.mean(pair_scores)) if pair_scores else float('nan')
                elif m == 'kuramoto':
                    members = list(getattr(sub, 'projector_member_ids', []))
                    if len(members) < 2:
                        row[m] = float('nan')
                    else:
                        phases = []
                        for sid in members:
                            ss = prog.runtime.substrates.get(sid)
                            if ss is None:
                                continue
                            psi_hat = _np.fft.fft(ss.psi)
                            P = _np.abs(psi_hat)
                            P[0] = 0
                            idx = int(_np.argmax(P))
                            phases.append(float(_np.angle(psi_hat[idx])))
                        if not phases:
                            row[m] = float('nan')
                        else:
                            phases = _np.array(phases)
                            z = _np.exp(1j * phases).mean()
                            row[m] = float(_np.abs(z))
                elif m == 'pcist':
                    from runtime.consciousness import lzc_of_trajectory
                    if sub.density_traj is not None:
                        row[m] = float(lzc_of_trajectory(sub.density_traj))
                    else:
                        row[m] = float('nan')
                elif m == 'metastability':
                    from runtime.consciousness import metastability_phases as _meta
                    if sub.density_traj is not None and sub.density_traj.shape[1] > 1:
                        peak_pos = sub.density_traj.argmax(axis=0) * sub.dx
                        row[m] = float(_np.var(peak_pos)) if peak_pos.size > 1 else 0.0
                    else:
                        row[m] = float('nan')
                elif m == 'slow_state':
                    if sub.macro_slow_state_t:
                        row[m] = float(sub.macro_slow_state_t[-1])
                        row['slow_state_late_mean'] = float(_np.mean(sub.macro_slow_state_t[-max(1, len(sub.macro_slow_state_t) // 4):]))
                    else:
                        row[m] = float('nan')
                elif m == 'slow_state_late_mean':
                    if m in row:
                        pass
                    elif sub.macro_slow_state_t:
                        row[m] = float(_np.mean(sub.macro_slow_state_t[-max(1, len(sub.macro_slow_state_t) // 4):]))
                    else:
                        row[m] = float('nan')
                else:
                    row[m] = f'<unknown metric: {m!r}>'
            out[name] = row
            if stream_to and sub.density_traj is not None and (sub.t_traj is not None):
                import csv as _csv
                rho_traj = sub.density_traj
                t_arr = sub.t_traj
                n_rec = len(t_arr)
                stream_metrics = []
                if 'peak' in metrics:
                    stream_metrics.append('peak')
                if 'norm' in metrics:
                    stream_metrics.append('norm')
                if 'IPR' in metrics:
                    stream_metrics.append('IPR')
                if 'FWHM' in metrics:
                    stream_metrics.append('FWHM')
                if 'crystallinity' in metrics:
                    stream_metrics.append('crystallinity')
                if not stream_metrics:
                    stream_metrics = ['peak']
                with open(stream_to, 'w', newline='') as fh:
                    w = _csv.writer(fh)
                    w.writerow(['t'] + stream_metrics)
                    for k in range(n_rec):
                        rho_k = rho_traj[..., k] if rho_traj.ndim > 1 else rho_traj
                        if rho_k.ndim == 0 or rho_traj.shape[1] == 0:
                            continue
                        rho_1d = rho_k.ravel()
                        row_vals = [float(t_arr[k])]
                        for mm_ in stream_metrics:
                            if mm_ == 'peak':
                                row_vals.append(float(rho_1d.max()))
                            elif mm_ == 'norm':
                                row_vals.append(float(rho_1d.sum() * sub.dx ** getattr(sub, 'D', 1)))
                            elif mm_ == 'IPR':
                                n2 = rho_1d.sum() * sub.dx
                                n4 = (rho_1d ** 2).sum() * sub.dx
                                row_vals.append(float(n4 / max(n2 * n2, 1e-30)))
                            elif mm_ == 'FWHM':
                                half = 0.5 * rho_1d.max()
                                row_vals.append(float((rho_1d > half).sum() * sub.dx))
                            elif mm_ == 'crystallinity':
                                psi_p = _np.sqrt(rho_1d).astype(_np.complex128)
                                row_vals.append(float(obs_C(psi_p, sub.dx)))
                        w.writerow(row_vals)
                row['_streamed_to'] = stream_to
        elif isinstance(entry, tuple) and entry[0] == 'CHECKPOINT':
            import numpy as _np
            _, name, path = entry
            slot = prog.mm.get(name)
            sub = prog.runtime.substrates[slot.substrate_id]
            _np.savez(path, psi=sub.psi, y=sub.y, x=sub.x, L=sub.params.L, N=sub.params.N, D=getattr(sub, 'D', 1), seed=sub.params.seed)
            out[f'checkpoint_{name}'] = path
            continue
        elif isinstance(entry, tuple) and entry[0] == 'ASSERT':
            _, name, predicate = entry
            slot = prog.mm.get(name)
            sub = prog.runtime.substrates[slot.substrate_id]
            import numpy as _np
            if predicate == 'persistent':
                norm = float((_np.abs(sub.psi) ** 2).sum() * sub.dx)
                ok = _np.isfinite(norm) and norm > 1e-06
            elif predicate == 'extended':
                from runtime.observables_atoms import atom_count_nd
                ok = atom_count_nd(sub.psi, sub.dx) >= 1
            elif predicate == 'structurally_open':
                ok = sub.Gamma_e > 0 and sub.f_FDT_e > 0 and _np.isfinite((_np.abs(sub.psi) ** 2).sum())
            elif predicate == 'non_trivial_memory':
                if len(sub.lam_arr) > 0 and sub.y.size > 0:
                    ok = float((sub.y ** 2).sum() * sub.dx) > 1e-06
                else:
                    ok = False
            elif predicate == 'atomic':
                from runtime.observables_atoms import atom_count_nd
                ok = atom_count_nd(sub.psi, sub.dx) >= 1
            elif predicate == 'anti_collapsed':
                from runtime.observables_atoms import atom_count_nd
                ok = atom_count_nd(sub.psi, sub.dx) >= 2
            else:
                ok = False
            label = f'assert_{predicate}({name})'
            out[label] = ok
    return out