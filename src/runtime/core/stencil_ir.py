"""TriadLang Stencil IR - MLIR-inspired stencil dialect for PDE operations.

Implements a multi-level IR representation for stencil computations:
  Level 0: TriadLang AST (for/loop with array accesses)
  Level 1: Stencil IR (stencil.apply, stencil.access, stencil.store)
  Level 2: Lowered IR (loop nests, vectorized ops, or GPU kernels)

The three pillars are represented:
  P1: stencil.access captures spatial derivatives (FFT/stencil patterns)
  P2: stencil.temp fields carry memory/self-reference state
  P3: noise injection via stencil.combine operations

Lowering targets:
  - CPU: NumPy vectorized operations
  - GPU: CuPy fused kernels
  - Native: C code with FFTW + BLAS

Reference: Open Earth Compiler stencil dialect (Gysi et al., ACM TACO 2021)
"The Open Earth Compiler uses a stencil dialect where two domain-specific
optimizations (500 lines of code) realized on top of MLIR suffice to
outperform state-of-the-art solutions."
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum

class StencilTarget(Enum):
    CPU = 'cpu'
    GPU = 'gpu'
    NATIVE = 'native'

@dataclass
class StencilAccess:
    """Represents a stencil access pattern: array[offsets...].

    Example: accessing psi[i-1], psi[i], psi[i+1] for a 3-point stencil.
    """
    array_name: str
    offsets: tuple[int, ...]
    field: str = ''   

    def __repr__(self):
        ofs = ','.join(str(o) for o in self.offsets)
        return f'stencil.access({self.array_name}[{ofs}])'

@dataclass
class StencilOp:
    """A single stencil operation: access + arithmetic + store."""
    inputs: list[StencilAccess]
    output: str
    operation: str  
    coefficients: dict[str, float] = field(default_factory=dict)

    def describe(self) -> str:
        ins = ', '.join(str(a) for a in self.inputs)
        return f'{self.output} = {self.operation}({ins})'

@dataclass
class StencilApply:
    """stencil.apply: the main stencil computation over a grid.

    Mirrors MLIR's stencil.apply operator: takes input fields,
    applies a stencil pattern, and produces an output field.
    """
    name: str
    inputs: list[str]
    output: str
    ops: list[StencilOp]
    grid_shape: tuple[int, ...]
    halo: int = 1  

    def ir_repr(self) -> str:
        """Generate MLIR-like IR representation."""
        ins = ", ".join("%" + i for i in self.inputs)
        header = '%' + self.output + ' = "stencil.apply"(' + ins + ') ({'
        lines = [header + '  // ' + self.name]
        for op in self.ops:
            lines.append('  ' + op.describe())
        types = ", ".join("field" for _ in self.inputs)
        lines.append('}) : (' + types + ') -> field')
        return '\n'.join(lines)

@dataclass
class StencilCombine:
    """Combines multiple stencil results (e.g., P2 memory + P1 kinetic)."""
    inputs: list[str]
    output: str
    combine_op: str  
    weights: list[float] = field(default_factory=list)

    def describe(self) -> str:
        return f'{self.output} = stencil.combine({", ".join(self.inputs)}, op={self.combine_op})'

class StencilIR:
    """Intermediate representation for PDE stencil computations.

    Captures the stencil operations from the TriadLang PDE solver
    and provides lowering to different execution targets.
    """

    def __init__(self, name: str = 'triad_stencil'):
        self.name = name
        self.applies: list[StencilApply] = []
        self.combines: list[StencilCombine] = []
        self.fields: dict[str, np.ndarray] = {}
        self._target = StencilTarget.CPU

    def add_field(self, name: str, data: np.ndarray):
        """Register a field array."""
        self.fields[name] = data

    def add_apply(self, apply: StencilApply):
        """Add a stencil application."""
        self.applies.append(apply)

    def add_combine(self, combine: StencilCombine):
        """Add a stencil combination."""
        self.combines.append(combine)

    def set_target(self, target: StencilTarget):
        """Set the lowering target."""
        self._target = target

    def emit_ir(self) -> str:
        """Emit the full stencil IR."""
        lines = [f'stencil.module @{self.name} {{']
        for name, data in self.fields.items():
            shape = 'x'.join(str(s) for s in data.shape)
            lines.append(f'  %{name} = stencil.temp <{shape}xf64>')
        for apply in self.applies:
            lines.append(f'  {apply.ir_repr()}')
        for combine in self.combines:
            lines.append(f'  {combine.describe()}')
        lines.append('}')
        return '\n'.join(lines)

    def lower_to_cpu(self) -> Callable:
        """Lower stencil IR to CPU execution via NumPy vectorized ops."""
        grid_shape = self.applies[0].grid_shape if self.applies else (1,)

        def _execute(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            results = dict(fields)
            for apply in self.applies:
                out = np.zeros(apply.grid_shape, dtype=np.complex128)
                for op in apply.ops:
                    if op.operation == 'laplacian_1d':
                        psi = results.get(op.inputs[0].array_name, np.zeros(apply.grid_shape))
                        lap = np.zeros_like(psi)
                        lap[1:-1] = psi[2:] - 2 * psi[1:-1] + psi[:-2]
                        if op.coefficients:
                            c = op.coefficients.get('coeff', 1.0)
                            lap = lap * c
                        results[op.output] = lap
                    elif op.operation == 'nonlinear':
                        psi = results.get(op.inputs[0].array_name, np.zeros(apply.grid_shape))
                        rho = np.abs(psi) ** 2
                        results[op.output] = rho
                    elif op.operation == 'exp_dissipate':
                        psi = results.get(op.inputs[0].array_name, np.zeros(apply.grid_shape))
                        gamma = op.coefficients.get('gamma', 0.0)
                        results[op.output] = psi * np.exp(-gamma)
                    elif op.operation == 'add_noise':
                        psi = results.get(op.inputs[0].array_name, np.zeros(apply.grid_shape))
                        amp = op.coefficients.get('amplitude', 0.0)
                        noise = amp * (np.random.randn(*psi.shape) + 1j * np.random.randn(*psi.shape))
                        results[op.output] = psi + noise
                    else:
                        for inp in op.inputs:
                            if inp.array_name in results:
                                results[op.output] = results[inp.array_name]
            for combine in self.combines:
                vals = [results.get(n, np.zeros(grid_shape)) for n in combine.inputs]
                if combine.combine_op == 'add':
                    results[combine.output] = sum(vals)
                elif combine.combine_op == 'multiply':
                    res = np.ones(grid_shape, dtype=np.complex128)
                    for v in vals:
                        res = res * v
                    results[combine.output] = res
                elif combine.combine_op == 'mix':
                    if combine.weights:
                        res = sum(w * v for w, v in zip(combine.weights, vals))
                    else:
                        res = sum(vals) / max(len(vals), 1)
                    results[combine.output] = res
            return results
        return _execute

    def lower_to_gpu(self) -> Optional[Callable]:
        """Lower stencil IR to GPU execution via CuPy."""
        try:
            import cupy as cp
        except ImportError:
            return None

        cpu_fn = self.lower_to_cpu()

        def _execute_gpu(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            gpu_fields = {}
            for name, data in fields.items():
                gpu_fields[name] = cp.asarray(data)
            gpu_results = cpu_fn(gpu_fields)
            cpu_results = {}
            for name, data in gpu_results.items():
                if hasattr(data, 'get'):
                    cpu_results[name] = data.get()
                else:
                    cpu_results[name] = np.asarray(data)
            return cpu_results
        return _execute_gpu

    def lower(self) -> Callable:
        """Lower to the configured target."""
        if self._target == StencilTarget.GPU:
            gpu_fn = self.lower_to_gpu()
            if gpu_fn is not None:
                return gpu_fn
        return self.lower_to_cpu()

def build_pde_stencil_1d(N: int, L: float, dt: float, hbar: float, m: float,
                         Lambda: float, Gamma: float, f_FDT: float) -> StencilIR:
    """Build the standard 1D PDE stencil for the Triad equation.

    All three pillars:
      P1: laplacian_1d (kinetic/FFT)
      P2: nonlinear (memory/self-reference |psi|^2)
      P3: exp_dissipate + add_noise (coupling/dissipation/FDT)
    """
    dx = L / N
    coeff = -1j * hbar * dt / (2 * m * dx ** 2)

    ir = StencilIR('pde_1d_step')
    ir.add_field('psi', np.zeros(N, dtype=np.complex128))

    apply_kinetic = StencilApply(
        name='kinetic_step',
        inputs=['psi'],
        output='psi_k',
        ops=[
            StencilOp(
                inputs=[StencilAccess('psi', (-1,)), StencilAccess('psi', (0,)), StencilAccess('psi', (1,))],
                output='laplacian',
                operation='laplacian_1d',
                coefficients={'coeff': coeff.real if isinstance(coeff, complex) else coeff}
            ),
        ],
        grid_shape=(N,),
        halo=1,
    )
    ir.add_apply(apply_kinetic)

    apply_memory = StencilApply(
        name='memory_step',
        inputs=['psi'],
        output='psi_nl',
        ops=[
            StencilOp(
                inputs=[StencilAccess('psi', (0,))],
                output='rho',
                operation='nonlinear',
            ),
        ],
        grid_shape=(N,),
        halo=0,
    )
    ir.add_apply(apply_memory)

    apply_dissipate = StencilApply(
        name='dissipation_step',
        inputs=['psi'],
        output='psi_d',
        ops=[
            StencilOp(
                inputs=[StencilAccess('psi', (0,))],
                output='psi_diss',
                operation='exp_dissipate',
                coefficients={'gamma': Gamma * dt},
            ),
            StencilOp(
                inputs=[StencilAccess('psi_diss', (0,))],
                output='psi_noisy',
                operation='add_noise',
                coefficients={'amplitude': f_FDT * np.sqrt(dt)},
            ),
        ],
        grid_shape=(N,),
        halo=0,
    )
    ir.add_apply(apply_dissipate)

    ir.add_combine(StencilCombine(
        inputs=['psi_k', 'psi_nl', 'psi_d'],
        output='psi_next',
        combine_op='add',
    ))

    return ir
