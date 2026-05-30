from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np

class QubitState:

    def __init__(self, n_qubits: int, amplitudes: Optional[np.ndarray]=None, label: str=''):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.label = label
        if amplitudes is None:
            self.psi = np.zeros(self.dim, dtype=np.complex128)
            self.psi[0] = 1.0
        else:
            self.psi = np.asarray(amplitudes, dtype=np.complex128).copy()
            if len(self.psi) != self.dim:
                raise ValueError(f'Need {self.dim} amplitudes for {n_qubits} qubits, got {len(self.psi)}')
            self._normalize()

    def _normalize(self):
        norm = np.sqrt(np.sum(np.abs(self.psi) ** 2))
        if norm > 1e-15:
            self.psi /= norm

    def copy(self) -> QubitState:
        return QubitState(self.n_qubits, self.psi.copy(), self.label)

    def probabilities(self) -> np.ndarray:
        return np.abs(self.psi) ** 2

    def fidelity(self, other: QubitState) -> float:
        return float(np.abs(self.psi.conj() @ other.psi) ** 2)

    def density_matrix(self) -> np.ndarray:
        return np.outer(self.psi, self.psi.conj())

    def entropy(self) -> float:
        p = self.probabilities()
        p = p[p > 1e-15]
        return float(-np.sum(p * np.log2(p)))

    def bloch_vector(self, qubit_idx: int=0) -> np.ndarray:
        if self.n_qubits < 1:
            raise ValueError('Need at least 1 qubit')
        reduced = self._partial_trace_one(qubit_idx)
        rx = 2.0 * np.real(reduced[0, 1])
        ry = 2.0 * np.imag(reduced[1, 0])
        rz = float(np.real(reduced[0, 0] - reduced[1, 1]))
        return np.array([rx, ry, rz])

    def _partial_trace_one(self, qubit_idx: int) -> np.ndarray:
        rho = self.density_matrix()
        if self.n_qubits == 1:
            return rho
        keep = list(range(self.n_qubits))
        keep.remove(qubit_idx)
        d_keep = 2 ** len(keep)
        d_trace = 2
        rho_reshaped = rho.reshape([2] * self.n_qubits + [2] * self.n_qubits)
        axes_keep = keep + [a + self.n_qubits for a in keep]
        axes_trace = [qubit_idx, qubit_idx + self.n_qubits]
        from functools import reduce
        rho_out = np.einsum(rho_reshaped, list(range(2 * self.n_qubits)), axes_trace, optimize=True)
        return rho_out.reshape(d_keep, d_keep) if d_keep > 1 else rho_out

    def measure(self, qubit_idx: Optional[int]=None, rng: Optional[np.random.Generator]=None) -> Union[int, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()
        if qubit_idx is not None:
            return self._measure_single(qubit_idx, rng)
        probs = self.probabilities()
        outcome = rng.choice(self.dim, p=probs)
        self.psi = np.zeros(self.dim, dtype=np.complex128)
        self.psi[outcome] = 1.0
        return outcome

    def _measure_single(self, qubit_idx: int, rng: np.random.Generator) -> int:
        p0 = 0.0
        for j in range(self.dim):
            if not j >> self.n_qubits - 1 - qubit_idx & 1:
                p0 += abs(self.psi[j]) ** 2
        outcome = 0 if rng.random() < p0 else 1
        mask_bit = 1 << self.n_qubits - 1 - qubit_idx
        for j in range(self.dim):
            if j >> self.n_qubits - 1 - qubit_idx & 1 == 1 - outcome:
                self.psi[j] = 0.0
        self._normalize()
        return outcome

    def __repr__(self) -> str:
        terms = []
        for j in range(self.dim):
            c = self.psi[j]
            if abs(c) < 1e-10:
                continue
            bits = format(j, f'0{self.n_qubits}b')
            terms.append(f'({c:.4f})|{bits}⟩')
        return f"QubitState({self.n_qubits}q): {(' + '.join(terms) if terms else '∅')}"

def encode_to_substrate(state: QubitState, N_grid: int, L: float=32.0) -> np.ndarray:
    dx = L / N_grid
    n = state.n_qubits
    dim = state.dim
    if dim > N_grid:
        raise ValueError(f'Need N_grid >= {dim} for {n} qubits, got {N_grid}')
    psi = np.zeros(N_grid, dtype=np.complex128)
    center = N_grid // 2 - dim // 2
    psi[center:center + dim] = state.psi
    tail_len = N_grid - center - dim
    if tail_len > 0 and dim > 0:
        last_amp = state.psi[-1]
        decay = np.exp(-np.arange(1, tail_len + 1) * 0.3)
        psi[center + dim:] = last_amp * decay
    if center > 0 and dim > 0:
        first_amp = state.psi[0]
        decay = np.exp(-np.arange(1, center + 1)[::-1] * 0.3)
        psi[:center] = first_amp * decay
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    if norm > 1e-15:
        psi /= norm
    return psi

def decode_from_substrate(psi: np.ndarray, n_qubits: int, L: float=32.0) -> QubitState:
    N = len(psi)
    dim = 2 ** n_qubits
    center = N // 2 - dim // 2
    amplitudes = psi[center:center + dim].copy()
    norm = np.sqrt(np.sum(np.abs(amplitudes) ** 2))
    if norm > 1e-15:
        amplitudes /= norm
    return QubitState(n_qubits, amplitudes)

def decode_probabilities_from_substrate(psi: np.ndarray, n_qubits: int) -> np.ndarray:
    N = len(psi)
    dim = 2 ** n_qubits
    center = N // 2 - dim // 2
    amps = psi[center:center + dim]
    probs = np.abs(amps) ** 2
    total = probs.sum()
    if total > 1e-15:
        probs /= total
    return probs

def _gate_1q(matrix: np.ndarray, state: QubitState, qubit: int) -> QubitState:
    n = state.n_qubits
    dim = state.dim
    U_full = np.eye(dim, dtype=np.complex128)
    for i in range(dim):
        for j in range(dim):
            if i >> n - 1 - qubit & 1 == 0 and j >> n - 1 - qubit & 1 == 0:
                mask = ~(1 << n - 1 - qubit)
                i0, j0 = (i & mask, j & mask)
                if i0 == j0:
                    U_full[i, j] = matrix[0, 0]
                    U_full[i | 1 << n - 1 - qubit, j] = matrix[1, 0]
                    U_full[i, j | 1 << n - 1 - qubit] = matrix[0, 1]
                    U_full[i | 1 << n - 1 - qubit, j | 1 << n - 1 - qubit] = matrix[1, 1]
    new_psi = U_full @ state.psi
    return QubitState(n, new_psi, state.label)

class Gates:
    I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)

    @staticmethod
    def rx(theta: float) -> np.ndarray:
        c, s = (np.cos(theta / 2), np.sin(theta / 2))
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)

    @staticmethod
    def ry(theta: float) -> np.ndarray:
        c, s = (np.cos(theta / 2), np.sin(theta / 2))
        return np.array([[c, -s], [s, c]], dtype=np.complex128)

    @staticmethod
    def rz(theta: float) -> np.ndarray:
        return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=np.complex128)

    @staticmethod
    def apply(state: QubitState, gate: np.ndarray, qubit: int) -> QubitState:
        return _gate_1q(gate, state, qubit)

    @staticmethod
    def cnot(state: QubitState, control: int, target: int) -> QubitState:
        n = state.n_qubits
        new_psi = state.psi.copy()
        c_bit = 1 << n - 1 - control
        t_bit = 1 << n - 1 - target
        for j in range(state.dim):
            if j & c_bit:
                swapped = j ^ t_bit
                new_psi[j], new_psi[swapped] = (state.psi[swapped].copy(), state.psi[j].copy())
        return QubitState(n, new_psi, state.label)

    @staticmethod
    def cz(state: QubitState, control: int, target: int) -> QubitState:
        n = state.n_qubits
        new_psi = state.psi.copy()
        c_bit = 1 << n - 1 - control
        t_bit = 1 << n - 1 - target
        for j in range(state.dim):
            if j & c_bit and j & t_bit:
                new_psi[j] = -new_psi[j]
        return QubitState(n, new_psi, state.label)

    @staticmethod
    def bell(n_qubits: int=2) -> QubitState:
        s = QubitState(n_qubits)
        s = Gates.apply(s, Gates.H, 0)
        s = Gates.cnot(s, 0, 1)
        return s

    @staticmethod
    def ghz(n_qubits: int) -> QubitState:
        s = QubitState(n_qubits)
        s = Gates.apply(s, Gates.H, 0)
        for i in range(1, n_qubits):
            s = Gates.cnot(s, 0, i)
        return s

def substrate_gate(state: QubitState, gate_type: str='evolve', T: float=1.0, dt: float=0.005, n_qubits: int=0, **kwargs) -> QubitState:
    n = n_qubits or state.n_qubits
    dim = 2 ** n
    N_grid = max(128, dim * 4)
    L = 32.0
    psi_sub = encode_to_substrate(state, N_grid, L)
    from runtime.solver import TriadParams, integrate
    regime = gate_type
    if gate_type == 'evolve':
        p = TriadParams(N=N_grid, L=L, T=T, dt=dt, mode='full', V_ext=None, seed=42)
    elif gate_type == 'kerr':
        p = TriadParams(N=N_grid, L=L, T=T, dt=dt, mode='full', Lambda=-1.0, V_ext=None, Gamma=0.0, f_FDT=0.0, alpha=0.0, nu=(0.1,), lam=(0.0,), seed=42)
    elif gate_type == 'decohere':
        p = TriadParams(N=N_grid, L=L, T=T, dt=dt, mode='full', Lambda=0.0, V_ext=None, Gamma=0.05, f_FDT=0.002, alpha=0.0, nu=(0.1,), lam=(0.0,), seed=42)
    elif gate_type == 'memory':
        p = TriadParams(N=N_grid, L=L, T=T, dt=dt, mode='full', Lambda=0.0, V_ext=None, Gamma=0.0, f_FDT=0.0, alpha=0.0, nu=(2.0, 0.5, 0.1), lam=(-0.3, -0.2, -0.1), seed=42)
    elif gate_type == 'full':
        p = TriadParams(N=N_grid, L=L, T=T, dt=dt, mode='full', V_ext=None, seed=42)
    else:
        raise ValueError(f'Unknown gate_type: {gate_type}')
    out = integrate(p, psi0=psi_sub, auto_halve_dt=False)
    psi_out = out['psi_final']
    return decode_from_substrate(psi_out, n, L)

@dataclass
class HardwareResult:
    counts: dict[str, int]
    shots: int
    backend_name: str
    raw: Optional[object] = None

class QuantumHardware:

    def __init__(self, backend_type: str='simulator', **kwargs):
        self.backend_type = backend_type
        self.config = kwargs
        self._backend = None

    def run(self, state: QubitState, shots: int=1024, seed: Optional[int]=None) -> HardwareResult:
        if self.backend_type == 'simulator':
            return self._run_simulator(state, shots, seed)
        elif self.backend_type == 'qiskit':
            return self._run_qiskit(state, shots, seed)
        elif self.backend_type == 'cirq':
            return self._run_cirq(state, shots, seed)
        else:
            raise ValueError(f'Unknown backend: {self.backend_type}')

    def _run_simulator(self, state: QubitState, shots: int, seed: Optional[int]) -> HardwareResult:
        rng = np.random.default_rng(seed)
        probs = state.probabilities()
        n = state.n_qubits
        outcomes = rng.multinomial(shots, probs)
        counts = {}
        for j, count in enumerate(outcomes):
            if count > 0:
                bits = format(j, f'0{n}b')
                counts[bits] = int(count)
        return HardwareResult(counts=counts, shots=shots, backend_name='triadlang_simulator')

    def _run_qiskit(self, state: QubitState, shots: int, seed: Optional[int]) -> HardwareResult:
        try:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
            from qiskit_aer import AerSimulator
        except ImportError:
            raise ImportError('Qiskit not installed. pip install qiskit qiskit-aer')
        n = state.n_qubits
        qc = QuantumCircuit(n)
        sv = Statevector(state.psi)
        qc.initialize(sv, range(n))
        qc.measure_all()
        sim = AerSimulator()
        if seed is not None:
            sim.set_options(seed_simulator=seed)
        job = sim.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts()
        return HardwareResult(counts=dict(counts), shots=shots, backend_name='qiskit_aer', raw=result)

    def _run_cirq(self, state: QubitState, shots: int, seed: Optional[int]) -> HardwareResult:
        try:
            import cirq
        except ImportError:
            raise ImportError('Cirq not installed. pip install cirq')
        n = state.n_qubits
        qubits = cirq.LineQubit.range(n)
        circuit = cirq.Circuit()
        state_matrix = state.psi.reshape([2] * n)
        circuit.append(cirq.StatePreparationChannel(state_matrix).on(*qubits))
        circuit.append(cirq.measure(*qubits, key='result'))
        simulator = cirq.Simulator(seed=seed)
        result = simulator.run(circuit, repetitions=shots)
        measurements = result.measurements['result']
        counts = {}
        for row in measurements:
            bits = ''.join((str(b) for b in row))
            counts[bits] = counts.get(bits, 0) + 1
        return HardwareResult(counts=counts, shots=shots, backend_name='cirq_simulator', raw=result)

    def read_qubits_from_result(self, result: HardwareResult) -> QubitState:
        total = sum(result.counts.values())
        n = len(list(result.counts.keys())[0])
        amplitudes = np.zeros(2 ** n, dtype=np.complex128)
        for bits, count in result.counts.items():
            j = int(bits, 2)
            amplitudes[j] = np.sqrt(count / total)
        return QubitState(n, amplitudes, label=f'from_{result.backend_name}')

    def read_real_qubits(self, n_qubits: int, circuit: Optional[object]=None, shots: int=8192) -> QubitState:
        if circuit is None:
            state = QubitState(n_qubits)
            result = self.run(state, shots)
            return self.read_qubits_from_result(result)
        if self.backend_type == 'qiskit':
            return self._read_qiskit_circuit(circuit, n_qubits, shots)
        elif self.backend_type == 'cirq':
            return self._read_cirq_circuit(circuit, n_qubits, shots)
        else:
            raise ValueError(f'Circuit execution not supported for {self.backend_type}')

    def _read_qiskit_circuit(self, circuit, n_qubits, shots):
        from qiskit_aer import AerSimulator
        sim = AerSimulator()
        if not any((op.name == 'measure' for op in circuit.data)):
            circuit.measure_all()
        job = sim.run(circuit, shots=shots)
        counts = dict(job.result().get_counts())
        result = HardwareResult(counts=counts, shots=shots, backend_name='qiskit_aer')
        return self.read_qubits_from_result(result)

    def _read_cirq_circuit(self, circuit, n_qubits, shots):
        import cirq
        sim = cirq.Simulator()
        result = sim.run(circuit, repetitions=shots)
        key = list(result.measurements.keys())[0]
        measurements = result.measurements[key]
        counts = {}
        for row in measurements:
            bits = ''.join((str(b) for b in row))
            counts[bits] = counts.get(bits, 0) + 1
        hr = HardwareResult(counts=counts, shots=shots, backend_name='cirq_simulator')
        return self.read_qubits_from_result(hr)

class TriadCircuit:

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self._ops = []

    def h(self, qubit: int):
        self._ops.append(('gate', Gates.H, qubit))
        return self

    def x(self, qubit: int):
        self._ops.append(('gate', Gates.X, qubit))
        return self

    def y(self, qubit: int):
        self._ops.append(('gate', Gates.Y, qubit))
        return self

    def z(self, qubit: int):
        self._ops.append(('gate', Gates.Z, qubit))
        return self

    def s(self, qubit: int):
        self._ops.append(('gate', Gates.S, qubit))
        return self

    def t(self, qubit: int):
        self._ops.append(('gate', Gates.T, qubit))
        return self

    def rx(self, qubit: int, theta: float):
        self._ops.append(('gate', Gates.rx(theta), qubit))
        return self

    def ry(self, qubit: int, theta: float):
        self._ops.append(('gate', Gates.ry(theta), qubit))
        return self

    def rz(self, qubit: int, theta: float):
        self._ops.append(('gate', Gates.rz(theta), qubit))
        return self

    def cnot(self, control: int, target: int):
        self._ops.append(('cnot', control, target))
        return self

    def cz(self, control: int, target: int):
        self._ops.append(('cz', control, target))
        return self

    def substrate_evolve(self, T: float=1.0, gate_type: str='full', dt: float=0.005, **kwargs):
        self._ops.append(('substrate', T, gate_type, dt, kwargs))
        return self

    def run(self, initial: Optional[QubitState]=None, shots: int=1024, seed: Optional[int]=None, measure: bool=True) -> Union[HardwareResult, QubitState]:
        state = initial or QubitState(self.n_qubits)
        for op in self._ops:
            if op[0] == 'gate':
                state = Gates.apply(state, op[1], op[2])
            elif op[0] == 'cnot':
                state = Gates.cnot(state, op[1], op[2])
            elif op[0] == 'cz':
                state = Gates.cz(state, op[1], op[2])
            elif op[0] == 'substrate':
                _, T, gtype, dt, kw = op
                state = substrate_gate(state, gate_type=gtype, T=T, dt=dt, n_qubits=self.n_qubits, **kw)
        if measure:
            hw = QuantumHardware('simulator')
            return hw.run(state, shots, seed)
        return state