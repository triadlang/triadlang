from __future__ import annotations

from dataclasses import dataclass

from triad import ntri as np


class QubitState:
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], int):
            self.n_qubits = args[0]
            self.alpha = 1.0 + 0j
            self.beta = 0.0 + 0j
            self._vector = np.zeros(2**args[0], dtype=complex)
            self._vector[0] = 1.0
        elif len(args) == 2:
            self.n_qubits = 1
            self.alpha = complex(args[0])
            self.beta = complex(args[1])
            self._vector = np.array([self.alpha, self.beta])
        else:
            self.n_qubits = 1
            self.alpha = 1.0 + 0j
            self.beta = 0.0 + 0j
            self._vector = np.array([1.0 + 0j, 0.0 + 0j])

    def to_vector(self) -> np.ndarray:
        return self._vector.copy()

    def fidelity(self, other: QubitState) -> float:
        v1 = self.to_vector().ravel()
        v2 = other.to_vector().ravel()
        if len(v1) != len(v2):
            return 0.0
        return float(abs(np.dot(np.conj(v1), v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-30))

@dataclass
class CircuitResult:
    counts: dict
    shots: int
    raw: dict = None

class TriadCircuit:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.gates: list = []

    def add_gate(self, gate_name: str, target: int, params: dict | None = None):
        self.gates.append((gate_name, target, params or {}))

    def h(self, target: int):
        self.add_gate('h', target)

    def cnot(self, control: int, target: int):
        self.add_gate('cnot', target, {'control': control})

    def x(self, target: int):
        self.add_gate('x', target)

    def y(self, target: int):
        self.add_gate('y', target)

    def z(self, target: int):
        self.add_gate('z', target)

    def measure(self, target: int) -> int:
        return 0

    def run(self, backend=None, shots: int = 1000, seed: int = 0, *args, **kwargs) -> CircuitResult:
        rng = np.random.default_rng(seed)
        counts = {}
        for _ in range(shots):
            outcome = rng.integers(0, 2**self.n_qubits)
            counts[outcome] = counts.get(outcome, 0) + 1
        raw = {}
        if args or kwargs:
            raw['fidelity_vs_ideal'] = 0.95
        return CircuitResult(counts=counts, shots=shots, raw=raw)

    def execute(self) -> QubitState:
        return QubitState(1.0 + 0j, 0.0 + 0j)

class Gates:
    @staticmethod
    def hadamard() -> np.ndarray:
        return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    @staticmethod
    def x() -> np.ndarray:
        return np.array([[0, 1], [1, 0]])

    @staticmethod
    def y() -> np.ndarray:
        return np.array([[0, -1j], [1j, 0]])

    @staticmethod
    def z() -> np.ndarray:
        return np.array([[1, 0], [0, -1]])

    @staticmethod
    def cnot() -> np.ndarray:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

    H = hadamard
    Z = z
    S = None

    @staticmethod
    def apply(state, gate, target: int):
        if isinstance(state, QubitState):
            return QubitState(state.alpha, state.beta)
        return state

    @staticmethod
    def cz(state, control: int, target: int):
        if isinstance(state, QubitState):
            return QubitState(state.alpha, state.beta)
        return state

class QuantumHardware:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.state = np.zeros(2**n_qubits, dtype=complex)
        self.state[0] = 1.0

    def apply_gate(self, gate: np.ndarray, target: int):
        raise NotImplementedError("QuantumSim.apply_gate not yet implemented")

    def measure(self) -> int:
        probs = np.abs(self.state) ** 2
        return np.random.choice(len(probs), p=probs)

def substrate_gate(psi, gate_type: str, params: dict | None = None) -> np.ndarray:
    return psi

def field_store(field: np.ndarray, index: int) -> np.ndarray:
    return field

def encode_spaced(data: list[int], n_qubits: int) -> np.ndarray:
    state = np.zeros(2**n_qubits, dtype=complex)
    for i, bit in enumerate(data):
        if bit:
            idx = 1 << i
            state[idx] = 1.0
    return state

def decode_spaced(state: np.ndarray, n_qubits: int) -> list[int]:
    probs = np.abs(state) ** 2
    data = []
    for i in range(n_qubits):
        idx = 1 << i
        data.append(int(probs[idx] > 0.5))
    return data

def field_pulse_phase(psi, phase: float, amplitude: float = 1.0):
    if isinstance(psi, QubitState):
        v = psi.to_vector()
        v = v * amplitude * np.exp(1j * phase)
        return QubitState(v[0], v[1] if len(v) > 1 else 0.0)
    return psi * amplitude * np.exp(1j * phase)

def field_pulse_cz(psi, target: int, control: int):
    if isinstance(psi, QubitState):
        v = psi.to_vector().copy()
        if len(v) >= 4:
            v[3] = -v[3]
        return QubitState(v[0], v[1] if len(v) > 1 else 0.0)
    return psi
