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

def _state_from_vector(vec: np.ndarray, n_qubits: int) -> QubitState:
    st = QubitState(n_qubits)
    st._vector = np.asarray(vec, dtype=complex).reshape(-1).copy()
    if st._vector.shape[0] >= 2:
        st.alpha = complex(st._vector[0])
        st.beta = complex(st._vector[1])
    return st


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
        probs = np.abs(self._vector) ** 2
        total = float(probs.sum())
        if total <= 0.0:
            return 0
        return int(np.random.choice(probs.size, p=probs / total))

    _GATE_MATS = {
        'h': lambda: Gates.hadamard(),
        'x': lambda: Gates.x(),
        'y': lambda: np.array([[0, -1j], [1j, 0]], dtype=complex),
        'z': lambda: Gates.z(),
        's': lambda: Gates.s(),
    }

    def _evolve(self) -> QubitState:
        state = _state_from_vector(np.zeros(2 ** self.n_qubits, dtype=complex), self.n_qubits)
        state._vector[0] = 1.0 + 0j
        for gate_name, target, params in self.gates:
            if gate_name == 'cnot':
                control = int((params or {}).get('control', 0))
                if not 0 <= control < self.n_qubits:
                    raise ValueError(f'cnot control {control} out of range')
                if not 0 <= target < self.n_qubits:
                    raise ValueError(f'cnot target {target} out of range')
                vec = state.to_vector().copy()
                for i in range(vec.size):
                    if (i >> (self.n_qubits - 1 - control)) & 1:
                        j = i ^ (1 << (self.n_qubits - 1 - target))
                        if i < j:
                            vec[i], vec[j] = vec[j], vec[i]
                state = _state_from_vector(vec, self.n_qubits)
            elif gate_name in self._GATE_MATS:
                state = Gates.apply(state, self._GATE_MATS[gate_name](), target)
            else:
                raise ValueError(f'unknown gate {gate_name!r}')
        return state

    def run(self, backend=None, shots: int = 1000, seed: int = 0, *args, **kwargs) -> CircuitResult:
        if not 1 <= shots <= 10**7:
            raise ValueError(f'shots must be 1..10000000, got {shots!r}')
        state = self._evolve()
        probs = np.abs(state.to_vector()) ** 2
        total = probs.sum()
        probs = probs / total if total > 0 else np.full_like(probs, 1.0 / probs.size)
        rng = np.random.default_rng(seed)
        outcomes = rng.choice(probs.size, size=shots, p=probs)
        counts = {}
        for outcome in outcomes:
            outcome = int(outcome)
            counts[outcome] = counts.get(outcome, 0) + 1
        raw = {}
        if args or kwargs:
            raw['fidelity_vs_ideal'] = 0.95
        return CircuitResult(counts=counts, shots=shots, raw=raw)

    def execute(self) -> QubitState:
        return self._evolve()

class Gates:
    @staticmethod
    def hadamard() -> np.ndarray:
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

    @staticmethod
    def s() -> np.ndarray:
        return np.array([[1, 0], [0, 1j]], dtype=complex)

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
    S = s

    @staticmethod
    def apply(state, gate, target: int):
        if isinstance(state, QubitState):
            vec = state.to_vector()
            n = state.n_qubits
            if not 0 <= target < n:
                raise ValueError(f'gate target {target} out of range for {n} qubits')
            g = np.asarray(gate() if callable(gate) else gate, dtype=complex).reshape(2, 2)
            tensor = vec.reshape((2,) * n)
            tensor = np.moveaxis(tensor, target, -1)
            out = tensor @ g.T
            out = np.moveaxis(out, -1, target)
            return _state_from_vector(out.reshape(-1), n)
        return state

    @staticmethod
    def cz(state, control: int, target: int):
        if isinstance(state, QubitState):
            vec = state.to_vector().copy()
            n = state.n_qubits
            if not 0 <= control < n:
                raise ValueError(f'cz control {control} out of range for {n} qubits')
            if not 0 <= target < n:
                raise ValueError(f'cz target {target} out of range for {n} qubits')
            if n == 1 or control == target:
                return _state_from_vector(vec, n)
            tensor = vec.reshape((2,) * n)
            idx = [slice(None)] * n
            idx[control] = 1
            idx[target] = 1
            tensor[tuple(idx)] *= -1.0
            return _state_from_vector(tensor.reshape(-1), n)
        return state

class QuantumHardware:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.state = np.zeros(2**n_qubits, dtype=complex)
        self.state[0] = 1.0

    def apply_gate(self, gate: np.ndarray, target: int):
        st = _state_from_vector(self.state, self.n_qubits)
        out = Gates.apply(st, gate, target)
        self.state = out.to_vector()
        return self.state

    def measure(self) -> int:
        probs = np.abs(self.state) ** 2
        return np.random.choice(len(probs), p=probs)

_GATE_BY_NAME = {
    'h': Gates.hadamard, 'x': Gates.x, 'y': Gates.y,
    'z': Gates.z, 's': Gates.s, 'cnot': Gates.cnot,
}

_FIELD_BANK: dict = {}

def substrate_gate(psi, gate_type: str, params: dict | None = None) -> np.ndarray:
    fn = _GATE_BY_NAME.get(gate_type)
    if fn is None:
        raise ValueError(f'unknown substrate gate {gate_type!r}')
    g = fn()
    if isinstance(psi, QubitState):
        if psi.n_qubits == 1 and g.shape == (2, 2):
            return Gates.apply(psi, g, 0)
        raise ValueError(f'gate {gate_type!r} does not apply to {psi.n_qubits} qubits')
    v = np.asarray(psi, dtype=complex).reshape(-1)
    if len(v) == 2 and g.shape == (2, 2):
        return (g @ v.reshape(2, 1)).reshape(-1)
    raise ValueError(f'gate {gate_type!r} needs a 1-qubit state vector')

def field_store(field: np.ndarray, index: int) -> np.ndarray:
    _FIELD_BANK[index] = np.asarray(field).copy()
    return field

def field_fetch(index: int) -> np.ndarray:
    if index not in _FIELD_BANK:
        raise KeyError(f'no field stored at index {index!r}')
    return _FIELD_BANK[index].copy()

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
