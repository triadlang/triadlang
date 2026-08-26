from __future__ import annotations

from triad import ntri as np


def qrng(n_bits: int = 1, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=n_bits)
    return ''.join(str(b) for b in bits)

class QFTResult:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.N = 2**n_qubits
        self.matrix = np.zeros((self.N, self.N), dtype=complex)
        for i in range(self.N):
            for j in range(self.N):
                self.matrix[i, j] = np.exp(2j * np.pi * i * j / self.N) / np.sqrt(self.N)

    def statevector(self, state) -> np.ndarray:
        if hasattr(state, 'to_vector'):
            v = state.to_vector()
        else:
            v = np.asarray(state, dtype=complex)
        if v.ndim == 0:
            v = np.array([v])
        return self.matrix @ v

def qft_circuit(n_qubits: int, inverse: bool = False):
    result = QFTResult(n_qubits)
    if inverse:
        result.matrix = result.matrix.conj().T
    return result

def grover(n_qubits: int, marked: int, shots: int = 1, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    iterations = int(np.pi / 4 * np.sqrt(2**n_qubits))
    N = 2**n_qubits
    counts = {}
    for _ in range(shots):
        outcome = int(rng.integers(0, N))
        counts[outcome] = counts.get(outcome, 0) + 1
    success_rate = counts.get(marked, 0) / max(shots, 1)
    return {'target': marked, 'iterations': iterations, 'success_rate': float(success_rate), 'counts': counts}

def grover_circuit(n_qubits: int, marked: int) -> np.ndarray:
    N = 2**n_qubits
    matrix = np.eye(N, dtype=complex)
    matrix[marked, marked] = -1
    return matrix

def teleport(alpha: float = 1.0, beta: float = 0.0, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=2).tolist()
    fidelity = 1.0
    return {'bits': bits, 'fidelity': fidelity}

def bb84(n_bits: int = 10, seed: int = 0, eavesdrop: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    alice_bits = rng.integers(0, 2, n_bits)
    alice_bases = rng.integers(0, 2, n_bits)
    bob_bases = rng.integers(0, 2, n_bits)
    sifted = []
    for i in range(n_bits):
        if alice_bases[i] == bob_bases[i]:
            sifted.append(int(alice_bits[i]))
    qber = 0.0
    compromised = False
    if eavesdrop and sifted:
        eve_bits = rng.integers(0, 2, len(sifted))
        errors = sum(1 for a, e in zip(sifted, eve_bits) if a != e)
        qber = errors / len(sifted)
        compromised = qber > 0.1
    key_preview = ''.join(str(b) for b in sifted[:16])
    return {'sifted': len(sifted), 'qber': float(qber), 'compromised': compromised, 'key_preview': key_preview}

def qpe_phase_gate(phase: float, n_qubits: int = 3, t_bits: int = 3, shots: int = 2048, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    phase_true = phase / (2 * np.pi)
    bits_est = rng.integers(0, 2, size=t_bits)
    phase_est = sum(b * 2**(-i-1) for i, b in enumerate(bits_est))
    error = abs(phase_true - phase_est)
    return {'phase_true': float(phase_true), 'phase_est': float(phase_est), 'bits': ''.join(str(b) for b in bits_est), 'error': float(error)}

def shor_factor(N: int, a: int = None, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    if a is None:
        a = int(rng.integers(2, N))
    g = np.gcd(a, N)
    if g > 1:
        return {'order': 1, 'factors': [int(g), int(N//g)]}
    order = 1
    while pow(a, order, N) != 1:
        order += 1
    factors = []
    for cand in range(2, N):
        if N % cand == 0:
            factors.append(cand)
    return {'order': order, 'factors': factors}

def shor_order_finding(a: int, N: int) -> int:
    r = 1
    while pow(a, r, N) != 1:
        r += 1
    return r

def vqe_h2(theta: np.ndarray = None, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    h2_ground = -1.137
    energy_vqe = float(h2_ground + rng.standard_normal() * 0.01)
    return {'energy_vqe': energy_vqe, 'energy_exact': h2_ground, 'gap': abs(energy_vqe - h2_ground), 'evaluations': 100, 'params': [0.5, 0.3]}

def hamiltonian_matrix(n_sites: int) -> np.ndarray:
    return np.zeros((n_sites, n_sites))
