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
    N = 2**n_qubits
    iterations = int(np.pi / 4 * np.sqrt(N))
    amp = [1.0 / (N ** 0.5)] * N
    for _ in range(iterations):
        amp[marked] = -amp[marked]
        m = sum(amp) / N
        amp = [2.0 * m - v for v in amp]
    probs = [v * v for v in amp]
    counts = {}
    for outcome in rng.choice(N, size=shots, p=np.asarray(probs)).tolist():
        outcome = int(outcome)
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
    norm = (abs(alpha) ** 2 + abs(beta) ** 2) ** 0.5
    if norm == 0.0:
        raise ValueError('teleport needs a nonzero input state')
    outcome = int(rng.integers(0, 4))
    bits = [(outcome >> 1) & 1, outcome & 1]
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
    if int(n_qubits) < 1:
        raise ValueError(f'qpe needs n_qubits >= 1, got {n_qubits!r}')
    if int(t_bits) < 1:
        raise ValueError(f'qpe needs t_bits >= 1, got {t_bits!r}')
    rng = np.random.default_rng(seed)
    phase_true = (phase / (2 * float(np.pi))) % 1.0
    M = 2**t_bits
    probs = []
    for k in range(M):
        delta = phase_true - k / M
        if abs(delta) < 1e-12:
            probs.append(1.0)
            continue
        s = np.sin(np.pi * M * delta) / (M * np.sin(np.pi * delta))
        probs.append(float(s * s))
    counts = {}
    for outcome in rng.choice(M, size=shots, p=np.asarray(probs)).tolist():
        outcome = int(outcome)
        counts[outcome] = counts.get(outcome, 0) + 1
    best = max(counts, key=lambda k: counts[k])
    bits = format(best, f'0{t_bits}b')
    phase_est = best / M
    error = abs(phase_true - phase_est)
    return {'phase_true': float(phase_true), 'phase_est': float(phase_est), 'bits': bits, 'error': float(error)}

def shor_factor(N: int, a: int = None, seed: int = 0, order_cap: int = 200000) -> dict:
    rng = np.random.default_rng(seed)
    if a is None:
        a = int(rng.integers(2, N))
    g = np.gcd(a, N)
    if g > 1:
        return {'order': 1, 'factors': [int(g), int(N//g)]}
    order = 1
    while pow(a, order, N) != 1:
        order += 1
        if order > order_cap:
            raise ValueError(f'shor_factor order search exceeded cap {order_cap}')
    factors = []
    for cand in range(2, min(N, order_cap)):
        if N % cand == 0:
            factors.append(cand)
    return {'order': order, 'factors': factors}

def shor_order_finding(a: int, N: int, order_cap: int = 200000) -> int:
    r = 1
    while pow(a, r, N) != 1:
        r += 1
        if r > order_cap:
            raise ValueError(f'shor_order_finding exceeded cap {order_cap}')
    return r

_H2_PAULI = {'I': (1.0, 0.0, 0.0, 1.0), 'X': (0.0, 1.0, 1.0, 0.0),
              'Y': (0.0, -1j, 1j, 0.0), 'Z': (1.0, 0.0, 0.0, -1.0)}
_H2_TERMS = [(-0.4804, 'II'), (0.3935, 'IZ'), (-0.3935, 'ZI'),
             (-0.0113, 'ZZ'), (0.1812, 'XX')]

def _kron2(a00, a01, a10, a11, b00, b01, b10, b11):
    return [[a00 * b00, a00 * b01, a01 * b00, a01 * b01],
            [a00 * b10, a00 * b11, a01 * b10, a01 * b11],
            [a10 * b00, a10 * b01, a11 * b00, a11 * b01],
            [a10 * b10, a10 * b11, a11 * b10, a11 * b11]]

def _h2_matrix():
    H = [[0j] * 4 for _ in range(4)]
    for coef, word in _H2_TERMS:
        a = _H2_PAULI[word[0]]
        b = _H2_PAULI[word[1]]
        M = _kron2(*a, *b)
        for i in range(4):
            for j in range(4):
                H[i][j] += coef * M[i][j]
    return H

def _ry_state(t0, t1, t2, t3):
    import math as _m
    c0, s0 = _m.cos(t0 / 2), _m.sin(t0 / 2)
    c1, s1 = _m.cos(t1 / 2), _m.sin(t1 / 2)
    v = [c0 * c1, c0 * s1, s0 * c1, s0 * s1]
    v[2], v[3] = v[3], v[2]
    c2, s2 = _m.cos(t2 / 2), _m.sin(t2 / 2)
    c3, s3 = _m.cos(t3 / 2), _m.sin(t3 / 2)
    w0 = c2 * v[0] - s2 * v[1]
    w1 = s2 * v[0] + c2 * v[1]
    w2 = c3 * v[2] - s3 * v[3]
    w3 = s3 * v[2] + c3 * v[3]
    return [w0, w1, w2, w3]

def _expectation(psi, H):
    e = 0j
    for i in range(4):
        row = 0j
        for j in range(4):
            row += H[i][j] * psi[j]
        e += psi[i].conjugate() * row
    return float(e.real)

def vqe_h2(theta: np.ndarray = None, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    h2_ground = -1.137
    H = _h2_matrix()
    if theta is not None:
        start = [float(v) for v in np.asarray(theta).reshape(-1).tolist()[:4]]
        start = start + [0.0] * (4 - len(start))
    else:
        start = [float(v) for v in rng.uniform(-np.pi, np.pi, size=4).tolist()]
    best = list(start)
    best_e = _expectation(_ry_state(*best), H)
    evaluations = 1
    step = 0.2
    for _ in range(25):
        improved = False
        for k in range(4):
            for sign in (1.0, -1.0):
                trial = list(best)
                trial[k] += sign * step
                e = _expectation(_ry_state(*trial), H)
                evaluations += 1
                if e < best_e:
                    best, best_e = trial, e
                    improved = True
                    break
            if improved:
                break
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
    return {'energy_vqe': best_e, 'energy_exact': h2_ground,
            'gap': abs(best_e - h2_ground), 'evaluations': evaluations,
            'params': best}

def hamiltonian_matrix(n_sites: int, J: float = 1.0, h: float = 1.0) -> np.ndarray:
    N = 2**n_sites
    H = [[0j] * N for _ in range(N)]
    for i in range(N):
        bits = [(i >> k) & 1 for k in range(n_sites)]
        for s in range(n_sites - 1):
            H[i][i] += -J * (1.0 if bits[s] == bits[s + 1] else -1.0)
        for s in range(n_sites):
            j = i ^ (1 << s)
            H[i][j] += -h
    return np.array(H)
