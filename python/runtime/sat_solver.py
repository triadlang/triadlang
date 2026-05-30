from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class SATInstance:
    n_vars: int
    clauses: list[list[int]]

    def evaluate(self, assignment: list[int]) -> int:
        violated = 0
        for clause in self.clauses:
            satisfied = False
            for lit in clause:
                var = abs(lit) - 1
                val = assignment[var]
                if lit > 0 and val == 1 or (lit < 0 and val == 0):
                    satisfied = True
                    break
            if not satisfied:
                violated += 1
        return violated

    def all_solutions(self) -> list[list[int]]:
        sols = []
        for j in range(2 ** self.n_vars):
            assignment = [j >> self.n_vars - 1 - k & 1 for k in range(self.n_vars)]
            if self.evaluate(assignment) == 0:
                sols.append(assignment)
        return sols

    @staticmethod
    def random(n_vars: int, n_clauses: int, k: int=3, seed: int=0) -> SATInstance:
        rng = np.random.default_rng(seed)
        clauses = []
        for _ in range(n_clauses):
            vars_ = rng.choice(n_vars, size=min(k, n_vars), replace=False) + 1
            clause = [int(v) * (1 if rng.random() > 0.5 else -1) for v in vars_]
            clauses.append(clause)
        return SATInstance(n_vars, clauses)

def _clause_wave(x: np.ndarray, clause: list[int], k_var: np.ndarray) -> np.ndarray:
    wave = np.zeros_like(x, dtype=np.complex128)
    for lit in clause:
        var = abs(lit) - 1
        k = k_var[var]
        if lit > 0:
            wave += np.exp(1j * k * x)
        else:
            wave += np.exp(-1j * k * x)
    return wave

def encode_sat_interference(sat: SATInstance, N_grid: Optional[int]=None, L: Optional[float]=None):
    n = sat.n_vars
    m = len(sat.clauses)
    if N_grid is None:
        N_grid = max(128, n * 16)
    if L is None:
        L = float(N_grid) * 0.5
    x = np.linspace(-L / 2, L / 2, N_grid, endpoint=False)
    dx = x[1] - x[0]
    k_var = 2 * np.pi * np.arange(1, n + 1) / L
    psi = np.zeros(N_grid, dtype=np.complex128)
    for clause in sat.clauses:
        psi += _clause_wave(x, clause, k_var)
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    if norm > 1e-15:
        psi /= norm
    V_ext = 0.01 * (x / (L / 2)) ** 2

    def V_ext_fn(x_arr):
        return V_ext
    return (V_ext_fn, psi, N_grid, L, None)

def encode_sat_spectral(sat: SATInstance, N_grid: Optional[int]=None, L: Optional[float]=None):
    n = sat.n_vars
    m = len(sat.clauses)
    if N_grid is None:
        N_grid = max(128, n * 16)
    if L is None:
        L = float(N_grid) * 0.5
    x = np.linspace(-L / 2, L / 2, N_grid, endpoint=False)
    dx = x[1] - x[0]
    k_var = 2 * np.pi * np.arange(1, n + 1) / L
    psi_hat = np.zeros(N_grid, dtype=np.complex128)
    k_fft = 2 * np.pi * np.fft.fftfreq(N_grid, d=dx)
    for clause in sat.clauses:
        for satisfying_combo in _clause_satisfying_phases(clause, n):
            mode_weight = np.zeros(N_grid, dtype=np.complex128)
            for var_idx, phase in enumerate(satisfying_combo):
                k_target = k_var[var_idx]
                idx = np.argmin(np.abs(k_fft - k_target))
                mode_weight[idx] += np.exp(1j * phase)
            psi_hat += mode_weight * np.conj(mode_weight)
    psi = np.fft.ifft(psi_hat)
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    if norm > 1e-15:
        psi /= norm
    V_ext = 0.01 * (x / (L / 2)) ** 2

    def V_ext_fn(x_arr):
        return V_ext
    return (V_ext_fn, psi, N_grid, L, None)

def _clause_satisfying_phases(clause: list[int], n_vars: int):
    vars_in_clause = [abs(lit) - 1 for lit in clause]
    signs = [1 if lit > 0 else -1 for lit in clause]
    for mask in range(1, 1 << len(clause)):
        phases = [0.0] * n_vars
        for bit_idx in range(len(clause)):
            var = vars_in_clause[bit_idx]
            if mask & 1 << bit_idx:
                phases[var] = 0.0 if signs[bit_idx] > 0 else np.pi
            else:
                phases[var] = np.pi if signs[bit_idx] > 0 else 0.0
        yield phases

def encode_sat_resonance(sat: SATInstance, N_grid: Optional[int]=None, L: Optional[float]=None):
    n = sat.n_vars
    m = len(sat.clauses)
    if N_grid is None:
        N_grid = max(128, n * 16)
    if L is None:
        L = float(N_grid) * 0.5
    x = np.linspace(-L / 2, L / 2, N_grid, endpoint=False)
    dx = x[1] - x[0]
    k_var = 2 * np.pi * np.arange(1, n + 1) / L
    psi = np.zeros(N_grid, dtype=np.complex128)
    for ci, clause in enumerate(sat.clauses):
        clause_wave = np.ones(N_grid, dtype=np.complex128)
        for lit in clause:
            var = abs(lit) - 1
            k = k_var[var]
            if lit > 0:
                clause_wave *= (1 + np.cos(k * x)) / 2.0
            else:
                clause_wave *= (1 + np.cos(k * x + np.pi)) / 2.0
        phase_offset = 2 * np.pi * ci / m
        psi += clause_wave * np.exp(1j * phase_offset)
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    if norm > 1e-15:
        psi /= norm
    V_ext = 0.01 * (x / (L / 2)) ** 2

    def V_ext_fn(x_arr):
        return V_ext
    return (V_ext_fn, psi, N_grid, L, None)

def decode_interference(psi_final: np.ndarray, sat: SATInstance, L: float) -> list[int]:
    n = sat.n_vars
    N_grid = len(psi_final)
    dx = L / N_grid
    k_var = 2 * np.pi * np.arange(1, n + 1) / L
    psi_hat = np.fft.fft(psi_final)
    k_fft = 2 * np.pi * np.fft.fftfreq(N_grid, d=dx)
    assignment = []
    for i, k_target in enumerate(k_var):
        idx = np.argmin(np.abs(k_fft - k_target))
        phase = np.angle(psi_hat[idx])
        bit = 0 if abs(phase) > np.pi / 2 else 1
        assignment.append(bit)
    complement = [1 - b for b in assignment]
    if sat.evaluate(complement) < sat.evaluate(assignment):
        return complement
    return assignment

def decode_multi_peak(psi_final: np.ndarray, sat: SATInstance, L: float, n_candidates: int=8) -> list[int]:
    n = sat.n_vars
    N_grid = len(psi_final)
    dx = L / N_grid
    k_var = 2 * np.pi * np.arange(1, n + 1) / L
    psi_hat = np.fft.fft(psi_final)
    k_fft = 2 * np.pi * np.fft.fftfreq(N_grid, d=dx)
    mode_phases = []
    mode_amps = []
    for i, k_target in enumerate(k_var):
        idx = np.argmin(np.abs(k_fft - k_target))
        mode_phases.append(np.angle(psi_hat[idx]))
        mode_amps.append(np.abs(psi_hat[idx]))
    best_assignment = [0] * n
    best_violations = sat.evaluate(best_assignment)
    for threshold_offset in np.linspace(-np.pi / 2, np.pi / 2, n_candidates):
        assignment = []
        for i in range(n):
            phase = mode_phases[i]
            threshold = np.pi / 2 + threshold_offset
            bit = 0 if abs(phase) > threshold else 1
            assignment.append(bit)
        v = sat.evaluate(assignment)
        if v < best_violations:
            best_violations = v
            best_assignment = assignment
        complement = [1 - b for b in assignment]
        vc = sat.evaluate(complement)
        if vc < best_violations:
            best_violations = vc
            best_assignment = complement
    return best_assignment

def decode_spatial(psi_final: np.ndarray, sat: SATInstance, L: float, max_flips: int=2, n_peaks: int=64) -> list[int]:
    n = sat.n_vars
    N_grid = len(psi_final)
    x = np.linspace(-L / 2, L / 2, N_grid, endpoint=False)
    k_var = 2 * np.pi * np.arange(1, n + 1) / L
    rho = np.abs(psi_final) ** 2
    try:
        from scipy.signal import find_peaks as _find_peaks
        peaks, _ = _find_peaks(rho, height=rho.max() * 0.05)
    except ImportError:
        peaks = np.argsort(-rho)[:n_peaks]
    if len(peaks) == 0:
        peaks = np.array([np.argmax(rho)])
    heights = rho[peaks]
    order = np.argsort(-heights)
    peaks = peaks[order[:min(n_peaks, len(peaks))]]
    best_assignment = [0] * n
    best_v = sat.evaluate(best_assignment)
    for pk in peaks:
        x_peak = x[pk]
        base = [1 if np.cos(k_var[i] * x_peak) > 0 else 0 for i in range(n)]
        v = sat.evaluate(base)
        if v < best_v:
            best_v = v
            best_assignment = base
        if v == 0:
            return best_assignment
        from itertools import combinations
        for nf in range(1, max_flips + 1):
            for flips in combinations(range(n), nf):
                a = base.copy()
                for f in flips:
                    a[f] = 1 - a[f]
                vf = sat.evaluate(a)
                if vf < best_v:
                    best_v = vf
                    best_assignment = a
                if vf == 0:
                    return best_assignment
            if best_v == 0:
                break
        if best_v == 0:
            break
    return best_assignment

def _build_penalty_potential(sat: SATInstance, N_grid: int, center: int) -> np.ndarray:
    V = np.zeros(N_grid)
    n_assign = 2 ** sat.n_vars
    scale = 2.0
    for j in range(n_assign):
        assignment = [j >> sat.n_vars - 1 - k & 1 for k in range(sat.n_vars)]
        violations = sat.evaluate(assignment)
        idx = center + j
        if 0 <= idx < N_grid:
            V[idx] = scale * violations
    return V

def encode_sat_direct(sat: SATInstance, N_grid: Optional[int]=None, L: Optional[float]=None):
    n_assign = 2 ** sat.n_vars
    if N_grid is None:
        N_grid = max(128, n_assign * 4)
    if L is None:
        L = float(N_grid) * 0.25
    center = N_grid // 2 - n_assign // 2
    V_ext = _build_penalty_potential(sat, N_grid, center)
    psi = np.zeros(N_grid, dtype=np.complex128)
    amp = 1.0 / np.sqrt(n_assign)
    for j in range(n_assign):
        idx = center + j
        if 0 <= idx < N_grid:
            psi[idx] = amp

    def V_ext_fn(x):
        return V_ext
    return (V_ext_fn, psi, N_grid, L, center)

def solve_sat(sat: SATInstance, encoding: str='interference', T: float=10.0, dt: float=0.005, Lambda: float=-3.0, Gamma: float=0.02, f_FDT: float=0.001, alpha: float=0.1, sigma: float=1.5, seed: int=42, decode: str='spatial') -> dict:
    from runtime.solver import TriadParams, integrate
    if encoding == 'interference':
        V_ext_fn, psi0, N, L, center = encode_sat_interference(sat)
    elif encoding == 'resonance':
        V_ext_fn, psi0, N, L, center = encode_sat_resonance(sat)
    elif encoding == 'spectral':
        V_ext_fn, psi0, N, L, center = encode_sat_spectral(sat)
    elif encoding == 'direct':
        V_ext_fn, psi0, N, L, center = encode_sat_direct(sat)
    elif encoding == 'compact':
        V_ext_fn, psi0, N, L, center = encode_sat_resonance(sat)
    else:
        raise ValueError(f'Unknown encoding: {encoding}')
    p = TriadParams(N=N, L=L, T=T, dt=dt, Lambda=Lambda, Gamma=Gamma, f_FDT=f_FDT, alpha=alpha, sigma=sigma, V_ext=V_ext_fn, nu=(2.0, 0.5, 0.1), lam=(-0.5, -0.3, -0.1), seed=seed)
    out = integrate(p, psi0=psi0, auto_halve_dt=False)
    psi_final = out['psi_final']
    rho = np.abs(psi_final) ** 2
    if encoding == 'direct' and center is not None:
        n_assign = 2 ** sat.n_vars
        segment = rho[center:center + n_assign]
        if len(segment) > 0:
            best_j = int(np.argmax(segment))
            best_assignment = [best_j >> sat.n_vars - 1 - k & 1 for k in range(sat.n_vars)]
            peak = float(segment[best_j])
        else:
            best_assignment = [0] * sat.n_vars
            peak = 0.0
    else:
        if decode == 'spatial':
            best_assignment = decode_spatial(psi_final, sat, L)
        elif decode == 'multi_peak':
            best_assignment = decode_multi_peak(psi_final, sat, L)
        else:
            best_assignment = decode_interference(psi_final, sat, L)
        peak = float(rho.max())
    violations = sat.evaluate(best_assignment)
    solutions = sat.all_solutions()
    is_correct = violations == 0
    return {'assignment': best_assignment, 'violations': violations, 'is_satisfiable': len(solutions) > 0, 'is_correct': is_correct, 'peak_density': peak, 'n_solutions': len(solutions), 'N_grid': N, 'encoding': encoding, 'T': T, 'psi_final': psi_final, 'density': rho}

def solve_sat_iterative(sat: SATInstance, max_rounds: int=4, seed: int=42) -> dict:
    from runtime.solver import TriadParams, integrate
    frozen = {}
    n = sat.n_vars
    for rnd in range(max_rounds):
        unfrozen = [i for i in range(n) if i not in frozen]
        if not unfrozen:
            break
        n_rem = len(unfrozen)
        rem_clauses = []
        for clause in sat.clauses:
            new_clause = []
            skip = False
            for lit in clause:
                var = abs(lit) - 1
                if var in frozen:
                    val = frozen[var]
                    sat_lit = lit > 0 and val == 1 or (lit < 0 and val == 0)
                    if sat_lit:
                        skip = True
                        break
                else:
                    new_idx = unfrozen.index(var)
                    sign = 1 if lit > 0 else -1
                    new_clause.append(sign * (new_idx + 1))
            if not skip and new_clause:
                rem_clauses.append(new_clause)
        if not rem_clauses:
            break
        rem_sat = SATInstance(n_vars=n_rem, clauses=rem_clauses)
        best_v = 999
        best_a = [0] * n_rem
        for s in [seed + rnd * 10, seed + rnd * 10 + 7, seed + rnd * 10 + 13]:
            V_ext_fn, psi0, Ng, L, _ = encode_sat_interference(rem_sat)
            for T in [20, 40]:
                for Lam in [-5, -10, -15]:
                    p = TriadParams(N=Ng, L=L, T=T, dt=0.005, Lambda=Lam, Gamma=0.02, f_FDT=0.001, alpha=0.1, sigma=1.5, V_ext=V_ext_fn, nu=(2.0, 0.5, 0.1), lam=(-0.5, -0.3, -0.1), seed=s)
                    out = integrate(p, psi0=psi0, auto_halve_dt=False)
                    a, v = _decode_with_confidence(out['psi_final'], rem_sat, L)
                    if v < best_v:
                        best_v = v
                        best_a = a
                    if v == 0:
                        break
                if best_v == 0:
                    break
            if best_v == 0:
                break
        for i, var in enumerate(unfrozen):
            frozen[var] = best_a[i]
        if best_v == 0:
            break
    assignment = [frozen.get(i, 0) for i in range(n)]
    violations = sat.evaluate(assignment)
    solutions = sat.all_solutions()
    return {'assignment': assignment, 'violations': violations, 'is_satisfiable': len(solutions) > 0, 'is_correct': violations == 0, 'n_solutions': len(solutions), 'encoding': 'iterative', 'rounds_used': rnd + 1}

def _decode_with_confidence(psi_final, sat, L, n_peaks=32, max_flips=2):
    from scipy.signal import find_peaks as _find_peaks
    n = sat.n_vars
    N_grid = len(psi_final)
    x = np.linspace(-L / 2, L / 2, N_grid, endpoint=False)
    k_var = 2 * np.pi * np.arange(1, n + 1) / L
    rho = np.abs(psi_final) ** 2
    try:
        peaks, _ = _find_peaks(rho, height=rho.max() * 0.05)
    except Exception:
        peaks = np.argsort(-rho)[:n_peaks]
    if len(peaks) == 0:
        peaks = np.array([np.argmax(rho)])
    heights = rho[peaks]
    peaks = peaks[np.argsort(-heights)[:min(n_peaks, len(peaks))]]
    best_v, best_a = (999, [0] * n)
    for pk in peaks:
        xp = x[pk]
        base = [1 if np.cos(k_var[i] * xp) > 0 else 0 for i in range(n)]
        v = sat.evaluate(base)
        if v < best_v:
            best_v, best_a = (v, base)
        if v == 0:
            return (best_a, 0)
        from itertools import combinations
        for nf in range(1, max_flips + 1):
            for flips in combinations(range(n), nf):
                a = base.copy()
                for f in flips:
                    a[f] = 1 - a[f]
                vf = sat.evaluate(a)
                if vf < best_v:
                    best_v, best_a = (vf, a)
                if vf == 0:
                    return (best_a, 0)
    return (best_a, best_v)

def solve_sat_batch(sat: SATInstance, encoding: str='interference', seed: int=42) -> list[dict]:
    results = []
    for T in [5.0, 10.0, 20.0, 40.0]:
        for Lam in [-1.0, -3.0, -5.0, -8.0]:
            for alpha in [0.0, 0.1, 0.2]:
                r = solve_sat(sat, encoding=encoding, T=T, Lambda=Lam, alpha=alpha, seed=seed)
                r['T'] = T
                r['Lambda'] = Lam
                r['alpha'] = alpha
                results.append(r)
    results.sort(key=lambda r: (r['violations'], -r['peak_density']))
    return results