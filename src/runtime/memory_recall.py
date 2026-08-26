
from __future__ import annotations

from dataclasses import dataclass

from runtime.core.solver import TriadParams, integrate
from triad import ntri as np


def overlap(y_a: np.ndarray, y_b: np.ndarray) -> float:

    a = np.asarray(y_a, dtype=np.float64).ravel()
    b = np.asarray(y_b, dtype=np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def capacity_estimate(N: int) -> float:

    return 0.3 * float(N) ** 1.2

def _corrupt_pattern(rho: np.ndarray, fraction: float,
                     rng: np.random.Generator) -> np.ndarray:

    rho = np.asarray(rho, dtype=np.float64).copy()
    mask = rng.random(rho.shape) < fraction
    noise_std = float(np.std(rho)) * 0.5
    rho[mask] += rng.standard_normal(mask.sum()) * noise_std
    rho = np.clip(rho, 0.0, None)
    return rho

def _density_from_psi(psi: np.ndarray) -> np.ndarray:

    return np.abs(psi) ** 2

def _psi_from_density(rho: np.ndarray, dx: float,
                      phase_rng: np.random.Generator | None = None) -> np.ndarray:

    rho = np.asarray(rho, dtype=np.float64)
    rho = np.clip(rho, 0.0, None)
    if phase_rng is not None:
        phase = phase_rng.uniform(0, 2 * np.pi, size=rho.shape)
    else:
        phase = np.zeros_like(rho)
    psi = np.sqrt(rho) * np.exp(1j * phase)
    norm = np.sqrt((np.abs(psi) ** 2).sum() * dx)
    if norm > 1e-30:
        psi /= norm
    return psi

@dataclass
class StoredPattern:

    index: int
    rho: np.ndarray
    y_stored: np.ndarray
    psi: np.ndarray

class HopfieldMemory:

    def __init__(self, params: TriadParams | None = None):
        if params is None:

            params = TriadParams(N=64, L=16.0, T=10.0, D=1)
        self.base_params = params

        self.params = params
        self.patterns: list[StoredPattern] = []
        self._rng = np.random.default_rng(params.seed)

    def store_pattern(self, rho: np.ndarray | None = None,
                      T_store: float = 10.0,
                      psi0: np.ndarray | None = None) -> StoredPattern:

        p = TriadParams(**{**self.base_params.__dict__,
                           'T': T_store, 'mode': 'triad'})

        if psi0 is None and rho is not None:
            dx_guess = p.L / p.N
            psi0 = _psi_from_density(rho, dx_guess, phase_rng=self._rng)

        result = integrate(p, psi0=psi0, record_y=True)

        rho_final = _density_from_psi(result['psi_final'])
        y_final = result['y_final']

        sp = StoredPattern(
            index=len(self.patterns),
            rho=rho_final,
            y_stored=y_final.copy(),
            psi=result['psi_final'].copy(),
        )
        self.patterns.append(sp)
        return sp

    def recall(self, partial_rho: np.ndarray,
               T_recall: float = 10.0,
               y0: np.ndarray | None = None) -> dict:

        if not self.patterns:
            raise ValueError("No patterns stored. Call store_pattern first.")

        p = TriadParams(**{**self.base_params.__dict__,
                           'T': T_recall, 'mode': 'triad'})

        dx = p.L / p.N
        psi0 = _psi_from_density(partial_rho, dx, phase_rng=self._rng)

        result = integrate(p, psi0=psi0, y0=y0, record_y=True)

        rho_final = _density_from_psi(result['psi_final'])

        overlaps = []
        for sp in self.patterns:

            ov = overlap(rho_final, sp.rho)
            overlaps.append(ov)

        best_idx = int(np.argmax(overlaps))
        return {
            'psi_final': result['psi_final'],
            'rho_final': rho_final,
            'y_final': result['y_final'],
            'overlaps': overlaps,
            'best_match': best_idx,
            'best_overlap': overlaps[best_idx],
            'T_recall': T_recall,
        }

    def recall_with_seed(self, seed: int,
                         T_recall: float = 10.0,
                         y0: np.ndarray | None = None) -> dict:

        if seed < 0 or seed >= len(self.patterns):
            raise IndexError(f"seed {seed} out of range [0, {len(self.patterns)})")
        sp = self.patterns[seed]
        corrupted = _corrupt_pattern(sp.rho, fraction=0.4, rng=self._rng)
        return self.recall(corrupted, T_recall=T_recall, y0=y0)

    def capacity(self) -> float:

        return capacity_estimate(self.base_params.N)

    def stored_count(self) -> int:

        return len(self.patterns)

    def pattern_overlaps(self) -> np.ndarray:

        K = len(self.patterns)
        mat = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                mat[i, j] = overlap(self.patterns[i].rho, self.patterns[j].rho)
        return mat

def batch_recall_accuracy(hm: HopfieldMemory,
                          n_trials: int = 10,
                          corruption: float = 0.4,
                          T_recall: float = 10.0,
                          threshold: float = 0.5) -> dict:

    K = hm.stored_count()
    if K == 0:
        return {'accuracy': 0.0, 'per_pattern': [], 'mean_overlap': 0.0}

    rng = np.random.default_rng(hm.base_params.seed + 999)
    per_pattern = []
    all_overlaps = []

    for pat_idx in range(K):
        sp = hm.patterns[pat_idx]
        successes = 0
        for trial in range(n_trials):
            corrupted = _corrupt_pattern(sp.rho, fraction=corruption, rng=rng)
            result = hm.recall(corrupted, T_recall=T_recall)
            if result['best_match'] == pat_idx and result['best_overlap'] >= threshold:
                successes += 1
            all_overlaps.append(result['best_overlap'])
        per_pattern.append(successes / n_trials)

    total_trials = K * n_trials
    accuracy = sum(p * n_trials for p in per_pattern) / total_trials if total_trials > 0 else 0.0
    mean_overlap = float(np.mean(all_overlaps)) if all_overlaps else 0.0

    return {
        'accuracy': accuracy,
        'per_pattern': per_pattern,
        'mean_overlap': mean_overlap,
    }
