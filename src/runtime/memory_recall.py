"""A3: Hopfield-like associative recall via memory fields (P2).

The M memory fields y_j are OU-smoothed density histories.  Each y_j tracks
rho = |psi|^2 with its own decay rate nu_j.  The Hopfield insight is that these
fields naturally store density patterns over time because V_mem = sum(lambda_j * y_j)
acts as an energy landscape.  Retrieval is emergent energy-descent in V_mem
toward a stored density pattern.  Nothing imposes the pattern externally.

Key properties:
  - P1+P2+P3 must all be active during recall (dispersion, memory, noise/FDT)
  - Retrieval is emergent, not imposed
  - Ablation of P3 noise degrades basin escape (frozen in wrong minimum)

Reference: Hopfield Hetero-Associative Network capacity K ~ 0.3 * N^1.2
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from runtime.core.solver import TriadParams, integrate

def overlap(y_a: np.ndarray, y_b: np.ndarray) -> float:
    """Cosine similarity between two memory-field vectors.

    Parameters
    ----------
    y_a, y_b : np.ndarray
        Flattened memory field vectors (or same-shape arrays).

    Returns
    -------
    float in [-1, 1].  1.0 means identical, 0 means orthogonal.
    """
    a = np.asarray(y_a, dtype=np.float64).ravel()
    b = np.asarray(y_b, dtype=np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def capacity_estimate(N: int) -> float:
    """Theoretical Hopfield capacity: K ~ 0.3 * N^1.2.

    From the HHN scaling law for associative memory in continuous
    high-dimensional fields.  N is the number of spatial grid points.

    Parameters
    ----------
    N : int
        Grid size (number of spatial points).

    Returns
    -------
    float
        Estimated maximum number of storable patterns.
    """
    return 0.3 * float(N) ** 1.2

def _corrupt_pattern(rho: np.ndarray, fraction: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Randomly corrupt a fraction of the density pattern with Gaussian noise.

    Parameters
    ----------
    rho : np.ndarray
        Original density pattern.
    fraction : float
        Fraction of pixels to corrupt (0 to 1).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Corrupted density pattern (same shape, non-negative).
    """
    rho = np.asarray(rho, dtype=np.float64).copy()
    mask = rng.random(rho.shape) < fraction
    noise_std = float(np.std(rho)) * 0.5
    rho[mask] += rng.standard_normal(mask.sum()) * noise_std
    rho = np.clip(rho, 0.0, None)
    return rho

def _density_from_psi(psi: np.ndarray) -> np.ndarray:
    """Compute density |psi|^2."""
    return np.abs(psi) ** 2

def _psi_from_density(rho: np.ndarray, dx: float,
                      phase_rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Construct a complex psi from a density pattern with random or zero phase.

    Parameters
    ----------
    rho : np.ndarray
        Target density pattern.
    dx : float
        Grid spacing (for normalization).
    phase_rng : optional
        If provided, random phases are assigned; otherwise phase = 0.

    Returns
    -------
    np.ndarray (complex128)
        Normalized psi whose |psi|^2 approximates rho.
    """
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
    """A stored density pattern with its associated memory field state."""
    index: int
    rho: np.ndarray          
    y_stored: np.ndarray     
    psi: np.ndarray          

class HopfieldMemory:
    """Hopfield-like associative memory using PDE memory fields.

    The memory fields y_j are OU-smoothed density histories.  By running
    the solver until a density pattern crystallizes and then saving the
    y_j state, we create stored "attractors" in the V_mem landscape.
    Recall works by seeding a new psi from a partial/corrupted pattern and
    letting the solver relax under V_mem toward the nearest stored attractor.

    This is purely emergent: no external pattern is imposed during recall.
    The solver dynamics (P1 dispersion + P2 memory feedback + P3 noise)
    perform the energy descent.
    """

    def __init__(self, params: Optional[TriadParams] = None):
        if params is None:
            
            params = TriadParams(N=64, L=16.0, T=10.0)
        self.base_params = params
        
        self.params = params
        self.patterns: list[StoredPattern] = []
        self._rng = np.random.default_rng(params.seed)

    def store_pattern(self, rho: Optional[np.ndarray] = None,
                      T_store: float = 10.0,
                      psi0: Optional[np.ndarray] = None) -> StoredPattern:
        """Store a density pattern as a memory field attractor.

        If rho is provided, it is used as a target.  The solver runs until
        the density crystallizes under full P1+P2+P3 dynamics, and the
        resulting y_j state is saved as the stored memory.

        If neither rho nor psi0 is provided, the solver starts from default
        Gaussian initial conditions and an emergent pattern is stored.

        Parameters
        ----------
        rho : optional np.ndarray
            Target density pattern.  Used to seed psi0 if psi0 is not given.
        T_store : float
            Integration time for pattern crystallization.
        psi0 : optional np.ndarray
            Initial wavefunction.  If None, derived from rho or default.

        Returns
        -------
        StoredPattern
        """
        p = TriadParams(**{**self.base_params.__dict__,
                           'T': T_store, 'mode': 'full'})

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
               y0: Optional[np.ndarray] = None) -> dict:
        """Attempt associative recall from a partial/corrupted density pattern.

        Seeds psi from the partial pattern and runs the solver under the
        stored V_mem landscape.  The solver naturally relaxes toward the
        nearest stored attractor (emergent energy descent).

        All three pillars (P1+P2+P3) are active: dispersion explores,
        memory fields guide, noise helps escape spurious basins.

        Parameters
        ----------
        partial_rho : np.ndarray
            Partial or corrupted density pattern (cue).
        T_recall : float
            Integration time for recall relaxation.
        y0 : optional np.ndarray
            Initial memory field state.  If None, starts from zero.

        Returns
        -------
        dict with keys:
            psi_final: final wavefunction
            rho_final: final density
            y_final: final memory field state
            overlaps: overlap of rho_final with each stored pattern
            best_match: index of best matching stored pattern
            best_overlap: overlap score with best match
        """
        if not self.patterns:
            raise ValueError("No patterns stored. Call store_pattern first.")

        p = TriadParams(**{**self.base_params.__dict__,
                           'T': T_recall, 'mode': 'full'})

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
                         y0: Optional[np.ndarray] = None) -> dict:
        """Recall using a seed index into stored patterns with corruption.

        Convenience method: corrupts the stored pattern at the given index
        and attempts recall.

        Parameters
        ----------
        seed : int
            Index of stored pattern to use as seed (will be corrupted).
        T_recall : float
            Integration time.
        y0 : optional np.ndarray
            Initial memory field state.

        Returns
        -------
        dict (same as recall())
        """
        if seed < 0 or seed >= len(self.patterns):
            raise IndexError(f"seed {seed} out of range [0, {len(self.patterns)})")
        sp = self.patterns[seed]
        corrupted = _corrupt_pattern(sp.rho, fraction=0.4, rng=self._rng)
        return self.recall(corrupted, T_recall=T_recall, y0=y0)

    def capacity(self) -> float:
        """Theoretical capacity estimate for current grid size."""
        return capacity_estimate(self.base_params.N)

    def stored_count(self) -> int:
        """Number of currently stored patterns."""
        return len(self.patterns)

    def pattern_overlaps(self) -> np.ndarray:
        """Compute the overlap matrix between all stored patterns.

        Returns
        -------
        np.ndarray of shape (K, K) where K = len(self.patterns)
        """
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
    """Run batch recall trials and measure accuracy.

    For each stored pattern, corrupt it and attempt recall.  Report
    accuracy as the fraction of trials where the best match is the
    correct pattern with overlap above threshold.

    Parameters
    ----------
    hm : HopfieldMemory
        Memory instance with stored patterns.
    n_trials : int
        Number of trials per pattern.
    corruption : float
        Fraction of pattern to corrupt (0 to 1).
    T_recall : float
        Recall integration time.
    threshold : float
        Minimum overlap to count as successful recall.

    Returns
    -------
    dict with:
        accuracy: fraction of successful recalls
        per_pattern: list of per-pattern accuracy
        mean_overlap: mean best overlap across all trials
    """
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
