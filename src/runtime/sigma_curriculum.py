"""Phase B1: Fractional-exponent sigma curriculum and pattern observables.

Treats the fractional exponent sigma in alpha*(-Delta)^(sigma/2) (P1)
as a curriculum variable. Begin chaotic (small sigma, broad dispersion)
and let sigma self-adjust via the A1 curvature-aware calibrator.

Adds Turing/Gierer-Meinhardt pattern observables:
- Turing wavelength: dominant spatial period of pattern
- Pattern entropy: Shannon entropy of normalized power spectrum

All three pillars active throughout. Sigma change is internal to P1.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional

import numpy as np

from runtime.core.solver import integrate, TriadParams
from runtime.physics.observables import crystallinity, dominant_wavenumber

def turing_wavelength(psi: np.ndarray, dx: float) -> float:
    """Compute the Turing wavelength (dominant spatial period) of the field.

    This is 2*pi / k_star where k_star is the dominant wavenumber.
    """
    rho = np.abs(psi) ** 2
    rho_hat = np.fft.rfft(rho)
    power = np.abs(rho_hat) ** 2
    N = len(rho)
    L = N * dx
    k_vals = 2.0 * np.pi * np.arange(len(power)) / L

    if len(power) < 2:
        return 0.0
    k_star = k_vals[1 + np.argmax(power[1:])]
    if k_star < 1e-10:
        return float("inf")
    return 2.0 * np.pi / k_star

def pattern_entropy(psi: np.ndarray, dx: float) -> float:
    """Shannon entropy of the normalized power spectrum.

    High entropy = flat spectrum (chaotic).
    Low entropy = concentrated spectrum (crystallized pattern).
    """
    rho = np.abs(psi) ** 2
    rho_hat = np.fft.rfft(rho)
    power = np.abs(rho_hat) ** 2

    total = power.sum()
    if total < 1e-20:
        return 0.0
    prob = power / total
    
    prob = prob[prob > 1e-20]
    return float(-np.sum(prob * np.log(prob)))

def pattern_metrics(psi: np.ndarray, dx: float) -> Dict[str, float]:
    """Compute all pattern observables."""
    return {
        "turing_wavelength": turing_wavelength(psi, dx),
        "pattern_entropy": pattern_entropy(psi, dx),
    }

@dataclass
class SigmaSchedule:
    """Curriculum schedule for the fractional exponent sigma.

    Parameters
    ----------
    sigma_start : float
        Initial sigma (small = broad dispersion, chaotic).
    sigma_end : float
        Target sigma (larger = sharper, more structured).
    mode : str
        'linear': linearly interpolate sigma each step.
        'emergent': adjust sigma based on crystallinity feedback.
    n_steps : int
        Number of curriculum steps.
    crystallinity_target : float
        For 'emergent' mode: target crystallinity before increasing sigma.
    """
    sigma_start: float = 0.5
    sigma_end: float = 2.0
    mode: str = "linear"
    n_steps: int = 10
    crystallinity_target: float = 0.5

    def sigma_at_step(self, step: int) -> float:
        """Get sigma for a given curriculum step."""
        if self.mode == "linear":
            frac = step / max(self.n_steps - 1, 1)
            return self.sigma_start + frac * (self.sigma_end - self.sigma_start)
        else:
            
            return self.sigma_start

@dataclass
class CurriculumResult:
    """Result of a curriculum run."""
    converged: bool
    n_steps: int
    history: List[Dict]
    final_params: TriadParams
    final_observables: Dict[str, float]

def run_curriculum(params: TriadParams,
                   schedule: SigmaSchedule = None,
                   T_per_step: float = 3.0,
                   verbose: bool = False) -> CurriculumResult:
    """Run sigma-curriculum integration.

    Starts with sigma_start (chaotic), gradually increases to sigma_end
    (structured). Each step integrates the PDE and measures observables.

    All three pillars active at every step.
    """
    if schedule is None:
        schedule = SigmaSchedule()

    history = []
    p = params
    converged = False

    for step in range(schedule.n_steps):
        sigma = schedule.sigma_at_step(step)
        p = replace(p, sigma=sigma, T=T_per_step, seed=(params.seed or 0) + step)

        r = integrate(p)
        dx = r["dx"]
        psi = r["psi_final"]

        obs = {
            "crystallinity": float(crystallinity(psi, dx)),
            "dominant_k": float(dominant_wavenumber(psi, dx)),
            "sigma": sigma,
        }
        obs.update(pattern_metrics(psi, dx))

        history.append({
            "step": step,
            "sigma": sigma,
            "observables": obs,
        })

        if verbose:
            print(f"  step {step}: sigma={sigma:.2f}, C={obs['crystallinity']:.4f}, "
                  f"entropy={obs['pattern_entropy']:.2f}")

        if schedule.mode == "emergent":
            if obs["crystallinity"] >= schedule.crystallinity_target:
                
                schedule.sigma_start = sigma
                if sigma >= schedule.sigma_end:
                    converged = True
                    break
        else:
            if step == schedule.n_steps - 1:
                converged = obs["crystallinity"] > 0.3

    final_obs = history[-1]["observables"] if history else {}
    return CurriculumResult(
        converged=converged,
        n_steps=len(history),
        history=history,
        final_params=p,
        final_observables=final_obs,
    )
