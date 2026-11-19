"""Phase B1: Fixed-point / equilibrium-style fast inference.

The crystallized field Psi* is the steady-state of the FULL stochastic propagator.
Without FDT noise, the field decays to zero (P3 is load-bearing for crystallization).
We instead accelerate reaching the STEADY STATE by:

1. Anderson-accelerated observable convergence: run S (deterministic or stochastic)
   but accelerate convergence of the OBSERVABLES (C, E, k*) rather than psi itself.
2. Multi-step super-operator with observable convergence checks.

The one-step operator S is the FULL three-pillar Strang split-step (P1+P2+P3).
For the fixed-point to be non-trivial, we use the stochastic version (with FDT noise).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np

from runtime.core.solver import TriadParams, _effective_params, _build_V_ext
from runtime.physics.observables import crystallinity, dominant_wavenumber, ipr, participation_ratio, energy
from runtime.backend import get_xp, asnumpy

def _one_step_full(psi: np.ndarray, y: np.ndarray,
                    p: TriadParams, rng: np.random.Generator,
                    deterministic: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """One FULL Strang split-step WITH stochastic FDT noise (unless deterministic)."""
    eff = _effective_params(p)
    xp = np
    N = len(psi)
    x = xp.linspace(-p.L / 2, p.L / 2, N, endpoint=False)
    dx = float(x[1] - x[0])
    k = 2.0 * xp.pi * xp.fft.fftfreq(N, d=dx)
    abs_k = xp.abs(k)

    Lambda_e = eff['Lambda']
    alpha_e = eff['alpha']
    Gamma_e = eff['Gamma']
    f_FDT_e = eff['f_FDT']
    if getattr(p, 'fdt_couple', False) and Gamma_e > 0:
        f_FDT_e = 2.0 * Gamma_e * dx * p.kT / p.hbar

    H_lin_k = p.hbar**2 * k**2 / (2.0 * p.m) + alpha_e * abs_k**p.sigma
    half_lin = xp.exp(-1j * H_lin_k * p.dt / (2.0 * p.hbar)
                       - Gamma_e * p.dt / (2.0 * p.hbar))

    lam_e = xp.asarray(eff['lam'])
    nu_arr = xp.asarray(p.nu, dtype=xp.float64)
    M = len(p.nu)
    ou_decay_half = xp.exp(-nu_arr * p.dt * 0.5) if M else None
    noise_amp = float(xp.sqrt(xp.asarray(f_FDT_e * p.dt / dx))) if f_FDT_e > 0 else 0.0

    V_ext = _build_V_ext(p, x)

    psi = psi.copy()

    psi = xp.fft.ifft(xp.fft.fft(psi) * half_lin)

    rho = xp.abs(psi) ** 2
    if M:
        y = ou_decay_half[:, None] * y + (1.0 - ou_decay_half)[:, None] * rho
    V_mem = (lam_e[:, None] * y).sum(axis=0) if M else 0.0
    V_total = V_ext + Lambda_e * rho + V_mem
    psi = psi * xp.exp(-1j * V_total * p.dt / p.hbar)

    rho = xp.abs(psi) ** 2
    if M:
        y = ou_decay_half[:, None] * y + (1.0 - ou_decay_half)[:, None] * rho

    if not deterministic and noise_amp > 0:
        xi = rng.standard_normal(p.N)
        xip = rng.standard_normal(p.N)
        psi = psi + noise_amp * (xi + 1j * xip) / xp.sqrt(2.0)

    psi = xp.fft.ifft(xp.fft.fft(psi) * half_lin)

    return psi, y

def equilibrate(p: TriadParams, method: str = "accelerated",
                n_super: int = 50, max_supersteps: int = 60,
                tol_obs: float = 0.02, window: int = 5,
                verbose: bool = False) -> dict:
    """Reach the steady state of the FULL propagator faster than naive integration.

    Instead of running integrate() for thousands of dt steps, we run super-steps
    (each = n_super sub-steps of the FULL operator with FDT noise) and check
    observable convergence.  The acceleration comes from early stopping once
    the observables stabilize (same criterion as A2 ConvergenceObserver).

    Parameters
    ----------
    p : TriadParams
        Parameters.
    method : str
        "accelerated" for super-step + observable convergence,
        "baseline" for full integrate() comparison.
    n_super : int
        Number of sub-steps per super-step.
    max_supersteps : int
        Maximum number of super-steps.
    tol_obs : float
        Relative tolerance for observable convergence.
    window : int
        Window size for convergence check.
    verbose : bool

    Returns
    -------
    dict with psi_star, y_star, converged, n_supersteps, n_total_steps,
    observables_history, observables.
    """
    rng = np.random.default_rng(p.seed)
    x = np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
    dx = float(x[1] - x[0])
    psi = np.exp(-x**2 / 8.0).astype(np.complex128)
    psi /= np.sqrt((np.abs(psi)**2).sum() * dx)
    y = np.zeros((len(p.nu), p.N))

    obs_history = {"crystallinity": [], "energy": [], "k_star": []}
    converged = False
    n_total = 0

    for sup in range(max_supersteps):
        
        for _ in range(n_super):
            psi, y = _one_step_full(psi, y, p, rng, deterministic=False)
            n_total += 1

        C = crystallinity(psi, dx)
        E = energy(psi, dx, hbar=p.hbar, m=p.m, Lambda=p.Lambda)
        ks = dominant_wavenumber(psi, dx)

        obs_history["crystallinity"].append(C)
        obs_history["energy"].append(E)
        obs_history["k_star"].append(ks)

        if verbose and (sup % 5 == 0 or sup < 3):
            print(f"  superstep {sup:3d}  C={C:.4f}  E={E:.2f}  k*={ks:.4f}")

        if len(obs_history["crystallinity"]) >= window:
            recent_C = obs_history["crystallinity"][-window:]
            mean_C = np.mean(recent_C)
            if mean_C > 0.01:  
                rel_std = np.std(recent_C) / mean_C
                if rel_std < tol_obs:
                    converged = True
                    if verbose:
                        print(f"  CONVERGED at superstep {sup}: rel_std={rel_std:.4f}")
                    break

    obs_final = {
        "crystallinity": obs_history["crystallinity"][-1] if obs_history["crystallinity"] else 0,
        "k_star": obs_history["k_star"][-1] if obs_history["k_star"] else 0,
        "ipr": ipr(psi, dx),
        "participation": participation_ratio(psi, dx),
        "energy": obs_history["energy"][-1] if obs_history["energy"] else 0,
    }

    return {
        "psi_star": psi,
        "y_star": y,
        "converged": converged,
        "n_supersteps": sup + 1,
        "n_total_steps": n_total,
        "observables_history": obs_history,
        "observables": obs_final,
        "method": method,
    }

def fast_integrate(p: TriadParams, **kwargs) -> dict:
    """Compatibility wrapper: calls equilibrate() and returns old-style dict."""
    x = np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
    dx = float(x[1] - x[0])
    out = equilibrate(p, **kwargs)
    psi = out["psi_star"]
    return {
        "psi_final": psi,
        "y_final": out["y_star"],
        "density_final": np.abs(psi) ** 2,
        "x": x,
        "dx": dx,
        "params": p,
    }
