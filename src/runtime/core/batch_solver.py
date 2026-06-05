"""Batch solver for GPU-accelerated parallel trajectories.

The standard solver processes one psi field at a time.  For ensemble methods
(SelfVerifier, parallel_relax, consistency distillation) we need K independent
trajectories running in parallel.  This module processes K fields of size N
as a (K, N) array, using batched FFTs that are 2-3x faster on GPU than
sequential CPU FFTs.

The physics is identical to solver.integrate().  Only the execution model
differs: all K fields share the same TriadParams but have independent noise
realizations and initial conditions.
"""

from __future__ import annotations
import numpy as np
from runtime.backend import get_xp, asnumpy
from runtime.core.solver import TriadParams
from runtime.physics.observables import crystallinity
from typing import Optional

def integrate_batch(
    p: TriadParams,
    K: int = 8,
    seeds: Optional[list[int]] = None,
    T: Optional[float] = None,
    backend: str = "auto",
    return_trajectory: bool = False,
    record_every: int = 200,
) -> dict:
    """Integrate K independent trajectories in parallel.

    Parameters
    ----------
    p : TriadParams
        Shared parameters for all trajectories.
    K : int
        Number of parallel trajectories.
    seeds : list[int] or None
        Per-trajectory RNG seeds.  If None, seeds = range(K).
    T : float or None
        Override p.T for this batch.
    backend : str
        'auto', 'cuda', or 'numpy'.
    return_trajectory : bool
        If True, record density snapshots.
    record_every : int
        Steps between trajectory snapshots.

    Returns
    -------
    dict with keys:
        psi_final : (K, N) complex array
        y_final : (K, M, N) float array
        observables : list[dict] per trajectory
        seeds : list[int]
        backend : str
    """
    xp = get_xp(backend)
    is_gpu = hasattr(xp, "cuda")

    if seeds is None:
        seeds = list(range(K))
    assert len(seeds) == K

    if T is not None:
        import dataclasses
        p = dataclasses.replace(p, T=T)

    N = p.N
    M = len(p.nu)
    dx = p.L / N
    n_steps = int(p.T / p.dt)
    x = xp.linspace(-p.L / 2, p.L / 2, N, endpoint=False)
    k = 2.0 * xp.pi * xp.fft.fftfreq(N, d=dx)
    abs_k = xp.abs(k)

    V_ext_np = _build_V_ext_host(p)
    V_ext = xp.asarray(V_ext_np)

    hbar, m, dt = p.hbar, p.m, p.dt
    H_lin_k = hbar ** 2 * k ** 2 / (2 * m) + p.alpha * abs_k ** p.sigma
    Gamma_e = p.Gamma * (1.0 if p.mode in ("full", "thermal") else 0.0)
    half_lin = xp.exp(
        -1j * H_lin_k * dt / (2 * hbar) - Gamma_e * dt / (2 * hbar)
    )
    ou_decay_half = xp.exp(-xp.asarray(p.nu, dtype=xp.float64) * dt * 0.5)
    lam_e = xp.asarray(p.lam, dtype=xp.float64)

    f_FDT_e = p.f_FDT
    if getattr(p, "fdt_couple", False):
        kT = 2.0 * p.Gamma / (p.Gamma + 0.1)
        f_FDT_e = 2 * p.Gamma * dx * kT / hbar
    noise_amp = float(xp.sqrt(xp.asarray(f_FDT_e * dt / dx))) if f_FDT_e > 0 else 0.0
    use_noise = p.mode in ("full", "thermal") and noise_amp > 0

    bc_mask = None
    if p.bc == "absorbing":
        bc_mask_np = np.ones(N, dtype=np.float64)
        w = int(N * 0.1)
        bc_mask_np[:w] = np.linspace(0, 1, w)
        bc_mask_np[-w:] = np.linspace(1, 0, w)
        bc_mask = xp.asarray(bc_mask_np)

    psi = xp.zeros((K, N), dtype=xp.complex128)
    rngs = []
    for i in range(K):
        rng = xp.random.default_rng(seeds[i])
        rngs.append(rng)
        psi0 = xp.exp(-x ** 2 / 8.0).astype(xp.complex128)
        psi0 = psi0 / xp.sqrt((xp.abs(psi0) ** 2).sum() * dx)
        psi[i, :] = psi0

    y = xp.zeros((K, M, N), dtype=xp.float64)

    rec_density = [] if return_trajectory else None
    rec_crystallinity = [] if return_trajectory else None

    half_lin_row = half_lin  

    for step in range(n_steps):
        
        psi = xp.fft.ifft(xp.fft.fft(psi, axis=1) * half_lin_row[None, :], axis=1)

        rho = xp.abs(psi) ** 2  

        if M > 0 and p.mode == "full":
            for j in range(M):
                y[:, j, :] = ou_decay_half[j] * y[:, j, :] + (1 - ou_decay_half[j]) * rho

        V_tot = V_ext[None, :] + p.Lambda * rho
        if M > 0 and p.mode == "full":
            V_mem = xp.einsum("m,kmn->kn", lam_e, y)
            V_tot = V_tot + V_mem

        if p.step_mode == "exptrap":
            V_start = V_tot
            psi_pred = psi * xp.exp(-1j * V_start * dt / hbar)
            rho_pred = xp.abs(psi_pred) ** 2
            V_pred = V_ext[None, :] + p.Lambda * rho_pred
            if M > 0 and p.mode == "full":
                V_pred = V_pred + V_mem
            V_avg = 0.5 * (V_start + V_pred)
            psi = psi * xp.exp(-1j * V_avg * dt / hbar)
        else:
            psi = psi * xp.exp(-1j * V_tot * dt / hbar)

        rho = xp.abs(psi) ** 2

        if M > 0 and p.mode == "full":
            for j in range(M):
                y[:, j, :] = ou_decay_half[j] * y[:, j, :] + (1 - ou_decay_half[j]) * rho

        if use_noise:
            for i in range(K):
                xi = rngs[i].standard_normal(N)
                xip = rngs[i].standard_normal(N)
                psi[i] += noise_amp * (xi + 1j * xip) / xp.sqrt(2.0)

        if bc_mask is not None:
            psi = psi * bc_mask[None, :]

        psi = xp.fft.ifft(xp.fft.fft(psi, axis=1) * half_lin_row[None, :], axis=1)

        if return_trajectory and step % record_every == 0:
            rho_np = asnumpy(xp.abs(psi) ** 2)
            rec_density.append(rho_np)
            cryst = [float(crystallinity(rho_np[i], dx)) for i in range(K)]
            rec_crystallinity.append(cryst)

    psi_np = asnumpy(psi)
    y_np = asnumpy(y)

    obs_list = []
    for i in range(K):
        obs_list.append({
            "crystallinity": float(crystallinity(psi_np[i], dx)),
            "energy": float(_batch_energy(psi_np[i], dx, p)),
        })

    result = {
        "psi_final": psi_np,
        "y_final": y_np,
        "observables": obs_list,
        "seeds": seeds,
        "K": K,
        "N": N,
        "backend": "cuda" if is_gpu else "numpy",
    }
    if return_trajectory:
        result["density_trajectory"] = rec_density
        result["crystallinity_trajectory"] = rec_crystallinity

    return result

def _build_V_ext_host(p: TriadParams) -> np.ndarray:
    """Build V_ext on CPU, same logic as solver.py."""
    import numpy as _np
    x = _np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
    if callable(getattr(p, "V_ext", None)):
        return _np.asarray(p.V_ext(x), dtype=_np.float64)
    if isinstance(p.V_ext, _np.ndarray):
        return p.V_ext.astype(_np.float64)
    
    return 0.5 * x ** 2

def _batch_energy(psi: np.ndarray, dx: float, p: TriadParams) -> float:
    """Compute energy for a single field state."""
    rho = np.abs(psi) ** 2
    
    dpsi = np.gradient(psi, dx)
    kinetic = 0.5 * np.sum(np.abs(dpsi) ** 2) * dx
    nonlinear = 0.5 * p.Lambda * np.sum(rho ** 2) * dx
    x = np.linspace(-p.L / 2, p.L / 2, len(psi), endpoint=False)
    V_ext = _build_V_ext_host(p)
    potential = np.sum(V_ext * rho) * dx
    return kinetic + nonlinear + potential
