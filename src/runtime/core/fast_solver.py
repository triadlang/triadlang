from __future__ import annotations

from runtime.core.solver import TriadParams, _build_V_ext, _effective_params, _validate_triad_params
from runtime.physics.observables import (
    crystallinity,
    dominant_wavenumber,
    energy,
    ipr,
    participation_ratio,
)
from triad import ntri as np


def _one_step_triad(psi: np.ndarray, y: np.ndarray,
                    p: TriadParams, rng: np.random.Generator,
                    deterministic: bool = False) -> tuple[np.ndarray, np.ndarray]:

    xp = np
    N = len(psi)
    x = xp.linspace(-p.L / 2, p.L / 2, N, endpoint=False)
    dx = float(x[1] - x[0])
    eff = _effective_params(p, dx=dx, D=1)
    k = 2.0 * xp.pi * xp.fft.fftfreq(N, d=dx)
    abs_k = xp.abs(k)

    Lambda_e = eff['Lambda']
    alpha_e = eff['alpha']
    Gamma_e = eff['Gamma']
    f_FDT_e = eff['f_FDT_e']

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
    _validate_triad_params(p)

    if method not in ("accelerated", "direct"):
        raise ValueError(f"equilibrate() method must be 'accelerated' or 'direct', got {method!r}")
    if method == "direct":
        return _equilibrate_direct(p, verbose=verbose)

    if p.D != 1:
        raise ValueError(f"equilibrate() e 1D mas recebeu D={p.D}")
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
            psi, y = _one_step_triad(psi, y, p, rng, deterministic=False)
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

def _equilibrate_direct(p: TriadParams, verbose: bool = False) -> dict:
    from runtime.core.solver import integrate
    r = integrate(p)
    psi = r["psi_final"]
    y = r["y_final"]
    x = np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
    dx = float(x[1] - x[0])
    obs_final = {
        "crystallinity": crystallinity(psi, dx),
        "k_star": dominant_wavenumber(psi, dx),
        "ipr": ipr(psi, dx),
        "participation": participation_ratio(psi, dx),
        "energy": energy(psi, dx, hbar=p.hbar, m=p.m, Lambda=p.Lambda),
    }
    if verbose:
        print(f'  direct: C={obs_final["crystallinity"]:.4f}  E={obs_final["energy"]:.2f}')
    return {
        "psi_star": psi,
        "y_star": y,
        "converged": False,
        "n_supersteps": 0,
        "n_total_steps": int(r.get("n_steps", 0)),
        "observables_history": {k: [v] for k, v in obs_final.items()},
        "observables": obs_final,
        "method": "direct",
    }

def fast_integrate(p: TriadParams, **kwargs) -> dict:
    _validate_triad_params(p)

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
