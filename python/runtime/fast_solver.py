from __future__ import annotations
import numpy as np
from runtime.solver import TriadParams, _effective_params, _build_V_ext

def fast_integrate(p: TriadParams, psi0: np.ndarray | None=None, y0: np.ndarray | None=None) -> dict:
    from runtime.backend import get_xp, asnumpy
    xp = get_xp(getattr(p, 'backend', 'auto'))
    x = xp.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
    dx = float(xp.diff(x[:2])[0]) if p.N > 1 else p.L
    k = 2.0 * xp.pi * xp.fft.fftfreq(p.N, d=dx)
    eff = _effective_params(p)
    Lambda_e = eff['Lambda']
    alpha_e = eff['alpha']
    Gamma_e = eff['Gamma']
    f_FDT_e = eff['f_FDT']
    lam_e = xp.asarray(eff['lam'])
    V_ext = xp.asarray(_build_V_ext(p, np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)))
    if psi0 is None:
        psi = xp.exp(-x ** 2 / 8.0).astype(xp.complex128)
        psi = psi / xp.sqrt((xp.abs(psi) ** 2).sum() * dx)
    else:
        psi = xp.asarray(psi0, dtype=xp.complex128).copy()
    nu_arr = xp.asarray(p.nu, dtype=xp.float64)
    M = len(p.nu)
    if y0 is None:
        y = xp.zeros((M, p.N), dtype=xp.float64)
    else:
        y = xp.asarray(y0, dtype=xp.float64).copy()
    abs_k = xp.abs(k)
    H_lin_k = p.hbar ** 2 * k ** 2 / (2.0 * p.m) + alpha_e * abs_k ** p.sigma
    half_lin = xp.exp((-1j * H_lin_k / p.hbar - Gamma_e / p.hbar) * (p.dt / 2.0))
    if M:
        ou_decay_half = xp.exp(-nu_arr * p.dt * 0.5)
        odh = ou_decay_half[:, None]
        odh_comp = 1.0 - odh
    if f_FDT_e > 0:
        noise_amp = float(xp.sqrt(xp.asarray(f_FDT_e * p.dt / dx)))
        rng = xp.random.default_rng(p.seed)
    else:
        noise_amp = 0.0
    n_steps = int(round(p.T / p.dt))
    dt_over_hbar = p.dt / p.hbar
    for _ in range(n_steps):
        psi = xp.fft.ifft(xp.fft.fft(psi) * half_lin)
        rho = xp.abs(psi) ** 2
        if M:
            y = odh * y + odh_comp * rho
            V_mem = (lam_e[:, None] * y).sum(axis=0)
        else:
            V_mem = 0.0
        psi = psi * xp.exp(-1j * (V_ext + Lambda_e * rho + V_mem) * dt_over_hbar)
        if M:
            rho = xp.abs(psi) ** 2
            y = odh * y + odh_comp * rho
        if noise_amp > 0:
            xi = rng.standard_normal((2, p.N))
            psi = psi + noise_amp * (xi[0] + 1j * xi[1]) * (1.0 / xp.sqrt(2.0))
        psi = xp.fft.ifft(xp.fft.fft(psi) * half_lin)
    return {'psi_final': asnumpy(psi), 'y_final': asnumpy(y), 'density_final': asnumpy(xp.abs(psi) ** 2), 'x': asnumpy(x), 'dx': dx, 'params': p}