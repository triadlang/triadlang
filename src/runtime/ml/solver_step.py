
from __future__ import annotations

from runtime.backend import get_xp, to_xp
from runtime.core.solver import TriadParams, _effective_params
from triad import ntri as np


def _build_ml_step_kernels(p: TriadParams):

    xp = get_xp('auto')
    N = p.N
    x = xp.linspace(-p.L / 2, p.L / 2, N, endpoint=False)
    dx = float(x[1] - x[0])
    k = 2.0 * xp.pi * xp.fft.fftfreq(N, d=dx)
    abs_k = xp.abs(k)
    eff = _effective_params(p, dx=dx, D=1)
    Lambda_e = eff['Lambda']
    alpha_e = eff['alpha']
    Gamma_e = eff['Gamma']
    f_FDT_e = eff['f_FDT_e']
    H_lin_k = p.hbar**2 * k**2 / (2.0 * p.m) + alpha_e * abs_k**p.sigma
    half_lin = xp.exp(-1j * H_lin_k * p.dt / (2.0 * p.hbar) - Gamma_e * p.dt / (2.0 * p.hbar))
    lam_e = xp.asarray(eff['lam'])
    nu_arr = xp.asarray(p.nu, dtype=xp.float64)
    M = len(p.nu)
    ou_decay_half = xp.exp(-nu_arr * p.dt * 0.5) if M else None
    noise_amp = float(xp.sqrt(f_FDT_e * p.dt / dx)) if f_FDT_e > 0 else 0.0
    V_ext = xp.zeros(N)
    if p.V_ext is not None:
        if callable(p.V_ext):
            V_ext = xp.asarray(p.V_ext(x))
        elif isinstance(p.V_ext, str):

            if p.V_ext == 'harmonic':
                V_ext = 0.5 * p.m * p.omega ** 2 * x ** 2
            else:
                V_ext = xp.zeros(N)
        else:
            V_ext = xp.asarray(p.V_ext)
    return {
        'N': N, 'dx': dx, 'half_lin': half_lin, 'Lambda_e': Lambda_e,
        'lam_e': lam_e, 'ou_decay_half': ou_decay_half, 'noise_amp': noise_amp,
        'V_ext': V_ext, 'M': M, 'hbar': p.hbar, 'dt': p.dt,
    }

def _solver_step_ml_batched(psi: np.ndarray, y: np.ndarray,
                            kernels: dict,
                            drive_re: np.ndarray, drive_im: np.ndarray,
                            rng: np.random.Generator = None,
                            custom_half_lin: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:

    xp = get_xp('auto')
    half_lin = custom_half_lin if custom_half_lin is not None else kernels['half_lin']
    half_lin = to_xp(half_lin, xp)
    Lambda_e = kernels['Lambda_e']
    lam_e = to_xp(kernels['lam_e'], xp)
    ou_decay_half = to_xp(kernels['ou_decay_half'], xp) if kernels['ou_decay_half'] is not None else None
    noise_amp = kernels['noise_amp']
    V_ext = to_xp(kernels['V_ext'], xp)
    M = kernels['M']
    hbar = kernels['hbar']
    dt = kernels['dt']

    psi = to_xp(psi, xp)
    y = to_xp(y, xp)
    drive_re = to_xp(drive_re, xp)
    drive_im = to_xp(drive_im, xp)

    psi = psi + (drive_re + 1j * drive_im)

    psi = xp.fft.ifft(xp.fft.fft(psi, axis=-1) * half_lin, axis=-1)

    rho = xp.abs(psi) ** 2
    if M:
        y = ou_decay_half[None, :, None] * y + (1.0 - ou_decay_half[None, :, None]) * rho[:, None, :]
        if lam_e.ndim == 1:
            V_mem = (lam_e[None, :, None] * y).sum(axis=1)
        else:
            V_mem = (lam_e[None, :, :] * y).sum(axis=1)
    else:
        V_mem = 0.0
    V_total = V_ext[None, :] + Lambda_e * rho + V_mem
    psi = psi * xp.exp(-1j * V_total * dt / hbar)

    rho = xp.abs(psi) ** 2
    if M:
        y = ou_decay_half[None, :, None] * y + (1.0 - ou_decay_half[None, :, None]) * rho[:, None, :]

    if rng is not None and noise_amp > 0:
        N = kernels['N']
        B = int(psi.shape[0])
        xi = rng.standard_normal((B, N))
        xip = rng.standard_normal((B, N))
        psi = psi + noise_amp * (xi + 1j * xip) / xp.sqrt(2.0)

    psi = xp.fft.ifft(xp.fft.fft(psi, axis=-1) * half_lin, axis=-1)

    return psi, y
