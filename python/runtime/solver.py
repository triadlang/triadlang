from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, Union
import numpy as np
VExtSpec = Union[None, str, Callable[[np.ndarray], np.ndarray]]

@dataclass
class TriadParams:
    L: float = 32.0
    N: int = 128
    dt: float = 0.005
    T: float = 20.0
    hbar: float = 1.0
    m: float = 1.0
    V_ext: VExtSpec = 'harmonic'
    omega: float = 0.05
    Lambda: float = -0.5
    alpha: float = 0.15
    sigma: float = 1.5
    Gamma: float = 0.05
    f_FDT: float = 0.002
    nu: tuple = (2.0, 0.5, 0.1)
    lam: tuple = (-0.3, -0.2, -0.1)
    mode: str = 'full'
    seed: int = 0
    record_every: int = 4
    D: int = 1
    backend: str = 'auto'
    bc: str = 'periodic'
    bc_width: float = 0.15

def _effective_params(p: TriadParams) -> dict:
    lam_arr = np.asarray(p.lam, dtype=np.float64)
    if p.mode == 'linear':
        return dict(Lambda=0.0, alpha=0.0, Gamma=0.0, f_FDT=0.0, lam=np.zeros_like(lam_arr))
    if p.mode == 'thermal':
        return dict(Lambda=0.0, alpha=0.0, Gamma=p.Gamma, f_FDT=p.f_FDT, lam=np.zeros_like(lam_arr))
    return dict(Lambda=p.Lambda, alpha=p.alpha, Gamma=p.Gamma, f_FDT=p.f_FDT, lam=lam_arr)

def _build_V_ext(p: TriadParams, x_or_grid) -> np.ndarray:
    spec = p.V_ext
    if isinstance(x_or_grid, np.ndarray) and x_or_grid.ndim == 1:
        x = x_or_grid
        if spec is None:
            return np.zeros_like(x)
        if spec == 'harmonic':
            return 0.5 * p.m * p.omega ** 2 * x ** 2
        if callable(spec):
            return np.asarray(spec(x), dtype=np.float64)
        raise ValueError(f'unknown V_ext spec: {spec!r}')
    grids = x_or_grid
    if spec is None:
        return np.zeros_like(grids[0])
    if spec == 'harmonic':
        r2 = sum((g * g for g in grids))
        return 0.5 * p.m * p.omega ** 2 * r2
    if spec == 'double_well':
        r2 = sum((g * g for g in grids))
        w2 = (p.L / 8.0) ** 2
        return 0.05 * (r2 - w2) ** 2
    if spec == 'gaussian_bump':
        r2 = sum((g * g for g in grids))
        return -2.0 * np.exp(-r2 / 2.0)
    if spec == 'ramp':
        return 0.05 * grids[0]
    if spec == 'lattice':
        k0 = 2.0 * np.pi / (p.L / 4.0)
        return float(0.5) * sum((np.cos(k0 * g) for g in grids))
    if callable(spec):
        return np.asarray(spec(*grids), dtype=np.float64)
    raise ValueError(f'unknown V_ext spec: {spec!r}')

def _build_absorbing_mask(N: int, L: float, bc_width_frac: float) -> np.ndarray:
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    edge = bc_width_frac * L / 2
    mask = np.ones(N)
    near_left = x < -L / 2 + edge
    near_right = x > L / 2 - edge
    mask[near_left] = np.cos(np.pi * (x[near_left] - (-L / 2 + edge)) / (2 * edge)) ** 2
    mask[near_right] = np.cos(np.pi * (x[near_right] - (L / 2 - edge)) / (2 * edge)) ** 2
    return mask

def integrate(p: TriadParams, psi0: Optional[np.ndarray]=None, y0: Optional[np.ndarray]=None, auto_halve_dt: bool=True, noise_provider: Optional[callable]=None, record_y: bool=False) -> dict:
    p = _maybe_halve_dt(p, auto_halve_dt)
    from runtime.backend import get_xp, asnumpy
    xp = get_xp(getattr(p, 'backend', 'auto'))
    rng = xp.random.default_rng(p.seed)
    x = xp.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
    dx = float(asnumpy(x[1] - x[0]))
    k = 2.0 * xp.pi * xp.fft.fftfreq(p.N, d=dx)
    abs_k = xp.abs(k)
    eff = _effective_params(p)
    Lambda_e = eff['Lambda']
    alpha_e = eff['alpha']
    Gamma_e = eff['Gamma']
    f_FDT_e = eff['f_FDT']
    lam_e_np = eff['lam']
    lam_e = xp.asarray(lam_e_np)
    V_ext_host = _build_V_ext(p, np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False))
    V_ext = xp.asarray(V_ext_host)
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
    H_lin_k = p.hbar ** 2 * k ** 2 / (2.0 * p.m) + alpha_e * abs_k ** p.sigma
    half_lin = xp.exp(-1j * H_lin_k * p.dt / (2.0 * p.hbar) - Gamma_e * p.dt / (2.0 * p.hbar))
    ou_decay_half = xp.exp(-nu_arr * p.dt * 0.5) if M else None
    noise_amp = float(xp.sqrt(xp.asarray(f_FDT_e * p.dt / dx))) if f_FDT_e > 0 else 0.0
    bc = getattr(p, 'bc', 'periodic')
    if bc == 'absorbing':
        bc_mask_np = _build_absorbing_mask(p.N, p.L, getattr(p, 'bc_width', 0.15))
        bc_mask = xp.asarray(bc_mask_np)
    else:
        bc_mask = None
    n_steps = int(round(p.T / p.dt))
    rec_indices = list(range(0, n_steps + 1, max(1, p.record_every)))
    if rec_indices[-1] != n_steps:
        rec_indices.append(n_steps)
    rec_set = set(rec_indices)
    rec_density = []
    rec_t = []
    rec_y = []
    for step in range(n_steps + 1):
        if step in rec_set:
            rec_density.append(asnumpy(xp.abs(psi) ** 2))
            rec_t.append(step * p.dt)
            if record_y:
                rec_y.append(asnumpy(y).copy())
        if step == n_steps:
            break
        psi = xp.fft.ifft(xp.fft.fft(psi) * half_lin)
        rho = xp.abs(psi) ** 2
        if M:
            y = ou_decay_half[:, None] * y + (1.0 - ou_decay_half)[:, None] * rho
        V_mem = (lam_e[:, None] * y).sum(axis=0) if M else xp.asarray(0.0)
        V_tot = V_ext + Lambda_e * rho + V_mem
        psi = psi * xp.exp(-1j * V_tot * p.dt / p.hbar)
        rho = xp.abs(psi) ** 2
        if M:
            y = ou_decay_half[:, None] * y + (1.0 - ou_decay_half)[:, None] * rho
        if noise_amp > 0:
            if noise_provider is not None:
                psi = psi + xp.asarray(noise_provider(step, p.dt, p.N, dx, f_FDT_e))
            else:
                xi = rng.standard_normal(p.N)
                xip = rng.standard_normal(p.N)
                psi = psi + noise_amp * (xi + 1j * xip) / xp.sqrt(2.0)
        psi = xp.fft.ifft(xp.fft.fft(psi) * half_lin)
        if bc_mask is not None:
            psi = psi * bc_mask
    result = {'t': np.asarray(rec_t), 'x': asnumpy(x), 'dx': dx, 'density': np.asarray(rec_density).T if rec_density else np.zeros((p.N, 0)), 'psi_final': asnumpy(psi), 'y_final': asnumpy(y), 'params': p}
    if record_y:
        if rec_y:
            ya = np.asarray(rec_y)
            result['y_traj'] = np.transpose(ya, (1, 2, 0))
        else:
            result['y_traj'] = np.zeros((M, p.N, 0))
    return result

def _maybe_halve_dt(p: TriadParams, auto_halve_dt: bool) -> TriadParams:
    if not auto_halve_dt:
        return p
    if abs(p.Lambda) >= 4.0 and p.dt > 0.0025:
        return TriadParams(**{**p.__dict__, 'dt': 0.0025})
    return p

def integrate_2d(p: TriadParams, psi0: Optional[np.ndarray]=None, y0: Optional[np.ndarray]=None, auto_halve_dt: bool=True, record_y: bool=False, record_density: bool=False) -> dict:
    p = _maybe_halve_dt(p, auto_halve_dt)
    if p.D != 2:
        p = TriadParams(**{**p.__dict__, 'D': 2})
    from runtime.backend import get_xp, asnumpy
    xp = get_xp(getattr(p, 'backend', 'auto'))
    rng = xp.random.default_rng(p.seed)
    xs = xp.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
    dx = float(asnumpy(xs[1] - xs[0]))
    X, Y = xp.meshgrid(xs, xs, indexing='ij')
    kvec = 2.0 * xp.pi * xp.fft.fftfreq(p.N, d=dx)
    KX, KY = xp.meshgrid(kvec, kvec, indexing='ij')
    k2 = KX * KX + KY * KY
    abs_k = xp.sqrt(k2)
    eff = _effective_params(p)
    Lambda_e, alpha_e, Gamma_e, f_FDT_e, lam_e_np = (eff['Lambda'], eff['alpha'], eff['Gamma'], eff['f_FDT'], eff['lam'])
    lam_e = xp.asarray(lam_e_np)
    V_ext_host = _build_V_ext(p, (np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False), np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)))
    V_ext = xp.asarray(V_ext_host)
    if psi0 is None:
        psi = xp.exp(-(X * X + Y * Y) / 8.0).astype(xp.complex128)
        psi = psi / xp.sqrt((xp.abs(psi) ** 2).sum() * dx * dx)
    else:
        psi = xp.asarray(psi0, dtype=xp.complex128).copy()
    nu_arr = xp.asarray(p.nu, dtype=xp.float64)
    M = len(p.nu)
    if y0 is None:
        y = xp.zeros((M, p.N, p.N), dtype=xp.float64)
    else:
        y = xp.asarray(y0, dtype=xp.float64).copy()
    H_lin_k = p.hbar ** 2 * k2 / (2.0 * p.m) + alpha_e * abs_k ** p.sigma
    half_lin = xp.exp(-1j * H_lin_k * p.dt / (2.0 * p.hbar) - Gamma_e * p.dt / (2.0 * p.hbar))
    ou_decay_half = xp.exp(-nu_arr * p.dt * 0.5) if M else None
    noise_amp = float(xp.sqrt(xp.asarray(f_FDT_e * p.dt / (dx * dx)))) if f_FDT_e > 0 else 0.0
    bc = getattr(p, 'bc', 'periodic')
    if bc == 'absorbing':
        mask_1d_np = _build_absorbing_mask(p.N, p.L, getattr(p, 'bc_width', 0.15))
        mask_1d = xp.asarray(mask_1d_np)
        bc_mask = mask_1d[:, None] * mask_1d[None, :]
    else:
        bc_mask = None
    n_steps = int(round(p.T / p.dt))
    rec_t = []
    rec_psi_final = None
    rec_y = []
    rec_density = []
    rec_every = max(1, p.record_every)
    for step in range(n_steps + 1):
        if (record_y or record_density) and (step % rec_every == 0 or step == n_steps):
            rec_t.append(step * p.dt)
            if record_y:
                rec_y.append(asnumpy(y).copy())
            if record_density:
                rec_density.append(asnumpy(xp.abs(psi) ** 2).copy())
        if step == n_steps:
            rec_psi_final = asnumpy(psi).copy()
            if not (record_y or record_density):
                rec_t.append(step * p.dt)
            break
        psi = xp.fft.ifft2(xp.fft.fft2(psi) * half_lin)
        rho = xp.abs(psi) ** 2
        if M:
            y = ou_decay_half[:, None, None] * y + (1.0 - ou_decay_half)[:, None, None] * rho
        V_mem = (lam_e[:, None, None] * y).sum(axis=0) if M else xp.asarray(0.0)
        V_tot = V_ext + Lambda_e * rho + V_mem
        psi = psi * xp.exp(-1j * V_tot * p.dt / p.hbar)
        rho = xp.abs(psi) ** 2
        if M:
            y = ou_decay_half[:, None, None] * y + (1.0 - ou_decay_half)[:, None, None] * rho
        if noise_amp > 0:
            xi = rng.standard_normal((p.N, p.N))
            xip = rng.standard_normal((p.N, p.N))
            psi = psi + noise_amp * (xi + 1j * xip) / xp.sqrt(2.0)
        psi = xp.fft.ifft2(xp.fft.fft2(psi) * half_lin)
        if bc_mask is not None:
            psi = psi * bc_mask
    result = {'t': np.asarray(rec_t), 'x': asnumpy(xs), 'dx': dx, 'psi_final': rec_psi_final, 'y_final': asnumpy(y), 'params': p}
    if record_y and rec_y:
        ya = np.asarray(rec_y)
        result['y_traj'] = np.transpose(ya, (1, 2, 3, 0))
    if record_density and rec_density:
        result['density'] = np.transpose(np.asarray(rec_density), (1, 2, 0))
    return result

def integrate_3d(p: TriadParams, psi0: Optional[np.ndarray]=None, y0: Optional[np.ndarray]=None, auto_halve_dt: bool=True, record_density: bool=False, record_y: bool=False) -> dict:
    p = _maybe_halve_dt(p, auto_halve_dt)
    if p.D != 3:
        p = TriadParams(**{**p.__dict__, 'D': 3})
    from runtime.backend import get_xp, asnumpy
    xp = get_xp(getattr(p, 'backend', 'auto'))
    rng = xp.random.default_rng(p.seed)
    xs = xp.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
    dx = float(asnumpy(xs[1] - xs[0]))
    X, Y, Z = xp.meshgrid(xs, xs, xs, indexing='ij')
    kvec = 2.0 * xp.pi * xp.fft.fftfreq(p.N, d=dx)
    KX, KY, KZ = xp.meshgrid(kvec, kvec, kvec, indexing='ij')
    k2 = KX * KX + KY * KY + KZ * KZ
    abs_k = xp.sqrt(k2)
    eff = _effective_params(p)
    Lambda_e, alpha_e, Gamma_e, f_FDT_e, lam_e_np = (eff['Lambda'], eff['alpha'], eff['Gamma'], eff['f_FDT'], eff['lam'])
    lam_e = xp.asarray(lam_e_np)
    V_ext_host = _build_V_ext(p, (np.array(asnumpy(X)), np.array(asnumpy(Y)), np.array(asnumpy(Z))))
    V_ext = xp.asarray(V_ext_host)
    if psi0 is None:
        psi = xp.exp(-(X * X + Y * Y + Z * Z) / 8.0).astype(xp.complex128)
        psi = psi / xp.sqrt((xp.abs(psi) ** 2).sum() * dx ** 3)
    else:
        psi = xp.asarray(psi0, dtype=xp.complex128).copy()
    nu_arr = xp.asarray(p.nu, dtype=xp.float64)
    M = len(p.nu)
    if y0 is None:
        y = xp.zeros((M, p.N, p.N, p.N), dtype=xp.float64)
    else:
        y = xp.asarray(y0, dtype=xp.float64).copy()
    H_lin_k = p.hbar ** 2 * k2 / (2.0 * p.m) + alpha_e * abs_k ** p.sigma
    half_lin = xp.exp(-1j * H_lin_k * p.dt / (2.0 * p.hbar) - Gamma_e * p.dt / (2.0 * p.hbar))
    ou_decay_half = xp.exp(-nu_arr * p.dt * 0.5) if M else None
    noise_amp = float(xp.sqrt(xp.asarray(f_FDT_e * p.dt / dx ** 3))) if f_FDT_e > 0 else 0.0
    bc = getattr(p, 'bc', 'periodic')
    if bc == 'absorbing':
        mask_1d_np = _build_absorbing_mask(p.N, p.L, getattr(p, 'bc_width', 0.15))
        bc_mask = xp.asarray(mask_1d_np[:, None, None] * mask_1d_np[None, :, None] * mask_1d_np[None, None, :])
    else:
        bc_mask = None
    n_steps = int(round(p.T / p.dt))
    peak_density_t = []
    participation_t = []
    t_arr = []
    rec_y = []
    for step in range(n_steps + 1):
        rho = xp.abs(psi) ** 2
        if step % max(1, p.record_every) == 0 or step == n_steps:
            peak_density_t.append(float(asnumpy(rho.max())))
            norm2 = float(asnumpy(rho.sum() * dx ** 3))
            ipr_denom = max(float(asnumpy((rho ** 2).sum() * dx ** 3)), 1e-300)
            participation_t.append(float(norm2 * norm2 / ipr_denom))
            t_arr.append(step * p.dt)
            if record_y:
                rec_y.append(asnumpy(y).copy())
        if step == n_steps:
            break
        psi = xp.fft.ifftn(xp.fft.fftn(psi) * half_lin)
        rho = xp.abs(psi) ** 2
        if M:
            y = ou_decay_half[:, None, None, None] * y + (1.0 - ou_decay_half)[:, None, None, None] * rho
        V_mem = (lam_e[:, None, None, None] * y).sum(axis=0) if M else 0.0
        V_tot = V_ext + Lambda_e * rho + V_mem
        psi = psi * xp.exp(-1j * V_tot * p.dt / p.hbar)
        rho = xp.abs(psi) ** 2
        if M:
            y = ou_decay_half[:, None, None, None] * y + (1.0 - ou_decay_half)[:, None, None, None] * rho
        if noise_amp > 0:
            xi = rng.standard_normal((p.N, p.N, p.N))
            xip = rng.standard_normal((p.N, p.N, p.N))
            psi = psi + noise_amp * (xi + 1j * xip) / xp.sqrt(2.0)
        psi = xp.fft.ifftn(xp.fft.fftn(psi) * half_lin)
        if bc_mask is not None:
            psi = psi * bc_mask
    result = {'t': np.asarray(t_arr), 'x': asnumpy(xs), 'dx': dx, 'psi_final': asnumpy(psi), 'y_final': asnumpy(y), 'peak_t': np.asarray(peak_density_t), 'participation_t': np.asarray(participation_t), 'params': p}
    if record_y and rec_y:
        ya = np.asarray(rec_y)
        result['y_traj'] = np.transpose(ya, (1, 2, 3, 4, 0))
    return result
if __name__ == '__main__':
    for mode in ('linear', 'thermal', 'full'):
        out = integrate(TriadParams(mode=mode, T=4.0, seed=0))
        norm_t = out['density'].sum(axis=0) * out['dx']
        print(f'mode={mode:8s}  norm in [{norm_t.min():.6f}, {norm_t.max():.6f}]')