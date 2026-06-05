"""GPU-fused kernels for the Triad split-step solver.

When the backend is cupy, these kernels replace the per-element Python
arithmetic in the main integration loop with fused CUDA kernels via
``cp.fuse()``, reducing kernel launches from ~15 per step to ~3:
  1. half-step FFT pair (cupy native)
  2. mega-fused nonlinear + OU + noise (single kernel)
  3. half-step FFT pair again

Works for 1D, 2D, and 3D.  The fused kernels are elementwise and
automatically handle any array shape.

All three pillars remain active:
  P1: spectral propagation via cupy FFT (unmodified)
  P2: memory fields updated inside fused kernel
  P3: FDT noise injected inside fused kernel

Fused kernels are compiled lazily on first call and cached by cupy.
"""
from __future__ import annotations

import numpy as np

try:
    import cupy as _cp
    _CUDA = True
except Exception:
    _cp = None
    _CUDA = False

def _build_fused_kernels():
    """Build and cache fused CUDA kernels via cp.fuse().

    The mega3 kernel fuses OU half-update + V_mem + nonlinear phase
    + noise injection + OU half-update into a single kernel launch.
    It works on arrays of any dimensionality (1D, 2D, 3D) because
    cp.fuse() generates elementwise kernels.
    """
    if not _CUDA:
        return None

    @_cp.fuse()
    def mega3(psi, V_ext,
              y0, y1, y2,
              d0, d1, d2,
              l0, l1, l2,
              Lambda, dt_hbar,
              na, xi_r, xi_i, sqrt2):
        rho = _cp.abs(psi) ** 2
        
        y0 = d0 * y0 + (1.0 - d0) * rho
        y1 = d1 * y1 + (1.0 - d1) * rho
        y2 = d2 * y2 + (1.0 - d2) * rho
        V_mem = l0 * y0 + l1 * y1 + l2 * y2
        V_total = V_ext + Lambda * rho + V_mem
        psi = psi * _cp.exp(-1j * V_total * dt_hbar)
        psi = psi + na * (xi_r + 1j * xi_i) / sqrt2
        
        rho_new = _cp.abs(psi) ** 2
        y0 = d0 * y0 + (1.0 - d0) * rho_new
        y1 = d1 * y1 + (1.0 - d1) * rho_new
        y2 = d2 * y2 + (1.0 - d2) * rho_new
        return psi, y0, y1, y2

    @_cp.fuse()
    def fused_nl(psi, V_ext, Lambda, V_mem,
                 dt_hbar, noise_amp,
                 xi_r, xi_i, sqrt2):
        rho = _cp.abs(psi) ** 2
        V_total = V_ext + Lambda * rho + V_mem
        psi = psi * _cp.exp(-1j * V_total * dt_hbar)
        psi = psi + noise_amp * (xi_r + 1j * xi_i) / sqrt2
        return psi

    @_cp.fuse()
    def fused_ou(y, rho, decay):
        return decay * y + (1.0 - decay) * rho

    @_cp.fuse()
    def fused_extrap_predictor(psi, V_ext, Lambda, V_mem, dt_hbar):
        """Predictor: psi_pred = psi * exp(-1j * V_start * dt_hbar)"""
        rho = _cp.abs(psi) ** 2
        V_start = V_ext + Lambda * rho + V_mem
        return psi * _cp.exp(-1j * V_start * dt_hbar)

    @_cp.fuse()
    def fused_extrap_corrector(psi, V_ext, Lambda, V_mem, V_start,
                               dt_hbar, trap_lam,
                               noise_amp, xi_r, xi_i, sqrt2):
        """Corrector: interpolate V_start/V_end, apply, add noise.
        Note: Gamma_dt is NOT applied here; it is already in half_lin."""
        rho_pred = _cp.abs(psi) ** 2
        V_end = V_ext + Lambda * rho_pred + V_mem
        V_tot = (1.0 - trap_lam) * V_start + trap_lam * V_end
        psi = psi * _cp.exp(-1j * V_tot * dt_hbar)
        psi = psi + noise_amp * (xi_r + 1j * xi_i) / sqrt2
        return psi

    return {
        "mega3": mega3,
        "fused_nl": fused_nl,
        "fused_ou": fused_ou,
        "fused_extrap_predictor": fused_extrap_predictor,
        "fused_extrap_corrector": fused_extrap_corrector,
    }

_kernels = None

def get_kernels():
    global _kernels
    if _kernels is None:
        _kernels = _build_fused_kernels()
    return _kernels

def _apply_fft(psi, half_lin, D):
    """Apply the half-step spectral propagator for dimension D."""
    if D == 1:
        return _cp.fft.ifft(_cp.fft.fft(psi) * half_lin)
    elif D == 2:
        return _cp.fft.ifft2(_cp.fft.fft2(psi) * half_lin)
    else:
        return _cp.fft.ifftn(_cp.fft.fftn(psi) * half_lin)

def _rng_normal(rng, shape):
    """Generate two independent normal arrays of given shape."""
    return rng.standard_normal(shape), rng.standard_normal(shape)

def _build_V_ext_gpu(p, xp):
    """Build V_ext directly on the GPU without CPU round-trip.

    Unlike solver._build_V_ext which takes numpy grids and returns
    numpy arrays, this version constructs everything on the xp backend.
    """
    N = p.N
    L = p.L
    spec = p.V_ext

    if p.D == 1:
        x = xp.linspace(-L / 2, L / 2, N, endpoint=False)
        if spec is None:
            return xp.zeros(N, dtype=xp.float64)
        if spec == 'harmonic':
            return 0.5 * p.m * p.omega ** 2 * x ** 2
        if callable(spec):
            from runtime.backend import asnumpy
            return xp.asarray(spec(asnumpy(x)), dtype=xp.float64)
        raise ValueError(f'unknown V_ext spec: {spec!r}')

    xs = xp.linspace(-L / 2, L / 2, N, endpoint=False)
    if p.D == 2:
        X, Y = xp.meshgrid(xs, xs, indexing='ij')
    else:
        X, Y, Z = xp.meshgrid(xs, xs, xs, indexing='ij')

    if spec is None:
        shape = (N,) * p.D
        return xp.zeros(shape, dtype=xp.float64)
    if spec == 'harmonic':
        if p.D == 2:
            r2 = X * X + Y * Y
        else:
            r2 = X * X + Y * Y + Z * Z
        return 0.5 * p.m * p.omega ** 2 * r2
    if spec == 'double_well':
        if p.D == 2:
            r2 = X * X + Y * Y
        else:
            r2 = X * X + Y * Y + Z * Z
        w2 = (L / 8.0) ** 2
        return 0.05 * (r2 - w2) ** 2
    if spec == 'gaussian_bump':
        if p.D == 2:
            r2 = X * X + Y * Y
        else:
            r2 = X * X + Y * Y + Z * Z
        return -2.0 * xp.exp(-r2 / 2.0)
    if spec == 'ramp':
        return 0.05 * X
    if spec == 'lattice':
        k0 = 2.0 * xp.pi / (L / 4.0)
        if p.D == 2:
            return float(0.5) * (xp.cos(k0 * X) + xp.cos(k0 * Y))
        else:
            return float(0.5) * (xp.cos(k0 * X) + xp.cos(k0 * Y) + xp.cos(k0 * Z))
    if callable(spec):
        from runtime.backend import asnumpy
        if p.D == 2:
            grids_np = (asnumpy(X), asnumpy(Y))
        else:
            grids_np = (asnumpy(X), asnumpy(Y), asnumpy(Z))
        return xp.asarray(spec(*grids_np), dtype=xp.float64)
    raise ValueError(f'unknown V_ext spec: {spec!r}')

def integrate_gpu_fused(p, psi0=None, y0=None, auto_halve_dt=True,
                        noise_provider=None, record_y=False,
                        record_density=False):
    """Full Strang split-step integration using fused GPU kernels.

    Identical physics to ``solver.integrate()``, but with all elementwise
    operations fused into ~3 kernel launches per step instead of ~15.

    Supports 1D, 2D, and 3D.  For the default 3-channel OU memory (M=3),
    uses a single mega-kernel that does OU update + V_mem + nonlinear +
    noise + OU update in one launch.  For M != 3, falls back to
    per-channel fused_ou + fused_nl.

    Parameters
    ----------
    p : TriadParams
    psi0 : ndarray, optional
    y0 : ndarray, optional
    auto_halve_dt : bool
    noise_provider : callable, optional
    record_y : bool
    record_density : bool

    Returns
    -------
    dict, same keys as ``solver.integrate()`` or ``solver.integrate_2d()``
    etc. depending on p.D.
    """
    from runtime.core.solver import (
        _effective_params, _build_absorbing_mask, _maybe_halve_dt,
        _resolve_trap_lambda,
    )
    from runtime.backend import get_xp, asnumpy

    p = _maybe_halve_dt(p, auto_halve_dt)
    D = getattr(p, 'D', 1)

    k = get_kernels()
    if k is None:
        raise RuntimeError("cupy not available")

    xp = get_xp(getattr(p, 'backend', 'auto'))
    if xp is not _cp:
        raise RuntimeError("backend is not cupy")

    eff = _effective_params(p)
    N = p.N
    dx = p.L / N
    n_steps = int(round(p.T / p.dt))
    M = len(p.nu)
    use_mega = (M == 3)

    xs = xp.linspace(-p.L / 2, p.L / 2, N, endpoint=False)

    if D == 1:
        kv = 2.0 * xp.pi * xp.fft.fftfreq(N, d=dx)
        abs_k = xp.abs(kv)
        H_lin = p.hbar**2 * kv**2 / (2.0 * p.m) + eff['alpha'] * abs_k**p.sigma
        half_lin = xp.exp(-1j * H_lin * p.dt / (2.0 * p.hbar)
                          - eff['Gamma'] * p.dt / (2.0 * p.hbar))
    elif D == 2:
        kvec = 2.0 * xp.pi * xp.fft.fftfreq(N, d=dx)
        KX, KY = xp.meshgrid(kvec, kvec, indexing='ij')
        k2 = KX * KX + KY * KY
        abs_k = xp.sqrt(k2)
        H_lin = p.hbar**2 * k2 / (2.0 * p.m) + eff['alpha'] * abs_k**p.sigma
        half_lin = xp.exp(-1j * H_lin * p.dt / (2.0 * p.hbar)
                          - eff['Gamma'] * p.dt / (2.0 * p.hbar))
    else:
        kvec = 2.0 * xp.pi * xp.fft.fftfreq(N, d=dx)
        KX, KY, KZ = xp.meshgrid(kvec, kvec, kvec, indexing='ij')
        k2 = KX * KX + KY * KY + KZ * KZ
        abs_k = xp.sqrt(k2)
        H_lin = p.hbar**2 * k2 / (2.0 * p.m) + eff['alpha'] * abs_k**p.sigma
        half_lin = xp.exp(-1j * H_lin * p.dt / (2.0 * p.hbar)
                          - eff['Gamma'] * p.dt / (2.0 * p.hbar))

    V_ext = _build_V_ext_gpu(p, xp)

    if D == 1:
        grid_r2 = xs ** 2
        vol = dx
    elif D == 2:
        X, Y = xp.meshgrid(xs, xs, indexing='ij')
        grid_r2 = X * X + Y * Y
        vol = dx * dx
    else:
        X, Y, Z = xp.meshgrid(xs, xs, xs, indexing='ij')
        grid_r2 = X * X + Y * Y + Z * Z
        vol = dx ** 3

    if psi0 is None:
        psi = xp.exp(-grid_r2 / 8.0).astype(xp.complex128)
        psi = psi / xp.sqrt((xp.abs(psi)**2).sum() * vol)
    else:
        psi = _cp.asarray(psi0, dtype=_cp.complex128).copy()

    lam_e = _cp.asarray(eff['lam'], dtype=_cp.float64)
    nu_e = _cp.asarray(p.nu, dtype=_cp.float64)
    ou_decay_half = _cp.exp(-nu_e * p.dt / 2.0)
    if M > 0:
        mem_shape = (M,) + (N,) * D
        if y0 is None:
            y = _cp.zeros(mem_shape, dtype=_cp.float64)
        else:
            y = _cp.asarray(y0, dtype=_cp.float64).copy()

    rng = xp.random.default_rng(p.seed)
    Gamma_e = eff['Gamma']
    f_FDT_e = eff['f_FDT']
    if getattr(p, 'fdt_couple', False) and Gamma_e > 0:
        f_FDT_e = 2.0 * Gamma_e * dx * p.kT / p.hbar
    if f_FDT_e > 0:
        noise_amp = float(xp.sqrt(xp.asarray(f_FDT_e * p.dt / vol)))
    else:
        noise_amp = 0.0

    sqrt2 = _cp.float64(1.4142135623730951)
    Lambda_e = _cp.float64(float(eff['Lambda']))
    dt_hbar = _cp.float64(p.dt / p.hbar)
    Gamma_dt = _cp.float64(float(Gamma_e) * p.dt / p.hbar)
    na = _cp.float64(noise_amp) if noise_amp > 0 else _cp.float64(0.0)

    bc = getattr(p, 'bc', 'periodic')
    bc_mask = None
    if bc == 'absorbing':
        mask_np = _build_absorbing_mask(N, p.L, getattr(p, 'bc_width', 0.15))
        mask_1d = _cp.asarray(mask_np, dtype=_cp.float64)
        if D == 1:
            bc_mask = mask_1d
        elif D == 2:
            bc_mask = mask_1d[:, None] * mask_1d[None, :]
        else:
            bc_mask = (mask_1d[:, None, None] * mask_1d[None, :, None]
                       * mask_1d[None, None, :])

    exptrap = getattr(p, 'step_mode', 'strang') == 'exptrap'
    trap_lambda_val = getattr(p, 'trap_lambda', 0.5)
    
    use_mega_actual = use_mega and not exptrap

    rec_every = max(1, getattr(p, 'record_every', 4))
    rec_density = []
    rec_t = []
    rec_y = []
    field_shape = (N,) * D

    for step in range(n_steps + 1):
        
        do_record = (step % rec_every == 0 or step == n_steps)
        if D == 3 and do_record:
            
            pass
        if do_record:
            if record_density or D == 1:
                rec_density.append(asnumpy(xp.abs(psi)**2))
            rec_t.append(step * p.dt)
            if record_y and M > 0:
                rec_y.append(asnumpy(y).copy())
        if step == n_steps:
            break

        psi = _apply_fft(psi, half_lin, D)

        if use_mega_actual:
            noise_shape = field_shape
            if noise_provider is not None:
                nd = _cp.asarray(noise_provider(step, p.dt, N, dx, f_FDT_e))
                xi_r, xi_i = nd.real, nd.imag
                _na = _cp.float64(1.0)
            else:
                xi_r = rng.standard_normal(noise_shape).astype(_cp.float64)
                xi_i = rng.standard_normal(noise_shape).astype(_cp.float64)
                _na = na

            psi, y[0], y[1], y[2] = k["mega3"](
                psi, V_ext,
                y[0], y[1], y[2],
                ou_decay_half[0], ou_decay_half[1], ou_decay_half[2],
                lam_e[0], lam_e[1], lam_e[2],
                Lambda_e, dt_hbar,
                _na, xi_r, xi_i, sqrt2,
            )
        else:
            rho = _cp.abs(psi) ** 2
            V_mem = _cp.zeros(field_shape, dtype=_cp.float64)
            for j in range(M):
                y[j] = k["fused_ou"](y[j], rho, ou_decay_half[j])
                V_mem = V_mem + lam_e[j] * y[j]

            if noise_provider is not None:
                nd = _cp.asarray(noise_provider(step, p.dt, N, dx, f_FDT_e))
                xi_r, xi_i = nd.real, nd.imag
                _na = _cp.float64(1.0)
            else:
                xi_r = rng.standard_normal(field_shape).astype(_cp.float64)
                xi_i = rng.standard_normal(field_shape).astype(_cp.float64)
                _na = na

            if exptrap:
                
                rho_s = _cp.abs(psi) ** 2
                V_start = V_ext + Lambda_e * rho_s + V_mem
                lam_t = _resolve_trap_lambda(trap_lambda_val, rho_s, _cp)
                trap_lam = _cp.float64(float(lam_t))
                
                psi = k["fused_extrap_corrector"](
                    psi, V_ext, Lambda_e, V_mem, V_start,
                    dt_hbar, trap_lam,
                    _na, xi_r, xi_i, sqrt2)
            else:
                psi = k["fused_nl"](psi, V_ext, Lambda_e, V_mem,
                                    dt_hbar, _na,
                                    xi_r, xi_i, sqrt2)

            rho2 = _cp.abs(psi) ** 2
            for j in range(M):
                y[j] = k["fused_ou"](y[j], rho2, ou_decay_half[j])

        psi = _apply_fft(psi, half_lin, D)

        if bc_mask is not None:
            psi = psi * bc_mask

    result = {
        'psi_final': asnumpy(psi),
        'params': p,
        'dx': dx,
    }

    if D == 1:
        result['t'] = np.asarray(rec_t)
        result['x'] = asnumpy(xs)
        result['density'] = np.asarray(rec_density).T if rec_density else np.zeros((N, 0))
        result['y_final'] = asnumpy(y) if M > 0 else np.zeros((M, N))
        if record_y and rec_y:
            ya = np.asarray(rec_y)
            result['y_traj'] = np.transpose(ya, (1, 2, 0))
    elif D == 2:
        result['t'] = np.asarray(rec_t)
        result['x'] = asnumpy(xs)
        result['y_final'] = asnumpy(y) if M > 0 else np.zeros((M, N, N))
        if record_density and rec_density:
            result['density'] = np.transpose(np.asarray(rec_density), (1, 2, 0))
        if record_y and rec_y:
            ya = np.asarray(rec_y)
            result['y_traj'] = np.transpose(ya, (1, 2, 3, 0))
    else:
        result['t'] = np.asarray(rec_t)
        result['x'] = asnumpy(xs)
        result['y_final'] = asnumpy(y) if M > 0 else np.zeros((M, N, N, N))
        
        rho_final = xp.abs(psi)**2
        peak_val = float(asnumpy(rho_final.max()))
        norm2 = float(asnumpy(rho_final.sum() * vol))
        ipr_denom = max(float(asnumpy((rho_final**2).sum() * vol)), 1e-300)
        result['peak_t'] = np.asarray([peak_val])
        result['participation_t'] = np.asarray([norm2**2 / ipr_denom])
        if record_density and rec_density:
            result['density'] = np.transpose(np.asarray(rec_density), (1, 2, 3, 0))
        if record_y and rec_y:
            ya = np.asarray(rec_y)
            result['y_traj'] = np.transpose(ya, (1, 2, 3, 4, 0))

    return result
