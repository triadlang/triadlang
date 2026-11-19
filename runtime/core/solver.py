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
    # mode is a solver-level numerical audit knob, not a Triad regime selector.
    # 'full' is the physical model: P1+P2+P3 always active together.
    # 'linear' and 'thermal' exist only so that test_solver_audit.py can verify
    # propagator unitarity (norm conservation to fp64) and dt-convergence, which
    # are mathematically checkable only when Gamma=0 and Lambda=0. no regime,
    # template, compiler config, or .tri program ever sets mode away from 'full':
    # the three pillars are never isolated by design. see _effective_params.
    seed: int = 0
    record_every: int = 4
    D: int = 1
    backend: str = 'auto'
    bc: str = 'periodic'
    bc_width: float = 0.15
    step_mode: str = 'strang'
    trap_lambda: object = 0.5
    fdt_couple: bool = True
    kT: float = 1.0

def _effective_params(p: TriadParams) -> dict:
    """Return the coefficients the integrator actually uses.

    The physical Triad model is the 'full' case: P1 (alpha, fractional
    dispersion), P2 (Lambda, lam memory) and P3 (Gamma, f_FDT) are all
    active together and never isolated. The 'linear' and 'thermal' cases
    are NOT regimes and are never reachable from a regime, template,
    compiler config or .tri program. they exist only as a numerical audit
    harness (test_solver_audit.py): linear zeroes the nonlinear and
    dissipative terms so the split-step propagator becomes exactly unitary,
    which is the only setting where norm conservation to fp64 and
    dt-convergence can be checked against analytic expectations. thermal
    keeps only P3 to check the dissipation half-step in isolation.
    """
    lam_arr = np.asarray(p.lam, dtype=np.float64)
    if p.mode == 'linear':
        return dict(Lambda=0.0, alpha=0.0, Gamma=0.0, f_FDT=0.0, lam=np.zeros_like(lam_arr))
    if p.mode == 'thermal':
        return dict(Lambda=0.0, alpha=0.0, Gamma=p.Gamma, f_FDT=p.f_FDT, lam=np.zeros_like(lam_arr))
    return dict(Lambda=p.Lambda, alpha=p.alpha, Gamma=p.Gamma, f_FDT=p.f_FDT, lam=lam_arr)

def _resolve_trap_lambda(trap_lambda, rho, xp) -> float:
    if callable(trap_lambda):
        
        from runtime.backend import asnumpy
        return float(trap_lambda(asnumpy(rho)))
    return float(trap_lambda)

def _build_V_ext(p: TriadParams, x_or_grid, xp=None):
    """Build external potential. Works with numpy or cupy arrays.

    When xp is provided, uses xp for math operations so arrays stay on GPU.
    When xp is None, infers from input type (falls back to numpy).
    """
    if xp is None:
        xp = np
    spec = p.V_ext
    
    is_1d = False
    try:
        if hasattr(x_or_grid, 'ndim') and x_or_grid.ndim == 1:
            is_1d = True
    except Exception:
        pass
    if is_1d:
        x = x_or_grid
        if spec is None:
            return xp.zeros_like(x)
        if spec == 'harmonic':
            return 0.5 * p.m * p.omega ** 2 * x ** 2
        if callable(spec):
            result = spec(asnumpy(x) if hasattr(x, 'get') else x)
            return xp.asarray(result, dtype=xp.float64)
        raise ValueError(f'unknown V_ext spec: {spec!r}')
    grids = x_or_grid
    if spec is None:
        return xp.zeros_like(grids[0])
    if spec == 'harmonic':
        r2 = sum((g * g for g in grids))
        return 0.5 * p.m * p.omega ** 2 * r2
    if spec == 'double_well':
        r2 = sum((g * g for g in grids))
        w2 = (p.L / 8.0) ** 2
        return 0.05 * (r2 - w2) ** 2
    if spec == 'gaussian_bump':
        r2 = sum((g * g for g in grids))
        return -2.0 * xp.exp(-r2 / 2.0)
    if spec == 'ramp':
        return 0.05 * grids[0]
    if spec == 'lattice':
        k0 = 2.0 * xp.pi / (p.L / 4.0)
        return 0.5 * sum((xp.cos(k0 * g) for g in grids))
    if callable(spec):
        grids_host = tuple(asnumpy(g) if hasattr(g, 'get') else g for g in grids)
        result = spec(*grids_host)
        return xp.asarray(result, dtype=xp.float64)
    raise ValueError(f'unknown V_ext spec: {spec!r}')

def _build_V_ext_on_device(p: TriadParams, r2, xp):
    """Build V_ext directly on the xp backend (cupy or numpy).

    Takes pre-computed r2 = sum(grid_i^2) so we avoid transferring
    coordinate grids to CPU and back.  Only handles the common
    radially-symmetric potentials that do not need individual grids.
    """
    spec = p.V_ext
    if spec is None:
        return xp.zeros_like(r2)
    if spec == 'harmonic':
        return 0.5 * p.m * p.omega ** 2 * r2
    if spec == 'double_well':
        w2 = (p.L / 8.0) ** 2
        return 0.05 * (r2 - w2) ** 2
    if spec == 'gaussian_bump':
        return -2.0 * xp.exp(-r2 / 2.0)
    
    from runtime.backend import asnumpy
    if p.D == 1:
        grids_np = np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
    elif p.D == 2:
        xs_np = np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
        grids_np = tuple(np.meshgrid(xs_np, xs_np, indexing='ij'))
    else:
        xs_np = np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
        grids_np = tuple(np.meshgrid(xs_np, xs_np, xs_np, indexing='ij'))
    V_host = _build_V_ext(p, grids_np)
    return xp.asarray(V_host)

def _build_absorbing_mask(N: int, L: float, bc_width_frac: float) -> np.ndarray:
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    edge = bc_width_frac * L / 2
    mask = np.ones(N)
    near_left = x < -L / 2 + edge
    near_right = x > L / 2 - edge
    mask[near_left] = np.cos(np.pi * (x[near_left] - (-L / 2 + edge)) / (2 * edge)) ** 2
    mask[near_right] = np.cos(np.pi * (x[near_right] - (L / 2 - edge)) / (2 * edge)) ** 2
    return mask

def _try_native_c(p, **kwargs):
    """Attempt to delegate to the native C solver via ctypes.

    The native C solver implements the same P1+P2+P3 split-step
    scheme as the Python solver but runs ~1.5x faster for 1D
    because it uses FFTW3 directly without Python overhead.

    Returns (result_dict, True) if delegation succeeded,
    or (None, False) if native path is unavailable.
    """
    backend = getattr(p, 'backend', 'auto')
    
    if backend not in ('cpu', 'numpy'):
        return None, False
    
    if getattr(p, 'D', 1) != 1:
        return None, False
    
    import ctypes, os, time
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        so_path = os.path.join(repo_root, 'native', 'c', 'libtriad_rt.so')
        if not os.path.exists(so_path):
            return None, False
        _lib = ctypes.CDLL(so_path)
    except (OSError, Exception):
        return None, False

    try:
        if not hasattr(_lib, 'triad_solve_1d'):
            return None, False
    except Exception:
        return None, False

    class _Cplx(ctypes.Structure):
        _fields_ = [('re', ctypes.c_double), ('im', ctypes.c_double)]

    class _SolverC(ctypes.Structure):
        _fields_ = [
            ('N', ctypes.c_int32), ('L', ctypes.c_double),
            ('dt', ctypes.c_double), ('T', ctypes.c_double),
            ('hbar', ctypes.c_double), ('m', ctypes.c_double),
            ('omega', ctypes.c_double), ('Lambda', ctypes.c_double),
            ('alpha', ctypes.c_double), ('sigma', ctypes.c_double),
            ('Gamma', ctypes.c_double), ('f_FDT', ctypes.c_double),
            ('M', ctypes.c_int32),
            ('nu', ctypes.POINTER(ctypes.c_double)),
            ('lam', ctypes.POINTER(ctypes.c_double)),
            ('mode', ctypes.c_int32), ('seed', ctypes.c_uint64),
            ('V_ext', ctypes.c_char_p),
            ('D', ctypes.c_int32),
            ('bc', ctypes.c_char_p),
            ('bc_width', ctypes.c_double),
        ]

    class _Result(ctypes.Structure):
        _fields_ = [
            ('psi_final', ctypes.POINTER(_Cplx)),
            ('y_final', ctypes.POINTER(ctypes.c_double)),
            ('density_final', ctypes.POINTER(ctypes.c_double)),
            ('x', ctypes.POINTER(ctypes.c_double)),
            ('dx', ctypes.c_double),
        ]

    _lib.triad_solve_1d.argtypes = [ctypes.POINTER(_SolverC)]
    _lib.triad_solve_1d.restype = _Result
    _lib.triad_solver_result_free.argtypes = [ctypes.POINTER(_Result)]
    _lib.triad_solver_result_free.restype = None

    v_ext_str = None
    if p.V_ext is None:
        v_ext_str = None
    elif callable(p.V_ext):
        v_ext_str = b'custom'  
    else:
        v_ext_str = str(p.V_ext).encode('utf-8')

    nu_arr = (ctypes.c_double * len(p.nu))(*p.nu)
    lam_arr = (ctypes.c_double * len(p.lam))(*p.lam)

    mode_map = {'linear': 0, 'thermal': 1, 'full': 2}
    bc_str = str(p.bc).encode('utf-8')

    sp = _SolverC()
    sp.N = p.N
    sp.L = p.L
    sp.dt = p.dt
    sp.T = p.T
    sp.hbar = p.hbar
    sp.m = p.m
    sp.omega = p.omega
    sp.Lambda = p.Lambda
    sp.alpha = p.alpha
    sp.sigma = p.sigma
    sp.Gamma = p.Gamma
    sp.f_FDT = p.f_FDT
    sp.M = len(p.nu)
    sp.nu = nu_arr
    sp.lam = lam_arr
    sp.mode = mode_map.get(p.mode, 2)
    sp.seed = p.seed if p.seed else 0
    sp.V_ext = v_ext_str
    sp.D = 1
    sp.bc = bc_str
    sp.bc_width = p.bc_width

    try:
        result = _lib.triad_solve_1d(sp)

        n = p.N
        psi = np.empty(n, dtype=np.complex128)
        for i in range(n):
            psi[i] = complex(result.psi_final[i].re, result.psi_final[i].im)
        x = np.array([result.x[i] for i in range(n)], dtype=np.float64)
        density = np.array([result.density_final[i] for i in range(n)], dtype=np.float64)
        dx = result.dx

        _lib.triad_solver_result_free(ctypes.byref(result))

        t_arr = np.array([0.0, p.T], dtype=np.float64)
        return dict(
            t=t_arr, x=x, dx=dx,
            density=np.array([density, density]),
            psi_final=psi,
            y_final=np.zeros((len(p.nu), n)),
            params=p,
        ), True
    except Exception:
        return None, False

def _try_gpu_fused(p, **kwargs):
    """Attempt to delegate to the GPU-fused solver.

    Returns (result_dict, True) if delegation succeeded,
    or (None, False) if fused path is unavailable or disabled.
    """
    backend = getattr(p, 'backend', 'auto')
    if backend == 'cpu':
        return None, False
    from runtime.backend import get_xp, cuda_available
    if not cuda_available():
        return None, False
    xp = get_xp(backend)
    
    try:
        import cupy as _cp
        if xp is not _cp:
            return None, False
    except ImportError:
        return None, False
    
    from runtime.ml.gpu_fused import integrate_gpu_fused
    result = integrate_gpu_fused(p, auto_halve_dt=True, **kwargs)
    return result, True

def integrate(p: TriadParams, psi0: Optional[np.ndarray]=None, y0: Optional[np.ndarray]=None, auto_halve_dt: bool=True, noise_provider: Optional[callable]=None, record_y: bool=False) -> dict:
    p = _maybe_halve_dt(p, auto_halve_dt)
    
    if getattr(p, 'D', 1) == 1:
        native_kwargs = {}
        native_result, delegated = _try_native_c(p, **native_kwargs)
        if delegated:
            return native_result
    
    if getattr(p, 'D', 1) == 1:
        fused_kwargs = {}
        if psi0 is not None:
            fused_kwargs['psi0'] = psi0
        if y0 is not None:
            fused_kwargs['y0'] = y0
        fused_kwargs['noise_provider'] = noise_provider
        fused_kwargs['record_y'] = record_y
        fused_result, delegated = _try_gpu_fused(p, **fused_kwargs)
        if delegated:
            return fused_result
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
    if getattr(p, 'fdt_couple', False) and Gamma_e > 0:
        f_FDT_e = 2.0 * Gamma_e * dx * p.kT / p.hbar
    lam_e_np = eff['lam']
    lam_e = xp.asarray(lam_e_np)
    V_ext = _build_V_ext(p, x, xp=xp)
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
    exptrap = getattr(p, 'step_mode', 'strang') == 'exptrap'
    trap_lambda_val = getattr(p, 'trap_lambda', 0.5)
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
        V_start = V_ext + Lambda_e * rho + V_mem
        if exptrap:
            lam_t = _resolve_trap_lambda(trap_lambda_val, rho, xp)
            psi_pred = psi * xp.exp(-1j * V_start * p.dt / p.hbar)
            rho_pred = xp.abs(psi_pred) ** 2
            V_end = V_ext + Lambda_e * rho_pred + V_mem
            V_tot = (1.0 - lam_t) * V_start + lam_t * V_end
        else:
            V_tot = V_start
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
    
    fused_kwargs = {}
    if psi0 is not None:
        fused_kwargs['psi0'] = psi0
    if y0 is not None:
        fused_kwargs['y0'] = y0
    fused_kwargs['record_y'] = record_y
    fused_kwargs['record_density'] = record_density
    fused_result, delegated = _try_gpu_fused(p, **fused_kwargs)
    if delegated:
        return fused_result
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
    if getattr(p, 'fdt_couple', False) and Gamma_e > 0:
        f_FDT_e = 2.0 * Gamma_e * dx ** 2 * p.kT / p.hbar
    lam_e = xp.asarray(lam_e_np)
    r2_2d = X * X + Y * Y
    V_ext = _build_V_ext_on_device(p, r2_2d, xp)
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
    exptrap_2d = getattr(p, 'step_mode', 'strang') == 'exptrap'
    trap_lambda_2d = getattr(p, 'trap_lambda', 0.5)
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
        V_start = V_ext + Lambda_e * rho + V_mem
        if exptrap_2d:
            lam_t = _resolve_trap_lambda(trap_lambda_2d, rho, xp)
            psi_pred = psi * xp.exp(-1j * V_start * p.dt / p.hbar)
            rho_pred = xp.abs(psi_pred) ** 2
            V_end = V_ext + Lambda_e * rho_pred + V_mem
            V_tot = (1.0 - lam_t) * V_start + lam_t * V_end
        else:
            V_tot = V_start
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
    
    fused_kwargs = {}
    if psi0 is not None:
        fused_kwargs['psi0'] = psi0
    if y0 is not None:
        fused_kwargs['y0'] = y0
    fused_kwargs['record_density'] = record_density
    fused_kwargs['record_y'] = record_y
    fused_result, delegated = _try_gpu_fused(p, **fused_kwargs)
    if delegated:
        return fused_result
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
    if getattr(p, 'fdt_couple', False) and Gamma_e > 0:
        f_FDT_e = 2.0 * Gamma_e * dx ** 3 * p.kT / p.hbar
    lam_e = xp.asarray(lam_e_np)

    r2 = X * X + Y * Y + Z * Z
    V_ext = _build_V_ext_on_device(p, r2, xp)
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
    exptrap_3d = getattr(p, 'step_mode', 'strang') == 'exptrap'
    trap_lambda_3d = getattr(p, 'trap_lambda', 0.5)
    n_steps = int(round(p.T / p.dt))
    peak_density_t = []
    participation_t = []
    t_arr = []
    rec_y = []
    rec_density = []
    for step in range(n_steps + 1):
        rho = xp.abs(psi) ** 2
        if step % max(1, p.record_every) == 0 or step == n_steps:
            peak_density_t.append(float(asnumpy(rho.max())))
            norm2 = float(asnumpy(rho.sum() * dx ** 3))
            ipr_denom = max(float(asnumpy((rho ** 2).sum() * dx ** 3)), 1e-300)
            participation_t.append(float(norm2 * norm2 / ipr_denom))
            t_arr.append(step * p.dt)
            if record_density:
                rec_density.append(asnumpy(rho).copy())
            if record_y:
                rec_y.append(asnumpy(y).copy())
        if step == n_steps:
            break
        psi = xp.fft.ifftn(xp.fft.fftn(psi) * half_lin)
        rho = xp.abs(psi) ** 2
        if M:
            y = ou_decay_half[:, None, None, None] * y + (1.0 - ou_decay_half)[:, None, None, None] * rho
        V_mem = (lam_e[:, None, None, None] * y).sum(axis=0) if M else 0.0
        V_start = V_ext + Lambda_e * rho + V_mem
        if exptrap_3d:
            lam_t = _resolve_trap_lambda(trap_lambda_3d, rho, xp)
            psi_pred = psi * xp.exp(-1j * V_start * p.dt / p.hbar)
            rho_pred = xp.abs(psi_pred) ** 2
            V_end = V_ext + Lambda_e * rho_pred + V_mem
            V_tot = (1.0 - lam_t) * V_start + lam_t * V_end
        else:
            V_tot = V_start
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
    if record_density and rec_density:
        da = np.asarray(rec_density)
        result['density'] = np.transpose(da, (1, 2, 3, 0))
    if record_y and rec_y:
        ya = np.asarray(rec_y)
        result['y_traj'] = np.transpose(ya, (1, 2, 3, 4, 0))
    return result
def _integrate_steps(p: TriadParams, psi0: Optional[np.ndarray]=None,
                     y0: Optional[np.ndarray]=None, auto_halve_dt: bool=True,
                     noise_provider: Optional[callable]=None,
                     record_every: int=0):
    """Generator that yields checkpoint states from the integration loop.

    Each yield produces a dict with keys:
        step, t, psi (numpy), y (numpy), x (numpy), dx (float), params

    The caller decides when to stop consuming.  No change to the split-step
    arithmetic compared to integrate().
    """
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
    if getattr(p, 'fdt_couple', False) and Gamma_e > 0:
        f_FDT_e = 2.0 * Gamma_e * dx * p.kT / p.hbar
    lam_e = xp.asarray(eff['lam'])
    V_ext_host = _build_V_ext(p, np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False))
    V_ext = xp.asarray(V_ext_host)
    if psi0 is None:
        psi = xp.exp(-x ** 2 / 8.0).astype(xp.complex128)
        psi = psi / xp.sqrt((xp.abs(psi) ** 2).sum() * dx)
    else:
        psi = xp.asarray(psi0, dtype=xp.complex128).copy()
    nu_arr = xp.asarray(p.nu, dtype=np.float64)
    M = len(p.nu)
    if y0 is None:
        y = xp.zeros((M, p.N), dtype=np.float64)
    else:
        y = xp.asarray(y0, dtype=np.float64).copy()
    H_lin_k = p.hbar ** 2 * k ** 2 / (2.0 * p.m) + alpha_e * abs_k ** p.sigma
    half_lin = xp.exp(-1j * H_lin_k * p.dt / (2.0 * p.hbar) - Gamma_e * p.dt / (2.0 * p.hbar))
    ou_decay_half = xp.exp(-nu_arr * p.dt * 0.5) if M else None
    noise_amp = float(xp.sqrt(xp.asarray(f_FDT_e * p.dt / dx))) if f_FDT_e > 0 else 0.0
    bc = getattr(p, 'bc', 'periodic')
    if bc == 'absorbing':
        bc_mask = xp.asarray(_build_absorbing_mask(p.N, p.L, getattr(p, 'bc_width', 0.15)))
    else:
        bc_mask = None
    exptrap = getattr(p, 'step_mode', 'strang') == 'exptrap'
    trap_lambda_val = getattr(p, 'trap_lambda', 0.5)
    rec_every = max(1, record_every) if record_every > 0 else max(1, p.record_every)
    n_steps = int(round(p.T / p.dt))

    for step in range(n_steps + 1):
        if step % rec_every == 0 or step == n_steps:
            yield {
                'step': step,
                't': step * p.dt,
                'psi': asnumpy(psi).copy(),
                'y': asnumpy(y).copy(),
                'x': asnumpy(x),
                'dx': dx,
                'params': p,
            }
        if step == n_steps:
            break
        psi = xp.fft.ifft(xp.fft.fft(psi) * half_lin)
        rho = xp.abs(psi) ** 2
        if M:
            y = ou_decay_half[:, None] * y + (1.0 - ou_decay_half)[:, None] * rho
        V_mem = (lam_e[:, None] * y).sum(axis=0) if M else xp.asarray(0.0)
        V_start = V_ext + Lambda_e * rho + V_mem
        if exptrap:
            lam_t = _resolve_trap_lambda(trap_lambda_val, rho, xp)
            psi_pred = psi * xp.exp(-1j * V_start * p.dt / p.hbar)
            rho_pred = xp.abs(psi_pred) ** 2
            V_end = V_ext + Lambda_e * rho_pred + V_mem
            V_tot = (1.0 - lam_t) * V_start + lam_t * V_end
        else:
            V_tot = V_start
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

def integrate_adaptive(p: TriadParams, stop_fn: Optional[callable]=None,
                       max_T: Optional[float]=None, psi0=None, y0=None,
                       auto_halve_dt: bool=True, noise_provider=None,
                       checkpoint_every: int=50) -> dict:
    """Adaptive integration: consume _integrate_steps until stop_fn returns True.

    Parameters
    ----------
    stop_fn : callable(list[checkpoint_dict]) -> bool
        Receives the full history of checkpoints so far.  Return True to stop.
        If None, runs to max_T.
    max_T : float or None
        Hard upper bound on integration time.  Defaults to p.T.
    checkpoint_every : int
        Steps between yields from the generator (controls checkpoint density).

    Returns
    -------
    dict with keys: stopped_at_T, steps_taken, psi_final, y_final, x, dx,
                    params, history (list of checkpoint dicts)
    """
    if max_T is not None:
        p = TriadParams(**{**p.__dict__, 'T': max_T})
    gen = _integrate_steps(p, psi0=psi0, y0=y0, auto_halve_dt=auto_halve_dt,
                           noise_provider=noise_provider,
                           record_every=checkpoint_every)
    history = []
    final = None
    for chk in gen:
        history.append(chk)
        final = chk
        if stop_fn is not None and stop_fn(history):
            break
    if final is None:
        final = {'psi': np.zeros(p.N, dtype=np.complex128),
                 'y': np.zeros((len(p.nu), p.N)),
                 'x': np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False),
                 'dx': p.L / p.N, 'step': 0, 't': 0.0, 'params': p}
    return {
        'stopped_at_T': final['t'],
        'steps_taken': final['step'],
        'psi_final': final['psi'],
        'y_final': final['y'],
        'x': final['x'],
        'dx': final['dx'],
        'params': final['params'],
        'history': history,
    }

if __name__ == '__main__':
    for mode in ('linear', 'thermal', 'full'):
        out = integrate(TriadParams(mode=mode, T=4.0, seed=0))
        norm_t = out['density'].sum(axis=0) * out['dx']
        print(f'mode={mode:8s}  norm in [{norm_t.min():.6f}, {norm_t.max():.6f}]')
