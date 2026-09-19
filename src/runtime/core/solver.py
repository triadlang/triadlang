from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import sys
from typing import Union

from runtime.env import env as _env
from runtime.backend import asnumpy, copy_array
from triad import ntri as np

VExtSpec = Union[None, str, Callable[[np.ndarray], np.ndarray]]

_NATIVE_LIB = None
_NATIVE_LIB_LOCK = __import__('threading').Lock()

def _default_D() -> int:
    d = _env('TRIADLANG_D', 0)
    if d == 0:
        from runtime.backend import cuda_available
        d = 3 if cuda_available() else 1
    if d < 1:
        raise ValueError(f'TRIADLANG_D={d} invalid (use D>=1 or 0=auto)')
    return d

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
    mode: str = 'triad'

    seed: int = 0
    record_every: int = 4
    D: int = field(default_factory=_default_D)
    backend: str = 'auto'
    bc: str = 'periodic'
    bc_width: float = 0.15
    step_mode: str = 'strang'
    trap_lambda: object = 0.5
    fdt_couple: bool = True
    kT: float = 1.0
    init_sigma: float = 2.0
    init_k0: tuple = (0.0, 0.0, 0.0)
    init: str = 'gaussian'

    def __post_init__(self):
        if self.N < 2:
            raise ValueError(f'TriadParams.N must be >= 2, got {self.N}')
        if self.dt <= 0:
            raise ValueError(f'TriadParams.dt must be > 0, got {self.dt}')
        if self.T < 0:
            raise ValueError(f'TriadParams.T must be >= 0, got {self.T}')
        if self.L <= 0:
            raise ValueError(f"TriadParams.L must be > 0, got {self.L}")
        nu = tuple(self.nu)
        lam = tuple(self.lam)
        if len(nu) != len(lam):
            raise ValueError(f'triad rule: len(nu) ({len(nu)}) must equal len(lam) ({len(lam)})')
        if len(nu) < 3:
            raise ValueError(f'triad rule: P2 memory field must have at least 3 time-scales (p1/p2/p3), got {len(nu)}')
        if self.mode != 'triad':
            raise ValueError(f"mode={self.mode!r} does not exist; the Triad Triad is 'triad' only")
        if self.Gamma <= 0:
            raise ValueError(f'triad rule: P3 bath requires Gamma > 0, got {self.Gamma}')
        if self.f_FDT <= 0:
            raise ValueError(f'triad rule: P3 bath requires f_FDT > 0 to keep η active, got {self.f_FDT}')
        if getattr(self, 'kT', 1.0) <= 0:
            raise ValueError(f'triad rule: P3 bath requires kT > 0 to keep η active, got {getattr(self, "kT", 1.0)}')

def _validate_triad_params(p: TriadParams) -> None:
    nu = tuple(p.nu) if not isinstance(p.nu, tuple) else p.nu
    lam = tuple(p.lam) if not isinstance(p.lam, tuple) else p.lam
    if len(nu) != len(lam):
        raise ValueError(f'triad rule: len(nu) ({len(nu)}) must equal len(lam) ({len(lam)})')
    if len(nu) < 3:
        raise ValueError(f'triad rule: P2 memory field must have at least 3 time-scales (p1/p2/p3), got {len(nu)}')
    if p.mode != 'triad':
        raise ValueError(f"mode={p.mode!r} does not exist; the Triad Triad is 'triad' only")
    if p.Gamma <= 0:
        raise ValueError(f'triad rule: P3 bath requires Gamma > 0, got {p.Gamma}')
    if p.f_FDT <= 0:
        raise ValueError(f'triad rule: P3 bath requires f_FDT > 0 to keep η active, got {p.f_FDT}')
    if getattr(p, 'kT', 1.0) <= 0:
        raise ValueError(f'triad rule: P3 bath requires kT > 0 to keep η active, got {getattr(p, "kT", 1.0)}')


_MIN_FDT = 1e-12


def _noise_amplitude(f_FDT_e: float, scale: float) -> float:
    return float(np.sqrt(np.asarray(max(float(f_FDT_e), _MIN_FDT) * scale)))

def _effective_params(p: TriadParams,
                        dx: float | None = None,
                        D: int | None = None) -> dict:
    if p.mode != 'triad':
        raise ValueError(
            f"mode={p.mode!r} does not exist; the Triad Triad is "
            f"'triad' only")
    lam_arr = np.asarray(p.lam, dtype=np.float64)
    f_FDT_e = float(p.f_FDT)
    if dx is not None and D is not None:
        if getattr(p, 'fdt_couple', False) and p.Gamma > 0.0:
            f_FDT_e = 2.0 * p.Gamma * (dx ** D) * getattr(p, 'kT', 1.0) / p.hbar
    return dict(
        Lambda=p.Lambda, alpha=p.alpha, Gamma=p.Gamma,
        f_FDT=p.f_FDT, f_FDT_e=f_FDT_e, lam=lam_arr,
    )

def _resolve_trap_lambda(trap_lambda, rho, xp) -> float:
    if callable(trap_lambda):

        return float(trap_lambda(asnumpy(rho)))
    return float(trap_lambda)

def _build_V_ext(p: TriadParams, x_or_grid, xp=None):

    if xp is None:
        xp = np
    spec = p.V_ext

    is_1d = False
    try:
        if hasattr(x_or_grid, 'ndim') and x_or_grid.ndim == 1:
            is_1d = True
    except (AttributeError, TypeError):
        is_1d = False
    if is_1d:
        x = x_or_grid
        if spec is None:
            return xp.zeros_like(x)
        if spec == 'harmonic':
            return 0.5 * p.m * p.omega ** 2 * x ** 2
        if spec == 'double_well':
            w2 = (p.L / 8.0) ** 2
            return 0.05 * (x ** 2 - w2) ** 2
        if spec == 'gaussian_bump':
            return -2.0 * xp.exp(-x ** 2 / 2.0)
        if spec == 'ramp':
            return 0.05 * x
        if spec == 'lattice':
            k0 = 2.0 * xp.pi / (p.L / 4.0)
            return 0.5 * xp.cos(k0 * x)
        if callable(spec):
            result = spec(asnumpy(x) if hasattr(x, 'get') else x)
            return xp.asarray(result, dtype=xp.float64)
        raise ValueError(f'unknown V_ext spec: {spec!r}')
    grids = x_or_grid
    if spec is None:
        return xp.zeros_like(grids[0])
    if spec == 'harmonic':
        r2 = sum(g * g for g in grids)
        return 0.5 * p.m * p.omega ** 2 * r2
    if spec == 'double_well':
        r2 = sum(g * g for g in grids)
        w2 = (p.L / 8.0) ** 2
        return 0.05 * (r2 - w2) ** 2
    if spec == 'gaussian_bump':
        r2 = sum(g * g for g in grids)
        return -2.0 * xp.exp(-r2 / 2.0)
    if spec == 'ramp':
        return 0.05 * grids[0]
    if spec == 'lattice':
        k0 = 2.0 * xp.pi / (p.L / 4.0)
        return 0.5 * sum(xp.cos(k0 * g) for g in grids)
    if callable(spec):
        grids_host = tuple(asnumpy(g) if hasattr(g, 'get') else g for g in grids)
        result = spec(*grids_host)
        return xp.asarray(result, dtype=xp.float64)
    raise ValueError(f'unknown V_ext spec: {spec!r}')

def _build_V_ext_on_device(p: TriadParams, r2, xp):

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

    if p.D == 1:
        grids_np = np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
    elif p.D == 2:
        xs_np = np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
        grids_np = tuple(np.meshgrid(xs_np, xs_np, indexing='ij'))
    elif p.D == 3:
        xs_np = np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
        grids_np = tuple(np.meshgrid(xs_np, xs_np, xs_np, indexing='ij'))
    else:
        xs_np = np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
        grids_np = tuple(np.meshgrid(*([xs_np] * p.D), indexing='ij'))
    V_host = _build_V_ext(p, grids_np)
    return xp.asarray(V_host)

def _build_V_ext_nd_on_device(p: TriadParams, grids, r2, xp):
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
    if spec == 'ramp':
        return 0.05 * grids[0]
    if spec == 'lattice':
        k0 = 2.0 * xp.pi / (p.L / 4.0)
        return 0.5 * sum(xp.cos(k0 * g) for g in grids)
    if callable(spec):
        from runtime.backend import asnumpy
        target_shape = r2.shape
        grids_host = tuple(asnumpy(xp.broadcast_to(g, target_shape)) for g in grids)
        return xp.asarray(spec(*grids_host), dtype=xp.float64)
    raise ValueError(f'unknown V_ext spec: {spec!r}')

def _axis_views(vec, D: int):
    N = int(vec.shape[0])
    for axis in range(D):
        shape = [1] * D
        shape[axis] = N
        yield vec.reshape(tuple(shape))

def _sum_axis_squares(vec, D: int, shape, xp):
    acc = xp.zeros(shape, dtype=xp.float64)
    for v in _axis_views(vec, D):
        acc = acc + v * v
    return acc

def _sum_weighted_axis_views(weights, vec, D: int, shape, xp):
    acc = xp.zeros(shape, dtype=xp.float64)
    for i, v in enumerate(_axis_views(vec, D)):
        acc = acc + float(weights[i]) * v
    return acc

def _build_absorbing_mask(N: int, L: float, bc_width_frac: float) -> np.ndarray:
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    border = bc_width_frac * L / 2
    mask = np.ones(N)
    near_left = x < -L / 2 + border
    near_right = x > L / 2 - border
    mask[near_left] = np.cos(np.pi * (x[near_left] - (-L / 2 + border)) / (2 * border)) ** 2
    mask[near_right] = np.cos(np.pi * (x[near_right] - (L / 2 - border)) / (2 * border)) ** 2
    return mask

def _native_lib():
    with _NATIVE_LIB_LOCK:
        import ctypes
        import glob
        import os
        global _NATIVE_LIB
        if _NATIVE_LIB is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            if sys.platform == 'darwin':
                lib_name = 'libtriad_rt.dylib'
            elif sys.platform == 'win32':
                lib_name = 'triad_rt.dll'
            else:
                lib_name = 'libtriad_rt.so'
            so_path = os.path.join(repo_root, 'native', 'c', lib_name)
            if not os.path.exists(so_path):
                raise RuntimeError(
                    f'native runtime missing: {so_path}; compile com: make -C native/c')
            try:
                lib = ctypes.CDLL(so_path)
            except OSError as e:
                err = str(e)
                lib = None
                if 'cublas' in err or 'cuda' in err:
                    roots = [os.environ.get('CUDA_HOME'), '/opt/cuda', '/usr/local/cuda']
                    if sys.platform == 'win32':
                        pf = os.environ.get('ProgramFiles', r'C:\Program Files')
                        roots = [os.environ.get('CUDA_PATH')] + sorted(
                            glob.glob(os.path.join(pf, 'NVIDIA GPU Computing Toolkit', 'CUDA', 'v*')))
                    for root in roots:
                        if not root:
                            continue
                        if sys.platform == 'win32':
                            patterns = [os.path.join(root, 'bin', 'cublasLt64_*.dll'),
                                        os.path.join(root, 'bin', 'cublas64_*.dll')]
                        else:
                            patterns = [os.path.join(root, 'lib64', 'libcublasLt.so.*')]
                        for pat in patterns:
                            for cand in sorted(glob.glob(pat)):
                                try:
                                    ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
                                    lib = ctypes.CDLL(so_path)
                                    break
                                except OSError:
                                    continue
                            if lib is not None:
                                break
                        if lib is not None:
                            break
                if lib is None:
                    raise RuntimeError(f'native runtime failed to load: {err}')
            if not hasattr(lib, 'triad_solve_1d'):
                raise RuntimeError(f'native runtime missing triad_solve_1d: {so_path}')
            _NATIVE_LIB = lib
        return _NATIVE_LIB

def _try_native_c(p, **kwargs):

    import logging as _logging
    _log = _logging.getLogger(__name__)

    backend = getattr(p, 'backend', 'auto')
    if backend not in ('cpu', 'numpy'):
        _log.debug('native C refused: backend=%r not in cpu/numpy', backend)
        return None, False

    D = p.D
    if D < 1:
        _log.debug('native C refused: D=%r invalid', D)
        return None, False

    if getattr(p, 'step_mode', 'strang') == 'exptrap':
        _log.debug('native C refused: step_mode=exptrap has no C path')
        return None, False

    spec = p.V_ext
    if callable(spec):
        _log.debug('native C refused: callable V_ext has no C path')
        return None, False
    if spec is not None:
        spec_name = str(spec)
        if D == 1:
            if spec_name != 'harmonic':
                _log.debug('native C refused: D=1 V_ext=%r unsupported in C', spec_name)
                return None, False
        elif spec_name not in ('harmonic', 'double_well', 'gaussian_bump', 'ramp', 'lattice'):
            _log.debug('native C refused: V_ext=%r unsupported in C', spec_name)
            return None, False

    import ctypes
    try:
        _lib = _native_lib()
    except (OSError, RuntimeError) as _e:
        _log.debug('native C refused: library load failed: %s', _e)
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
            ('fdt_couple', ctypes.c_int32),
            ('kT', ctypes.c_double),
            ('init_sigma', ctypes.c_double),
            ('init_k0', ctypes.c_double * 3),
            ('init_mode', ctypes.c_int32),
            ('record_every', ctypes.c_int32),
            ('init_k0_ext', ctypes.POINTER(ctypes.c_double)),
            ('init_k0_ext_len', ctypes.c_int32),
        ]

    class _Result1D(ctypes.Structure):
        _fields_ = [
            ('psi_final', ctypes.POINTER(_Cplx)),
            ('y_final', ctypes.POINTER(ctypes.c_double)),
            ('density_final', ctypes.POINTER(ctypes.c_double)),
            ('x', ctypes.POINTER(ctypes.c_double)),
            ('dx', ctypes.c_double),
            ('density_t', ctypes.POINTER(ctypes.c_double)),
            ('n_records', ctypes.c_int32),
        ]

    class _Result2D(ctypes.Structure):
        _fields_ = [
            ('psi_final', ctypes.POINTER(_Cplx)),
            ('y_final', ctypes.POINTER(ctypes.c_double)),
            ('density_final', ctypes.POINTER(ctypes.c_double)),
            ('x', ctypes.POINTER(ctypes.c_double)),
            ('dx', ctypes.c_double),
        ]

    class _Result3D(ctypes.Structure):
        _fields_ = [
            ('psi_final', ctypes.POINTER(_Cplx)),
            ('y_final', ctypes.POINTER(ctypes.c_double)),
            ('density_final', ctypes.POINTER(ctypes.c_double)),
            ('x', ctypes.POINTER(ctypes.c_double)),
            ('dx', ctypes.c_double),
            ('peak_t', ctypes.POINTER(ctypes.c_double)),
            ('participation_t', ctypes.POINTER(ctypes.c_double)),
            ('n_records', ctypes.c_int32),
        ]

    class _ResultND(ctypes.Structure):
        _fields_ = [
            ('psi_final', ctypes.POINTER(_Cplx)),
            ('y_final', ctypes.POINTER(ctypes.c_double)),
            ('density_final', ctypes.POINTER(ctypes.c_double)),
            ('x', ctypes.POINTER(ctypes.c_double)),
            ('dx', ctypes.c_double),
            ('peak_t', ctypes.POINTER(ctypes.c_double)),
            ('participation_t', ctypes.POINTER(ctypes.c_double)),
            ('n_records', ctypes.c_int32),
        ]

    v_ext_str = None
    if p.V_ext is None:
        v_ext_str = None
    elif callable(p.V_ext):
        v_ext_str = b'custom'
    else:
        v_ext_str = str(p.V_ext).encode('utf-8')

    nu_arr = (ctypes.c_double * len(p.nu))(*p.nu)
    lam_arr = (ctypes.c_double * len(p.lam))(*p.lam)
    bc_str = str(p.bc).encode('utf-8')

    sp = _SolverC()
    sp.N = p.N; sp.L = p.L; sp.dt = p.dt; sp.T = p.T
    sp.hbar = p.hbar; sp.m = p.m; sp.omega = p.omega
    sp.Lambda = p.Lambda; sp.alpha = p.alpha; sp.sigma = p.sigma
    sp.Gamma = p.Gamma; sp.f_FDT = p.f_FDT
    sp.M = len(p.nu); sp.nu = nu_arr; sp.lam = lam_arr
    sp.mode = 2
    sp.seed = p.seed if p.seed else 0
    sp.V_ext = v_ext_str; sp.D = D
    sp.bc = bc_str; sp.bc_width = p.bc_width
    sp.fdt_couple = 1 if getattr(p, 'fdt_couple', False) else 0
    sp.kT = getattr(p, 'kT', 1.0)
    sp.init_sigma = getattr(p, 'init_sigma', 2.0)
    k0 = getattr(p, 'init_k0', (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0)
    k0 = tuple(k0) + (0.0, 0.0, 0.0)
    sp.init_k0 = (ctypes.c_double * 3)(k0[0], k0[1], k0[2])
    sp.init_mode = 1 if getattr(p, 'init', 'gaussian') == 'chaos' else 0
    rec_every = max(1, int(getattr(p, 'record_every', 4)))
    sp.record_every = rec_every
    k0_ext = k0 + (0.0,) * max(0, D - len(k0))
    k0_ext_arr = (ctypes.c_double * D)(*k0_ext[:D])
    sp.init_k0_ext = k0_ext_arr
    sp.init_k0_ext_len = D

    def _rec_times():
        dt_eff = p.dt
        if abs(p.Lambda) >= 4.0 and dt_eff > 0.0025:
            dt_eff = 0.0025
        n_steps = int(round(p.T / dt_eff))
        idx = list(range(0, n_steps + 1, rec_every))
        if idx[-1] != n_steps:
            idx.append(n_steps)
        return idx, dt_eff

    def _cplx_to_numpy(ptr, size, shape):

        dbuf = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))
        flat = np.ctypeslib.as_array(dbuf, shape=(size * 2,)).copy()
        return (flat[0::2] + 1j * flat[1::2]).reshape(shape)

    def _double_to_numpy(ptr, shape):
        return np.ctypeslib.as_array(ptr, shape=shape).copy()

    try:
        n = p.N
        M = len(p.nu)
        t_arr = np.array([0.0, p.T], dtype=np.float64)

        if D == 1:
            _lib.triad_solve_1d.argtypes = [ctypes.POINTER(_SolverC)]
            _lib.triad_solve_1d.restype = _Result1D
            _lib.triad_solver_result_free.argtypes = [ctypes.POINTER(_Result1D)]
            _lib.triad_solver_result_free.restype = None
            r = _lib.triad_solve_1d(sp)
            nrec = int(r.n_records)
            rec_idx, dt_eff = _rec_times()
            if nrec < 1 or not r.density_t or len(rec_idx) != nrec:
                _lib.triad_solver_result_free(ctypes.byref(r))
                return None, False
            psi = _cplx_to_numpy(r.psi_final, n, (n,))
            x = _double_to_numpy(r.x, (n,))
            density = _double_to_numpy(r.density_t, (nrec, n)).T
            y_final = _double_to_numpy(r.y_final, (M, n)) if M else np.zeros((M, n))
            dx = r.dx
            _lib.triad_solver_result_free(ctypes.byref(r))
            return dict(
                t=np.asarray(rec_idx, dtype=np.float64) * dt_eff,
                x=x, dx=dx,
                density=density,
                psi_final=psi,
                y_final=y_final,
                params=p,
            ), True

        elif D == 2:
            if not hasattr(_lib, 'triad_solve_2d'):
                return None, False
            _lib.triad_solve_2d.argtypes = [ctypes.POINTER(_SolverC)]
            _lib.triad_solve_2d.restype = _Result2D
            _lib.triad_solver_result_2d_free.argtypes = [ctypes.POINTER(_Result2D)]
            _lib.triad_solver_result_2d_free.restype = None
            r = _lib.triad_solve_2d(sp)
            psi = _cplx_to_numpy(r.psi_final, n * n, (n, n))
            x = _double_to_numpy(r.x, (n,))
            density = _double_to_numpy(r.density_final, (n, n))
            y_final = _double_to_numpy(r.y_final, (M, n, n)) if M else np.zeros((M, n, n))
            dx = r.dx
            _lib.triad_solver_result_2d_free(ctypes.byref(r))
            return dict(
                t=np.array([p.T], dtype=np.float64), x=x, dx=dx,
                density=density[..., None],
                psi_final=psi,
                y_final=y_final,
                params=p,
            ), True

        elif D == 3:
            if not hasattr(_lib, 'triad_solve_3d'):
                return None, False
            _lib.triad_solve_3d.argtypes = [ctypes.POINTER(_SolverC)]
            _lib.triad_solve_3d.restype = _Result3D
            _lib.triad_solver_result_3d_free.argtypes = [ctypes.POINTER(_Result3D)]
            _lib.triad_solver_result_3d_free.restype = None
            r = _lib.triad_solve_3d(sp)
            psi = _cplx_to_numpy(r.psi_final, n * n * n, (n, n, n))
            x = _double_to_numpy(r.x, (n,))
            density = _double_to_numpy(r.density_final, (n, n, n))
            y_final = _double_to_numpy(r.y_final, (M, n, n, n)) if M else np.zeros((M, n, n, n))
            nrec = int(r.n_records)
            rec_idx, dt_eff = _rec_times()
            if nrec < 1 or len(rec_idx) != nrec:
                _lib.triad_solver_result_3d_free(ctypes.byref(r))
                return None, False
            peak_t = _double_to_numpy(r.peak_t, (nrec,))
            participation_t = _double_to_numpy(r.participation_t, (nrec,))
            t_rec = np.asarray(rec_idx, dtype=np.float64) * dt_eff
            dx = r.dx
            _lib.triad_solver_result_3d_free(ctypes.byref(r))
            return dict(
                t=t_rec, x=x, dx=dx,
                density=density[..., None],
                psi_final=psi,
                y_final=y_final,
                peak_t=peak_t,
                participation_t=participation_t,
                params=p,
            ), True

        else:
            if not hasattr(_lib, 'triad_solve_nd'):
                return None, False
            _lib.triad_solve_nd.argtypes = [ctypes.POINTER(_SolverC)]
            _lib.triad_solve_nd.restype = _ResultND
            _lib.triad_solver_result_nd_free.argtypes = [ctypes.POINTER(_ResultND)]
            _lib.triad_solver_result_nd_free.restype = None
            r = _lib.triad_solve_nd(sp)
            shape = (n,) * D
            total = n ** D
            psi = _cplx_to_numpy(r.psi_final, total, shape)
            x = _double_to_numpy(r.x, (n,))
            dx = r.dx
            y_shape = (M,) + shape
            y_final = _double_to_numpy(r.y_final, y_shape) if M else np.zeros(y_shape)
            density_final = _double_to_numpy(r.density_final, shape)
            nrec = int(r.n_records)
            rec_idx, dt_eff = _rec_times()
            if nrec < 1 or len(rec_idx) != nrec:
                _lib.triad_solver_result_nd_free(ctypes.byref(r))
                return None, False
            peak_t = _double_to_numpy(r.peak_t, (nrec,))
            participation_t = _double_to_numpy(r.participation_t, (nrec,))
            _lib.triad_solver_result_nd_free(ctypes.byref(r))
            return dict(
                t=np.asarray(rec_idx, dtype=np.float64) * dt_eff,
                x=x, dx=dx,
                psi_final=psi,
                y_final=y_final,
                density=density_final[..., None],
                peak_t=peak_t,
                participation_t=participation_t,
                params=p,
            ), True

    except (ValueError, RuntimeError, TypeError):
        return None, False

def _try_gpu_fused(p, **kwargs):

    backend = getattr(p, 'backend', 'auto')
    if backend == 'cpu':
        return None, False
    from runtime.backend import cuda_available, get_xp
    if not cuda_available():
        return None, False
    xp = get_xp(backend)

    try:
        import cupy as _cp
        if xp is not _cp:
            return None, False

        _cp.cuda.cufft
    except (ImportError, Exception):
        return None, False

    from runtime.ml.gpu_fused import integrate_gpu_fused
    result = integrate_gpu_fused(p, auto_halve_dt=True, **kwargs)
    return result, True

def integrate(p: TriadParams, psi0: np.ndarray | None=None, y0: np.ndarray | None=None, auto_halve_dt: bool=True, noise_provider: Callable | None=None, record_y: bool=False) -> dict:
    _validate_triad_params(p)
    if p.D == 2:
        return integrate_2d(p, psi0=psi0, y0=y0, auto_halve_dt=auto_halve_dt, record_y=record_y, record_density=False)
    if p.D == 3:
        return integrate_3d(p, psi0=psi0, y0=y0, auto_halve_dt=auto_halve_dt, record_density=False, record_y=record_y)
    if p.D >= 4:
        return _integrate_highd(p, psi0=psi0, y0=y0, auto_halve_dt=auto_halve_dt, record_density=False, record_y=record_y)
    p = _maybe_halve_dt(p, auto_halve_dt)

    if (psi0 is None and y0 is None and not record_y
            and noise_provider is None):
        native_kwargs = {}
        native_result, delegated = _try_native_c(p, **native_kwargs)
        if delegated:
            return native_result

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
    from runtime.backend import asnumpy, get_xp
    xp = get_xp(getattr(p, 'backend', 'auto'))
    rng = xp.random.default_rng(p.seed)
    x = xp.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
    dx = float(asnumpy(x[1] - x[0]))
    k = 2.0 * xp.pi * xp.fft.fftfreq(p.N, d=dx)
    abs_k = xp.abs(k)
    eff = _effective_params(p, dx=dx, D=1)
    Lambda_e = eff['Lambda']
    alpha_e = eff['alpha']
    Gamma_e = eff['Gamma']
    f_FDT_e = eff['f_FDT_e']
    lam_e_np = eff['lam']
    lam_e = xp.asarray(lam_e_np)
    V_ext = _build_V_ext(p, x, xp=xp)
    if psi0 is None:
        if getattr(p, 'init', 'gaussian') == 'chaos':
            psi = (rng.standard_normal(p.N) + 1j * rng.standard_normal(p.N)).astype(xp.complex128)
        else:
            s = getattr(p, 'init_sigma', 2.0) or 2.0
            psi = xp.exp(-x ** 2 / (2.0 * s * s)).astype(xp.complex128)
            k0 = getattr(p, 'init_k0', (0.0, 0.0, 0.0))
            if k0 and k0[0]:
                psi = psi * xp.exp(1j * k0[0] * x)
        psi = psi / xp.sqrt((xp.abs(psi) ** 2).sum() * dx)
    else:
        psi = copy_array(xp.asarray(psi0, dtype=xp.complex128))
    nu_arr = xp.asarray(p.nu, dtype=xp.float64)
    M = len(p.nu)
    if y0 is None:
        y = xp.zeros((M, p.N), dtype=xp.float64)
    else:
        y = copy_array(xp.asarray(y0, dtype=xp.float64))
    H_lin_k = p.hbar ** 2 * k ** 2 / (2.0 * p.m) + alpha_e * abs_k ** p.sigma
    half_lin = xp.exp(-1j * H_lin_k * p.dt / (2.0 * p.hbar) - Gamma_e * p.dt / (2.0 * p.hbar))
    ou_decay_half = xp.exp(-nu_arr * p.dt * 0.5) if M else None
    noise_amp = _noise_amplitude(f_FDT_e, p.dt / dx)
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

def integrate_2d(p: TriadParams, psi0: np.ndarray | None=None, y0: np.ndarray | None=None, auto_halve_dt: bool=True, record_y: bool=False, record_density: bool=False) -> dict:
    _validate_triad_params(p)
    p = _maybe_halve_dt(p, auto_halve_dt)
    if p.D != 2:
        p = TriadParams(**{**p.__dict__, 'D': 2})

    if psi0 is None and y0 is None and not record_y and not record_density:
        native_result, delegated = _try_native_c(p)
        if delegated:
            return native_result

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
    from runtime.backend import asnumpy, get_xp
    xp = get_xp(getattr(p, 'backend', 'auto'))
    rng = xp.random.default_rng(p.seed)
    xs = xp.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
    dx = float(asnumpy(xs[1] - xs[0]))
    X, Y = xp.meshgrid(xs, xs, indexing='ij')
    kvec = 2.0 * xp.pi * xp.fft.fftfreq(p.N, d=dx)
    KX, KY = xp.meshgrid(kvec, kvec, indexing='ij')
    k2 = KX * KX + KY * KY
    abs_k = xp.sqrt(k2)
    eff = _effective_params(p, dx=dx, D=2)
    Lambda_e, alpha_e, Gamma_e, f_FDT_e, lam_e_np = (eff['Lambda'], eff['alpha'], eff['Gamma'], eff['f_FDT_e'], eff['lam'])
    lam_e = xp.asarray(lam_e_np)
    r2_2d = X * X + Y * Y
    V_ext = _build_V_ext_on_device(p, r2_2d, xp)
    if psi0 is None:
        if getattr(p, 'init', 'gaussian') == 'chaos':
            psi = (rng.standard_normal((p.N, p.N)) + 1j * rng.standard_normal((p.N, p.N))).astype(xp.complex128)
        else:
            s = getattr(p, 'init_sigma', 2.0) or 2.0
            psi = xp.exp(-(X * X + Y * Y) / (2.0 * s * s)).astype(xp.complex128)
            k0 = getattr(p, 'init_k0', (0.0, 0.0, 0.0))
            if k0 and (k0[0] or k0[1]):
                psi = psi * xp.exp(1j * (k0[0] * X + k0[1] * Y))
        psi = psi / xp.sqrt((xp.abs(psi) ** 2).sum() * dx * dx)
    else:
        psi = copy_array(xp.asarray(psi0, dtype=xp.complex128))
    nu_arr = xp.asarray(p.nu, dtype=xp.float64)
    M = len(p.nu)
    if y0 is None:
        y = xp.zeros((M, p.N, p.N), dtype=xp.float64)
    else:
        y = copy_array(xp.asarray(y0, dtype=xp.float64))
    H_lin_k = p.hbar ** 2 * k2 / (2.0 * p.m) + alpha_e * abs_k ** p.sigma
    half_lin = xp.exp(-1j * H_lin_k * p.dt / (2.0 * p.hbar) - Gamma_e * p.dt / (2.0 * p.hbar))
    ou_decay_half = xp.exp(-nu_arr * p.dt * 0.5) if M else None
    noise_amp = _noise_amplitude(f_FDT_e, p.dt / (dx * dx))
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

def integrate_3d(p: TriadParams, psi0: np.ndarray | None=None, y0: np.ndarray | None=None, auto_halve_dt: bool=True, record_density: bool=False, record_y: bool=False) -> dict:
    _validate_triad_params(p)
    p = _maybe_halve_dt(p, auto_halve_dt)
    if p.D != 3:
        p = TriadParams(**{**p.__dict__, 'D': 3})

    if psi0 is None and y0 is None and not record_density and not record_y:
        native_result, delegated = _try_native_c(p)
        if delegated:
            return native_result

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
    from runtime.backend import asnumpy, get_xp
    xp = get_xp(getattr(p, 'backend', 'auto'))
    rng = xp.random.default_rng(p.seed)
    xs = xp.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
    dx = float(asnumpy(xs[1] - xs[0]))
    X, Y, Z = xp.meshgrid(xs, xs, xs, indexing='ij')
    kvec = 2.0 * xp.pi * xp.fft.fftfreq(p.N, d=dx)
    KX, KY, KZ = xp.meshgrid(kvec, kvec, kvec, indexing='ij')
    k2 = KX * KX + KY * KY + KZ * KZ
    abs_k = xp.sqrt(k2)
    eff = _effective_params(p, dx=dx, D=3)
    Lambda_e, alpha_e, Gamma_e, f_FDT_e, lam_e_np = (eff['Lambda'], eff['alpha'], eff['Gamma'], eff['f_FDT_e'], eff['lam'])
    lam_e = xp.asarray(lam_e_np)

    r2 = X * X + Y * Y + Z * Z
    V_ext = _build_V_ext_on_device(p, r2, xp)
    if psi0 is None:
        if getattr(p, 'init', 'gaussian') == 'chaos':
            psi = (rng.standard_normal((p.N, p.N, p.N)) + 1j * rng.standard_normal((p.N, p.N, p.N))).astype(xp.complex128)
        else:
            s = getattr(p, 'init_sigma', 2.0) or 2.0
            psi = xp.exp(-(X * X + Y * Y + Z * Z) / (2.0 * s * s)).astype(xp.complex128)
            k0 = getattr(p, 'init_k0', (0.0, 0.0, 0.0))
            if k0 and (k0[0] or k0[1] or k0[2]):
                psi = psi * xp.exp(1j * (k0[0] * X + k0[1] * Y + k0[2] * Z))
        psi = psi / xp.sqrt((xp.abs(psi) ** 2).sum() * dx ** 3)
    else:
        psi = copy_array(xp.asarray(psi0, dtype=xp.complex128))
    nu_arr = xp.asarray(p.nu, dtype=xp.float64)
    M = len(p.nu)
    if y0 is None:
        y = xp.zeros((M, p.N, p.N, p.N), dtype=xp.float64)
    else:
        y = copy_array(xp.asarray(y0, dtype=xp.float64))
    H_lin_k = p.hbar ** 2 * k2 / (2.0 * p.m) + alpha_e * abs_k ** p.sigma
    half_lin = xp.exp(-1j * H_lin_k * p.dt / (2.0 * p.hbar) - Gamma_e * p.dt / (2.0 * p.hbar))
    ou_decay_half = xp.exp(-nu_arr * p.dt * 0.5) if M else None
    noise_amp = _noise_amplitude(f_FDT_e, p.dt / dx ** 3)
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

def _integrate_highd(p: TriadParams, psi0: np.ndarray | None=None,
                     y0: np.ndarray | None=None,
                     auto_halve_dt: bool=True,
                     record_density: bool=False,
                     record_y: bool=False) -> dict:
    _validate_triad_params(p)
    p = _maybe_halve_dt(p, auto_halve_dt)
    if p.D < 4:
        raise ValueError(f'_integrate_highd() supports D>=4, got D={p.D}')

    if psi0 is None and y0 is None and not record_density and not record_y:
        native_result, delegated = _try_native_c(p)
        if delegated:
            return native_result

    from runtime.backend import asnumpy, get_xp
    xp = get_xp(getattr(p, 'backend', 'auto'))
    rng = xp.random.default_rng(p.seed)
    D = int(p.D)
    N = int(p.N)
    field_shape = (N,) * D
    xs = xp.linspace(-p.L / 2, p.L / 2, N, endpoint=False)
    dx = float(asnumpy(xs[1] - xs[0]))

    grids = tuple(_axis_views(xs, D))
    r2 = _sum_axis_squares(xs, D, field_shape, xp)
    kvec = 2.0 * xp.pi * xp.fft.fftfreq(N, d=dx)
    k2 = _sum_axis_squares(kvec, D, field_shape, xp)
    abs_k = xp.sqrt(k2)

    eff = _effective_params(p, dx=dx, D=D)
    Lambda_e = eff['Lambda']
    alpha_e = eff['alpha']
    Gamma_e = eff['Gamma']
    f_FDT_e = eff['f_FDT_e']
    lam_e = xp.asarray(eff['lam'])
    V_ext = _build_V_ext_nd_on_device(p, grids, r2, xp)

    vol = dx ** D
    if psi0 is None:
        if getattr(p, 'init', 'gaussian') == 'chaos':
            psi = (rng.standard_normal(field_shape) +
                   1j * rng.standard_normal(field_shape)).astype(xp.complex128)
        else:
            s = getattr(p, 'init_sigma', 2.0) or 2.0
            psi = xp.exp(-r2 / (2.0 * s * s)).astype(xp.complex128)
            k0 = tuple(getattr(p, 'init_k0', (0.0,)) or (0.0,))
            k0 = k0 + (0.0,) * max(0, D - len(k0))
            if any(k0[:D]):
                phase = _sum_weighted_axis_views(k0, xs, D, field_shape, xp)
                psi = psi * xp.exp(1j * phase)
        psi = psi / xp.sqrt((xp.abs(psi) ** 2).sum() * vol)
    else:
        psi = copy_array(xp.asarray(psi0, dtype=xp.complex128))
        if tuple(psi.shape) != field_shape:
            raise ValueError(f'psi0 shape {tuple(psi.shape)} incompatible with D={D}, N={N}')

    nu_arr = xp.asarray(p.nu, dtype=xp.float64)
    M = len(p.nu)
    y_shape = (M,) + field_shape
    if y0 is None:
        y = xp.zeros(y_shape, dtype=xp.float64)
    else:
        y = copy_array(xp.asarray(y0, dtype=xp.float64))
        if tuple(y.shape) != y_shape:
            raise ValueError(f'y0 shape {tuple(y.shape)} incompatible with memory shape {y_shape}')

    H_lin_k = p.hbar ** 2 * k2 / (2.0 * p.m) + alpha_e * abs_k ** p.sigma
    half_lin = xp.exp(-1j * H_lin_k * p.dt / (2.0 * p.hbar) -
                      Gamma_e * p.dt / (2.0 * p.hbar))
    ou_decay_half = xp.exp(-nu_arr * p.dt * 0.5) if M else None
    noise_amp = _noise_amplitude(f_FDT_e, p.dt / vol)

    bc = getattr(p, 'bc', 'periodic')
    if bc == 'absorbing':
        mask_1d = xp.asarray(_build_absorbing_mask(N, p.L, getattr(p, 'bc_width', 0.15)))
        bc_mask = xp.ones(field_shape, dtype=xp.float64)
        for v in _axis_views(mask_1d, D):
            bc_mask = bc_mask * v
    else:
        bc_mask = None

    exptrap = getattr(p, 'step_mode', 'strang') == 'exptrap'
    trap_lambda_val = getattr(p, 'trap_lambda', 0.5)
    n_steps = int(round(p.T / p.dt))
    rec_every = max(1, p.record_every)
    lam_shape = (M,) + (1,) * D

    peak_density_t = []
    participation_t = []
    t_arr = []
    rec_density = []
    rec_y = []

    for step in range(n_steps + 1):
        rho = xp.abs(psi) ** 2
        if step % rec_every == 0 or step == n_steps:
            peak_density_t.append(float(asnumpy(rho.max())))
            norm2 = float(asnumpy(rho.sum() * vol))
            ipr_denom = max(float(asnumpy((rho ** 2).sum() * vol)), 1e-300)
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
            decay = ou_decay_half.reshape(lam_shape)
            y = decay * y + (1.0 - decay) * rho
        V_mem = (lam_e.reshape(lam_shape) * y).sum(axis=0) if M else xp.asarray(0.0)
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
            decay = ou_decay_half.reshape(lam_shape)
            y = decay * y + (1.0 - decay) * rho
        xi = rng.standard_normal(field_shape)
        xip = rng.standard_normal(field_shape)
        psi = psi + noise_amp * (xi + 1j * xip) / xp.sqrt(2.0)
        psi = xp.fft.ifftn(xp.fft.fftn(psi) * half_lin)
        if bc_mask is not None:
            psi = psi * bc_mask

    result = {
        't': np.asarray(t_arr),
        'x': asnumpy(xs),
        'dx': dx,
        'psi_final': asnumpy(psi),
        'y_final': asnumpy(y),
        'peak_t': np.asarray(peak_density_t),
        'participation_t': np.asarray(participation_t),
        'params': p,
    }
    if record_density and rec_density:
        result['density'] = np.moveaxis(np.asarray(rec_density), 0, -1)
    if record_y and rec_y:
        result['y_traj'] = np.moveaxis(np.asarray(rec_y), 0, -1)
    return result

def integrate_nd(p: TriadParams, **kwargs) -> dict:

    _validate_triad_params(p)
    if p.D == 1:
        return integrate(p, **kwargs)
    if p.D == 2:
        return integrate_2d(p, **kwargs)
    if p.D == 3:
        return integrate_3d(p, **kwargs)
    if p.D >= 4:
        return _integrate_highd(p, **kwargs)
    raise ValueError(f'D={p.D} invalid (use D>=1)')

def _integrate_steps(p: TriadParams, psi0: np.ndarray | None=None,
                     y0: np.ndarray | None=None, auto_halve_dt: bool=True,
                     noise_provider: Callable | None=None,
                     record_every: int=0):
    _validate_triad_params(p)

    if p.D != 1:
        raise ValueError(
            f'_integrate_steps() is 1D but received D={p.D}')
    p = _maybe_halve_dt(p, auto_halve_dt)
    from runtime.backend import asnumpy, get_xp
    xp = get_xp(getattr(p, 'backend', 'auto'))
    rng = xp.random.default_rng(p.seed)
    x = xp.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
    dx = float(asnumpy(x[1] - x[0]))
    k = 2.0 * xp.pi * xp.fft.fftfreq(p.N, d=dx)
    abs_k = xp.abs(k)
    eff = _effective_params(p, dx=dx, D=1)
    Lambda_e = eff['Lambda']
    alpha_e = eff['alpha']
    Gamma_e = eff['Gamma']
    f_FDT_e = eff['f_FDT_e']
    lam_e = xp.asarray(eff['lam'])
    V_ext_host = _build_V_ext(p, np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False))
    V_ext = xp.asarray(V_ext_host)
    if psi0 is None:
        if getattr(p, 'init', 'gaussian') == 'chaos':
            psi = (rng.standard_normal(p.N) + 1j * rng.standard_normal(p.N)).astype(xp.complex128)
        else:
            s = getattr(p, 'init_sigma', 2.0) or 2.0
            psi = xp.exp(-x ** 2 / (2.0 * s * s)).astype(xp.complex128)
            k0 = getattr(p, 'init_k0', (0.0, 0.0, 0.0))
            if k0 and k0[0]:
                psi = psi * xp.exp(1j * k0[0] * x)
        psi = psi / xp.sqrt((xp.abs(psi) ** 2).sum() * dx)
    else:
        psi = copy_array(xp.asarray(psi0, dtype=xp.complex128))
    nu_arr = xp.asarray(p.nu, dtype=np.float64)
    M = len(p.nu)
    if y0 is None:
        y = xp.zeros((M, p.N), dtype=np.float64)
    else:
        y = copy_array(xp.asarray(y0, dtype=np.float64))
    H_lin_k = p.hbar ** 2 * k ** 2 / (2.0 * p.m) + alpha_e * abs_k ** p.sigma
    half_lin = xp.exp(-1j * H_lin_k * p.dt / (2.0 * p.hbar) - Gamma_e * p.dt / (2.0 * p.hbar))
    ou_decay_half = xp.exp(-nu_arr * p.dt * 0.5) if M else None
    noise_amp = _noise_amplitude(f_FDT_e, p.dt / dx)
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
        if noise_provider is not None:
            psi = psi + xp.asarray(noise_provider(step, p.dt, p.N, dx, f_FDT_e))
        else:
            xi = rng.standard_normal(p.N)
            xip = rng.standard_normal(p.N)
            psi = psi + noise_amp * (xi + 1j * xip) / xp.sqrt(2.0)
        psi = xp.fft.ifft(xp.fft.fft(psi) * half_lin)
        if bc_mask is not None:
            psi = psi * bc_mask

def integrate_adaptive(p: TriadParams, stop_fn: Callable | None=None,
                       max_T: float | None=None, psi0=None, y0=None,
                       auto_halve_dt: bool=True, noise_provider=None,
                       checkpoint_every: int=50) -> dict:
    _validate_triad_params(p)

    if p.D != 1:
        raise ValueError(
            f'integrate_adaptive() is 1D but received D={p.D}')
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
    out = integrate(TriadParams(T=4.0, seed=0, D=1))
    norm_t = out['density'].sum(axis=0) * out['dx']
    print(f'norm in [{norm_t.min():.6f}, {norm_t.max():.6f}]')
