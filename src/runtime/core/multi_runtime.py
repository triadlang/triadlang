from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

from runtime.backend import asnumpy, get_xp
from runtime.core.solver import (
    TriadParams,
    _effective_params,
    _resolve_trap_lambda,
    _validate_triad_params,
)
from triad import ntri as np


def _project_to_shape(src_rho: np.ndarray, dst_shape: tuple) -> np.ndarray:
    src_D = src_rho.ndim
    dst_D = len(dst_shape)
    if src_D == dst_D:
        return src_rho
    if src_D == 3 and dst_D == 1:
        return src_rho.sum(axis=(1, 2))
    if src_D == 2 and dst_D == 1:
        return src_rho.sum(axis=1)
    if src_D == 1 and dst_D == 3:
        N = src_rho.shape[0]
        env = np.exp(-(np.arange(N) - N // 2) ** 2 / (2 * (N / 8) ** 2))
        env = env / env.sum()
        proj_2d = src_rho[:, None] * env[None, :]
        proj_3d = proj_2d[:, :, None] * env[None, None, :]
        return proj_3d
    if src_D == 1 and dst_D == 2:
        N = src_rho.shape[0]
        env = np.exp(-(np.arange(N) - N // 2) ** 2 / (2 * (N / 8) ** 2))
        env = env / env.sum()
        return src_rho[:, None] * env[None, :]
    raise ValueError(f'projection from src_D={src_D} para dst_D={dst_D} not implemented; combinacoes suportadas: 1->1, 1->2, 1->3, 2->1, 3->1')
from runtime.physics.observables import crystallinity, dominant_wavenumber, ipr, peak_density


@dataclass
class Substrate:
    id: int
    name: str
    params: TriadParams
    psi: np.ndarray
    y: np.ndarray
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    dx: float = 0.0
    k: np.ndarray = field(default_factory=lambda: np.array([]))
    half_lin: np.ndarray = field(default_factory=lambda: np.array([]))
    nu_arr: np.ndarray = field(default_factory=lambda: np.array([]))
    lam_arr: np.ndarray = field(default_factory=lambda: np.array([]))
    Lambda_e: float = 0.0
    alpha_e: float = 0.0
    Gamma_e: float = 0.0
    f_FDT_e: float = 0.0
    use_ou: bool = False
    ou_decay: np.ndarray | None = None
    noise_amp: float = 0.0
    rng: np.random.Generator | None = None
    V_ext_static: np.ndarray | None = None
    active: bool = True
    density_traj: np.ndarray | None = None
    t_traj: np.ndarray | None = None
    _rec_buf: list = field(default_factory=list)
    _t_buf: list = field(default_factory=list)
    projector: Callable | None = None
    projector_member_ids: list = field(default_factory=list)
    macro_slow_state_t: list = field(default_factory=list)
    macro_slow_state_t_grid: list = field(default_factory=list)
    use_exptrap: bool = False
    trap_lambda_val: object = 0.5

    def initialise_arrays(self):
        p = self.params
        self.D = int(p.D)
        M = len(p.nu)
        if M < 3 or len(p.lam) < 3:
            raise ValueError(f'triad rule: P2 memory field needs at least 3 scales, got nu={M}, lam={len(p.lam)}')
        bk = getattr(p, 'backend', 'auto')
        self.xp = get_xp(bk)
        xp = self.xp
        self.x = xp.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
        self.dx = float(asnumpy(self.x[1] - self.x[0]))
        kvec = 2.0 * xp.pi * xp.fft.fftfreq(p.N, d=self.dx)
        if self.D == 1:
            self.k = kvec
            abs_k = xp.abs(self.k)
            self.grid_shape = (p.N,)
        elif self.D == 2:
            KX, KY = xp.meshgrid(kvec, kvec, indexing='ij')
            self.k2 = KX * KX + KY * KY
            abs_k = xp.sqrt(self.k2)
            self.grid_shape = (p.N, p.N)
        elif self.D == 3:
            KX, KY, KZ = xp.meshgrid(kvec, kvec, kvec, indexing='ij')
            self.k2 = KX * KX + KY * KY + KZ * KZ
            abs_k = xp.sqrt(self.k2)
            self.grid_shape = (p.N, p.N, p.N)
        else:
            raise ValueError(f'unsupported D={self.D}; need 1, 2, or 3')
        eff = _effective_params(p, dx=self.dx, D=self.D)
        self.Lambda_e = eff['Lambda']
        self.alpha_e = eff['alpha']
        self.Gamma_e = eff['Gamma']
        self.f_FDT_e = eff['f_FDT_e']
        self.lam_arr = xp.asarray(eff['lam'])
        self.nu_arr = xp.asarray(p.nu, dtype=xp.float64)
        if self.D == 1:
            H_lin_k = p.hbar ** 2 * self.k ** 2 / (2 * p.m) + self.alpha_e * abs_k ** p.sigma
        else:
            H_lin_k = p.hbar ** 2 * self.k2 / (2 * p.m) + self.alpha_e * abs_k ** p.sigma
        self.half_lin = xp.exp(-1j * H_lin_k * p.dt / (2 * p.hbar) - self.Gamma_e * p.dt / (2 * p.hbar))
        self.use_ou = True
        self.ou_decay_half = xp.exp(-self.nu_arr * p.dt * 0.5)
        if self.D == 1:
            noise_denom = self.dx
        else:
            noise_denom = self.dx ** self.D
        self.noise_amp = float(xp.sqrt(xp.asarray(self.f_FDT_e * p.dt / noise_denom))) if self.f_FDT_e > 0 else 0.0
        self.rng = xp.random.default_rng(p.seed)
        self.V_ext_static = self._build_V_ext_static()
        self.use_exptrap = getattr(p, 'step_mode', 'strang') == 'exptrap'
        self.trap_lambda_val = getattr(p, 'trap_lambda', 0.5)
        self.psi = xp.asarray(self.psi, dtype=xp.complex128)
        expected_y_shape = (len(p.nu),) + self.grid_shape
        if self.y.size == 0 or self.y.shape != expected_y_shape:
            self.y = xp.zeros(expected_y_shape, dtype=xp.float64)
        if self.D == 2 and self.psi.ndim == 1:
            psi_1d = self.psi
            x = self.x
            Xg, Yg = xp.meshgrid(x, x, indexing='ij')
            psi_2d = psi_1d[:, None] * xp.exp(-Yg ** 2 / 8.0)
            self.psi = psi_2d.astype(xp.complex128)
            self.psi = self.psi / xp.sqrt((xp.abs(self.psi) ** 2).sum() * self.dx ** 2)
        if self.D == 3 and self.psi.ndim == 1:
            psi_1d = self.psi
            x = self.x
            Xg, Yg, Zg = xp.meshgrid(x, x, x, indexing='ij')
            psi_3d = psi_1d[:, None, None] * xp.exp(-(Yg ** 2 + Zg ** 2) / 8.0)
            self.psi = psi_3d.astype(xp.complex128)
            self.psi = self.psi / xp.sqrt((xp.abs(self.psi) ** 2).sum() * self.dx ** 3)

    def _build_V_ext_static(self):
        xp = self.xp
        v = self.params.V_ext
        if self.D == 1:
            x = self.x
            if v is None:
                return xp.zeros(self.params.N)
            if v == 'harmonic':
                return 0.5 * self.params.m * self.params.omega ** 2 * x ** 2
            if callable(v):
                return xp.asarray(v(x), dtype=xp.float64)
        elif self.D == 2:
            xs = self.x
            X, Y = xp.meshgrid(xs, xs, indexing='ij')
            if v is None:
                return xp.zeros(self.grid_shape)
            if v == 'harmonic':
                r2 = X * X + Y * Y
                return 0.5 * self.params.m * self.params.omega ** 2 * r2
            if callable(v):
                return xp.asarray(v(X, Y), dtype=xp.float64)
        else:
            xs = self.x
            X, Y, Z = xp.meshgrid(xs, xs, xs, indexing='ij')
            if v is None:
                return xp.zeros(self.grid_shape)
            if v == 'harmonic':
                r2 = X * X + Y * Y + Z * Z
                return 0.5 * self.params.m * self.params.omega ** 2 * r2
            if callable(v):
                return xp.asarray(v(X, Y, Z), dtype=xp.float64)
        raise ValueError(f'unknown V_ext spec: {v!r}')

@dataclass
class CouplingLink:
    src_id: int
    dst_id: int
    kappa: float
    kappa_modulator: Callable[[dict], float] | None = None
    coupling_mode: str = 'density'
    k_target: float = 0.0

@dataclass
class Segment:
    t_start: float
    t_end: float
    links: list[CouplingLink] = field(default_factory=list)
    active_ids: set[int] | Callable | None = None
    v_ext_override: dict[int, Callable[[np.ndarray], np.ndarray]] = field(default_factory=dict)
    router: object = None

import ctypes as _ct


class _CCplx(_ct.Structure):
    _fields_ = [('re', _ct.c_double), ('im', _ct.c_double)]


def _ct_f64(a):
    flat = np.ascontiguousarray(np.asarray(a, dtype=np.float64)).ravel().tolist()
    if not isinstance(flat, list):
        flat = [flat]
    return (_ct.c_double * len(flat))(*[float(v) for v in flat])


def _ct_cplx(a):
    flat = np.ascontiguousarray(np.asarray(a, dtype=np.complex128)).ravel().tolist()
    if not isinstance(flat, list):
        flat = [flat]
    return (_CCplx * len(flat))(*[_CCplx(float(v.real), float(v.imag)) for v in flat])

class _CLink(_ct.Structure):
    _fields_ = [('src_id', _ct.c_int), ('dst_id', _ct.c_int),
                ('kappa', _ct.c_double), ('mode', _ct.c_int),
                ('k_target', _ct.c_double),
                ('kappa_modulator', _ct.c_void_p),
                ('kappa_modulator_data', _ct.c_void_p)]

class _CSeg(_ct.Structure):
    _fields_ = [('t_start', _ct.c_double), ('t_end', _ct.c_double),
                ('links', _ct.POINTER(_CLink)), ('n_links', _ct.c_int),
                ('active_ids', _ct.POINTER(_ct.c_int)), ('n_active', _ct.c_int),
                ('router', _ct.c_void_p),
                ('v_ext_override_ids', _ct.POINTER(_ct.c_int)),
                ('v_ext_overrides', _ct.c_void_p),
                ('n_v_ext_overrides', _ct.c_int)]

class _CSub(_ct.Structure):
    _fields_ = [
        ('id', _ct.c_int), ('name', _ct.c_char_p), ('D', _ct.c_int),
        ('N', _ct.c_int), ('grid_size', _ct.c_int64),
        ('L', _ct.c_double), ('dt', _ct.c_double),
        ('hbar', _ct.c_double), ('m', _ct.c_double), ('omega', _ct.c_double),
        ('Lambda', _ct.c_double), ('alpha', _ct.c_double),
        ('sigma', _ct.c_double), ('Gamma', _ct.c_double), ('f_FDT', _ct.c_double),
        ('fdt_couple', _ct.c_int), ('kT', _ct.c_double),
        ('M', _ct.c_int),
        ('nu', _ct.POINTER(_ct.c_double)), ('lam', _ct.POINTER(_ct.c_double)),
        ('mode', _ct.c_int), ('seed', _ct.c_uint64), ('V_ext', _ct.c_char_p),
        ('dx', _ct.c_double), ('x', _ct.POINTER(_ct.c_double)),
        ('V_ext_static', _ct.POINTER(_ct.c_double)),
        ('half_lin', _ct.POINTER(_CCplx)),
        ('ou_decay_half', _ct.POINTER(_ct.c_double)),
        ('noise_amp', _ct.c_double),
        ('psi', _ct.POINTER(_CCplx)), ('y', _ct.POINTER(_ct.c_double)),
        ('psi_scratch', _ct.POINTER(_CCplx)), ('psi_freq', _ct.POINTER(_CCplx)),
        ('V_couple', _ct.POINTER(_ct.c_double)), ('V_mem', _ct.POINTER(_ct.c_double)),
        ('Lambda_e', _ct.c_double), ('alpha_e', _ct.c_double),
        ('Gamma_e', _ct.c_double), ('f_FDT_e', _ct.c_double),
        ('lam_e', _ct.POINTER(_ct.c_double)),
        ('active', _ct.c_int), ('record_every', _ct.c_int),
        ('n_records', _ct.c_int), ('cap_records', _ct.c_int),
        ('density_traj', _ct.POINTER(_ct.c_double)),
        ('t_traj', _ct.POINTER(_ct.c_double)),
    ]

class _CRt(_ct.Structure):
    _fields_ = [
        ('dt', _ct.c_double), ('record_every', _ct.c_int), ('global_t', _ct.c_double),
        ('n_substrates', _ct.c_int), ('cap_substrates', _ct.c_int),
        ('substrates', _ct.POINTER(_ct.POINTER(_CSub))),
        ('n_segments', _ct.c_int), ('cap_segments', _ct.c_int),
        ('segments', _ct.POINTER(_CSeg)),
        ('diverged', _ct.c_int), ('diverged_segment', _ct.c_int),
        ('diverged_step', _ct.c_int),
        ('diverged_name', _ct.c_char_p), ('diverged_norm', _ct.c_double),
    ]

_COUPLING_MODES = {'density': 0, 'dc_subtracted': 1, 'phase_coherent': 2}
_KAPPA_MOD_CB = _ct.CFUNCTYPE(
    _ct.c_double,
    _ct.POINTER(_ct.POINTER(_ct.c_double)),
    _ct.c_int,
    _ct.c_void_p)

_kappa_bridges: dict[int, _KappaModulatorBridge] = {}
_kappa_bridge_next_id = 1

class _KappaModulatorBridge:
    __slots__ = ('_id', '_fn', '_grid_sizes', '_D', '_N', '_keep')

    def __init__(self, fn, substrates: dict):
        global _kappa_bridge_next_id
        self._id = _kappa_bridge_next_id
        _kappa_bridge_next_id += 1
        self._fn = fn
        self._grid_sizes = {}
        self._D = {}
        self._N = {}
        for sid, sub in substrates.items():
            self._grid_sizes[sid] = int(sub.grid_shape[0]) if sub.D == 1 else (
                int(sub.params.N) * int(sub.params.N) if sub.D == 2 else
                int(sub.params.N) * int(sub.params.N) * int(sub.params.N))
            self._D[sid] = int(sub.D)
            self._N[sid] = int(sub.params.N)
        self._keep = []

    @staticmethod
    @_KAPPA_MOD_CB
    def _c_callback(rho_snapshots_ptr, n_subs, user_data):
        bridge_id = user_data or 0
        bridge = _kappa_bridges.get(bridge_id)
        if bridge is None:
            return 1.0
        try:
            rho_dict = {}
            for i in range(n_subs):
                ptr = rho_snapshots_ptr[i]
                if not ptr:
                    continue
                gs = bridge._grid_sizes.get(i)
                if gs is None:
                    continue
                D = bridge._D.get(i, 1)
                N = bridge._N.get(i, 64)
                flat = np.ctypeslib.as_array(ptr, shape=(gs,))
                rho_arr = flat.copy()
                if D == 2:
                    rho_arr = rho_arr.reshape(N, N)
                elif D == 3:
                    rho_arr = rho_arr.reshape(N, N, N)
                rho_dict[i] = rho_arr
            return float(bridge._fn(rho_dict))
        except (ValueError, TypeError, RuntimeError):
            return 1.0

    def register(self):
        _kappa_bridges[self._id] = self
        return self._c_callback, self._id

    def unregister(self):
        _kappa_bridges.pop(self._id, None)

_libc_handle = None

def _libc():
    global _libc_handle
    if _libc_handle is not None:
        return _libc_handle
    libgc = None
    for name in ("libgc.so.1", "libgc.so"):
        try:
            libgc = _ct.CDLL(name, mode=_ct.RTLD_GLOBAL)
            break
        except OSError:
            continue
    if libgc is not None and hasattr(libgc, "GC_malloc"):
        libgc.malloc = libgc.GC_malloc
        libgc.malloc.restype = _ct.c_void_p
        libgc.malloc.argtypes = [_ct.c_size_t]
        libgc.free = libgc.GC_free
        libgc.free.restype = None
        libgc.free.argtypes = [_ct.c_void_p]
        _libc_handle = libgc
    else:
        _libc_handle = _ct.CDLL(None)
        _libc_handle.malloc.restype = _ct.c_void_p
        _libc_handle.malloc.argtypes = [_ct.c_size_t]
        _libc_handle.free.restype = None
        _libc_handle.free.argtypes = [_ct.c_void_p]
    return _libc_handle

def _native_alloc(lib, size):
    use_gc = getattr(lib, '_triad_use_gc', None)
    if use_gc is None:
        try:
            lib.GC_malloc_uncollectable.restype = _ct.c_void_p
            lib.GC_malloc_uncollectable.argtypes = [_ct.c_size_t]
            use_gc = True
        except AttributeError:
            use_gc = False
        lib._triad_use_gc = use_gc
    if use_gc:
        return lib.GC_malloc_uncollectable(size)
    return _libc().malloc(size)

def _malloc_copy(lib, buf):
    n = _ct.sizeof(buf)
    ptr = _native_alloc(lib, n if n else 1)
    if not ptr:
        raise MemoryError('allocation failed in native bridge')
    _ct.memmove(ptr, buf, n)
    return ptr

def _native_mr_lib():
    from runtime.core.solver import _native_lib
    try:
        lib = _native_lib()
    except (OSError, RuntimeError):
        raise RuntimeError('native runtime failed to load: CUDA libraries not available')
    if not hasattr(lib, 'triad_mr_new'):
        raise RuntimeError('native runtime missing triad_mr_new; recompile com: make -C native/c')
    if not getattr(lib, '_mr_bound', False):
        lib.triad_mr_new.restype = _ct.POINTER(_CRt)
        lib.triad_mr_new.argtypes = [_ct.c_double, _ct.c_int]
        lib.triad_mr_free.restype = None
        lib.triad_mr_free.argtypes = [_ct.POINTER(_CRt)]
        lib.triad_mr_add_substrate.restype = _ct.c_int
        lib.triad_mr_add_substrate.argtypes = [
            _ct.POINTER(_CRt), _ct.c_char_p, _ct.c_int,
            _ct.c_int, _ct.c_double, _ct.c_double, _ct.c_double,
            _ct.c_double, _ct.c_double, _ct.c_double,
            _ct.c_double, _ct.c_double, _ct.c_double,
            _ct.c_int, _ct.c_double, _ct.c_int,
            _ct.POINTER(_ct.c_double), _ct.POINTER(_ct.c_double),
            _ct.c_int, _ct.c_uint64, _ct.c_char_p, _ct.POINTER(_CCplx)]
        lib.triad_mr_add_segment.restype = None
        lib.triad_mr_add_segment.argtypes = [
            _ct.POINTER(_CRt), _ct.c_double, _ct.c_double,
            _ct.c_void_p, _ct.c_int, _ct.c_void_p, _ct.c_int]
        lib.triad_mr_set_v_ext.restype = _ct.c_int
        lib.triad_mr_set_v_ext.argtypes = [_ct.POINTER(_CRt), _ct.c_int, _ct.POINTER(_ct.c_double)]
        lib.triad_mr_run.restype = None
        lib.triad_mr_run.argtypes = [_ct.POINTER(_CRt)]
        lib.triad_mr_get.restype = _ct.POINTER(_CSub)
        lib.triad_mr_get.argtypes = [_ct.POINTER(_CRt), _ct.c_int]
        lib.triad_mr_segment_set_router.restype = None
        lib.triad_mr_segment_set_router.argtypes = [
            _ct.POINTER(_CRt), _ct.c_int, _ct.c_int, _ct.c_double]
        lib.triad_mr_segment_add_v_ext_override.restype = _ct.c_int
        lib.triad_mr_segment_add_v_ext_override.argtypes = [
            _ct.POINTER(_CRt), _ct.c_int, _ct.c_int, _ct.POINTER(_ct.c_double)]
        lib._mr_bound = True
    return lib

class MultiRuntime:

    def __init__(self, dt: float=0.005, record_every: int=4):
        self.substrates: dict[int, Substrate] = {}
        self.segments: list[Segment] = []
        self.dt = dt
        self.record_every = record_every
        self._next_id = 0
        self.global_t = 0.0
        self.norm_threshold = 100.0
        self.on_segment_end: Callable[[Segment, int], None] | None = None
        self._coupling_links: list[CouplingLink] = []
        self._router_type: str = 'precision_weighted'

    def add_substrate(self, name: str, params: TriadParams, psi: np.ndarray | None=None) -> Substrate:
        _validate_triad_params(params)
        sid = self._next_id
        self._next_id += 1
        if abs(params.dt - self.dt) > 1e-12:
            params = TriadParams(**{**params.__dict__, 'dt': self.dt})
        D = int(params.D)
        if psi is None:
            x = np.linspace(-params.L / 2, params.L / 2, params.N, endpoint=False)
            dx = x[1] - x[0]
            if D == 1:
                psi = np.exp(-x ** 2 / 8.0).astype(np.complex128)
                psi /= np.sqrt((np.abs(psi) ** 2).sum() * dx)
            elif D == 2:
                Xg, Yg = np.meshgrid(x, x, indexing='ij')
                psi = np.exp(-(Xg ** 2 + Yg ** 2) / 8.0).astype(np.complex128)
                psi /= np.sqrt((np.abs(psi) ** 2).sum() * dx ** 2)
            elif D == 3:
                Xg, Yg, Zg = np.meshgrid(x, x, x, indexing='ij')
                psi = np.exp(-(Xg ** 2 + Yg ** 2 + Zg ** 2) / 8.0).astype(np.complex128)
                psi /= np.sqrt((np.abs(psi) ** 2).sum() * dx ** 3)
            else:
                raise ValueError(f'unsupported D={D}')
        sub = Substrate(id=sid, name=name, params=params, psi=np.asarray(psi, dtype=np.complex128).copy(), y=np.array([]))
        sub.initialise_arrays()
        self.substrates[sid] = sub
        return sub

    def couple(self, substrate_ids: list[int], kappa: float = 1.0,
               router: str = 'precision_weighted',
               coupling_mode: str = 'density') -> dict:

        from runtime.core.field_router import FieldRouter

        field_router = FieldRouter(method=router)

        for sid in substrate_ids:
            sub = self.substrates.get(sid)
            if sub is None:
                raise ValueError(f"substrate id {sid} not found")
            if sub.params.mode != 'triad':
                sub.params = TriadParams(**{**sub.params.__dict__, 'mode': 'triad'})

                from runtime.core.solver import _effective_params
                eff = _effective_params(sub.params, dx=sub.dx, D=sub.D)
                sub.Lambda_e = eff['Lambda']
                sub.alpha_e = eff['alpha']
                sub.Gamma_e = eff['Gamma']
                sub.f_FDT_e = eff['f_FDT_e']
                sub.lam_arr = sub.xp.asarray(eff['lam'])

        self._router_type = router
        self._coupling_links = []

        for i, sid_a in enumerate(substrate_ids):
            for sid_b in substrate_ids[i + 1:]:
                link_ab = CouplingLink(
                    src_id=sid_a, dst_id=sid_b, kappa=kappa,
                    coupling_mode=coupling_mode,
                )
                link_ba = CouplingLink(
                    src_id=sid_b, dst_id=sid_a, kappa=kappa,
                    coupling_mode=coupling_mode,
                )
                self._coupling_links.extend([link_ab, link_ba])

        seg = Segment(
            t_start=self.global_t,
            t_end=self.global_t + 1.0,
            links=list(self._coupling_links),
            active_ids=set(substrate_ids),
            router=field_router,
        )
        self.segments.append(seg)

        return {
            'links': self._coupling_links,
            'router': field_router,
            'router_type': router,
            'n_links': len(self._coupling_links),
            'substrate_ids': substrate_ids,
        }

    def add_segment(self, segment: Segment):
        self.segments.append(segment)

    def _apply_segment_setup(self, seg: Segment):
        active_ids = seg.active_ids
        if callable(active_ids):
            rho_now = {sid: np.abs(s.psi) ** 2 for sid, s in self.substrates.items()}
            active_ids = active_ids(rho_now)
        for sid, sub in self.substrates.items():
            if active_ids is None:
                sub.active = True
            else:
                sub.active = sid in active_ids
        for sid, fn in seg.v_ext_override.items():
            if sid in self.substrates:
                self.substrates[sid].V_ext_static = np.asarray(fn(self.substrates[sid].x), dtype=np.float64)
    def _apply_router_gating(self, seg: Segment):

        from runtime.core.field_router import gather_observables
        router = seg.router
        active = {sid: s for sid, s in self.substrates.items() if s.active}
        if not active:
            return
        obs = gather_observables(active)
        effective = router.route(seg.links, obs)
        for e in seg.links:
            eff = effective.get((e.src_id, e.dst_id))
            if eff is None or e.kappa == 0:
                e._router_weight = 1.0
            else:
                e._router_weight = eff / e.kappa

    def _native_supported(self) -> bool:
        if self.on_segment_end is not None:
            return False
        try:
            from runtime.core.solver import _native_lib
            _native_lib()
        except (OSError, RuntimeError):
            return False
        n = len(self.substrates)
        if n == 0 or sorted(self.substrates) != list(range(n)):
            return False
        shapes = {}
        for sid, sub in self.substrates.items():
            if getattr(sub, 'xp', None) is not np:
                return False
            if sub.use_exptrap or sub.projector is not None:
                return False
            if sub.params.mode != 'triad':
                return False
            if sub.V_ext_static is None:
                return False
            if int(sub.D) not in (1, 2, 3):
                return False
            shapes[sid] = (int(sub.D), int(sub.params.N))
        for seg in self.segments:
            if callable(seg.active_ids):
                return False
            for e in seg.links:
                if e.coupling_mode not in _COUPLING_MODES:
                    return False
                if e.src_id not in shapes or e.dst_id not in shapes:
                    return False
                if shapes[e.src_id] != shapes[e.dst_id]:
                    sD, sN = shapes[e.src_id]
                    dD, dN = shapes[e.dst_id]
                    if not (sD == 1 and sN == dN):
                        return False
        return True

    def _run_native(self) -> dict:
        lib = _native_mr_lib()
        n_subs = len(self.substrates)
        rt = lib.triad_mr_new(float(self.dt), int(self.record_every))
        keep = []
        try:
            for sid in range(n_subs):
                sub = self.substrates[sid]
                p = sub.params
                nu = _ct_f64(p.nu)
                lam = _ct_f64(p.lam)
                M = len(p.nu)
                psi = _ct_cplx(sub.psi)
                name_b = sub.name.encode()
                vext_b = b'harmonic' if p.V_ext == 'harmonic' else b'none'
                keep.extend([nu, lam, psi, name_b, vext_b])
                rid = lib.triad_mr_add_substrate(
                    rt, name_b, int(sub.D), int(p.N), float(p.L),
                    float(p.hbar), float(p.m), float(p.omega),
                    float(p.Lambda), float(p.alpha), float(p.sigma),
                    float(p.Gamma), float(sub.f_FDT_e),
                    1 if getattr(p, 'fdt_couple', False) else 0,
                    float(getattr(p, 'kT', 1.0)), M,
                    _ct.cast(nu, _ct.POINTER(_ct.c_double)),
                    _ct.cast(lam, _ct.POINTER(_ct.c_double)),
                    2, int(p.seed) & 0xFFFFFFFFFFFFFFFF, vext_b,
                    _ct.cast(psi, _ct.POINTER(_CCplx)))
                if rid != sid:
                    raise RuntimeError(f'ponte nativa: substrato id {rid} != {sid}')
                vex = _ct_f64(sub.V_ext_static)
                keep.append(vex)
                rc = lib.triad_mr_set_v_ext(rt, sid, _ct.cast(vex, _ct.POINTER(_ct.c_double)))
                if rc != 0:
                    raise RuntimeError(f'ponte nativa: set_v_ext falhou para substrato {sid}')
            for seg in self.segments:
                n_links = len(seg.links)
                links_ptr = None
                if n_links:
                    arr = (_CLink * n_links)()
                    seg_bridges = []
                    for i, e in enumerate(seg.links):
                        mod_cb = _ct.c_void_p(0)
                        mod_data = _ct.c_void_p(0)
                        if e.kappa_modulator is not None:
                            bridge = _KappaModulatorBridge(e.kappa_modulator, self.substrates)
                            cb_fn, cb_id = bridge.register()
                            mod_cb = _ct.cast(cb_fn, _ct.c_void_p)
                            mod_data = _ct.c_void_p(cb_id)
                            seg_bridges.append(bridge)
                        arr[i] = _CLink(int(e.src_id), int(e.dst_id), float(e.kappa),
                                        _COUPLING_MODES[e.coupling_mode], float(e.k_target),
                                        mod_cb, mod_data)
                    links_ptr = _malloc_copy(lib, arr)
                    keep.append(seg_bridges)
                active_ptr = None
                n_active = 0
                if seg.active_ids is not None:
                    ids = sorted(int(i) for i in seg.active_ids)
                    n_active = len(ids)
                    ids_arr = (_ct.c_int * max(n_active, 1))(*ids)
                    active_ptr = _malloc_copy(lib, ids_arr)
                lib.triad_mr_add_segment(rt, float(seg.t_start), float(seg.t_end),
                                         links_ptr, n_links, active_ptr, n_active)
                if links_ptr is not None:
                    try:
                        _libc().free(links_ptr)
                    except Exception as e:
                        import warnings
                        warnings.warn(f'free(links_ptr) failed: {e}')
                if active_ptr is not None:
                    try:
                        _libc().free(active_ptr)
                    except Exception as e:
                        import warnings
                        warnings.warn(f'free(active_ptr) failed: {e}')
                links_ptr = None
                active_ptr = None
                seg_idx = self.segments.index(seg)
                if seg.router is not None:
                    router_method = 0
                    if hasattr(seg.router, 'method') and seg.router.method != 'precision_weighted':
                        router_method = 1
                    router_temp = float(getattr(seg.router, 'temperature', 0.0))
                    lib.triad_mr_segment_set_router(rt, seg_idx, router_method, router_temp)
                for sid, vfn in seg.v_ext_override.items():
                    if sid in self.substrates:
                        sub = self.substrates[sid]
                        V_new = np.ascontiguousarray(
                            np.asarray(vfn(sub.x), dtype=np.float64)).ravel()
                        keep.append(V_new)
                        lib.triad_mr_segment_add_v_ext_override(
                            rt, seg_idx, int(sid),
                            V_new.ctypes.data_as(_ct.POINTER(_ct.c_double)))
            lib.triad_mr_run(rt)
            for item in keep:
                if isinstance(item, list) and item and isinstance(item[0], _KappaModulatorBridge):
                    for b in item:
                        b.unregister()
            rtc = rt.contents
            for sid in range(n_subs):
                sub = self.substrates[sid]
                sp = lib.triad_mr_get(rt, sid).contents
                G = int(sp.grid_size)
                raw_psi = _ct.cast(sp.psi, _ct.POINTER(_ct.c_double * (2 * G))).contents
                psi_flat = np.frombuffer(raw_psi, dtype=np.float64).copy().view(np.complex128)
                sub.psi = psi_flat.reshape(sub.grid_shape)
                M = int(sp.M)
                if M < 3:
                    raise ValueError(f'triad rule: C substrate must have at least 3 memory scales, got {M}')
                raw_y = _ct.cast(sp.y, _ct.POINTER(_ct.c_double * (M * G))).contents
                sub.y = np.frombuffer(raw_y, dtype=np.float64).copy().reshape((M,) + sub.grid_shape)
                nrec = int(sp.n_records)
                if nrec > 0:
                    raw_d = _ct.cast(sp.density_traj, _ct.POINTER(_ct.c_double * (nrec * G))).contents
                    dens = np.frombuffer(raw_d, dtype=np.float64).copy().reshape((nrec,) + sub.grid_shape)
                    raw_t = _ct.cast(sp.t_traj, _ct.POINTER(_ct.c_double * nrec)).contents
                    sub.density_traj = np.array(list(dens)).T
                    sub.t_traj = np.frombuffer(raw_t, dtype=np.float64).copy()
                    sub._rec_buf = []
                    sub._t_buf = []
            diverged = bool(rtc.diverged)
            self.global_t += float(rtc.global_t)
            result = {'diverged': diverged, 'global_t': self.global_t, 'engine': 'native'}
            if diverged:
                dname = rtc.diverged_name.decode() if rtc.diverged_name else ''
                result['diverged_at'] = (int(rtc.diverged_segment), int(rtc.diverged_step),
                                         dname, float(rtc.diverged_norm))
            return result
        finally:
            lib.triad_mr_free(rt)

    def run(self, verbose: bool=False) -> dict:
        if not self.segments:
            return {'diverged': False, 'global_t': self.global_t}
        if self._native_supported():
            return self._run_native()
        diverged = False
        diverged_at = None
        for seg_i, seg in enumerate(self.segments):
            self._apply_segment_setup(seg)
            links_by_dst: dict[int, list[CouplingLink]] = {}
            for e in seg.links:
                links_by_dst.setdefault(e.dst_id, []).append(e)
            duration = seg.t_end - seg.t_start
            n_steps = int(round(duration / self.dt))
            seg_t = seg.t_start
            for step in range(n_steps):
                rho_now = {sid: s.xp.abs(s.psi) ** 2 for sid, s in self.substrates.items() if s.active}
                if seg.router is not None:
                    self._apply_router_gating(seg)
                for sid, sub in self.substrates.items():
                    if not sub.active:
                        continue
                    self._step_one(sub, rho_now, links_by_dst.get(sid, []))
                    if sub.D == 1:
                        norm = float((sub.xp.abs(sub.psi) ** 2).sum() * sub.dx)
                    else:
                        norm = float((sub.xp.abs(sub.psi) ** 2).sum() * sub.dx ** sub.D)
                    N_grid = sub.params.N ** sub.D
                    plateau = N_grid * sub.f_FDT_e / (2 * max(sub.Gamma_e, 1e-30)) if sub.f_FDT_e > 0 else 0
                    thresh = max(100.0, 10.0 * plateau)
                    xp_mod = sub.xp
                    if not xp_mod.isfinite(xp_mod.asarray(norm)) or norm > thresh:
                        diverged = True
                        diverged_at = (seg_i, step, sub.name, norm)
                        break
                if diverged:
                    break
                if step % self.record_every == 0:
                    for sub in self.substrates.values():
                        sub._rec_buf.append(asnumpy(sub.xp.abs(sub.psi) ** 2))
                        sub._t_buf.append(seg_t)
                    for macro in self.substrates.values():
                        if macro.projector is None:
                            continue
                        members = [self.substrates[mid] for mid in macro.projector_member_ids if mid in self.substrates]
                        try:
                            slow = float(macro.projector(members))
                        except (ValueError, TypeError, RuntimeError):
                            slow = float('nan')
                        macro.macro_slow_state_t.append(slow)
                        macro.macro_slow_state_t_grid.append(seg_t)
                seg_t += self.dt
                self.global_t += self.dt
            if diverged:
                break
            if self.on_segment_end is not None:
                try:
                    self.on_segment_end(seg, seg_i)
                except (ValueError, TypeError, RuntimeError):
                    warnings.warn("triad: on_segment_end callback failed", RuntimeWarning, stacklevel=2)
        for sub in self.substrates.values():
            if sub._rec_buf:
                sub.density_traj = np.array(sub._rec_buf).T
                sub.t_traj = np.array(sub._t_buf)
        result = {'diverged': diverged, 'global_t': self.global_t, 'engine': 'python'}
        if diverged:
            result['diverged_at'] = diverged_at
        return result

    def _step_one(self, sub: Substrate, rho_now: dict[int, np.ndarray], inbound_links: list[CouplingLink]):
        if sub.D == 2:
            self._step_one_2d(sub, rho_now, inbound_links)
            return
        if sub.D == 3:
            self._step_one_3d(sub, rho_now, inbound_links)
            return
        xp = sub.xp
        p = sub.params
        psi = sub.psi
        M = len(sub.nu_arr)
        if M < 3:
            raise ValueError(f'triad rule: substrate must have at least 3 memory scales, got {M}')
        psi = xp.fft.ifft(xp.fft.fft(psi) * sub.half_lin)
        rho = xp.abs(psi) ** 2
        sub.y = sub.ou_decay_half[:, None] * sub.y + (1.0 - sub.ou_decay_half)[:, None] * rho
        V_mem = (sub.lam_arr[:, None] * sub.y).sum(axis=0)
        V_couple = xp.zeros(sub.grid_shape)
        for e in inbound_links:
            src_rho = rho_now.get(e.src_id)
            if src_rho is None:
                continue
            eff_kappa = e.kappa * getattr(e, '_router_weight', 1.0)
            if e.kappa_modulator is not None:
                try:
                    eff_kappa = e.kappa * float(e.kappa_modulator(rho_now))
                except (ValueError, TypeError, RuntimeError):
                    warnings.warn("triad: kappa_modulator failed, using original kappa", RuntimeWarning, stacklevel=2)
            src_rho_xp = xp.asarray(src_rho) if not isinstance(src_rho, type(xp.zeros(1))) else src_rho
            if src_rho_xp.shape != sub.grid_shape:
                src_rho_xp = xp.asarray(_project_to_shape(asnumpy(src_rho_xp), sub.grid_shape))
            mode = getattr(e, 'coupling_mode', 'density')
            if mode == 'density':
                V_couple = V_couple + eff_kappa * src_rho_xp
            elif mode == 'dc_subtracted':
                src_mean = float(xp.mean(src_rho_xp))
                V_couple = V_couple + eff_kappa * (src_rho_xp - src_mean)
            elif mode == 'phase_coherent':
                src_sub = self.substrates.get(e.src_id)
                if src_sub is not None:
                    x = src_sub.x
                    proj = xp.real(src_sub.psi * xp.exp(-1j * e.k_target * x))
                    V_couple = V_couple + eff_kappa * proj
            else:
                V_couple = V_couple + eff_kappa * src_rho_xp
        V_start = sub.V_ext_static + sub.Lambda_e * rho + V_mem + V_couple
        if sub.use_exptrap:
            lam_t = _resolve_trap_lambda(sub.trap_lambda_val, rho, xp)
            psi_pred = psi * xp.exp(-1j * V_start * p.dt / p.hbar)
            rho_pred = xp.abs(psi_pred) ** 2
            V_end = sub.V_ext_static + sub.Lambda_e * rho_pred + V_mem + V_couple
            V_tot = (1.0 - lam_t) * V_start + lam_t * V_end
        else:
            V_tot = V_start
        psi = psi * xp.exp(-1j * V_tot * p.dt / sub.params.hbar)
        rho = xp.abs(psi) ** 2
        sub.y = sub.ou_decay_half[:, None] * sub.y + (1.0 - sub.ou_decay_half)[:, None] * rho
        if sub.noise_amp > 0:
            xi = sub.rng.standard_normal(p.N)
            xip = sub.rng.standard_normal(p.N)
            psi = psi + sub.noise_amp * (xi + 1j * xip) / xp.sqrt(2.0)
        psi = xp.fft.ifft(xp.fft.fft(psi) * sub.half_lin)
        sub.psi = psi

    def _step_one_2d(self, sub, rho_now, inbound_links):
        xp = sub.xp
        p = sub.params
        psi = sub.psi
        M = len(sub.nu_arr)
        if M < 3:
            raise ValueError(f'triad rule: substrate must have at least 3 memory scales, got {M}')
        psi = xp.fft.ifft2(xp.fft.fft2(psi) * sub.half_lin)
        rho = xp.abs(psi) ** 2
        sub.y = sub.ou_decay_half[:, None, None] * sub.y + (1.0 - sub.ou_decay_half)[:, None, None] * rho
        V_mem = (sub.lam_arr[:, None, None] * sub.y).sum(axis=0)
        V_couple = xp.zeros(sub.grid_shape)
        for e in inbound_links:
            src_rho = rho_now.get(e.src_id)
            if src_rho is None:
                continue
            src_rho_xp = xp.asarray(src_rho)
            if src_rho_xp.shape != sub.grid_shape:
                continue
            eff_kappa = e.kappa * getattr(e, '_router_weight', 1.0)
            if e.kappa_modulator is not None:
                try:
                    eff_kappa = e.kappa * float(e.kappa_modulator(rho_now))
                except (ValueError, TypeError, RuntimeError):
                    warnings.warn("triad: kappa_modulator failed, using original kappa", RuntimeWarning, stacklevel=2)
            mode = getattr(e, 'coupling_mode', 'density')
            if mode == 'density':
                V_couple = V_couple + eff_kappa * src_rho_xp
            elif mode == 'dc_subtracted':
                src_mean = float(xp.mean(src_rho_xp))
                V_couple = V_couple + eff_kappa * (src_rho_xp - src_mean)
            elif mode == 'phase_coherent':
                src_sub = self.substrates.get(e.src_id)
                if src_sub is not None and xp.asarray(src_sub.psi).shape == sub.grid_shape:
                    x = src_sub.x
                    r = x[:, None] + x[None, :]
                    proj = xp.real(src_sub.psi * xp.exp(-1j * e.k_target * r))
                    V_couple = V_couple + eff_kappa * proj
            else:
                V_couple = V_couple + eff_kappa * src_rho_xp
        V_start = sub.V_ext_static + sub.Lambda_e * rho + V_mem + V_couple
        if sub.use_exptrap:
            lam_t = _resolve_trap_lambda(sub.trap_lambda_val, rho, xp)
            psi_pred = psi * xp.exp(-1j * V_start * p.dt / sub.params.hbar)
            rho_pred = xp.abs(psi_pred) ** 2
            V_end = sub.V_ext_static + sub.Lambda_e * rho_pred + V_mem + V_couple
            V_tot = (1.0 - lam_t) * V_start + lam_t * V_end
        else:
            V_tot = V_start
        psi = psi * xp.exp(-1j * V_tot * p.dt / sub.params.hbar)
        rho = xp.abs(psi) ** 2
        sub.y = sub.ou_decay_half[:, None, None] * sub.y + (1.0 - sub.ou_decay_half)[:, None, None] * rho
        if sub.noise_amp > 0:
            xi = sub.rng.standard_normal(sub.grid_shape)
            xip = sub.rng.standard_normal(sub.grid_shape)
            psi = psi + sub.noise_amp * (xi + 1j * xip) / xp.sqrt(2.0)
        psi = xp.fft.ifft2(xp.fft.fft2(psi) * sub.half_lin)
        sub.psi = psi

    def _step_one_3d(self, sub, rho_now, inbound_links):
        xp = sub.xp
        p = sub.params
        psi = sub.psi
        M = len(sub.nu_arr)
        if M < 3:
            raise ValueError(f'triad rule: substrate must have at least 3 memory scales, got {M}')
        psi = xp.fft.ifftn(xp.fft.fftn(psi) * sub.half_lin)
        rho = xp.abs(psi) ** 2
        sub.y = sub.ou_decay_half[:, None, None, None] * sub.y + (1.0 - sub.ou_decay_half)[:, None, None, None] * rho
        V_mem = (sub.lam_arr[:, None, None, None] * sub.y).sum(axis=0)
        V_couple = xp.zeros(sub.grid_shape)
        for e in inbound_links:
            src_rho = rho_now.get(e.src_id)
            if src_rho is None:
                continue
            src_rho_xp = xp.asarray(src_rho)
            eff_kappa = e.kappa * getattr(e, '_router_weight', 1.0)
            if e.kappa_modulator is not None:
                try:
                    eff_kappa = e.kappa * float(e.kappa_modulator(rho_now))
                except (ValueError, TypeError, RuntimeError):
                    warnings.warn("triad: kappa_modulator failed, using original kappa", RuntimeWarning, stacklevel=2)
            if src_rho_xp.shape != sub.grid_shape:
                src_rho_xp = xp.asarray(_project_to_shape(asnumpy(src_rho_xp), sub.grid_shape))
            mode = getattr(e, 'coupling_mode', 'density')
            if mode == 'density':
                V_couple = V_couple + eff_kappa * src_rho_xp
            elif mode == 'dc_subtracted':
                src_mean = float(xp.mean(src_rho_xp))
                V_couple = V_couple + eff_kappa * (src_rho_xp - src_mean)
            elif mode == 'phase_coherent':
                src_sub = self.substrates.get(e.src_id)
                if src_sub is not None and xp.asarray(src_sub.psi).shape == sub.grid_shape:
                    x = src_sub.x
                    r = x[:, None, None] + x[None, :, None] + x[None, None, :]
                    proj = xp.real(src_sub.psi * xp.exp(-1j * e.k_target * r))
                    V_couple = V_couple + eff_kappa * proj
            else:
                V_couple = V_couple + eff_kappa * src_rho_xp
        V_start = sub.V_ext_static + sub.Lambda_e * rho + V_mem + V_couple
        if sub.use_exptrap:
            lam_t = _resolve_trap_lambda(sub.trap_lambda_val, rho, xp)
            psi_pred = psi * xp.exp(-1j * V_start * p.dt / sub.params.hbar)
            rho_pred = xp.abs(psi_pred) ** 2
            V_end = sub.V_ext_static + sub.Lambda_e * rho_pred + V_mem + V_couple
            V_tot = (1.0 - lam_t) * V_start + lam_t * V_end
        else:
            V_tot = V_start
        psi = psi * xp.exp(-1j * V_tot * p.dt / p.hbar)
        rho = xp.abs(psi) ** 2
        sub.y = sub.ou_decay_half[:, None, None, None] * sub.y + (1.0 - sub.ou_decay_half)[:, None, None, None] * rho
        if sub.noise_amp > 0:
            xi = sub.rng.standard_normal(sub.grid_shape)
            xip = sub.rng.standard_normal(sub.grid_shape)
            psi = psi + sub.noise_amp * (xi + 1j * xip) / xp.sqrt(2.0)
        psi = xp.fft.ifftn(xp.fft.fftn(psi) * sub.half_lin)
        sub.psi = psi

    def parallel_relax(self, K: int = 5, T: float = 15.0,
                       select: str = "min_energy",
                       prune: str = "stalled",
                       prune_patience: int = 20,
                       verbose: bool = False) -> dict:

        import time as _time

        from runtime.core.solver import TriadParams, integrate_nd
        from runtime.physics.observables import energy as _energy

        if self.substrates:
            base_sub = list(self.substrates.values())[0]
            base_p = base_sub.params
        else:
            base_p = TriadParams(T=T, seed=0, N=64)

        results = []
        for k in range(K):
            p_k = TriadParams(**{**base_p.__dict__, 'T': T,
                                  'seed': base_p.seed + k * 777})
            if verbose:
                t0 = _time.perf_counter()

            if prune == "stalled":

                r = self._run_with_pruning(p_k, prune_patience, verbose)
            else:
                r = integrate_nd(p_k)
                dx = r['dx']
                r['_observables'] = {
                    'crystallinity': crystallinity(r['psi_final'], dx),
                    'energy': _energy(r['psi_final'], dx, hbar=p_k.hbar,
                                      m=p_k.m, Lambda=p_k.Lambda),
                    'k_star': dominant_wavenumber(r['psi_final'], dx),
                    'ipr': ipr(r['psi_final'], dx),
                }
                r['_pruned'] = False

            if verbose:
                elapsed = _time.perf_counter() - t0
                obs = r.get('_observables', {})
                print(f"  member {k}: C={obs.get('crystallinity',0):.4f} "
                      f"E={obs.get('energy',0):.2f} pruned={r.get('_pruned',False)} "
                      f"t={elapsed:.3f}s")
            results.append(r)

        member_obs = [r.get('_observables', {}) for r in results]
        if select == "min_energy":
            energies = [o.get('energy', float('inf')) for o in member_obs]
            best_idx = int(np.argmin(energies))
        elif select == "max_crystallinity":
            crysts = [o.get('crystallinity', 0) for o in member_obs]
            best_idx = int(np.argmax(crysts))
        else:
            best_idx = 0

        return {
            "selected_idx": best_idx,
            "selected_result": results[best_idx],
            "selected_observables": member_obs[best_idx],
            "per_member_observables": member_obs,
            "n_members": K,
            "criterion": select,
        }

    def _run_with_pruning(self, p: TriadParams, patience: int,
                          verbose: bool) -> dict:

        from runtime.core.solver import TriadParams, integrate_nd
        from runtime.physics.observables import energy as _energy

        chunk_T = 1.0
        n_chunks = int(round(p.T / chunk_T))
        best_C = 0.0
        stall_count = 0
        pruned = False

        x = np.linspace(-p.L/2, p.L/2, p.N, endpoint=False)
        dx = float(x[1] - x[0])
        psi = np.exp(-x**2 / 8.0).astype(np.complex128)
        psi /= np.sqrt((np.abs(psi)**2).sum() * dx)
        y = None

        for c in range(n_chunks):
            p_chunk = TriadParams(**{**p.__dict__, 'T': chunk_T,
                                      'seed': p.seed + c * 100})
            r = integrate_nd(p_chunk, psi0=psi, y0=y)
            psi = r['psi_final']
            y = r['y_final']

            C = crystallinity(psi, dx)
            if C > best_C + 0.001:
                best_C = C
                stall_count = 0
            else:
                stall_count += 1

            if stall_count >= patience and c > patience:
                pruned = True
                break

        return {
            'psi_final': psi,
            'y_final': y,
            'dx': dx,
            '_observables': {
                'crystallinity': crystallinity(psi, dx),
                'energy': _energy(psi, dx, hbar=p.hbar, m=p.m, Lambda=p.Lambda),
                'k_star': dominant_wavenumber(psi, dx),
                'ipr': ipr(psi, dx),
            },
            '_pruned': pruned,
        }
