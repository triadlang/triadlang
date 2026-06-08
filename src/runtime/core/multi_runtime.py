from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
from runtime.core.solver import TriadParams, _effective_params, _resolve_trap_lambda
from runtime.backend import get_xp, asnumpy, cuda_available

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
    return np.zeros(dst_shape)
from runtime.physics.observables import crystallinity, ipr, dominant_wavenumber, peak_density

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
    ou_decay: Optional[np.ndarray] = None
    noise_amp: float = 0.0
    rng: Optional[np.random.Generator] = None
    V_ext_static: Optional[np.ndarray] = None
    active: bool = True
    density_traj: Optional[np.ndarray] = None
    t_traj: Optional[np.ndarray] = None
    _rec_buf: list = field(default_factory=list)
    _t_buf: list = field(default_factory=list)
    projector: Optional[Callable] = None
    projector_member_ids: list = field(default_factory=list)
    macro_slow_state_t: list = field(default_factory=list)
    macro_slow_state_t_grid: list = field(default_factory=list)
    use_exptrap: bool = False
    trap_lambda_val: object = 0.5

    def initialise_arrays(self):
        p = self.params
        self.D = int(getattr(p, 'D', 1))
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
        eff = _effective_params(p)
        self.Lambda_e = eff['Lambda']
        self.alpha_e = eff['alpha']
        self.Gamma_e = eff['Gamma']
        self.f_FDT_e = eff['f_FDT']
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
class CouplingEdge:
    src_id: int
    dst_id: int
    kappa: float
    kappa_modulator: Optional[Callable[[dict], float]] = None
    coupling_mode: str = 'density'
    k_target: float = 0.0

@dataclass
class Segment:
    t_start: float
    t_end: float
    edges: list[CouplingEdge] = field(default_factory=list)
    active_ids: Optional[object] = None
    v_ext_override: dict[int, Callable[[np.ndarray], np.ndarray]] = field(default_factory=dict)
    probes: list = field(default_factory=list)
    
    router: object = None

@dataclass
class ProbeRecord:
    t: float
    label: str
    substrate_name: str
    psi: np.ndarray
    k_star: float
    ipr: float
    crystallinity: float
    peak: float

class MultiRuntime:

    def __init__(self, dt: float=0.005, record_every: int=4):
        self.substrates: dict[int, Substrate] = {}
        self.segments: list[Segment] = []
        self.dt = dt
        self.record_every = record_every
        self._next_id = 0
        self.global_t = 0.0
        self.norm_threshold = 100.0
        self.probe_log: list[ProbeRecord] = []
        self.on_segment_end: Optional[Callable[[Segment, int], None]] = None
        self._coupling_edges: list[CouplingEdge] = []
        self._router_type: str = 'precision_weighted'

    def add_substrate(self, name: str, params: TriadParams, psi: Optional[np.ndarray]=None) -> Substrate:
        sid = self._next_id
        self._next_id += 1
        if abs(params.dt - self.dt) > 1e-12:
            params = TriadParams(**{**params.__dict__, 'dt': self.dt})
        D = int(getattr(params, 'D', 1))
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
        """Set up precision-weighted MoE coupling between substrates.

        Creates bidirectional CouplingEdges between all pairs of the given
        substrate ids.  During simulation, the FieldRouter reads each
        substrate's FDT precision (1/variance) and adjusts coupling
        strengths so that information flows preferentially from high-
        precision substrates.

        Every substrate runs full mode (P1+P2+P3 active).  Gating is
        derived from each substrate's own thermal/memory state.

        Parameters
        ----------
        substrate_ids : list[int]
            Substrate ids to couple together (bidirectional).
        kappa : float
            Base coupling strength.
        router : str
            Router type: 'precision_weighted' for FDT-based precision
            gating, 'uniform' for equal weights.
        coupling_mode : str
            Coupling mode passed to CouplingEdge ('density', 'dc_subtracted',
            'phase_coherent').

        Returns
        -------
        dict with edges created and router info.
        """
        from runtime.core.field_router import FieldRouter

        field_router = FieldRouter(method=router)

        for sid in substrate_ids:
            sub = self.substrates.get(sid)
            if sub is None:
                raise ValueError(f"substrate id {sid} not found")
            if sub.params.mode != 'full':
                sub.params = TriadParams(**{**sub.params.__dict__, 'mode': 'full'})
                
                from runtime.core.solver import _effective_params
                eff = _effective_params(sub.params)
                sub.Lambda_e = eff['Lambda']
                sub.alpha_e = eff['alpha']
                sub.Gamma_e = eff['Gamma']
                sub.f_FDT_e = eff['f_FDT']
                sub.lam_arr = sub.xp.asarray(eff['lam'])

        self._router_type = router
        self._coupling_edges = []

        for i, sid_a in enumerate(substrate_ids):
            for sid_b in substrate_ids[i + 1:]:
                edge_ab = CouplingEdge(
                    src_id=sid_a, dst_id=sid_b, kappa=kappa,
                    coupling_mode=coupling_mode,
                )
                edge_ba = CouplingEdge(
                    src_id=sid_b, dst_id=sid_a, kappa=kappa,
                    coupling_mode=coupling_mode,
                )
                self._coupling_edges.extend([edge_ab, edge_ba])

        seg = Segment(
            t_start=self.global_t,
            t_end=self.global_t + 1.0,
            edges=list(self._coupling_edges),
            active_ids=set(substrate_ids),
            router=field_router,
        )
        self.segments.append(seg)

        return {
            'edges': self._coupling_edges,
            'router': field_router,
            'router_type': router,
            'n_edges': len(self._coupling_edges),
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
        for sid, label in seg.probes:
            sub = self.substrates.get(sid)
            if sub is None:
                continue
            self.probe_log.append(ProbeRecord(t=self.global_t, label=label, substrate_name=sub.name, psi=asnumpy(sub.psi).copy(), k_star=float(dominant_wavenumber(asnumpy(sub.psi), sub.dx, k_min=2 * np.pi / sub.params.L)), ipr=float(ipr(asnumpy(sub.psi), sub.dx)), crystallinity=float(crystallinity(asnumpy(sub.psi), sub.dx)), peak=float(peak_density(asnumpy(sub.psi)))))

    def _apply_router_gating(self, seg: Segment):
        """Recompute precision-weighted gating and stamp each edge with its
        effective weight.

        The FieldRouter reads each active substrate's FDT precision (and other
        observables) and returns per-edge effective kappa = base_kappa * weight.
        we store the weight on the edge so the per-step coupling in _step_one
        scales by it. all substrates keep running full mode; the router only
        modulates how strongly their fields couple (MoE gating), it never
        isolates a pillar.
        """
        from runtime.core.field_router import gather_observables
        router = seg.router
        active = {sid: s for sid, s in self.substrates.items() if s.active}
        if not active:
            return
        obs = gather_observables(active)
        effective = router.route(seg.edges, obs)
        for e in seg.edges:
            eff = effective.get((e.src_id, e.dst_id))
            if eff is None or e.kappa == 0:
                e._router_weight = 1.0
            else:
                e._router_weight = eff / e.kappa

    def run(self, verbose: bool=False) -> dict:
        if not self.segments:
            return {'diverged': False, 'global_t': self.global_t}
        diverged = False
        diverged_at = None
        for seg_i, seg in enumerate(self.segments):
            self._apply_segment_setup(seg)
            edges_by_dst: dict[int, list[CouplingEdge]] = {}
            for e in seg.edges:
                edges_by_dst.setdefault(e.dst_id, []).append(e)
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
                    self._step_one(sub, rho_now, edges_by_dst.get(sid, []))
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
                        except Exception:
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
                except Exception:
                    pass
        for sub in self.substrates.values():
            if sub._rec_buf:
                sub.density_traj = np.array(sub._rec_buf).T
                sub.t_traj = np.array(sub._t_buf)
        result = {'diverged': diverged, 'global_t': self.global_t}
        if diverged:
            result['diverged_at'] = diverged_at
        return result

    def _step_one(self, sub: Substrate, rho_now: dict[int, np.ndarray], inbound_edges: list[CouplingEdge]):
        if sub.D == 2:
            self._step_one_2d(sub, rho_now, inbound_edges)
            return
        if sub.D == 3:
            self._step_one_3d(sub, rho_now, inbound_edges)
            return
        xp = sub.xp
        p = sub.params
        psi = sub.psi
        M = len(sub.nu_arr)
        psi = xp.fft.ifft(xp.fft.fft(psi) * sub.half_lin)
        rho = xp.abs(psi) ** 2
        if M > 0:
            sub.y = sub.ou_decay_half[:, None] * sub.y + (1.0 - sub.ou_decay_half)[:, None] * rho
            V_mem = (sub.lam_arr[:, None] * sub.y).sum(axis=0)
        else:
            V_mem = xp.asarray(0.0)
        V_couple = xp.zeros(p.N)
        for e in inbound_edges:
            src_rho = rho_now.get(e.src_id)
            if src_rho is None:
                continue
            eff_kappa = e.kappa * getattr(e, '_router_weight', 1.0)
            if e.kappa_modulator is not None:
                try:
                    eff_kappa = e.kappa * float(e.kappa_modulator(rho_now))
                except Exception:
                    pass
            src_rho_xp = xp.asarray(src_rho) if not isinstance(src_rho, type(xp.zeros(1))) else src_rho
            if src_rho_xp.shape != (p.N,):
                src_rho_xp = xp.asarray(_project_to_shape(asnumpy(src_rho_xp), (p.N,)))
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
        if M > 0:
            sub.y = sub.ou_decay_half[:, None] * sub.y + (1.0 - sub.ou_decay_half)[:, None] * rho
        if sub.noise_amp > 0:
            xi = sub.rng.standard_normal(p.N)
            xip = sub.rng.standard_normal(p.N)
            psi = psi + sub.noise_amp * (xi + 1j * xip) / xp.sqrt(2.0)
        psi = xp.fft.ifft(xp.fft.fft(psi) * sub.half_lin)
        sub.psi = psi

    def _step_one_2d(self, sub, rho_now, inbound_edges):
        xp = sub.xp
        p = sub.params
        psi = sub.psi
        M = len(sub.nu_arr)
        psi = xp.fft.ifft2(xp.fft.fft2(psi) * sub.half_lin)
        rho = xp.abs(psi) ** 2
        if M > 0:
            sub.y = sub.ou_decay_half[:, None, None] * sub.y + (1.0 - sub.ou_decay_half)[:, None, None] * rho
            V_mem = (sub.lam_arr[:, None, None] * sub.y).sum(axis=0)
        else:
            V_mem = xp.asarray(0.0)
        V_couple = xp.zeros(sub.grid_shape)
        for e in inbound_edges:
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
                except Exception:
                    pass
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
        if M > 0:
            sub.y = sub.ou_decay_half[:, None, None] * sub.y + (1.0 - sub.ou_decay_half)[:, None, None] * rho
        if sub.noise_amp > 0:
            xi = sub.rng.standard_normal(sub.grid_shape)
            xip = sub.rng.standard_normal(sub.grid_shape)
            psi = psi + sub.noise_amp * (xi + 1j * xip) / xp.sqrt(2.0)
        psi = xp.fft.ifft2(xp.fft.fft2(psi) * sub.half_lin)
        sub.psi = psi

    def _step_one_3d(self, sub, rho_now, inbound_edges):
        xp = sub.xp
        p = sub.params
        psi = sub.psi
        M = len(sub.nu_arr)
        psi = xp.fft.ifftn(xp.fft.fftn(psi) * sub.half_lin)
        rho = xp.abs(psi) ** 2
        if M > 0:
            sub.y = sub.ou_decay_half[:, None, None, None] * sub.y + (1.0 - sub.ou_decay_half)[:, None, None, None] * rho
            V_mem = (sub.lam_arr[:, None, None, None] * sub.y).sum(axis=0)
        else:
            V_mem = xp.asarray(0.0)
        V_couple = xp.zeros(sub.grid_shape)
        for e in inbound_edges:
            src_rho = rho_now.get(e.src_id)
            if src_rho is None:
                continue
            src_rho_xp = xp.asarray(src_rho)
            eff_kappa = e.kappa * getattr(e, '_router_weight', 1.0)
            if e.kappa_modulator is not None:
                try:
                    eff_kappa = e.kappa * float(e.kappa_modulator(rho_now))
                except Exception:
                    pass
            if src_rho_xp.shape != sub.grid_shape:
                src_rho_xp = xp.asarray(_project_to_shape(asnumpy(src_rho_xp), sub.grid_shape))
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
        if M > 0:
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
        """Launch K FULL integrations in parallel with different seeds.

        Each substrate gets a different noise realization.  After all finish,
        select the best by emergent criterion (min_energy or max_crystallinity).
        Optionally prune stalled trajectories early.

        Parameters
        ----------
        K : int
            Number of parallel runs (each uses a different seed offset).
        T : float
            Integration time per run.
        select : str
            "min_energy" or "max_crystallinity" for the selection criterion.
        prune : str
            "stalled" to prune runs where crystallinity stalls for
            prune_patience consecutive checkpoints, "none" to disable.
        prune_patience : int
            Number of consecutive non-improvement checkpoints before pruning.
        verbose : bool

        Returns
        -------
        dict with selected index, per-member observables, selection criterion.
        """
        from runtime.core.solver import integrate, TriadParams
        from runtime.physics.observables import energy as _energy
        import time as _time

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
                r = integrate(p_k)
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
        """Run integrate() with early stopping when crystallinity stalls."""
        from runtime.core.solver import integrate, TriadParams
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
            r = integrate(p_chunk, psi0=psi, y0=y)
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
