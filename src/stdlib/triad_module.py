from dataclasses import dataclass

from runtime.backend import asnumpy as _asnumpy
from runtime.core.fast_solver import fast_integrate
from runtime.core.multi_runtime import CouplingLink, MultiRuntime, Segment
from runtime.core.solver import TriadParams
from runtime.physics.observables import (
    crystallinity,
    dominant_wavenumber,
    fwhm,
    ipr,
    participation_ratio,
    peak_density,
    power_spectrum,
)
from runtime.physics.observables import norm as obs_norm
from stdlib.regimes import resolve_regime
from triad import ntri as np


@dataclass
class _FastResult:
    psi: object = None
    x: object = None
    dx: float = 0.0
    params: object = None
    density: object = None
    k_star: float = 0.0
    crystallinity: float = 0.0
    peak: float = 0.0
    ipr: float = 0.0
    norm: float = 0.0
    fwhm: float = float('nan')

class _SolveResult:

    def __init__(self, sub, runtime):
        self._sub = sub
        self._rt = runtime
        self.psi = sub.psi
        self.x = sub.x
        self.dx = sub.dx
        self.params = sub.params
        L = sub.params.L
        k_min = 2.0 * np.pi / L
        # observáveis pós-processados no hospedeiro: materializa o estado do backend.
        psi_h = _asnumpy(sub.psi)
        self.k_star = float(dominant_wavenumber(psi_h, sub.dx, k_min=k_min))
        self.crystallinity = float(crystallinity(psi_h, sub.dx))
        self.peak = float(peak_density(psi_h))
        self.ipr = float(ipr(psi_h, sub.dx))
        self.norm = float((np.abs(psi_h) ** 2).sum() * sub.dx)
        try:
            self.fwhm = float(fwhm(psi_h, sub.dx))
        except (ValueError, ArithmeticError, TypeError):
            self.fwhm = float('nan')
        self.density = np.abs(psi_h) ** 2

    def __repr__(self):
        return f'SolveResult(k_star={self.k_star:.4f}, crystallinity={self.crystallinity:.4f}, peak={self.peak:.4f}, norm={self.norm:.4f})'

def _has_result_attrs(obj):
    return hasattr(obj, 'psi') and hasattr(obj, 'dx') and hasattr(obj, 'k_star')

def regime(name, **overrides):
    p = resolve_regime(name, seed=0, L=32.0, N=128, dt=0.005)
    if overrides:
        d = p.__dict__.copy()
        d.update(overrides)
        p = TriadParams(**d)
    return p

def solve(params, T=None):
    if T is not None:
        d = params.__dict__.copy()
        d['T'] = T
        params = TriadParams(**d)
    rt = MultiRuntime(dt=params.dt, record_every=getattr(params, 'record_every', 4))
    sub = rt.add_substrate('main', params)
    duration = params.T
    rt.add_segment(Segment(t_start=0.0, t_end=duration, links=[]))
    rt.global_t = duration
    rt.run(verbose=False)
    return _SolveResult(sub, rt)

def fast_solve(params, T=None):
    if T is not None:
        d = params.__dict__.copy()
        d['T'] = T
        params = TriadParams(**d)
    out = fast_integrate(params)
    psi = out['psi_final']
    dx = out['dx']
    L = params.L
    k_min = 2.0 * np.pi / L
    density = np.abs(psi) ** 2

    r = _FastResult()
    r.psi = psi
    r.x = out['x']
    r.dx = dx
    r.params = params
    r.density = density
    r.k_star = float(dominant_wavenumber(psi, dx, k_min=k_min))
    r.crystallinity = float(crystallinity(psi, dx))
    r.peak = float(peak_density(psi))
    r.ipr = float(ipr(psi, dx))
    r.norm = float(density.sum() * dx)
    try:
        r.fwhm = float(fwhm(psi, dx))
    except (ValueError, ArithmeticError, TypeError):
        r.fwhm = float('nan')
    return r

def solve_coupled(substrates, links, T=None):
    if not substrates:
        raise ValueError('need at least one substrate')
    dt = substrates[0][1].dt
    rt = MultiRuntime(dt=dt, record_every=4)
    subs = {}
    for name, params in substrates:
        subs[name] = rt.add_substrate(name, params)
    duration = T or substrates[0][1].T
    ce = []
    for src, dst, kappa in links:
        ce.append(CouplingLink(src_id=subs[src].id, dst_id=subs[dst].id, kappa=kappa))
    rt.add_segment(Segment(t_start=0.0, t_end=duration, links=ce))
    rt.run(verbose=False)
    return {name: _SolveResult(sub, rt) for name, sub in subs.items()}

def k_star(psi_or_result, dx=None):
    if _has_result_attrs(psi_or_result):
        return psi_or_result.k_star
    if dx is None:
        raise ValueError('dx required when passing raw psi array')
    L = dx * len(psi_or_result)
    return float(dominant_wavenumber(psi_or_result, dx, k_min=2 * np.pi / L))

def crystal(psi_or_result, dx=None):
    if _has_result_attrs(psi_or_result):
        return psi_or_result.crystallinity
    if dx is None:
        raise ValueError('dx required when passing raw psi array')
    return float(crystallinity(psi_or_result, dx))

def peak(psi_or_result):
    if _has_result_attrs(psi_or_result):
        return psi_or_result.peak
    return float(peak_density(psi_or_result))

def norm(psi_or_result, dx=None):
    if _has_result_attrs(psi_or_result):
        return psi_or_result.norm
    if dx is None:
        raise ValueError('dx required when passing raw psi array')
    return float(obs_norm(psi_or_result, dx))

def get_ipr(psi_or_result, dx=None):
    if _has_result_attrs(psi_or_result):
        return psi_or_result.ipr
    if dx is None:
        raise ValueError('dx required when passing raw psi array')
    return float(ipr(psi_or_result, dx))

def get_fwhm(psi_or_result, dx=None):
    if _has_result_attrs(psi_or_result):
        return psi_or_result.fwhm
    if dx is None:
        raise ValueError('dx required when passing raw psi array')
    return float(fwhm(psi_or_result, dx))

def spectrum(psi_or_result, dx=None):
    if _has_result_attrs(psi_or_result):
        psi = psi_or_result.psi
        dx = psi_or_result.dx
    else:
        psi = psi_or_result
    if dx is None:
        raise ValueError('dx required when passing raw psi array')
    k, S = power_spectrum(psi, dx)
    # pós-processamento no hospedeiro: mesma exibição em qualquer backend.
    return {'k': _asnumpy(k), 'S': _asnumpy(S)}

def pr(psi_or_result, dx=None):
    if _has_result_attrs(psi_or_result):
        return float(participation_ratio(psi_or_result.psi, psi_or_result.dx))
    if dx is None:
        raise ValueError('dx required when passing raw psi array')
    return float(participation_ratio(psi_or_result, dx))
