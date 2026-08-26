
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from runtime.core.solver import TriadParams
from runtime.ml.tensor import TriadTensor
from triad import ntri as np


class ParamBridge:

    def __init__(self, base: TriadParams):
        self.base = base
        self._registrations: dict[str, tuple[str, Callable]] = {}

    def register(self, field_name: str, param: TriadTensor, reducer: str = 'mean'):

        if callable(reducer):
            fn = reducer
        elif reducer == 'mean':
            fn = lambda d: float(np.mean(d))
        elif reducer == 'first':
            fn = lambda d: float(d.flat[0])
        elif reducer == 'tuple':
            fn = lambda d: tuple(float(v) for v in np.asarray(d).reshape(-1))
        elif reducer == 'array':
            fn = lambda d: np.asarray(d)
        else:
            raise ValueError(f"unknown reducer {reducer!r}")
        self._registrations[field_name] = (param, fn)

    def build(self) -> TriadParams:

        overrides = {}
        for field_name, (param, fn) in self._registrations.items():
            overrides[field_name] = fn(param._data)
        return replace(self.base, **overrides)

    def update(self, p: TriadParams) -> TriadParams:

        return self.build()

def ssm_bridge(layer, base: TriadParams | None = None) -> ParamBridge:

    if base is None:
        base = TriadParams(N=layer.d_state, dt=0.05, T=0.1, mode='triad')
    b = ParamBridge(base)
    b.register('omega', layer.omega, reducer='array')
    b.register('Lambda', layer.Lambda, reducer='mean')
    b.register('nu', layer.nu, reducer='tuple')
    b.register('lam', layer.lam, reducer=lambda d: tuple(float(v) for v in np.asarray(d).reshape(-1)))
    return b

def triadblock_bridge(layer, base: TriadParams | None = None) -> ParamBridge:

    if base is None:
        base = TriadParams(N=layer.N, dt=layer.dt, T=layer.dt*2, mode='triad')
    b = ParamBridge(base)
    b.register('omega', layer.p_omega, reducer='array')
    b.register('Lambda', layer.p_Lambda, reducer='mean')
    b.register('Gamma', layer.p_Gamma, reducer='mean')
    b.register('alpha', layer.p_alpha, reducer='mean')
    b.register('sigma', layer.p_sigma, reducer=lambda d: float(np.mean(d)) + 0.5)
    b.register('hbar', layer.p_hbar, reducer=lambda d: float(abs(d.item())) + 0.1)
    b.register('m', layer.p_m, reducer=lambda d: float(abs(d.item())) + 0.1)
    b.register('kT', layer.p_kT, reducer=lambda d: float(abs(d.item())))
    b.register('nu', layer.p_nu, reducer='tuple')
    b.register('lam', layer.p_lam, reducer=lambda d: tuple(float(v) for v in np.asarray(d).reshape(-1)))
    return b
