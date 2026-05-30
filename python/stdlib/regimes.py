from __future__ import annotations
from typing import Callable
from runtime.solver import TriadParams

def regime_B0(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    return TriadParams(L=L, N=N, dt=dt, T=20.0, hbar=1.0, m=1.0, V_ext='harmonic', omega=0.05, Lambda=-0.5, alpha=0.15, sigma=1.5, Gamma=0.05, f_FDT=0.002, nu=(2.0, 0.5, 0.1), lam=(-0.3, -0.2, -0.1), mode='full', seed=seed, record_every=4)

def regime_dispersive(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    return TriadParams(L=L, N=N, dt=dt, T=20.0, hbar=1.0, m=1.0, V_ext=None, omega=0.0, Lambda=-0.01, alpha=0.0, sigma=2.0, Gamma=0.05, f_FDT=0.002, nu=(2.0,), lam=(-0.01,), mode='full', seed=seed, record_every=4)

def regime_anti_collapse(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    return TriadParams(L=L, N=N, dt=dt, T=6.0, hbar=1.0, m=1.0, V_ext=None, omega=0.0, Lambda=-10.0, alpha=0.0, sigma=2.0, Gamma=0.01, f_FDT=0.001, nu=(10.0, 0.5), lam=(3.0, 1.0), mode='full', seed=seed, record_every=20)

def regime_R5_crystal(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    return TriadParams(L=L, N=N, dt=dt, T=15.0, hbar=1.0, m=1.0, V_ext=None, omega=0.0, Lambda=-8.0, alpha=0.0, sigma=2.0, Gamma=0.01, f_FDT=0.001, nu=(10.0, 0.5), lam=(1.125, 0.375), mode='full', seed=seed, record_every=4)

def regime_thermal_pure(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    return TriadParams(L=L, N=N, dt=dt, T=20.0, hbar=1.0, m=1.0, V_ext='harmonic', omega=0.05, Lambda=0.0, alpha=0.0, sigma=2.0, Gamma=0.05, f_FDT=0.002, nu=(1.0,), lam=(0.0,), mode='thermal', seed=seed, record_every=4)

def regime_B0_3d(seed: int=0, L: float=12.0, N: int=24, dt: float=0.005) -> TriadParams:
    return TriadParams(L=L, N=N, dt=dt, T=2.0, hbar=1.0, m=1.0, V_ext='harmonic', omega=0.3, Lambda=-0.3, alpha=0.15, sigma=1.5, Gamma=0.05, f_FDT=0.002, nu=(2.0, 0.5, 0.1), lam=(-0.2, -0.1, -0.05), mode='full', seed=seed, record_every=4)

def regime_register_legacy(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005, bit_width: int=1) -> TriadParams:
    return TriadParams(L=L, N=N, dt=dt, T=1.0, hbar=1.0, m=1.0, V_ext='harmonic', omega=0.005, Lambda=-0.05, alpha=0.0, sigma=2.0, Gamma=0.0, f_FDT=0.0, nu=(2.0, 0.2), lam=(-0.05, -0.05), mode='full', seed=seed, record_every=4)

def regime_HodgkinHuxley(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    return TriadParams(L=L, N=N, dt=dt, T=20.0, hbar=1.0, m=1.0, V_ext='harmonic', omega=0.05, Lambda=-1.2, alpha=0.0, sigma=2.0, Gamma=0.04, f_FDT=0.0016, nu=(5.0, 0.3, 0.04), lam=(-0.4, -0.25, -0.1), mode='full', seed=seed, record_every=4)

def regime_MaxwellWiechert(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    return TriadParams(L=L, N=N, dt=dt, T=30.0, hbar=1.0, m=1.0, V_ext='harmonic', omega=0.02, Lambda=-0.1, alpha=0.0, sigma=2.0, Gamma=0.02, f_FDT=0.001, nu=(5.0, 0.5, 0.05, 0.005), lam=(-0.05, -0.08, -0.12, -0.15), mode='full', seed=seed, record_every=4)

def regime_ENSO_recharge(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    return TriadParams(L=L, N=N, dt=dt, T=50.0, hbar=1.0, m=1.0, V_ext='harmonic', omega=0.03, Lambda=-0.4, alpha=0.0, sigma=2.0, Gamma=0.03, f_FDT=0.0015, nu=(0.25, 0.05), lam=(-0.5, -0.3), mode='full', seed=seed, record_every=4)

def regime_LSV_market(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    return TriadParams(L=L, N=N, dt=dt, T=30.0, hbar=1.0, m=1.0, V_ext=None, omega=0.0, Lambda=-0.2, alpha=0.0, sigma=2.0, Gamma=0.02, f_FDT=0.004, nu=(1.0, 0.1, 0.01), lam=(-0.2, -0.3, -0.4), mode='full', seed=seed, record_every=4)

def regime_Eigen_hypercycle(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    return TriadParams(L=L, N=N, dt=dt, T=30.0, hbar=1.0, m=1.0, V_ext='harmonic', omega=0.04, Lambda=-0.6, alpha=0.0, sigma=2.0, Gamma=0.03, f_FDT=0.0012, nu=(2.0, 0.2), lam=(-0.3, -0.15), mode='full', seed=seed, record_every=4)

def regime_Belousov_Zhabotinsky(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    return TriadParams(L=L, N=N, dt=dt, T=25.0, hbar=1.0, m=1.0, V_ext='harmonic', omega=0.03, Lambda=-1.5, alpha=0.0, sigma=2.0, Gamma=0.04, f_FDT=0.0016, nu=(5.0, 0.5), lam=(-0.4, -0.3), mode='full', seed=seed, record_every=4)

def regime_Cepheid_pulsator(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    return TriadParams(L=L, N=N, dt=dt, T=40.0, hbar=1.0, m=1.0, V_ext='harmonic', omega=0.15, Lambda=-0.05, alpha=0.0, sigma=2.0, Gamma=0.02, f_FDT=0.0008, nu=(0.5,), lam=(-0.1,), mode='full', seed=seed, record_every=4)

def regime_Cosmological_inflation(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    return TriadParams(L=L, N=N, dt=dt, T=10.0, hbar=1.0, m=1.0, V_ext=None, omega=0.0, Lambda=-2.5, alpha=0.0, sigma=2.0, Gamma=0.02, f_FDT=0.005, nu=(8.0, 0.8), lam=(-0.6, -0.4), mode='full', seed=seed, record_every=4)

def regime_DarkMatter_halo(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    return TriadParams(L=L, N=N, dt=dt, T=60.0, hbar=1.0, m=1.0, V_ext='harmonic', omega=0.01, Lambda=-0.15, alpha=0.0, sigma=2.0, Gamma=0.005, f_FDT=0.0001, nu=(0.05, 0.005), lam=(-0.3, -0.5), mode='full', seed=seed, record_every=4)

def regime_England_autopoietic(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    return TriadParams(L=L, N=N, dt=dt, T=25.0, hbar=1.0, m=1.0, V_ext='harmonic', omega=0.03, Lambda=-0.8, alpha=0.15, sigma=1.5, Gamma=0.08, f_FDT=0.003, nu=(3.0, 0.3, 0.03), lam=(-0.3, -0.5, -0.7), mode='full', seed=seed, record_every=4)
_REGISTRY: dict[str, Callable[..., TriadParams]] = {'B0': regime_B0, 'B0_3d': regime_B0_3d, 'dispersive': regime_dispersive, 'anti_collapse': regime_anti_collapse, 'R5_crystal': regime_R5_crystal, 'thermal_pure': regime_thermal_pure, 'register_legacy': regime_register_legacy, 'HodgkinHuxley': regime_HodgkinHuxley, 'MaxwellWiechert': regime_MaxwellWiechert, 'ENSO_recharge': regime_ENSO_recharge, 'LSV_market': regime_LSV_market, 'England_autopoietic': regime_England_autopoietic, 'Eigen_hypercycle': regime_Eigen_hypercycle, 'Belousov_Zhabotinsky': regime_Belousov_Zhabotinsky, 'Cepheid_pulsator': regime_Cepheid_pulsator, 'Cosmological_inflation': regime_Cosmological_inflation, 'DarkMatter_halo': regime_DarkMatter_halo}

def resolve_regime(name: str, **kwargs) -> TriadParams:
    if name not in _REGISTRY:
        raise KeyError(f'unknown regime {name!r}; available: {sorted(_REGISTRY)}')
    if name in _NONCOMPLIANT_REGIMES:
        import sys as _sys
        print(f'triadlang: warning — regime {name!r} is NOT Triad-compliant (P3 absent); kept for v1/v2 back-compat only.  New programs should use one of: B0, B0_3d, anti_collapse, R5_crystal, HodgkinHuxley, MaxwellWiechert, ENSO_recharge, LSV_market, England_autopoietic.', file=_sys.stderr)
    factory = _REGISTRY[name]
    import inspect
    sig = inspect.signature(factory)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return factory(**accepted)
_NONCOMPLIANT_REGIMES = {'register_legacy'}

def list_regimes() -> list[str]:
    return sorted(_REGISTRY)