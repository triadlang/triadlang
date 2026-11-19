from __future__ import annotations
from typing import Callable
from runtime.core.solver import TriadParams

# every regime is Triad-compliant by construction: the three pillars are
# present and coupled, never isolated.
#   P1 (kinetic + fractional dispersion): alpha > 0 and sigma != 2.0, so the
#      fractional laplacian (-Delta)^(sigma/2) contributes a non-trivial term.
#      sigma = 2.0 would collapse it back to the plain laplacian, so it is
#      avoided. sub-diffusive / anomalous-transport phenomenology uses lower
#      sigma (closer to 1), wave-like phenomenology uses sigma nearer 2.
#   P2 (memory / self-reference): Lambda != 0 (cubic self-interaction) and at
#      least one non-zero lam entry (memory-field feedback y_j).
#   P3 (open system): Gamma > 0 (dissipation) and f_FDT > 0 (FDT noise).
# alpha values are kept small relative to the kinetic term so the named
# phenomenology of each regime is preserved; the fractional part shapes the
# dispersion tail rather than dominating it.

def regime_B0(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    return TriadParams(L=L, N=N, dt=dt, T=20.0, hbar=1.0, m=1.0, V_ext='harmonic', omega=0.05, Lambda=-0.5, alpha=0.15, sigma=1.5, Gamma=0.05, f_FDT=0.002, nu=(2.0, 0.5, 0.1), lam=(-0.3, -0.2, -0.1), mode='full', seed=seed, record_every=4)

def regime_dispersive(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    # dispersive transport: anomalous (super-diffusive) spreading, so sigma
    # below 2 gives a heavier dispersion tail than ordinary diffusion.
    return TriadParams(L=L, N=N, dt=dt, T=20.0, hbar=1.0, m=1.0, V_ext=None, omega=0.0, Lambda=-0.05, alpha=0.2, sigma=1.6, Gamma=0.02, f_FDT=0.001, nu=(2.0, 0.5), lam=(-0.05, -0.03), mode='full', seed=seed, record_every=4)

def regime_anti_collapse(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    # strong attractive cubic pushes toward collapse; fractional dispersion
    # plus memory feedback resist it. sigma below 2 strengthens long-range
    # dispersion that counters the local collapse.
    return TriadParams(L=L, N=N, dt=dt, T=6.0, hbar=1.0, m=1.0, V_ext=None, omega=0.0, Lambda=-10.0, alpha=0.3, sigma=1.5, Gamma=0.01, f_FDT=0.001, nu=(10.0, 0.5), lam=(3.0, 1.0), mode='full', seed=seed, record_every=20)

def regime_R5_crystal(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    # crystallizing regime: standing-wave-like structure, sigma near 2 keeps
    # the dispersion close to wave behaviour while staying strictly fractional.
    return TriadParams(L=L, N=N, dt=dt, T=15.0, hbar=1.0, m=1.0, V_ext=None, omega=0.0, Lambda=-8.0, alpha=0.12, sigma=1.8, Gamma=0.01, f_FDT=0.001, nu=(10.0, 0.5), lam=(1.125, 0.375), mode='full', seed=seed, record_every=4)

def regime_thermal_pure(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    # named for its strong thermalization (large Gamma, FDT noise) but still a
    # full Triad system: nonlinear self-interaction and memory feedback are
    # active alongside the open-system dissipation.
    return TriadParams(L=L, N=N, dt=dt, T=20.0, hbar=1.0, m=1.0, V_ext='harmonic', omega=0.05, Lambda=-0.3, alpha=0.15, sigma=1.5, Gamma=0.08, f_FDT=0.003, nu=(1.0, 0.2), lam=(-0.1, -0.05), mode='full', seed=seed, record_every=4)

def regime_B0_3d(seed: int=0, L: float=12.0, N: int=24, dt: float=0.005) -> TriadParams:
    return TriadParams(L=L, N=N, dt=dt, T=2.0, hbar=1.0, m=1.0, V_ext='harmonic', omega=0.3, Lambda=-0.3, alpha=0.15, sigma=1.5, Gamma=0.05, f_FDT=0.002, nu=(2.0, 0.5, 0.1), lam=(-0.2, -0.1, -0.05), mode='full', seed=seed, record_every=4)

def regime_register_legacy(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005, bit_width: int=1) -> TriadParams:
    # register-style narrow well used to hold bit patterns. previously this
    # zeroed P3 for back-compat; now it carries dissipation and FDT noise like
    # every other regime, with a short integration time so the stored pattern
    # is still legible.
    return TriadParams(L=L, N=N, dt=dt, T=1.0, hbar=1.0, m=1.0, V_ext='harmonic', omega=0.005, Lambda=-0.05, alpha=0.08, sigma=1.7, Gamma=0.01, f_FDT=0.0004, nu=(2.0, 0.2), lam=(-0.05, -0.05), mode='full', seed=seed, record_every=4)

def regime_HodgkinHuxley(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    # excitable membrane dynamics: pulse-like propagation, moderate dispersion.
    return TriadParams(L=L, N=N, dt=dt, T=20.0, hbar=1.0, m=1.0, V_ext='harmonic', omega=0.05, Lambda=-1.2, alpha=0.12, sigma=1.6, Gamma=0.04, f_FDT=0.0016, nu=(5.0, 0.3, 0.04), lam=(-0.4, -0.25, -0.1), mode='full', seed=seed, record_every=4)

def regime_MaxwellWiechert(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    # viscoelastic relaxation spectrum: many memory timescales, slow creep, so
    # a heavier dispersion tail (sigma well below 2) matches the anomalous
    # stress relaxation.
    return TriadParams(L=L, N=N, dt=dt, T=30.0, hbar=1.0, m=1.0, V_ext='harmonic', omega=0.02, Lambda=-0.1, alpha=0.1, sigma=1.4, Gamma=0.02, f_FDT=0.001, nu=(5.0, 0.5, 0.05, 0.005), lam=(-0.05, -0.08, -0.12, -0.15), mode='full', seed=seed, record_every=4)

def regime_ENSO_recharge(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    # delayed-oscillator climate cycle: slow recharge memory, anomalous
    # transport of heat content across the basin.
    return TriadParams(L=L, N=N, dt=dt, T=50.0, hbar=1.0, m=1.0, V_ext='harmonic', omega=0.03, Lambda=-0.4, alpha=0.12, sigma=1.5, Gamma=0.03, f_FDT=0.0015, nu=(0.25, 0.05), lam=(-0.5, -0.3), mode='full', seed=seed, record_every=4)

def regime_LSV_market(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    # local-stochastic-volatility market: heavy-tailed jumps, so a low sigma
    # captures the super-diffusive (Levy-like) spreading of the density.
    return TriadParams(L=L, N=N, dt=dt, T=30.0, hbar=1.0, m=1.0, V_ext=None, omega=0.0, Lambda=-0.2, alpha=0.18, sigma=1.3, Gamma=0.02, f_FDT=0.004, nu=(1.0, 0.1, 0.01), lam=(-0.2, -0.3, -0.4), mode='full', seed=seed, record_every=4)

def regime_Eigen_hypercycle(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    # autocatalytic hypercycle: cooperative growth with memory feedback.
    return TriadParams(L=L, N=N, dt=dt, T=30.0, hbar=1.0, m=1.0, V_ext='harmonic', omega=0.04, Lambda=-0.6, alpha=0.13, sigma=1.6, Gamma=0.03, f_FDT=0.0012, nu=(2.0, 0.2), lam=(-0.3, -0.15), mode='full', seed=seed, record_every=4)

def regime_Belousov_Zhabotinsky(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    # oscillating chemical reaction with travelling waves: dispersion near
    # wave-like but strictly fractional.
    return TriadParams(L=L, N=N, dt=dt, T=25.0, hbar=1.0, m=1.0, V_ext='harmonic', omega=0.03, Lambda=-1.5, alpha=0.12, sigma=1.7, Gamma=0.04, f_FDT=0.0016, nu=(5.0, 0.5), lam=(-0.4, -0.3), mode='full', seed=seed, record_every=4)

def regime_Cepheid_pulsator(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    # radial stellar pulsation: strongly wave-like, sigma near 2 but fractional.
    return TriadParams(L=L, N=N, dt=dt, T=40.0, hbar=1.0, m=1.0, V_ext='harmonic', omega=0.15, Lambda=-0.05, alpha=0.1, sigma=1.85, Gamma=0.02, f_FDT=0.0008, nu=(0.5, 0.1), lam=(-0.1, -0.05), mode='full', seed=seed, record_every=4)

def regime_Cosmological_inflation(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    # rapid early expansion: strong cubic drive, broad dispersion.
    return TriadParams(L=L, N=N, dt=dt, T=10.0, hbar=1.0, m=1.0, V_ext=None, omega=0.0, Lambda=-2.5, alpha=0.2, sigma=1.5, Gamma=0.02, f_FDT=0.005, nu=(8.0, 0.8), lam=(-0.6, -0.4), mode='full', seed=seed, record_every=4)

def regime_DarkMatter_halo(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    # self-gravitating fuzzy-dark-matter halo: long-range, slow, anomalous
    # transport, so a low sigma gives the extended dispersion of the halo.
    return TriadParams(L=L, N=N, dt=dt, T=60.0, hbar=1.0, m=1.0, V_ext='harmonic', omega=0.01, Lambda=-0.15, alpha=0.1, sigma=1.4, Gamma=0.005, f_FDT=0.0001, nu=(0.05, 0.005), lam=(-0.3, -0.5), mode='full', seed=seed, record_every=4)

def regime_England_autopoietic(seed: int=0, L: float=32.0, N: int=128, dt: float=0.005) -> TriadParams:
    return TriadParams(L=L, N=N, dt=dt, T=25.0, hbar=1.0, m=1.0, V_ext='harmonic', omega=0.03, Lambda=-0.8, alpha=0.15, sigma=1.5, Gamma=0.08, f_FDT=0.003, nu=(3.0, 0.3, 0.03), lam=(-0.3, -0.5, -0.7), mode='full', seed=seed, record_every=4)
_REGISTRY: dict[str, Callable[..., TriadParams]] = {'B0': regime_B0, 'B0_3d': regime_B0_3d, 'dispersive': regime_dispersive, 'anti_collapse': regime_anti_collapse, 'R5_crystal': regime_R5_crystal, 'thermal_pure': regime_thermal_pure, 'register_legacy': regime_register_legacy, 'HodgkinHuxley': regime_HodgkinHuxley, 'MaxwellWiechert': regime_MaxwellWiechert, 'ENSO_recharge': regime_ENSO_recharge, 'LSV_market': regime_LSV_market, 'England_autopoietic': regime_England_autopoietic, 'Eigen_hypercycle': regime_Eigen_hypercycle, 'Belousov_Zhabotinsky': regime_Belousov_Zhabotinsky, 'Cepheid_pulsator': regime_Cepheid_pulsator, 'Cosmological_inflation': regime_Cosmological_inflation, 'DarkMatter_halo': regime_DarkMatter_halo}

def resolve_regime(name: str, **kwargs) -> TriadParams:
    if name not in _REGISTRY:
        raise KeyError(f'unknown regime {name!r}; available: {sorted(_REGISTRY)}')
    factory = _REGISTRY[name]
    import inspect
    sig = inspect.signature(factory)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return factory(**accepted)

def list_regimes() -> list[str]:
    return sorted(_REGISTRY)
