from __future__ import annotations
import numpy as np
from runtime.solver import TriadParams
DEFAULT_L = 32.0
DEFAULT_N = 128
DEFAULT_DT = 0.005

def register_params(seed: int=0, L: float=DEFAULT_L, N: int=DEFAULT_N, dt: float=DEFAULT_DT, bit_width: int=1) -> TriadParams:
    return TriadParams(L=L, N=N, dt=dt, T=1.0, hbar=1.0, m=1.0, V_ext='harmonic', omega=0.05, Lambda=-0.5, alpha=0.15, sigma=1.5, Gamma=0.05, f_FDT=0.002, nu=(2.0, 0.5, 0.1), lam=(-0.3, -0.2, -0.1), mode='full', seed=seed, record_every=4)

def gate_params(seed: int=0, L: float=DEFAULT_L, N: int=DEFAULT_N, dt: float=DEFAULT_DT) -> TriadParams:
    return TriadParams(L=L, N=N, dt=dt, T=1.0, hbar=1.0, m=1.0, V_ext=None, omega=0.0, Lambda=-2.0, alpha=0.15, sigma=1.5, Gamma=0.05, f_FDT=0.002, nu=(5.0, 0.5, 0.1), lam=(-0.4, -0.2, -0.1), mode='full', seed=seed, record_every=4)

def memory_cell_params(seed: int=0, L: float=DEFAULT_L, N: int=DEFAULT_N, dt: float=DEFAULT_DT) -> TriadParams:
    return TriadParams(L=L, N=N, dt=dt, T=1.0, hbar=1.0, m=1.0, V_ext='harmonic', omega=0.03, Lambda=-0.3, alpha=0.15, sigma=1.5, Gamma=0.01, f_FDT=0.0004, nu=(0.5, 0.05, 0.005), lam=(-0.3, -0.5, -0.8), mode='full', seed=seed, record_every=4)

def vext_add_gate(L: float, N: int, *, src_a_idx: int=0, src_b_idx: int=1, delta_a: float=-3.0, delta_b: float=+3.0, bump_w: float=1.5, bump_amp: float=-1.0, omega: float=0.05) -> callable:

    def f(x):
        V_harm = 0.5 * omega ** 2 * x ** 2
        bump_a = bump_amp * np.exp(-(x - delta_a) ** 2 / (2 * bump_w ** 2))
        bump_b = bump_amp * np.exp(-(x - delta_b) ** 2 / (2 * bump_w ** 2))
        return V_harm + bump_a + bump_b
    return f

def vext_sub_gate(L: float, N: int, *, delta_a: float=-3.0, delta_b: float=+3.0, bump_w: float=1.5, bump_amp_pos: float=-1.0, bump_amp_neg: float=+1.0, omega: float=0.05) -> callable:

    def f(x):
        V_harm = 0.5 * omega ** 2 * x ** 2
        bump_a = bump_amp_pos * np.exp(-(x - delta_a) ** 2 / (2 * bump_w ** 2))
        bump_b = bump_amp_neg * np.exp(-(x - delta_b) ** 2 / (2 * bump_w ** 2))
        return V_harm + bump_a + bump_b
    return f

def vext_double_well(L: float, N: int, *, well_sep: float=6.0, barrier_h: float=2.0, well_w: float=1.5) -> callable:

    def f(x):
        well_a = -np.exp(-(x + well_sep / 2) ** 2 / (2 * well_w ** 2))
        well_b = -np.exp(-(x - well_sep / 2) ** 2 / (2 * well_w ** 2))
        barrier = barrier_h * np.exp(-x ** 2 / 0.5)
        return well_a + well_b + barrier
    return f

def vext_single_wide_well(L: float, N: int, *, well_w: float=4.0) -> callable:

    def f(x):
        return -np.exp(-x ** 2 / (2 * well_w ** 2))
    return f

def vext_ramp(L: float, N: int, *, beta: float=0.05, omega: float=0.05) -> callable:

    def f(x):
        return 0.5 * omega ** 2 * x ** 2 + beta * x
    return f

def vext_zero(L: float, N: int) -> callable:

    def f(x):
        return np.zeros_like(x)
    return f
GATE_CATALOGUE = {'ADD_K': dict(vext=vext_add_gate, T=5.0, kappa=-3.0, bump_amp=-1.0), 'SUB_K': dict(vext=vext_sub_gate, T=5.0, kappa=-3.0, bump_amp=-1.0), 'MUL_K': dict(vext=vext_add_gate, T=8.0, kappa=-3.0, bump_amp=-1.0), 'AND_C': dict(vext=vext_double_well, T=3.0, kappa=-3.0), 'OR_C': dict(vext=vext_single_wide_well, T=3.0, kappa=-3.0), 'NOT_C': dict(vext=vext_zero, T=3.0, kappa=-3.0), 'CMP_SHIFT': dict(vext=vext_ramp, T=4.0, kappa=-3.0), 'MOV_COPY': dict(vext=vext_zero, T=5.0, kappa=-3.0), 'ZERO': dict(vext=vext_zero, T=2.0, kappa=0.0), 'INC': dict(vext=vext_add_gate, T=3.0, kappa=-3.0, bump_amp=-1.0), 'DEC': dict(vext=vext_sub_gate, T=3.0, kappa=-3.0, bump_amp=-1.0), 'XOR': dict(vext=vext_double_well, T=3.0, kappa=-3.0), 'SHIFT_LEFT': dict(vext=vext_zero, T=3.0, kappa=-3.0), 'SHIFT_RIGHT': dict(vext=vext_zero, T=3.0, kappa=-3.0)}