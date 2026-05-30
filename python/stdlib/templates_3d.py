from __future__ import annotations
import numpy as np

def vext_double_well_3d(L: float, N: int, *, well_sep: float=6.0, barrier_h: float=2.0, well_w: float=1.5):

    def f(X, Y, Z):
        well_a = -np.exp(-(X + well_sep / 2) ** 2 / (2 * well_w ** 2) - (Y ** 2 + Z ** 2) / (2 * well_w ** 2))
        well_b = -np.exp(-(X - well_sep / 2) ** 2 / (2 * well_w ** 2) - (Y ** 2 + Z ** 2) / (2 * well_w ** 2))
        barrier = barrier_h * np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / 0.5)
        return well_a + well_b + barrier
    return f

def vext_gaussian_bump_3d(L: float, N: int, *, amp: float=-1.0, w: float=1.5, center=(0.0, 0.0, 0.0)):
    cx, cy, cz = center

    def f(X, Y, Z):
        return amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) / (2 * w ** 2))
    return f

def vext_ramp_3d(L: float, N: int, *, beta_x: float=0.05, beta_y: float=0.0, beta_z: float=0.0, omega: float=0.05):

    def f(X, Y, Z):
        return 0.5 * omega ** 2 * (X * X + Y * Y + Z * Z) + beta_x * X + beta_y * Y + beta_z * Z
    return f

def vext_lattice_SC(L: float, N: int, *, a: float=None, amp: float=-1.0):
    if a is None:
        a = L / 4.0

    def f(X, Y, Z):
        k = 2 * np.pi / a
        return amp * (np.cos(k * X) + np.cos(k * Y) + np.cos(k * Z))
    return f

def vext_lattice_BCC(L: float, N: int, *, a: float=None, amp: float=-1.0):
    if a is None:
        a = L / 4.0

    def f(X, Y, Z):
        k = 2 * np.pi / a
        return amp * (np.cos(k * X) + np.cos(k * Y) + np.cos(k * Z) + np.cos(k * (X + Y) / np.sqrt(2)) + np.cos(k * (Y + Z) / np.sqrt(2)) + np.cos(k * (X + Z) / np.sqrt(2)))
    return f

def vext_lattice_FCC(L: float, N: int, *, a: float=None, amp: float=-1.0):
    if a is None:
        a = L / 4.0

    def f(X, Y, Z):
        k = 2 * np.pi / a
        return amp * (np.cos(k * X) * np.cos(k * Y) + np.cos(k * Y) * np.cos(k * Z) + np.cos(k * Z) * np.cos(k * X))
    return f