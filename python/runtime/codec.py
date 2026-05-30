from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from runtime import observables as obs
DEFAULT_K_MIN = 0.0
DEFAULT_K_STEP = 0.196349541

@dataclass
class IntCalib:
    k_min: float = DEFAULT_K_MIN
    k_step: float = DEFAULT_K_STEP

@dataclass
class FloatCalib:
    a: float = 1.0
    b: float = 0.0

def encode_int(value: int, calib: IntCalib, L: float, N: int, envelope_width: Optional[float]=None, modulation: float=0.9) -> np.ndarray:
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    dx = x[1] - x[0]
    k_n = calib.k_min + value * calib.k_step
    if value == 0 or k_n == 0:
        psi = np.ones(N, dtype=np.complex128)
    else:
        psi = (1.0 + modulation * np.cos(k_n * x)).astype(np.complex128)
    psi /= np.sqrt((np.abs(psi) ** 2).sum() * dx)
    return psi

def decode_int(psi: np.ndarray, dx: float, calib: IntCalib) -> int:
    N = len(psi)
    L = N * dx
    P = np.abs(np.fft.fft(psi)) ** 2
    half = P[1:N // 2]
    if half.size == 0:
        return 0
    vbin = int(np.argmax(half)) + 1
    if P[vbin] < 1e-06 * P[0]:
        return 0
    k_star = vbin * (2.0 * np.pi / L)
    return int(round((k_star - calib.k_min) / calib.k_step))

def encode_bool(value: bool, L: float, N: int) -> np.ndarray:
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    dx = x[1] - x[0]
    if value:
        env = np.exp(-x ** 2 / (2.0 * (L / 6.0) ** 2))
        psi = (env * np.cos(3.0 * x)).astype(np.complex128)
    else:
        psi = np.exp(-x ** 2 / (2.0 * (L / 3.0) ** 2)).astype(np.complex128)
    psi /= np.sqrt((np.abs(psi) ** 2).sum() * dx)
    return psi

def decode_bool(psi: np.ndarray, dx: float) -> Optional[bool]:
    c = obs.crystallinity(psi, dx)
    if c > 0.5:
        return True
    if c < 0.2:
        return False
    return None

def encode_float(value: float, calib: FloatCalib, L: float, N: int) -> np.ndarray:
    target_ipr = (value - calib.b) / calib.a
    target_ipr = max(target_ipr, 0.001)
    w = 1.0 / (target_ipr * np.sqrt(2.0 * np.pi))
    w = float(np.clip(w, 0.05, L / 4.0))
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    dx = x[1] - x[0]
    psi = np.exp(-x ** 2 / (2.0 * w ** 2)).astype(np.complex128)
    psi /= np.sqrt((np.abs(psi) ** 2).sum() * dx)
    return psi

def decode_float(psi: np.ndarray, dx: float, calib: FloatCalib) -> float:
    ipr = float((np.abs(psi) ** 4).sum() * dx / max((np.abs(psi) ** 2).sum() * dx, 1e-30) ** 2)
    return calib.a * ipr + calib.b

def encode_bits(value: int, bit_width: int, calib: IntCalib, L: float, N: int) -> np.ndarray:
    return encode_int(value, calib, L, N)