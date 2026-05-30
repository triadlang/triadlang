from __future__ import annotations
from typing import Optional
import numpy as np
FAMILIES = ('SC', 'BCC', 'FCC', 'HCP')

def _gaussian_at(x, y, z, x0, sigma):
    return np.exp(-((x - x0[0]) ** 2 + (y - x0[1]) ** 2 + (z - x0[2]) ** 2) / (2 * sigma ** 2))

def encode_bravais(family: str, L: float, N: int, sigma: float=0.5) -> np.ndarray:
    xs = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij')
    dx = xs[1] - xs[0]
    a = L / 2.0
    sites = []
    if family == 'SC':
        for sx in (-a / 2, +a / 2):
            for sy in (-a / 2, +a / 2):
                for sz in (-a / 2, +a / 2):
                    sites.append((sx, sy, sz))
    elif family == 'BCC':
        sites.append((0.0, 0.0, 0.0))
        for sx in (-a / 2, +a / 2):
            for sy in (-a / 2, +a / 2):
                for sz in (-a / 2, +a / 2):
                    sites.append((sx, sy, sz))
    elif family == 'FCC':
        for sx in (-a / 2, +a / 2):
            for sy in (-a / 2, +a / 2):
                for sz in (-a / 2, +a / 2):
                    sites.append((sx, sy, sz))
        for s in (-a / 2, +a / 2):
            sites.append((s, 0.0, 0.0))
            sites.append((0.0, s, 0.0))
            sites.append((0.0, 0.0, s))
    elif family == 'HCP':
        c = L / 4.0
        for z0 in (-c / 2, +c / 2):
            for phi_n in range(6):
                phi = phi_n * np.pi / 3
                sites.append((c * np.cos(phi), c * np.sin(phi), z0))
    else:
        raise ValueError(f'unknown family: {family!r}')
    psi = np.zeros_like(X, dtype=np.complex128)
    for s in sites:
        psi = psi + _gaussian_at(X, Y, Z, s, sigma)
    norm2 = (np.abs(psi) ** 2).sum() * dx ** 3
    if norm2 > 0:
        psi /= np.sqrt(norm2)
    return psi.astype(np.complex128)

def encode_int_bravais(value: int, L: float, N: int) -> np.ndarray:
    family = FAMILIES[value % len(FAMILIES)]
    return encode_bravais(family, L, N)

def _family_directions(family: str):
    if family == 'SC':
        return np.array([[+1, 0, 0], [-1, 0, 0], [0, +1, 0], [0, -1, 0], [0, 0, +1], [0, 0, -1]], dtype=float)
    if family == 'BCC':
        r2 = 1.0 / np.sqrt(2.0)
        d = []
        for ix in (-1, +1):
            for iy in (-1, +1):
                d.append([ix * r2, iy * r2, 0])
                d.append([ix * r2, 0, iy * r2])
                d.append([0, ix * r2, iy * r2])
        return np.array(d, dtype=float)
    if family == 'FCC':
        r3 = 1.0 / np.sqrt(3.0)
        return np.array([[s1 * r3, s2 * r3, s3 * r3] for s1 in (-1, +1) for s2 in (-1, +1) for s3 in (-1, +1)], dtype=float)
    if family == 'HCP':
        d = []
        for ph in range(6):
            phi = ph * np.pi / 3
            d.append([np.cos(phi), np.sin(phi), 0.0])
        d.append([0.0, 0.0, +1.0])
        d.append([0.0, 0.0, -1.0])
        return np.array(d, dtype=float)
    raise ValueError(family)

def bravais_family(psi: np.ndarray, dx: float, shell_tol: float=0.15) -> str:
    N = psi.shape[0]
    L = N * dx
    psi_hat = np.fft.fftn(psi)
    P = np.abs(psi_hat) ** 2
    kvec = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    KX, KY, KZ = np.meshgrid(kvec, kvec, kvec, indexing='ij')
    k_mag = np.sqrt(KX * KX + KY * KY + KZ * KZ)
    mask = k_mag >= 2.0 * np.pi / L
    if not mask.any():
        return 'SC'
    idx_max = np.unravel_index(int(np.argmax(P * mask)), P.shape)
    k_star = float(k_mag[idx_max])
    if k_star <= 0:
        return 'SC'
    shell = (np.abs(k_mag - k_star) / max(k_star, 1e-30) < shell_tol) & mask
    eps = 1e-12
    kx = KX[shell] / np.maximum(k_mag[shell], eps)
    ky = KY[shell] / np.maximum(k_mag[shell], eps)
    kz = KZ[shell] / np.maximum(k_mag[shell], eps)
    w = P[shell]
    scores = {}
    for fam in FAMILIES:
        dirs = _family_directions(fam)
        s_fam = 0.0
        for d in dirs:
            dot = kx * d[0] + ky * d[1] + kz * d[2]
            aligned = dot > 1.0 - shell_tol
            s_fam += float(w[aligned].sum())
        scores[fam] = s_fam
    return max(scores, key=scores.get)

def decode_int_bravais(psi: np.ndarray, dx: float) -> int:
    fam = bravais_family(psi, dx)
    return FAMILIES.index(fam)