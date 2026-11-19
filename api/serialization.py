from __future__ import annotations
import base64
import io
import numpy as np


def ndarray_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def b64_to_ndarray(s: str) -> np.ndarray:
    raw = base64.b64decode(s)
    return np.load(io.BytesIO(raw), allow_pickle=False)


def compute_standard_observables(psi_final: np.ndarray, dx: float, L: float) -> dict:
    from runtime.physics.observables import (
        crystallinity, dominant_wavenumber, peak_density,
        ipr, fwhm, norm, participation_ratio,
    )
    k_min = 2.0 * np.pi / L
    psi_1d = psi_final.ravel()
    return {
        'crystallinity': float(crystallinity(psi_1d, dx)),
        'k_star': float(dominant_wavenumber(psi_1d, dx, k_min=k_min)),
        'peak_density': float(peak_density(psi_1d)),
        'norm': float(norm(psi_1d, dx)),
        'ipr': float(ipr(psi_1d, dx)),
        'fwhm': float(fwhm(psi_1d, dx)),
    }
