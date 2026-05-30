from __future__ import annotations
import numpy as np

def norm(psi: np.ndarray, dx: float) -> float:
    return float((np.abs(psi) ** 2).sum() * dx)

def peak_density(psi: np.ndarray) -> float:
    return float((np.abs(psi) ** 2).max())

def fwhm(psi: np.ndarray, dx: float) -> float:
    rho = np.abs(psi) ** 2
    if rho.max() <= 0:
        return 0.0
    half = 0.5 * rho.max()
    idx_peak = int(np.argmax(rho))
    left = idx_peak
    while left > 0 and rho[left] > half:
        left -= 1
    right = idx_peak
    while right < len(rho) - 1 and rho[right] > half:
        right += 1
    return float(right - left) * dx

def ipr(psi: np.ndarray, dx: float) -> float:
    n2 = (np.abs(psi) ** 2).sum() * dx
    n4 = (np.abs(psi) ** 4).sum() * dx
    return float(n4 / max(n2 * n2, 1e-30))

def participation_ratio(psi: np.ndarray, dx: float) -> float:
    n2 = (np.abs(psi) ** 2).sum() * dx
    n4 = (np.abs(psi) ** 4).sum() * dx
    return float(n2 * n2 / max(n4, 1e-30))

def power_spectrum(psi: np.ndarray, dx: float):
    N = len(psi)
    psi_hat = np.fft.fft(psi)
    P = np.abs(psi_hat) ** 2
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    order = np.argsort(k)
    return (k[order], P[order])

def dominant_wavenumber(psi: np.ndarray, dx: float, k_min: float=0.0) -> float:
    k, P = power_spectrum(psi, dx)
    mask = np.abs(k) >= k_min
    if not mask.any():
        return 0.0
    k_m = k[mask]
    P_m = P[mask]
    return float(abs(k_m[int(np.argmax(P_m))]))

def crystallinity(psi: np.ndarray, dx: float, k_cutoff: float=1.0) -> float:
    k, P = power_spectrum(psi, dx)
    total = P.sum()
    if total <= 0:
        return 0.0
    structured = P[np.abs(k) > k_cutoff].sum()
    return float(structured / total)

def stabilization_score(observable_t: np.ndarray) -> float:
    arr = np.asarray(observable_t, dtype=float)
    if arr.size < 4:
        return 0.0
    half = arr.size // 2
    early = arr[:half]
    late = arr[half:]

    def cv(x):
        mu = float(np.mean(x))
        if abs(mu) < 1e-30:
            return 0.0
        return float(np.std(x) / abs(mu))
    cv_e = cv(early)
    cv_l = cv(late)
    if cv_e < 1e-12:
        return 1.0 if cv_l < 1e-12 else 0.0
    return float(max(0.0, min(1.0, 1.0 - cv_l / cv_e)))

def time_to_stabilize(observable_t: np.ndarray, t: np.ndarray, tolerance: float=0.1) -> float:
    arr = np.asarray(observable_t, dtype=float)
    tt = np.asarray(t, dtype=float)
    if arr.size == 0 or tt.size == 0:
        return 0.0
    if arr.size != tt.size:
        m = min(arr.size, tt.size)
        arr = arr[:m]
        tt = tt[:m]
    late = arr[max(arr.size * 3 // 4, 1):]
    mean = float(np.mean(late))
    band = tolerance * abs(mean) if abs(mean) > 1e-30 else tolerance
    for i in range(arr.size):
        if abs(arr[i] - mean) <= band and np.all(np.abs(arr[i:] - mean) <= band):
            return float(tt[i])
    return float(tt[-1])