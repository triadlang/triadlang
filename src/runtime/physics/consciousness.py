from __future__ import annotations

from triad import ntri as np


def consciousness_report(psi, dt: float = 0.005) -> dict:
    density = np.abs(psi) ** 2
    entropy = -np.sum(density * np.log(density + 1e-10))
    return {
        'integrated_information': float(integrated_information(psi)),
        'entropy': float(entropy),
        'stability': float(stabilization_score(density))
    }

def integrated_information(psi) -> float:
    density = np.abs(psi) ** 2
    entropy = -np.sum(density * np.log(density + 1e-10))
    return entropy

def lempel_ziv_complexity(series: list[float] | np.ndarray) -> float:
    s = ''.join('1' if x > 0.5 else '0' for x in series)
    n = len(s)
    subs = set()
    l = 0
    c = 1
    while l < n:
        if s[l:c+l] not in subs and c+l <= n:
            subs.add(s[l:c+l])
            c += 1
        else:
            l += c
            c = 1
    return float(len(subs))

def causal_density(psi, dx: float = 0.1) -> float:
    density = np.abs(psi) ** 2
    gradient = np.gradient(density, dx)
    return float(np.mean(np.abs(gradient)))

def metastability(psi, dt: float = 0.005, window: int = 100) -> float:
    density = np.abs(psi) ** 2
    if len(density) < window * 2:
        return 0.0
    early = density[:window]
    late = density[-window:]
    return 1.0 - float(np.abs(early.mean() - late.mean()))

def workspace_ignition(psi, threshold: float = 0.5) -> float:
    density = np.abs(psi) ** 2
    peaks = density > threshold
    return float(np.sum(peaks) / len(peaks))

def stabilization_score(observable_t) -> float:
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
