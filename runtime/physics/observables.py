"""Field measurement functions (observables).

All functions accept arrays from either numpy (CPU) or cupy (GPU).
They auto-detect the backend via ``backend.xp_of`` and compute on the
same device, avoiding GPU-CPU round-trips when the solver runs on CUDA.

Scalar returns are plain Python floats.  Array returns follow the
caller's backend (GPU in, GPU out).
"""
from __future__ import annotations
import numpy as _np

from runtime.backend import xp as _default_xp, asnumpy as _asnumpy, xp_of as _xp_of, to_xp as _to_xp

def _xp(arr):
    """Return the array module (numpy or cupy) matching *arr*."""
    return _xp_of(arr) if arr is not None else _default_xp

def norm(psi, dx: float) -> float:
    np = _xp(psi)
    return float(_asnumpy((np.abs(psi) ** 2).sum() * dx))

def peak_density(psi) -> float:
    np = _xp(psi)
    return float(_asnumpy((np.abs(psi) ** 2).max()))

def fwhm(psi, dx: float) -> float:
    np = _xp(psi)
    rho = np.abs(psi) ** 2
    if float(_asnumpy(rho.max())) <= 0:
        return 0.0
    half = 0.5 * rho.max()
    idx_peak = int(_asnumpy(np.argmax(rho)))
    rho_host = _asnumpy(rho)
    left = idx_peak
    while left > 0 and rho_host[left] > half:
        left -= 1
    right = idx_peak
    while right < len(rho_host) - 1 and rho_host[right] > half:
        right += 1
    return float(right - left) * dx

def ipr(psi, dx: float) -> float:
    np = _xp(psi)
    n2 = float(_asnumpy((np.abs(psi) ** 2).sum() * dx))
    n4 = float(_asnumpy((np.abs(psi) ** 4).sum() * dx))
    return n4 / max(n2 * n2, 1e-30)

def participation_ratio(psi, dx: float) -> float:
    np = _xp(psi)
    n2 = float(_asnumpy((np.abs(psi) ** 2).sum() * dx))
    n4 = float(_asnumpy((np.abs(psi) ** 4).sum() * dx))
    return n2 * n2 / max(n4, 1e-30)

def power_spectrum(psi, dx: float):
    """Return (k_sorted, P_sorted) on the same backend as *psi*."""
    np = _xp(psi)
    N = len(psi)
    psi_hat = np.fft.fft(psi)
    P = np.abs(psi_hat) ** 2
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    order = np.argsort(k)
    return (k[order], P[order])

def dominant_wavenumber(psi, dx: float, k_min: float = 0.0) -> float:
    np = _xp(psi)
    k, P = power_spectrum(psi, dx)
    mask = np.abs(k) >= k_min
    if not bool(_asnumpy(mask.any())):
        return 0.0
    k_m = k[mask]
    P_m = P[mask]
    return float(abs(float(_asnumpy(k_m[int(_asnumpy(np.argmax(P_m)))]))))

def crystallinity(psi, dx: float, k_cutoff: float = 1.0) -> float:
    np = _xp(psi)
    k, P = power_spectrum(psi, dx)
    total = float(_asnumpy(P.sum()))
    if total <= 0:
        return 0.0
    structured = float(_asnumpy(P[np.abs(k) > k_cutoff].sum()))
    return structured / total

def stabilization_score(observable_t) -> float:
    arr = _np.asarray(observable_t, dtype=float)
    if arr.size < 4:
        return 0.0
    half = arr.size // 2
    early = arr[:half]
    late = arr[half:]

    def cv(x):
        mu = float(_np.mean(x))
        if abs(mu) < 1e-30:
            return 0.0
        return float(_np.std(x) / abs(mu))
    cv_e = cv(early)
    cv_l = cv(late)
    if cv_e < 1e-12:
        return 1.0 if cv_l < 1e-12 else 0.0
    return float(max(0.0, min(1.0, 1.0 - cv_l / cv_e)))

def time_to_stabilize(observable_t, t, tolerance: float = 0.1) -> float:
    arr = _np.asarray(observable_t, dtype=float)
    tt = _np.asarray(t, dtype=float)
    if arr.size == 0 or tt.size == 0:
        return 0.0
    if arr.size != tt.size:
        m = min(arr.size, tt.size)
        arr = arr[:m]
        tt = tt[:m]
    late = arr[max(arr.size * 3 // 4, 1):]
    mean = float(_np.mean(late))
    band = tolerance * abs(mean) if abs(mean) > 1e-30 else tolerance
    for i in range(arr.size):
        if abs(arr[i] - mean) <= band and _np.all(_np.abs(arr[i:] - mean) <= band):
            return float(tt[i])
    return float(tt[-1])

def time_delay_embedding(series, dim: int = 3, tau: int = 1):
    s = _np.asarray(series, dtype=float).ravel()
    M = s.size - (dim - 1) * tau
    if M <= 1:
        return _np.zeros((0, dim))
    return _np.stack([s[i * tau:i * tau + M] for i in range(dim)], axis=1)

def _autocorr(series):
    s = _np.asarray(series, dtype=float).ravel()
    s = s - s.mean()
    n = s.size
    if n < 2 or _np.allclose(s, 0.0):
        return _np.array([1.0])
    ac = _np.correlate(s, s, mode='full')[n - 1:]
    return ac / ac[0]

def memory_persistence(series, dt: float = 1.0) -> float:
    ac = _autocorr(series)
    below = _np.where(ac < 1.0 / _np.e)[0]
    if below.size == 0:
        return float((len(ac) - 1) * dt)
    return float(below[0] * dt)

def slow_state_late_mean(series, frac: float = 0.5) -> float:
    s = _np.asarray(series, dtype=float).ravel()
    if s.size == 0:
        return 0.0
    k = max(1, int(s.size * (1.0 - frac)))
    return float(_np.mean(s[k:]))

def correlation_dimension(series, dim: int = 4, tau: int = 1,
                          n_points: int = 400, seed: int = 0) -> float:
    emb = time_delay_embedding(series, dim=dim, tau=tau)
    M = emb.shape[0]
    if M < 20:
        return 0.0
    rng = _np.random.default_rng(seed)
    if M > n_points:
        emb = emb[rng.choice(M, n_points, replace=False)]
        M = n_points
    d = _np.sqrt(((emb[:, None, :] - emb[None, :, :]) ** 2).sum(-1))
    dist = d[_np.triu_indices(M, k=1)]
    dist = dist[dist > 0]
    if dist.size < 10:
        return 0.0
    rmin, rmax = _np.percentile(dist, 5), _np.percentile(dist, 50)
    if rmax <= rmin:
        return 0.0
    rs = _np.logspace(_np.log10(rmin), _np.log10(rmax), 10)
    C = _np.array([(dist < r).mean() for r in rs])
    good = C > 0
    if good.sum() < 3:
        return 0.0
    coef = _np.polyfit(_np.log(rs[good]), _np.log(C[good]), 1)
    return float(coef[0])

def attractor_geometry_invariants(series, dim: int = 4, tau: int = 1) -> dict:
    emb = time_delay_embedding(series, dim=dim, tau=tau)
    if emb.shape[0] < 10:
        return {'corr_dim': 0.0, 'radius': 0.0, 'pca_spectrum': [0.0] * dim}
    c = emb - emb.mean(0, keepdims=True)
    if not _np.all(_np.isfinite(c)):
        finite_mask = _np.all(_np.isfinite(c), axis=1)
        c = c[finite_mask]
        if c.shape[0] < 10:
            return {'corr_dim': 0.0, 'radius': 0.0, 'pca_spectrum': [0.0] * dim}
    cov = (c.T @ c) / max(c.shape[0] - 1, 1)
    try:
        eig = _np.sort(_np.linalg.eigvalsh(cov))[::-1]
    except _np.linalg.LinAlgError:
        return {'corr_dim': 0.0, 'radius': 0.0, 'pca_spectrum': [0.0] * dim}
    sp = eig / (eig.sum() + 1e-30)
    radius = float(_np.sqrt((c ** 2).sum(1).mean()))
    return {'corr_dim': correlation_dimension(series, dim=dim, tau=tau),
            'radius': radius, 'pca_spectrum': [float(v) for v in sp]}

def lyapunov_proxy(series, dim: int = 4, tau: int = 1, dt: float = 1.0,
                   theiler: int = 5, n_ref: int = 200, horizon: int = 10,
                   seed: int = 0) -> float:
    emb = time_delay_embedding(series, dim=dim, tau=tau)
    M = emb.shape[0]
    if M < theiler + horizon + 5:
        return 0.0
    rng = _np.random.default_rng(seed)
    refs = rng.choice(M - horizon, size=min(n_ref, M - horizon), replace=False)
    div = _np.zeros(horizon)
    cnt = _np.zeros(horizon)
    for i in refs:
        d = _np.sqrt(((emb - emb[i]) ** 2).sum(-1))
        d[max(0, i - theiler):i + theiler + 1] = _np.inf
        d[M - horizon:] = _np.inf
        j = int(_np.argmin(d))
        if not _np.isfinite(d[j]) or d[j] <= 0:
            continue
        for h in range(horizon):
            dd = _np.sqrt(((emb[i + h] - emb[j + h]) ** 2).sum())
            if dd > 0:
                div[h] += _np.log(dd)
                cnt[h] += 1
    good = cnt > 0
    if good.sum() < 3:
        return 0.0
    h_idx = _np.arange(horizon)[good]
    mean_log = div[good] / cnt[good]
    slope = _np.polyfit(h_idx, mean_log, 1)[0]
    return float(slope / dt)

def reservoir_memory_capacity(states, inputs,
                              max_delay: int = None, train_frac: float = 0.7,
                              reg: float = 1e-6) -> dict:
    states = _np.asarray(states, dtype=float)
    u = _np.asarray(inputs, dtype=float).ravel()
    Tn, F = states.shape
    if max_delay is None:
        max_delay = min(2 * F, Tn // 3)
    X = _np.hstack([states, _np.ones((Tn, 1))])
    per = []
    for k in range(1, max_delay + 1):
        Xk = X[k:]
        yk = u[:-k]
        ntr = int(len(Xk) * train_frac)
        if ntr < F + 2 or len(Xk) - ntr < 5:
            break
        Xtr, Xte = Xk[:ntr], Xk[ntr:]
        ytr, yte = yk[:ntr], yk[ntr:]
        W = _np.linalg.solve(Xtr.T @ Xtr + reg * _np.eye(Xtr.shape[1]), Xtr.T @ ytr)
        pred = Xte @ W
        ss_res = ((yte - pred) ** 2).sum()
        ss_tot = ((yte - yte.mean()) ** 2).sum() + 1e-30
        per.append(max(0.0, 1.0 - ss_res / ss_tot))
    per = _np.asarray(per)
    return {'memory_capacity': float(per.sum()), 'per_delay': per}

def triad_phase_synchronization(psi, dx: float,
                                shell_width: int = 2) -> float:
    """Phase synchronization of Fourier triads."""
    np = _xp(psi)
    psi_k = np.fft.fft(psi)
    N = len(psi_k)
    k_all = np.fft.fftfreq(N, d=dx) * (2.0 * np.pi)
    amp = np.abs(psi_k)
    mask = amp > amp.max() * 1e-6
    phase = np.angle(psi_k)

    mask_h = _asnumpy(mask)
    amp_h = _asnumpy(amp)
    phase_h = _asnumpy(phase)

    sum_phase = []
    for i in range(N):
        if not mask_h[i] or amp_h[i] < 1e-30:
            continue
        for j in range(max(0, i - shell_width), min(N, i + shell_width + 1)):
            if i == j or not mask_h[j] or amp_h[j] < 1e-30:
                continue
            s = (i + j) % N
            if not mask_h[s] or amp_h[s] < 1e-30:
                continue
            triad_ph = phase_h[i] + phase_h[j] - phase_h[s]
            sum_phase.append(complex(_np.exp(1j * triad_ph)))
    if len(sum_phase) < 3:
        return 0.0
    return float(abs(_np.mean(sum_phase)))

def spectral_flux(psi, dx: float, n_shells: int = 20,
                  k_min: float = 0.0):
    """Spectral energy flux per |k| shell.  Returns numpy array (host)."""
    np = _xp(psi)
    psi_k = np.fft.fft(psi)
    k_all = 2.0 * np.pi * np.fft.fftfreq(len(psi_k), d=dx)
    abs_k = np.abs(k_all)
    P = np.abs(psi_k) ** 2
    k_max = float(_asnumpy(abs_k.max()))
    if k_max <= k_min or n_shells < 2:
        return _np.zeros(0)
    edges = _np.linspace(k_min, k_max, n_shells + 1)
    abs_k_h = _asnumpy(abs_k)
    P_h = _asnumpy(P)
    flux = _np.zeros(n_shells)
    for s in range(n_shells):
        sel = (abs_k_h >= edges[s]) & (abs_k_h < edges[s + 1])
        if sel.any():
            flux[s] = float(P_h[sel].sum())
    return flux

def memory_overlap(y_a, y_b) -> float:
    """Cosine similarity between two memory-field or density vectors."""
    a = _np.asarray(_asnumpy(y_a), dtype=_np.float64).ravel()
    b = _np.asarray(_asnumpy(y_b), dtype=_np.float64).ravel()
    na = _np.linalg.norm(a)
    nb = _np.linalg.norm(b)
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(_np.dot(a, b) / (na * nb))

def capacity_observable(N: int) -> float:
    """Theoretical Hopfield capacity estimate: K ~ 0.3 * N^1.2."""
    return 0.3 * float(N) ** 1.2

def energy(psi, dx: float, hbar: float = 1.0, m: float = 1.0,
           Lambda: float = -0.5, V_ext=None, V_mem=None) -> float:
    """Hamiltonian energy of the field.

    E[psi] = integral (hbar^2/(2m) |grad psi|^2 + V_ext |psi|^2
                       + (Lambda/2) |psi|^4 + V_mem |psi|^2) dx

    Reads the field; never modifies dynamics.  Kinetic term computed in
    k-space via Parseval so it stays O(N log N) with the rest of the solver.
    """
    np = _xp(psi)
    N = len(psi)
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    psi_k = np.fft.fft(psi)
    kinetic = float(_asnumpy(
        hbar ** 2 / (2.0 * m) * (np.abs(k) ** 2 * np.abs(psi_k) ** 2).sum()
        * dx / N))
    rho = np.abs(psi) ** 2
    nonlin = float(_asnumpy(0.5 * Lambda * (rho ** 2).sum() * dx))
    pot = 0.0
    if V_ext is not None:
        V_ext = _to_xp(V_ext, np)
        pot += float(_asnumpy((V_ext * rho).sum() * dx))
    if V_mem is not None:
        V_mem = _to_xp(V_mem, np)
        pot += float(_asnumpy((V_mem * rho).sum() * dx))
    return kinetic + nonlin + pot
