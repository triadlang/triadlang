from __future__ import annotations
from typing import Optional
import numpy as np

def extract_probes(density: np.ndarray, n_probes: int=8, dx: float=0.25) -> np.ndarray:
    N_grid = density.shape[0]
    if n_probes > N_grid:
        n_probes = N_grid
    indices = np.linspace(0, N_grid - 1, n_probes, dtype=int)
    return density[indices, :]

def extract_phase_probes(psi_complex: np.ndarray, n_probes: int=8) -> np.ndarray:
    N_grid = psi_complex.shape[0]
    indices = np.linspace(0, N_grid - 1, n_probes, dtype=int)
    return np.angle(psi_complex[indices, :])

def integrated_information(probes: np.ndarray, n_bins: int=2) -> float:
    n_probes, n_t = probes.shape
    if n_probes < 2 or n_t < 10:
        return 0.0
    binarised = _binarise(probes, n_bins)
    h_whole = _entropy_joint(binarised)
    if h_whole < 1e-10:
        return 0.0
    min_mi = float('inf')
    mid = n_probes // 2
    for split in range(1, n_probes):
        if abs(split - mid) > mid:
            continue
        part_a = binarised[:split, :]
        part_b = binarised[split:, :]
        h_a = _entropy_joint(part_a)
        h_b = _entropy_joint(part_b)
        h_ab = _entropy_joint(np.vstack([_encode_group(part_a), _encode_group(part_b)]))
        mi = h_a + h_b - h_ab
        if mi < min_mi:
            min_mi = mi
    h_parts = 0.0
    part_a = binarised[:mid, :]
    part_b = binarised[mid:, :]
    h_parts = _entropy_joint(part_a) + _entropy_joint(part_b)
    phi = max(0.0, h_parts - h_whole)
    return float(phi)

def lempel_ziv_complexity(probes: np.ndarray, threshold: Optional[float]=None) -> float:
    n_probes, n_t = probes.shape
    if threshold is None:
        threshold = np.median(probes)
    binary = (probes > threshold).astype(int).flatten()
    n = len(binary)
    if n < 2:
        return 0.0
    c = 1
    u = 1
    v = 1
    vmax = v
    while u + v <= n:
        if binary[u + v - 1] == binary[v - 1]:
            v += 1
        else:
            vmax = max(vmax, v)
            u += 1
            if u == u + vmax - 1:
                c += 1
                u = u + vmax
                vmax = 1
            else:
                v = 1
    if v != 1:
        c += 1
    norm = n / np.log2(n) if n > 1 else 1
    return float(c / norm)

def perturbational_complexity(density: np.ndarray, perturbation_idx: Optional[int]=None, perturbation_amp: float=0.5, dx: float=0.25, dt: float=0.005, record_every: int=4) -> float:
    N_grid, n_t = density.shape
    if perturbation_idx is None:
        perturbation_idx = N_grid // 2
    mid = n_t // 2
    pre = density[:, :mid]
    post = density[:, mid:]
    if pre.shape[1] < 4 or post.shape[1] < 4:
        return 0.0
    probes_pre = extract_probes(pre, n_probes=min(8, N_grid))
    probes_post = extract_probes(post, n_probes=min(8, N_grid))
    lz_pre = lempel_ziv_complexity(probes_pre)
    lz_post = lempel_ziv_complexity(probes_post)
    lz_total = max(lz_pre + lz_post, 1e-10)
    return float(lz_post / lz_total)

def workspace_ignition(probes: np.ndarray, threshold_sigma: float=1.0) -> float:
    n_probes, n_t = probes.shape
    if n_t < 2:
        return 0.0
    mean_per_probe = probes.mean(axis=1, keepdims=True)
    std_per_probe = probes.std(axis=1, keepdims=True) + 1e-10
    active = probes > mean_per_probe + threshold_sigma * std_per_probe
    frac_active = active.sum(axis=0) / n_probes
    ignited = frac_active > 0.5
    return float(ignited.sum() / n_t)

def causal_density(probes: np.ndarray, lag: int=5) -> float:
    n_probes, n_t = probes.shape
    if n_probes < 2 or n_t < lag * 2:
        return 0.0
    total_gc = 0.0
    count = 0
    for i in range(n_probes):
        for j in range(n_probes):
            if i == j:
                continue
            gc = _granger(probes[j], probes[i], lag)
            total_gc += gc
            count += 1
    return float(total_gc / max(count, 1))

def _granger(y: np.ndarray, x: np.ndarray, lag: int) -> float:
    n = len(y)
    if n < lag * 3:
        return 0.0
    Y = np.array([y[lag + t:n - (lag - t)] if lag - t > 0 else y[lag + t:] for t in range(lag)]).T
    T = Y.shape[0]
    if T < lag + 2:
        return 0.0
    Y_target = y[2 * lag:2 * lag + T]
    if len(Y_target) != T:
        min_len = min(len(Y_target), T)
        Y_target = Y_target[:min_len]
        Y = Y[:min_len]
    X_lag = np.array([x[lag + t:lag + t + T] for t in range(lag)]).T
    if X_lag.shape[0] != T:
        min_len = min(X_lag.shape[0], T)
        X_lag = X_lag[:min_len]
        Y = Y[:min_len]
        Y_target = Y_target[:min_len]
    ones = np.ones((T, 1))
    A_r = np.linalg.lstsq(np.hstack([ones, Y]), Y_target, rcond=None)[0]
    err_r = Y_target - np.hstack([ones, Y]) @ A_r
    A_u = np.linalg.lstsq(np.hstack([ones, Y, X_lag]), Y_target, rcond=None)[0]
    err_u = Y_target - np.hstack([ones, Y, X_lag]) @ A_u
    ss_r = np.sum(err_r ** 2) + 1e-10
    ss_u = np.sum(err_u ** 2) + 1e-10
    gc = np.log(ss_r / ss_u)
    return float(max(0.0, gc))

def metastability(probes: np.ndarray) -> float:
    n_probes, n_t = probes.shape
    if n_probes < 2 or n_t < 10:
        return 0.0
    phases = np.zeros_like(probes)
    for k in range(n_probes):
        sig = probes[k] - probes[k].mean()
        analytic = _hilbert(sig)
        phases[k] = np.angle(analytic)
    R_t = np.abs(np.mean(np.exp(1j * phases), axis=0))
    return float(np.var(R_t))

def _hilbert(x: np.ndarray) -> np.ndarray:
    n = len(x)
    X = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = 1
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    return np.fft.ifft(X * h)

def consciousness_report(density: np.ndarray, n_probes: int=8, dx: float=0.25, dt: float=0.005) -> dict:
    probes = extract_probes(density, n_probes, dx)
    report = {'phi': integrated_information(probes), 'lzc': lempel_ziv_complexity(probes), 'pci': perturbational_complexity(density, dx=dx, dt=dt), 'workspace': workspace_ignition(probes), 'causal_density': causal_density(probes), 'metastability': metastability(probes), 'n_probes': n_probes}
    return report

def _binarise(probes: np.ndarray, n_bins: int=2) -> np.ndarray:
    n_p, n_t = probes.shape
    out = np.zeros((n_p, n_t), dtype=int)
    for k in range(n_p):
        if n_bins == 2:
            median = np.median(probes[k])
            out[k] = (probes[k] > median).astype(int)
        else:
            out[k] = np.digitize(probes[k], np.linspace(probes[k].min(), probes[k].max(), n_bins + 1)[1:-1])
    return out

def _entropy_joint(binarised: np.ndarray) -> float:
    n_p, n_t = binarised.shape
    if n_t < 2:
        return 0.0
    encoded = _encode_group(binarised)
    _, counts = np.unique(encoded, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-15)))

def _encode_group(binarised: np.ndarray) -> np.ndarray:
    n_p, n_t = binarised.shape
    encoded = np.zeros(n_t, dtype=int)
    for k in range(n_p):
        encoded = encoded * 2 + binarised[k]
    return encoded

def _lz76_binary(binary_seq: str) -> int:
    if not binary_seq:
        return 0
    n = len(binary_seq)
    c = 1
    l = 1
    i = 0
    k = 1
    k_max = 1
    while True:
        if l + k > n:
            c += 1
            break
        if binary_seq[i + k - 1] == binary_seq[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
    return c

def lzc_of_trajectory(rho_trajectory: np.ndarray) -> float:
    peak_t = rho_trajectory.max(axis=0)
    median = float(np.median(peak_t))
    bin_seq = ''.join(('1' if x > median else '0' for x in peak_t))
    raw = _lz76_binary(bin_seq)
    n = len(bin_seq)
    if n < 2:
        return 0.0
    return float(raw / (n / np.log2(n)))

def kuramoto_order(phase_traj: np.ndarray) -> float:
    if phase_traj.size == 0:
        return 0.0
    phases = phase_traj[:, -1]
    z = np.exp(1j * phases)
    return float(np.abs(z.mean()))

def metastability_phases(phase_traj: np.ndarray) -> float:
    if phase_traj.shape[1] < 2:
        return 0.0
    order_t = np.abs(np.exp(1j * phase_traj).mean(axis=0))
    return float(np.var(order_t))

def phi_id_proxy(rho_trajectory: np.ndarray) -> float:
    if rho_trajectory.ndim != 2 or rho_trajectory.shape[1] < 4:
        return 0.0
    N = rho_trajectory.shape[0]
    half = N // 2
    left = rho_trajectory[:half].sum(axis=0)
    right = rho_trajectory[half:].sum(axis=0)
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    corr = float(np.corrcoef(left, right)[0, 1])
    return float(1.0 - abs(abs(corr) - 0.5) / 0.5)

def pcist_proxy(rho_baseline: np.ndarray, rho_perturbed: np.ndarray) -> float:
    if rho_baseline.shape != rho_perturbed.shape:
        return 0.0
    diff_peak = (rho_perturbed - rho_baseline).max(axis=0)
    median = float(np.median(diff_peak))
    bin_seq = ''.join(('1' if x > median else '0' for x in diff_peak))
    raw = _lz76_binary(bin_seq)
    n = len(bin_seq)
    if n < 2:
        return 0.0
    return float(raw / (n / np.log2(n)))

def causal_density_pair(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    if len(rho_a) < 5 or len(rho_a) != len(rho_b):
        return 0.0
    a, b = (rho_a, rho_b)
    a_lag = a[:-1]
    a_pred = a[1:]
    cov_aa = float(np.cov(a_lag, a_pred)[0, 1])
    var_a = float(np.var(a_lag) + 1e-30)
    a_resid = a_pred - cov_aa / var_a * a_lag
    b_lag = b[:-1]
    cov_ba = float(np.cov(b_lag, a_pred)[0, 1])
    var_b = float(np.var(b_lag) + 1e-30)
    cross_resid = a_pred - cov_ba / var_b * b_lag
    var_self = float(np.var(a_resid) + 1e-30)
    var_cross = float(np.var(cross_resid) + 1e-30)
    return float(np.log(var_cross / var_self))