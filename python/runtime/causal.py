from __future__ import annotations
from typing import Optional
import numpy as np

def memory_causal_strength(y_traj: np.ndarray, n_regions: int=4) -> np.ndarray:
    M, N_grid, n_t = y_traj.shape
    if n_t < 10:
        return np.zeros((n_regions, n_regions))
    region_size = N_grid // n_regions
    causal = np.zeros((n_regions, n_regions))
    for i in range(n_regions):
        for j in range(n_regions):
            ri_start = i * region_size
            ri_end = min((i + 1) * region_size, N_grid)
            rj_start = j * region_size
            rj_end = min((j + 1) * region_size, N_grid)
            y_source = y_traj[:, ri_start:ri_end, :].mean(axis=(0, 1))
            y_target = y_traj[:, rj_start:rj_end, :].mean(axis=(0, 1))
            lag = min(5, n_t // 4)
            if lag < 1:
                continue
            y_past = y_source[:-lag]
            y_future = y_target[lag:]
            min_len = min(len(y_past), len(y_future))
            if min_len < 3:
                continue
            y_past = y_past[:min_len]
            y_future = y_future[:min_len]
            corr = np.corrcoef(y_past, y_future)[0, 1]
            if np.isfinite(corr):
                causal[i, j] = abs(corr)
    return causal

def memory_timescale_map(y_traj: np.ndarray, nu: tuple=(2.0, 0.5, 0.1)) -> np.ndarray:
    M, N_grid, n_t = y_traj.shape
    nu_arr = np.array(nu[:M])
    timescales = 1.0 / nu_arr
    dominant_ts = np.zeros(N_grid)
    for i in range(N_grid):
        energy = np.array([np.sum(y_traj[j, i, :] ** 2) for j in range(M)])
        if energy.sum() > 0:
            dominant_ts[i] = timescales[energy.argmax()]
        else:
            dominant_ts[i] = timescales.mean()
    return dominant_ts

def transfer_entropy(source: np.ndarray, target: np.ndarray, lag: int=1, n_bins: int=3) -> float:
    n = min(len(source), len(target))
    if n < lag + 3:
        return 0.0
    y_future = _discretise(target[lag:n], n_bins)
    y_past = _discretise(target[:n - lag], n_bins)
    x_past = _discretise(source[:n - lag], n_bins)
    h_y_given_y = _conditional_entropy(y_future, y_past)
    joint_past = np.array([y_past[k] * n_bins + x_past[k] for k in range(len(y_past))])
    h_y_given_yx = _conditional_entropy(y_future, joint_past)
    return float(max(0.0, h_y_given_y - h_y_given_yx))

def transfer_entropy_matrix(density: np.ndarray, n_regions: int=4, lag: int=3, n_bins: int=3) -> np.ndarray:
    N_grid, n_t = density.shape
    if n_t < lag * 3:
        return np.zeros((n_regions, n_regions))
    region_size = N_grid // n_regions
    te_matrix = np.zeros((n_regions, n_regions))
    for i in range(n_regions):
        for j in range(n_regions):
            si = i * region_size
            ei = min((i + 1) * region_size, N_grid)
            sj = j * region_size
            ej = min((j + 1) * region_size, N_grid)
            src_ts = density[si:ei, :].mean(axis=0)
            tgt_ts = density[sj:ej, :].mean(axis=0)
            te_matrix[i, j] = transfer_entropy(src_ts, tgt_ts, lag, n_bins)
    return te_matrix

def estimate_causal_lag(source: np.ndarray, target: np.ndarray, max_lag: int=20) -> dict:
    n = min(len(source), len(target))
    if n < max_lag + 3:
        max_lag = n // 3
    te_curve = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        te_curve[lag - 1] = transfer_entropy(source, target, lag)
    optimal = int(np.argmax(te_curve)) + 1
    return {'optimal_lag': optimal, 'max_te': float(te_curve[optimal - 1]), 'te_curve': te_curve}

def detect_causal_channels(y_traj: np.ndarray, threshold: float=0.5) -> list[dict]:
    M, N_grid, n_t = y_traj.shape
    total_memory_energy = np.sum(y_traj ** 2, axis=(0, 2))
    percentile = np.percentile(total_memory_energy, threshold * 100)
    above = total_memory_energy > percentile
    channels = []
    in_channel = False
    start = 0
    for i in range(N_grid):
        if above[i] and (not in_channel):
            start = i
            in_channel = True
        elif not above[i] and in_channel:
            energy = float(total_memory_energy[start:i].mean())
            channels.append({'start': int(start), 'end': int(i - 1), 'width': int(i - start), 'energy': energy})
            in_channel = False
    if in_channel:
        channels.append({'start': int(start), 'end': int(N_grid - 1), 'width': int(N_grid - start), 'energy': float(total_memory_energy[start:].mean())})
    return channels

def causal_report(density: np.ndarray, y_traj: Optional[np.ndarray]=None, nu: tuple=(2.0, 0.5, 0.1), n_regions: int=4) -> dict:
    N_grid, n_t = density.shape
    M = len(nu)
    if y_traj is None:
        y_traj = _compute_memory(density, nu)
    causal_mat = memory_causal_strength(y_traj, n_regions)
    te_mat = transfer_entropy_matrix(density, n_regions)
    ts_map = memory_timescale_map(y_traj, nu)
    channels = detect_causal_channels(y_traj)
    total_causal = float(causal_mat.sum() / (n_regions * n_regions))
    total_te = float(te_mat.sum() / (n_regions * (n_regions - 1) + 1e-10))
    return {'causal_matrix': causal_mat, 'transfer_entropy_matrix': te_mat, 'total_causal_strength': total_causal, 'total_transfer_entropy': total_te, 'timescale_map': ts_map, 'causal_channels': channels, 'n_channels': len(channels), 'n_regions': n_regions}

def _compute_memory(density: np.ndarray, nu: tuple=(2.0, 0.5, 0.1), dt: float=0.005) -> np.ndarray:
    N_grid, n_t = density.shape
    M = len(nu)
    y = np.zeros((M, N_grid, n_t))
    for j in range(M):
        decay = np.exp(-nu[j] * dt)
        for t in range(1, n_t):
            y[j, :, t] = decay * y[j, :, t - 1] + (1 - decay) * density[:, t]
    return y

def _discretise(x: np.ndarray, n_bins: int) -> np.ndarray:
    if n_bins <= 1 or len(x) < 2:
        return np.zeros(len(x), dtype=int)
    edges = np.linspace(x.min() - 1e-10, x.max() + 1e-10, n_bins + 1)
    return np.digitize(x, edges[1:-1])

def _conditional_entropy(future: np.ndarray, past: np.ndarray) -> float:
    n = len(future)
    if n < 2:
        return 0.0
    joint = np.array([past[k] * 100 + future[k] for k in range(n)])
    _, joint_counts = np.unique(joint, return_counts=True)
    _, past_counts = np.unique(past, return_counts=True)
    h_joint = -np.sum(joint_counts / n * np.log2(joint_counts / n + 1e-15))
    h_past = -np.sum(past_counts / n * np.log2(past_counts / n + 1e-15))
    return float(max(0.0, h_joint - h_past))