from __future__ import annotations

from triad import ntri as np


def atom_count_nd(psi, dx: float, threshold_frac: float | None = None) -> float:
    density = np.abs(psi) ** 2
    if threshold_frac is not None and threshold_frac > 0:
        threshold = density.max() * threshold_frac
        density = density * (density > threshold)
    return float(np.sum(density) * dx)

def atom_distribution(psi, dx: float = 0.1) -> np.ndarray:
    density = np.abs(psi) ** 2
    total = np.sum(density) * dx
    if total == 0:
        return np.zeros_like(density)
    return density / total

def atom_clustering(psi, dx: float = 0.1, n_neighbors: int = 5) -> float:
    density = np.abs(psi) ** 2
    peaks = density > density.mean() + density.std()
    n_peaks = np.sum(peaks)
    if n_peaks < 2:
        return 0.0
    peak_indices = np.argwhere(peaks).flatten()
    distances = []
    for i in range(len(peak_indices)):
        for j in range(i + 1, len(peak_indices)):
            dist = abs(peak_indices[j] - peak_indices[i]) * dx
            distances.append(dist)
    if len(distances) == 0:
        return 0.0
    return float(np.std(distances) / (np.mean(distances) + 1e-10))

def atom_radial_distribution(psi, dx: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    density = np.abs(psi) ** 2
    N = len(density)
    r = np.arange(N) * dx
    return r, density

def local_atom_density(psi, radius: int = 3) -> np.ndarray:
    density = np.abs(psi) ** 2
    local_density = np.zeros_like(density)
    for i in range(len(density)):
        start = max(0, i - radius)
        end = min(len(density), i + radius + 1)
        local_density[i] = np.mean(density[start:end])
    return local_density

def atom_centroids_nd(psi, dx: float, threshold_frac: float | None = None) -> list[dict]:
    density = np.abs(psi) ** 2
    if threshold_frac is None:
        threshold_frac = 0.5
    threshold = density.max() * threshold_frac
    peaks = density > threshold
    if not peaks.any():
        return []
    peak_indices = np.argwhere(peaks)
    centroids = []
    for idx in peak_indices:
        pos = (idx.astype(float) + 0.5) * dx
        val = float(density[tuple(idx)])
        centroids.append({'position': pos.tolist(), 'density': val})
    return centroids

def atom_separation(psi, dx: float) -> float:
    centroids = atom_centroids_nd(psi, dx)
    if len(centroids) < 2:
        return 0.0
    positions = [np.array(c['position']) for c in centroids]
    min_dist = float('inf')
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            min_dist = min(min_dist, dist)
    return float(min_dist)

def atoms_per_region(psi, dx: float, n_regions: int = 4) -> list[int]:
    density = np.abs(psi) ** 2
    N = len(density)
    region_size = N // n_regions
    counts = []
    for r in range(n_regions):
        start = r * region_size
        end = start + region_size if r < n_regions - 1 else N
        region_density = density[start:end]
        threshold = region_density.max() * 0.5
        counts.append(int((region_density > threshold).sum()))
    return counts
