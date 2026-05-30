from __future__ import annotations
import numpy as np

def _label_periodic(above: np.ndarray):
    from scipy.ndimage import label as nd_label
    labels, n = nd_label(above)
    if n <= 1:
        return (labels, int(n))
    parent = list(range(n + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = (find(a), find(b))
        if ra != rb:
            parent[ra] = rb
    D = labels.ndim
    for axis in range(D):
        slice_lo = [slice(None)] * D
        slice_lo[axis] = 0
        slice_hi = [slice(None)] * D
        slice_hi[axis] = labels.shape[axis] - 1
        lo = labels[tuple(slice_lo)]
        hi = labels[tuple(slice_hi)]
        mask = (lo > 0) & (hi > 0)
        if not mask.any():
            continue
        for a, b in zip(lo[mask].ravel(), hi[mask].ravel()):
            union(int(a), int(b))
    remap = {}
    next_id = 0
    new_labels = np.zeros_like(labels)
    for lbl in range(1, n + 1):
        root = find(lbl)
        if root not in remap:
            next_id += 1
            remap[root] = next_id
        new_labels[labels == lbl] = remap[root]
    return (new_labels, next_id)

def atom_count_nd(psi: np.ndarray, dx: float, threshold_frac: float=0.25) -> int:
    if psi.ndim == 1:
        return atom_count(psi, dx, threshold_frac)
    if psi.ndim == 2:
        return atom_count_2d(psi, dx, threshold_frac)
    if psi.ndim == 3:
        return atom_count_3d(psi, dx, threshold_frac)
    raise ValueError(f'unsupported psi.ndim = {psi.ndim}')

def atomicity_ratio(psi_macro: np.ndarray, psi_aggregate: np.ndarray, dx: float, threshold_frac: float=0.25) -> float:
    if psi_macro.ndim != psi_aggregate.ndim:
        raise ValueError('psi_macro and psi_aggregate must have the same ndim')
    macro_density = atoms_per_region(psi_macro, dx, threshold_frac)
    agg_density = atoms_per_region(psi_aggregate, dx, threshold_frac)
    if agg_density <= 0:
        return 0.0
    return macro_density / agg_density

def atoms_per_region(psi: np.ndarray, dx: float, threshold_frac: float=0.25) -> float:
    if psi.ndim == 1:
        n_at = atom_count(psi, dx, threshold_frac)
        rho = np.abs(psi) ** 2
        thr = threshold_frac * float(rho.max()) if rho.max() > 0 else 0
        occupied = float((rho > thr).sum() * dx)
    elif psi.ndim == 2:
        n_at = atom_count_2d(psi, dx, threshold_frac)
        rho = np.abs(psi) ** 2
        thr = threshold_frac * float(rho.max()) if rho.max() > 0 else 0
        occupied = float((rho > thr).sum() * dx ** 2)
    elif psi.ndim == 3:
        n_at = atom_count_3d(psi, dx, threshold_frac)
        rho = np.abs(psi) ** 2
        thr = threshold_frac * float(rho.max()) if rho.max() > 0 else 0
        occupied = float((rho > thr).sum() * dx ** 3)
    else:
        raise ValueError(f'unsupported psi.ndim = {psi.ndim}')
    if occupied <= 0:
        return 0.0
    return float(n_at) / occupied

def atom_centroids_nd(psi: np.ndarray, dx: float, threshold_frac: float=0.25) -> np.ndarray:
    if psi.ndim == 1:
        return atom_centroids(psi, dx, threshold_frac)
    if psi.ndim == 2:
        return atom_centroids_2d(psi, dx, threshold_frac)
    if psi.ndim == 3:
        return atom_centroids_3d(psi, dx, threshold_frac)
    raise ValueError(f'unsupported psi.ndim = {psi.ndim}')

def atom_count(psi: np.ndarray, dx: float, threshold_frac: float=0.25) -> int:
    return count_clusters(psi, dx, threshold_frac)

def atom_centroids(psi: np.ndarray, dx: float, threshold_frac: float=0.25) -> np.ndarray:
    return cluster_centroids(psi, dx, threshold_frac)

def atom_separation(psi: np.ndarray, dx: float, threshold_frac: float=0.25) -> float:
    cs = cluster_centroids(psi, dx, threshold_frac)
    if cs.size < 2:
        return 0.0
    sorted_cs = np.sort(cs)
    gaps = np.diff(sorted_cs)
    return float(np.mean(gaps))

def atom_persistence_late(density_traj: np.ndarray, dx: float, threshold_frac: float=0.25) -> float:
    if density_traj.ndim != 2 or density_traj.shape[1] < 4:
        return 0.0
    nrec = density_traj.shape[1]
    late_window = density_traj[:, nrec * 3 // 4:]
    counts = []
    for k in range(late_window.shape[1]):
        rho = late_window[:, k]
        thresh = threshold_frac * float(rho.max()) if rho.max() > 0 else 0.0
        above = rho > thresh
        if not above.any():
            counts.append(0)
            continue
        if above.all():
            counts.append(1)
            continue
        idx0 = int(np.argmin(above))
        rolled = np.roll(above, -idx0)
        diffs = np.diff(rolled.astype(int))
        counts.append(int((diffs == 1).sum()))
    arr = np.asarray(counts, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(arr.var())

def atom_count_3d(psi: np.ndarray, dx: float, threshold_frac: float=0.25) -> int:
    if psi.ndim != 3:
        raise ValueError(f'atom_count_3d expects 3D psi, got shape {psi.shape}')
    rho = np.abs(psi) ** 2
    if rho.max() <= 0:
        return 0
    thresh = threshold_frac * float(rho.max())
    above = rho > thresh
    if not above.any():
        return 0
    try:
        _, n = _label_periodic(above)
        return int(n)
    except ImportError:
        Nx, Ny, Nz = above.shape
        visited = np.zeros_like(above, dtype=bool)
        n_atoms = 0
        from collections import deque
        offs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    if above[i, j, k] and (not visited[i, j, k]):
                        n_atoms += 1
                        q = deque([(i, j, k)])
                        visited[i, j, k] = True
                        while q:
                            ci, cj, ck = q.popleft()
                            for di, dj, dk in offs:
                                ni = (ci + di) % Nx
                                nj = (cj + dj) % Ny
                                nk = (ck + dk) % Nz
                                if above[ni, nj, nk] and (not visited[ni, nj, nk]):
                                    visited[ni, nj, nk] = True
                                    q.append((ni, nj, nk))
        return int(n_atoms)

def atom_count_2d(psi: np.ndarray, dx: float, threshold_frac: float=0.25) -> int:
    if psi.ndim != 2:
        raise ValueError(f'atom_count_2d expects 2D psi, got shape {psi.shape}')
    rho = np.abs(psi) ** 2
    if rho.max() <= 0:
        return 0
    thresh = threshold_frac * float(rho.max())
    above = rho > thresh
    if not above.any():
        return 0
    try:
        _, n = _label_periodic(above)
        return int(n)
    except ImportError:
        Nx, Ny = above.shape
        visited = np.zeros_like(above, dtype=bool)
        n_atoms = 0
        from collections import deque
        offs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for i in range(Nx):
            for j in range(Ny):
                if above[i, j] and (not visited[i, j]):
                    n_atoms += 1
                    q = deque([(i, j)])
                    visited[i, j] = True
                    while q:
                        ci, cj = q.popleft()
                        for di, dj in offs:
                            ni = (ci + di) % Nx
                            nj = (cj + dj) % Ny
                            if above[ni, nj] and (not visited[ni, nj]):
                                visited[ni, nj] = True
                                q.append((ni, nj))
        return int(n_atoms)

def atom_centroids_3d(psi: np.ndarray, dx: float, threshold_frac: float=0.25) -> np.ndarray:
    if psi.ndim != 3:
        raise ValueError(f'atom_centroids_3d expects 3D psi, got {psi.shape}')
    rho = np.abs(psi) ** 2
    if rho.max() <= 0:
        return np.zeros((0, 3))
    thresh = threshold_frac * float(rho.max())
    above = rho > thresh
    if not above.any():
        return np.zeros((0, 3))
    Nx, Ny, Nz = above.shape
    try:
        from scipy.ndimage import center_of_mass
        lab, n = _label_periodic(above)
        if n == 0:
            return np.zeros((0, 3))
        coms = center_of_mass(rho, lab, range(1, n + 1))
        out = np.zeros((n, 3))
        for i, (gi, gj, gk) in enumerate(coms):
            out[i, 0] = (gi - Nx / 2) * dx
            out[i, 1] = (gj - Ny / 2) * dx
            out[i, 2] = (gk - Nz / 2) * dx
        return out
    except ImportError:
        visited = np.zeros_like(above, dtype=bool)
        centroids = []
        from collections import deque
        offs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    if above[i, j, k] and (not visited[i, j, k]):
                        q = deque([(i, j, k)])
                        visited[i, j, k] = True
                        comp = []
                        while q:
                            ci, cj, ck = q.popleft()
                            comp.append((ci, cj, ck))
                            for di, dj, dk in offs:
                                ni = (ci + di) % Nx
                                nj = (cj + dj) % Ny
                                nk = (ck + dk) % Nz
                                if above[ni, nj, nk] and (not visited[ni, nj, nk]):
                                    visited[ni, nj, nk] = True
                                    q.append((ni, nj, nk))
                        comp = np.asarray(comp)
                        w = rho[comp[:, 0], comp[:, 1], comp[:, 2]]
                        cx = float((comp[:, 0] * w).sum() / w.sum() - Nx / 2) * dx
                        cy = float((comp[:, 1] * w).sum() / w.sum() - Ny / 2) * dx
                        cz = float((comp[:, 2] * w).sum() / w.sum() - Nz / 2) * dx
                        centroids.append((cx, cy, cz))
        return np.asarray(centroids) if centroids else np.zeros((0, 3))

def atom_centroids_2d(psi: np.ndarray, dx: float, threshold_frac: float=0.25) -> np.ndarray:
    if psi.ndim != 2:
        raise ValueError(f'atom_centroids_2d expects 2D psi, got {psi.shape}')
    rho = np.abs(psi) ** 2
    if rho.max() <= 0:
        return np.zeros((0, 2))
    thresh = threshold_frac * float(rho.max())
    above = rho > thresh
    if not above.any():
        return np.zeros((0, 2))
    Nx, Ny = above.shape
    try:
        from scipy.ndimage import center_of_mass
        lab, n = _label_periodic(above)
        if n == 0:
            return np.zeros((0, 2))
        coms = center_of_mass(rho, lab, range(1, n + 1))
        out = np.zeros((n, 2))
        for i, (gi, gj) in enumerate(coms):
            out[i, 0] = (gi - Nx / 2) * dx
            out[i, 1] = (gj - Ny / 2) * dx
        return out
    except ImportError:
        visited = np.zeros_like(above, dtype=bool)
        centroids = []
        from collections import deque
        offs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for i in range(Nx):
            for j in range(Ny):
                if above[i, j] and (not visited[i, j]):
                    q = deque([(i, j)])
                    visited[i, j] = True
                    comp = []
                    while q:
                        ci, cj = q.popleft()
                        comp.append((ci, cj))
                        for di, dj in offs:
                            ni = (ci + di) % Nx
                            nj = (cj + dj) % Ny
                            if above[ni, nj] and (not visited[ni, nj]):
                                visited[ni, nj] = True
                                q.append((ni, nj))
                    comp = np.asarray(comp)
                    w = rho[comp[:, 0], comp[:, 1]]
                    cx = float((comp[:, 0] * w).sum() / w.sum() - Nx / 2) * dx
                    cy = float((comp[:, 1] * w).sum() / w.sum() - Ny / 2) * dx
                    centroids.append((cx, cy))
        return np.asarray(centroids) if centroids else np.zeros((0, 2))

def count_clusters(psi: np.ndarray, dx: float, threshold_frac: float=0.25) -> int:
    rho = np.abs(psi) ** 2
    if rho.max() <= 0:
        return 0
    threshold = threshold_frac * rho.max()
    above = rho > threshold
    if not above.any():
        return 0
    if above.all():
        return 1
    idx = int(np.argmin(above))
    rolled = np.roll(above, -idx)
    diffs = np.diff(rolled.astype(int))
    return int((diffs == 1).sum())

def cluster_centroids(psi: np.ndarray, dx: float, threshold_frac: float=0.25) -> np.ndarray:
    rho = np.abs(psi) ** 2
    if rho.max() <= 0:
        return np.zeros(0)
    threshold = threshold_frac * rho.max()
    above = rho > threshold
    N = len(rho)
    x = (np.arange(N) - N // 2) * dx
    centroids = []
    in_cluster = False
    cur_idx = []
    for i in range(N):
        if above[i]:
            in_cluster = True
            cur_idx.append(i)
        elif in_cluster:
            idxs = np.array(cur_idx)
            w = rho[idxs]
            centroids.append(float(np.sum(x[idxs] * w) / max(w.sum(), 1e-30)))
            cur_idx = []
            in_cluster = False
    if in_cluster:
        idxs = np.array(cur_idx)
        w = rho[idxs]
        centroids.append(float(np.sum(x[idxs] * w) / max(w.sum(), 1e-30)))
    return np.asarray(centroids)