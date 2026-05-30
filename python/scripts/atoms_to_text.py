import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from runtime.observables_atoms import atom_count_nd, atom_centroids_nd, atom_separation, atoms_per_region

def _f(x):
    return repr(float(x))

def _emit_centroids(arr: np.ndarray, D: int) -> str:
    if arr is None or arr.size == 0:
        return '0'
    arr = np.asarray(arr)
    if D == 1:
        arr = np.sort(arr)
        parts = [str(arr.shape[0])] + [_f(v) for v in arr]
    else:
        rows = [tuple(r) for r in arr]
        rows.sort()
        parts = [str(len(rows))]
        for r in rows:
            for v in r:
                parts.append(_f(v))
    return ' '.join(parts)

def dump_fixture(name: str, psi: np.ndarray, dx: float, D: int, N: int, thr: float, lines: list) -> None:
    lines.append(f'fixture {name}')
    lines.append(f'D {D} N {N} dx {_f(dx)} thr {_f(thr)}')
    c = atom_count_nd(psi, dx, threshold_frac=thr)
    lines.append(f'count {int(c)}')
    apr = atoms_per_region(psi, dx, threshold_frac=thr)
    lines.append(f'per_region {_f(apr)}')
    if D == 1:
        sep = atom_separation(psi, dx, threshold_frac=thr)
        lines.append(f'separation_1d {_f(sep)}')
    cs = atom_centroids_nd(psi, dx, threshold_frac=thr)
    if hasattr(cs, 'shape') and cs.ndim == 2:
        lines.append(f'centroids {_emit_centroids(cs, D)}')
    else:
        lines.append(f'centroids {_emit_centroids(np.asarray(cs), D)}')

def fixture_1d_three_gaussians():
    N = 64
    L = 32.0
    dx = L / N
    x = (np.arange(N) - N // 2) * dx
    psi = (np.exp(-(x + 10) ** 2 / (2 * 0.8 ** 2)) + np.exp(-x ** 2 / (2 * 0.8 ** 2)) + np.exp(-(x - 10) ** 2 / (2 * 0.8 ** 2))).astype(np.complex128)
    return (psi, dx, N)

def fixture_1d_wraps_boundary():
    N = 64
    L = 32.0
    dx = L / N
    x = (np.arange(N) - N // 2) * dx
    psi = (np.exp(-(x - 15) ** 2 / (2 * 0.6 ** 2)) + np.exp(-(x + 15) ** 2 / (2 * 0.6 ** 2))).astype(np.complex128)
    return (psi, dx, N)

def fixture_1d_empty():
    N = 32
    L = 16.0
    dx = L / N
    psi = np.zeros(N, dtype=np.complex128)
    return (psi, dx, N)

def fixture_2d_four_blobs():
    N = 32
    L = 16.0
    dx = L / N
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x, indexing='ij')
    w = 0.6
    psi = (np.exp(-((X - 4) ** 2 + (Y - 4) ** 2) / (2 * w * w)) + np.exp(-((X + 4) ** 2 + (Y - 4) ** 2) / (2 * w * w)) + np.exp(-((X - 4) ** 2 + (Y + 4) ** 2) / (2 * w * w)) + np.exp(-((X + 4) ** 2 + (Y + 4) ** 2) / (2 * w * w))).astype(np.complex128)
    return (psi, dx, N)

def fixture_2d_single_blob():
    N = 32
    L = 16.0
    dx = L / N
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x, indexing='ij')
    psi = np.exp(-(X * X + Y * Y) / (2 * 0.8 * 0.8)).astype(np.complex128)
    return (psi, dx, N)

def fixture_3d_two_blobs():
    N = 16
    L = 8.0
    dx = L / N
    x = (np.arange(N) - N // 2) * dx
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    w = 0.5
    psi = (np.exp(-((X - 2) ** 2 + (Y - 2) ** 2 + (Z - 2) ** 2) / (2 * w * w)) + np.exp(-((X + 2) ** 2 + (Y + 2) ** 2 + (Z + 2) ** 2) / (2 * w * w))).astype(np.complex128)
    return (psi, dx, N)

def fixture_3d_single_blob():
    N = 16
    L = 8.0
    dx = L / N
    x = (np.arange(N) - N // 2) * dx
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    psi = np.exp(-(X * X + Y * Y + Z * Z) / (2 * 0.8 * 0.8)).astype(np.complex128)
    return (psi, dx, N)

def run() -> str:
    lines: list[str] = []
    psi, dx, N = fixture_1d_three_gaussians()
    dump_fixture('1d_three_gaussians', psi, dx, 1, N, 0.25, lines)
    psi, dx, N = fixture_1d_wraps_boundary()
    dump_fixture('1d_wraps_boundary', psi, dx, 1, N, 0.25, lines)
    psi, dx, N = fixture_1d_empty()
    dump_fixture('1d_empty', psi, dx, 1, N, 0.25, lines)
    psi, dx, N = fixture_2d_four_blobs()
    dump_fixture('2d_four_blobs', psi, dx, 2, N, 0.25, lines)
    psi, dx, N = fixture_2d_single_blob()
    dump_fixture('2d_single_blob', psi, dx, 2, N, 0.25, lines)
    psi, dx, N = fixture_3d_two_blobs()
    dump_fixture('3d_two_blobs', psi, dx, 3, N, 0.25, lines)
    psi, dx, N = fixture_3d_single_blob()
    dump_fixture('3d_single_blob', psi, dx, 3, N, 0.25, lines)
    return '\n'.join(lines) + '\n'
if __name__ == '__main__':
    sys.stdout.write(run())