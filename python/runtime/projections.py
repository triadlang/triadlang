from __future__ import annotations
import numpy as np

def proj_centroid_density(subs) -> float:
    num = 0.0
    den = 0.0
    for s in subs:
        if s.D == 1:
            rho = np.abs(s.psi) ** 2
            x = s.x
            num += float((x * rho).sum() * s.dx)
            den += float(rho.sum() * s.dx)
        else:
            rho = np.abs(s.psi) ** 2
            num += float(rho.sum() * s.dx ** s.D)
            den += float(rho.sum() * s.dx ** s.D)
    return num / max(den, 1e-30)

def proj_integrated_density(subs) -> float:
    total = 0.0
    for s in subs:
        D = getattr(s, 'D', 1)
        total += float((np.abs(s.psi) ** 2).sum() * s.dx ** D)
    return total

def proj_dominant_k_star(subs) -> float:
    if not subs:
        return 0.0
    s0 = subs[0]
    dx = s0.dx
    N = s0.params.N
    L = N * dx
    k_min = 2.0 * np.pi / L
    if s0.D == 1:
        kvec = 2 * np.pi * np.fft.fftfreq(N, d=dx)
        agg = np.zeros(N)
        for s in subs:
            P = np.abs(np.fft.fft(s.psi)) ** 2
            agg = agg + P
        mask = np.abs(kvec) >= k_min
        if not mask.any():
            return 0.0
        idx_local = int(np.argmax(agg[mask]))
        return float(abs(kvec[mask][idx_local]))
    else:
        kvec = 2 * np.pi * np.fft.fftfreq(N, d=dx)
        KX, KY, KZ = np.meshgrid(kvec, kvec, kvec, indexing='ij')
        kmag = np.sqrt(KX * KX + KY * KY + KZ * KZ)
        agg = np.zeros(s0.psi.shape)
        for s in subs:
            agg = agg + np.abs(np.fft.fftn(s.psi)) ** 2
        mask = kmag >= k_min
        return float(kmag[mask][int(np.argmax(agg[mask]))])

def proj_fourier_band(subs, k_lo: float=0.1, k_hi: float=1.0) -> float:
    if not subs:
        return 0.0
    s0 = subs[0]
    dx = s0.dx
    N = s0.params.N
    if s0.D == 1:
        kvec = 2 * np.pi * np.fft.fftfreq(N, d=dx)
        total = 0.0
        for s in subs:
            P = np.abs(np.fft.fft(s.psi)) ** 2
            mask = (np.abs(kvec) >= k_lo) & (np.abs(kvec) <= k_hi)
            total += float(P[mask].sum())
        return total
    else:
        kvec = 2 * np.pi * np.fft.fftfreq(N, d=dx)
        KX, KY, KZ = np.meshgrid(kvec, kvec, kvec, indexing='ij')
        kmag = np.sqrt(KX * KX + KY * KY + KZ * KZ)
        total = 0.0
        for s in subs:
            P = np.abs(np.fft.fftn(s.psi)) ** 2
            mask = (kmag >= k_lo) & (kmag <= k_hi)
            total += float(P[mask].sum())
        return total

def proj_atoms_of_atoms(subs) -> float:
    from runtime.observables_atoms import atom_count
    total = 0
    for s in subs:
        if s.D == 1:
            total += atom_count(s.psi, s.dx)
        else:
            rho = np.abs(s.psi) ** 2
            marginal = rho.sum(axis=tuple(range(1, s.D)))
            thr = 0.25 * float(marginal.max()) if marginal.max() > 0 else 0
            above = marginal > thr
            if above.any():
                idx0 = int(np.argmin(above)) if not above.all() else 0
                rolled = np.roll(above, -idx0)
                total += int((np.diff(rolled.astype(int)) == 1).sum()) or 1
    return float(total)

def proj_none(subs) -> float:
    return 0.0
PROJECTOR_REGISTRY = {'centroid_density': proj_centroid_density, 'integrated_density': proj_integrated_density, 'dominant_k_star': proj_dominant_k_star, 'fourier_band': proj_fourier_band, 'atoms_of_atoms': proj_atoms_of_atoms, 'none': proj_none}

def resolve_projector(name: str):
    if name not in PROJECTOR_REGISTRY:
        raise KeyError(f'unknown projection {name!r}; available: {sorted(PROJECTOR_REGISTRY)}')
    return PROJECTOR_REGISTRY[name]