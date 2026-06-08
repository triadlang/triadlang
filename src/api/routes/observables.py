from __future__ import annotations
from fastapi import APIRouter, HTTPException
from api.models import ObservablesRequest, ObservablesResult, PowerSpectrumRequest, PowerSpectrumResult
from api.serialization import b64_to_ndarray, ndarray_to_b64

router = APIRouter()

@router.post('/compute', response_model=ObservablesResult, summary='Compute all observables from a wavefunction field')
async def compute_observables(req: ObservablesRequest):
    try:
        psi = b64_to_ndarray(req.psi_b64)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f'failed to decode psi_b64: {exc}')

    psi_1d = psi.ravel()
    dx = req.dx
    L = req.L if req.L is not None else dx * len(psi_1d)
    k_min = 2.0 * 3.141592653589793 / L

    try:
        from runtime.physics.observables import (
            crystallinity, dominant_wavenumber, peak_density,
            ipr, fwhm, norm, participation_ratio,
        )
        return ObservablesResult(
            crystallinity=float(crystallinity(psi_1d, dx, k_cutoff=req.k_cutoff)),
            k_star=float(dominant_wavenumber(psi_1d, dx, k_min=k_min)),
            peak_density=float(peak_density(psi_1d)),
            norm=float(norm(psi_1d, dx)),
            ipr=float(ipr(psi_1d, dx)),
            fwhm=float(fwhm(psi_1d, dx)),
            participation_ratio=float(participation_ratio(psi_1d, dx)),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.post('/power_spectrum', response_model=PowerSpectrumResult, summary='Return FFT power spectrum (k, P) arrays')
async def power_spectrum_endpoint(req: PowerSpectrumRequest):
    try:
        psi = b64_to_ndarray(req.psi_b64)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f'failed to decode psi_b64: {exc}')

    try:
        from runtime.physics.observables import power_spectrum
        from runtime.backend import asnumpy
        import numpy as np
        k, P = power_spectrum(psi.ravel(), req.dx)
        k_np = asnumpy(k).astype(np.float64)
        P_np = asnumpy(P).astype(np.float64)
        return PowerSpectrumResult(
            k_b64=ndarray_to_b64(k_np),
            P_b64=ndarray_to_b64(P_np),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
