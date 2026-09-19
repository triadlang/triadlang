from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from triad import ntri as np

router = APIRouter()

_TIMEOUT = float(os.environ.get('TRIAD_API_TIMEOUT', '120'))

class PlotRequest(BaseModel):
    N: int = 128
    T: float = 10.0
    dt: float = 0.005
    L: float = 32.0
    Lambda: float = -0.5
    Gamma: float = 0.05
    sigma: float = 1.5
    alpha: float = 0.15
    f_FDT: float = 0.002
    regime: str | None = None
    V_ext: str | None = 'harmonic'
    seed: int = 0
    record_every: int = 20

def _build_plot_response(raw, p, record_every):

    psi = raw['psi_final']
    x = raw.get('x', np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False))
    dx = float(raw.get('dx', p.L / p.N))

    rho = np.abs(np.asarray(psi).ravel()) ** 2
    x_arr = np.asarray(x).ravel()

    from runtime.physics.observables import power_spectrum
    k, P = power_spectrum(np.asarray(psi).ravel(), dx)
    from runtime.backend import asnumpy
    k_arr = asnumpy(k).ravel()
    P_arr = asnumpy(P).ravel()

    from api.serialization import compute_standard_observables
    obs = compute_standard_observables(np.asarray(psi), dx, p.L)

    timeline = []
    history = raw.get('history', [])
    for chk in history:
        rho_t = np.abs(np.asarray(chk.get('psi', [])).ravel()) ** 2
        if len(rho_t) == 0:
            continue
        timeline.append({
            'step': int(chk.get('step', 0)),
            't': float(chk.get('t', 0.0)),
            'crystallinity': float(compute_standard_observables(np.asarray(chk['psi']), dx, p.L).get('crystallinity', 0)),
            'peak': float(rho_t.max()),
            'norm': float(rho_t.sum() * dx),
        })

    return {
        'x': x_arr.tolist(),
        'density': rho.tolist(),
        'k': k_arr.tolist(),
        'power_spectrum': P_arr.tolist(),
        'dx': dx,
        'L': float(p.L),
        't_final': float(raw.get('t_final', p.T)),
        'observables': obs,
        'timeline': timeline,
    }

def _run_plot_sync(req: PlotRequest) -> dict:
    from runtime.core.solver import TriadParams, integrate_adaptive
    from stdlib.regimes import resolve_regime

    if req.regime:
        p = resolve_regime(req.regime)
        p.seed = req.seed
        p.N = req.N
        p.L = req.L
        p.dt = req.dt
        p.T = req.T
        _set = req.model_fields_set if hasattr(req, 'model_fields_set') else set()
        for _k in ('Lambda', 'Gamma', 'sigma', 'alpha', 'f_FDT', 'V_ext'):
            if _k in _set:
                setattr(p, _k, getattr(req, _k))
    else:
        p = TriadParams(
            N=req.N, L=req.L, dt=req.dt, T=req.T,
            Lambda=req.Lambda, Gamma=req.Gamma, sigma=req.sigma,
            alpha=req.alpha, f_FDT=req.f_FDT,
            V_ext=req.V_ext, seed=req.seed,
        )

    raw = integrate_adaptive(p, checkpoint_every=req.record_every)
    return _build_plot_response(raw, p, req.record_every)

@router.post('/plot', summary='Run PDE solver and return JSON arrays for plotting (density, spectrum, timeline)')
async def plot(req: PlotRequest):
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _run_plot_sync, req),
            timeout=_TIMEOUT,
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail=f'solver timed out after {_TIMEOUT}s')
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result
