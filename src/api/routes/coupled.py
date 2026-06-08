from __future__ import annotations
import asyncio
import os
import numpy as np
from fastapi import APIRouter, HTTPException
from api.models import CoupledRunRequest, CoupledRunResult, SubstrateResult
from api.serialization import ndarray_to_b64, compute_standard_observables

router = APIRouter()

_TIMEOUT = float(os.environ.get('TRIAD_API_TIMEOUT', '120'))

_COUPLING_MAP = {
    'ring': 'ring',
    'all2all': 'full',
    'none': 'none',
}

def _run_coupled_sync(req: CoupledRunRequest) -> CoupledRunResult:
    from runtime.core.equation_runtime import EquationRuntime
    from runtime.backend import asnumpy

    coupling = _COUPLING_MAP.get(req.coupling, 'ring')
    er = EquationRuntime(
        n_substrates=req.n_substrates,
        regime=req.regime,
        N=req.N,
        coupling=coupling,
        kappa=req.kappa,
        seed=req.seed,
    )
    raw = er.run(T=req.T)
    mr = er._mr
    elapsed = float(raw['elapsed'])

    substrate_results = []
    for i, (sid, sub) in enumerate(mr.substrates.items()):
        psi = asnumpy(sub.psi)
        dx = float(sub.dx)
        L = float(sub.params.L)
        obs = compute_standard_observables(psi, dx, L)
        substrate_results.append(SubstrateResult(
            substrate_id=i,
            name=sub.name if hasattr(sub, 'name') else f's{i}',
            psi_final_b64=ndarray_to_b64(np.asarray(psi)),
            crystallinity=obs['crystallinity'],
            k_star=obs['k_star'],
            peak_density=obs['peak_density'],
            norm=obs['norm'],
            ipr=obs['ipr'],
            fwhm=obs['fwhm'],
        ))

    return CoupledRunResult(substrates=substrate_results, elapsed=elapsed)

@router.post('/run', response_model=CoupledRunResult, summary='Run coupled multi-substrate Triad system')
async def run_coupled(req: CoupledRunRequest):
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _run_coupled_sync, req),
            timeout=_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f'coupled solver timed out after {_TIMEOUT}s')
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result

@router.get('/regimes', summary='List all available regime names for coupled runs')
async def list_coupled_regimes():
    from stdlib.regimes import list_regimes
    return {'regimes': list_regimes()}
