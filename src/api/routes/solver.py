from __future__ import annotations

import asyncio
import json
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.models import (
    BatchSolveRequest,
    BatchSolveResult,
    SolveRequest,
    SolveResult,
    TriadParamsModel,
)
from api.serialization import b64_to_ndarray, compute_standard_observables, ndarray_to_b64
from triad import ntri as np

router = APIRouter()

_TIMEOUT = float(os.environ.get('TRIAD_API_TIMEOUT', '120'))

def _resolve_params(req: SolveRequest):
    if req.regime:
        from stdlib.regimes import resolve_regime
        base = resolve_regime(req.regime)
        overrides = req.params.model_dump(exclude_unset=True)
        if 'nu' in overrides:
            overrides['nu'] = tuple(overrides['nu'])
        if 'lam' in overrides:
            overrides['lam'] = tuple(overrides['lam'])
        for k, v in overrides.items():
            setattr(base, k, v)
        return base
    return req.params.to_triad_params()

def _build_solve_result(raw: dict, p) -> SolveResult:
    psi = raw['psi_final']
    x = raw.get('x', np.array([]))
    dx = float(raw.get('dx', p.L / p.N))
    t_arr = raw.get('t', raw.get('t_traj', np.array([p.T])))
    t_final = float(t_arr[-1]) if len(t_arr) > 0 else float(p.T)
    obs = compute_standard_observables(psi, dx, p.L)
    return SolveResult(
        psi_final_b64=ndarray_to_b64(np.asarray(psi)),
        x_b64=ndarray_to_b64(np.asarray(x)),
        dx=dx,
        t_final=t_final,
        params_used=TriadParamsModel.from_triad_params(p),
        **obs,
    )

def _run_1d_sync(req: SolveRequest) -> SolveResult:
    from runtime.core.solver import integrate
    p = _resolve_params(req)
    if p.D != 1:
        raise HTTPException(status_code=400, detail=f'D mismatch: endpoint is 1D but params D={p.D}')
    psi0 = b64_to_ndarray(req.psi0_b64) if req.psi0_b64 else None
    raw = integrate(p, psi0=psi0)
    return _build_solve_result(raw, p)

def _run_2d_sync(req: SolveRequest) -> SolveResult:
    from runtime.core.solver import integrate_2d
    p = _resolve_params(req)
    if p.D != 2:
        raise HTTPException(status_code=400, detail=f'D mismatch: endpoint is 2D but params D={p.D}')
    psi0 = b64_to_ndarray(req.psi0_b64) if req.psi0_b64 else None
    raw = integrate_2d(p, psi0=psi0)
    return _build_solve_result(raw, p)

def _run_3d_sync(req: SolveRequest) -> SolveResult:
    from runtime.core.solver import integrate_3d
    p = _resolve_params(req)
    if p.D != 3:
        raise HTTPException(status_code=400, detail=f'D mismatch: endpoint is 3D but params D={p.D}')
    raw = integrate_3d(p)
    return _build_solve_result(raw, p)

def _run_batch_sync(req: BatchSolveRequest) -> BatchSolveResult:
    from runtime.core.solver import integrate_nd
    seeds = req.seeds if req.seeds else list(range(req.K))
    results = []
    for s in seeds:
        if req.regime:
            from stdlib.regimes import resolve_regime
            p = resolve_regime(req.regime, seed=s, N=req.params.N, L=req.params.L, dt=req.params.dt)
            overrides = req.params.model_dump(exclude_unset=True)
            overrides.pop('seed', None)
            if 'nu' in overrides:
                overrides['nu'] = tuple(overrides['nu'])
            if 'lam' in overrides:
                overrides['lam'] = tuple(overrides['lam'])
            for k, v in overrides.items():
                setattr(p, k, v)
        else:
            p = req.params.to_triad_params()
            p.seed = s
        raw = integrate_nd(p)
        results.append(_build_solve_result(raw, p))
    crystallinities = [r.crystallinity for r in results]
    return BatchSolveResult(
        results=results,
        ensemble_crystallinity_mean=float(np.mean(crystallinities)),
        ensemble_crystallinity_std=float(np.std(crystallinities)),
    )

async def _stream_adaptive(req: SolveRequest):
    from runtime.core.solver import _integrate_steps
    p = _resolve_params(req)

    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    sentinel = object()
    loop = asyncio.get_event_loop()

    def _thread():
        try:
            for chk in _integrate_steps(p):
                psi = chk.get('psi')
                if psi is None:
                    continue
                dx = float(chk.get('dx', p.L / p.N))
                obs = compute_standard_observables(psi, dx, p.L)
                payload = {
                    'step': int(chk.get('step', 0)),
                    't': float(chk.get('t', 0.0)),
                    'psi_b64': ndarray_to_b64(np.asarray(psi)),
                    **obs,
                }
                asyncio.run_coroutine_threadsafe(queue.put(payload), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop).result()

    loop.run_in_executor(None, _thread)

    async def event_stream():
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            yield f'data: {json.dumps(item)}\n\n'

    return StreamingResponse(event_stream(), media_type='text/event-stream')

@router.post('/run', response_model=SolveResult, summary='Run 1D Triad PDE solver (P1+P2+P3)')
async def run_1d(req: SolveRequest):
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _run_1d_sync, req),
            timeout=_TIMEOUT,
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail=f'solver timed out after {_TIMEOUT}s')
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result

@router.post('/run/2d', response_model=SolveResult, summary='Run 2D Triad PDE solver (P1+P2+P3)')
async def run_2d(req: SolveRequest):
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _run_2d_sync, req),
            timeout=_TIMEOUT,
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail=f'solver timed out after {_TIMEOUT}s')
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result

@router.post('/run/3d', response_model=SolveResult, summary='Run 3D Triad PDE solver — use small N (default 24)')
async def run_3d(req: SolveRequest):
    if req.params.N > 64:
        raise HTTPException(
            status_code=422,
            detail='N > 64 for 3D produces very large responses. Set N <= 64 explicitly to proceed.',
        )
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _run_3d_sync, req),
            timeout=_TIMEOUT,
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail=f'solver timed out after {_TIMEOUT}s')
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result

@router.post('/run/adaptive', summary='Run adaptive Triad PDE solver with SSE streaming checkpoints')
async def run_adaptive(req: SolveRequest):
    try:
        from runtime.core.solver import _integrate_steps
    except ImportError:
        raise HTTPException(status_code=501, detail='_integrate_steps not available in this build')
    return await _stream_adaptive(req)

@router.post('/batch', response_model=BatchSolveResult, summary='Run K-seed ensemble of 1D Triad solver')
async def run_batch(req: BatchSolveRequest):
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _run_batch_sync, req),
            timeout=_TIMEOUT * req.K,
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail='batch timed out')
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result
