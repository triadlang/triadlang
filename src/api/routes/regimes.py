from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.models import RegimeInfo, TriadParamsModel

router = APIRouter()

@router.get('/', summary='List all available regime names')
async def list_regimes():
    from stdlib.regimes import list_regimes as _list
    return {'regimes': _list()}

@router.get('/{name}', response_model=RegimeInfo, summary='Get TriadParams for a named regime')
async def get_regime(name: str):
    from stdlib.regimes import list_regimes as _list
    from stdlib.regimes import resolve_regime
    if name not in _list():
        raise HTTPException(status_code=404, detail=f'regime {name!r} not found; available: {_list()}')
    try:
        p = resolve_regime(name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return RegimeInfo(
        name=name,
        params=TriadParamsModel.from_triad_params(p),
    )
