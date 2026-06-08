from __future__ import annotations
import sys
import os

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    import runtime.core.solver as _s
    import runtime.physics.observables as _o
    import stdlib.regimes as _r
    yield

app = FastAPI(
    title='TriadLang API',
    description='REST API for the Triad PDE runtime — P1 (phase/FFT) + P2 (memory/self-reference) + P3 (dissipation/FDT noise).',
    version='1.0.0',
    lifespan=lifespan,
)

from api.routes import solver, coupled, observables, lang, regimes, adapter, templates, plot  

app.include_router(solver.router, prefix='/solver', tags=['solver'])
app.include_router(coupled.router, prefix='/coupled', tags=['coupled'])
app.include_router(observables.router, prefix='/observables', tags=['observables'])
app.include_router(lang.router, prefix='/lang', tags=['lang'])
app.include_router(regimes.router, prefix='/regimes', tags=['regimes'])
app.include_router(adapter.router, prefix='/adapter', tags=['adapter'])
app.include_router(templates.router, prefix='/api/templates', tags=['templates'])
app.include_router(plot.router, prefix='/solver', tags=['plot'])

_DASHBOARD_PATH = os.path.join(os.path.dirname(__file__), 'dashboard.html')
try:
    with open(_DASHBOARD_PATH, 'r') as _f:
        _DASHBOARD_HTML = _f.read()
except Exception:
    _DASHBOARD_HTML = '<h1>Dashboard not found</h1>'

from fastapi.responses import HTMLResponse

@app.get('/', response_class=HTMLResponse)
async def dashboard():
    return _DASHBOARD_HTML

@app.get('/health')
async def health():
    return {'status': 'ok', 'runtime': 'triad-p1p2p3'}
