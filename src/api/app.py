from __future__ import annotations

import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.middleware import AuthMiddleware, RateLimitMiddleware, UnsafeBlockMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

def create_app():
    app = FastAPI(
        title='TriadLang API',
        description='REST API for the Triad PDE runtime — P1 (phase/FFT) + P2 (memory/self-reference) + P3 (dissipation/FDT noise).',
        version='1.0.0',
        lifespan=lifespan,
    )
    app.add_middleware(AuthMiddleware)
    app.add_middleware(UnsafeBlockMiddleware)
    app.add_middleware(RateLimitMiddleware)
    return app

def get_app():
    return app

app = create_app()

from api.routes import (
    adapter,
    coupled,
    engine3d,
    lang,
    observables,
    plot,
    regimes,
    solver,
    templates,
)
from api.routes import chain as chain_routes

app.include_router(solver.router, prefix='/solver', tags=['solver'])
app.include_router(coupled.router, prefix='/coupled', tags=['coupled'])
app.include_router(observables.router, prefix='/observables', tags=['observables'])
app.include_router(lang.router, prefix='/lang', tags=['lang'])
app.include_router(regimes.router, prefix='/regimes', tags=['regimes'])
app.include_router(adapter.router, prefix='/adapter', tags=['adapter'])
app.include_router(templates.router, prefix='/api/templates', tags=['templates'])
app.include_router(plot.router, prefix='/solver', tags=['plot'])
app.include_router(engine3d.router, prefix='/engine3d', tags=['engine3d'])
app.include_router(chain_routes.router, prefix='/chain', tags=['chain'])

app.include_router(solver.router, prefix='/v1/solve', tags=['solver-native'])

try:
    from api.routes import ml as ml_router
    app.include_router(ml_router.router, prefix='/v1/ml', tags=['ml-interop'])
except ImportError as _e:
    import sys as _sys
    print(f'api: ML router nao disponivel ({_e!r}); endpoints /v1/ml desabilitados', file=_sys.stderr)

_DASHBOARD_PATH = os.path.join(os.path.dirname(__file__), 'dashboard.html')
try:
    with open(_DASHBOARD_PATH) as _f:
        _DASHBOARD_HTML = _f.read()
except OSError:
    _DASHBOARD_HTML = '<h1>Dashboard not found</h1>'

from fastapi.responses import HTMLResponse

_ENGINE_PATH = os.path.join(os.path.dirname(__file__), 'engine.html')

@app.get('/engine', response_class=HTMLResponse, include_in_schema=False)
def engine_page():
    try:
        with open(_ENGINE_PATH) as f:
            return f.read()
    except OSError:
        return '<h1>engine.html not found</h1>'

@app.get('/', response_class=HTMLResponse)
async def dashboard():
    return _DASHBOARD_HTML

@app.get('/health')
async def health():
    return {'status': 'ok', 'runtime': 'triad-p1p2p3'}
