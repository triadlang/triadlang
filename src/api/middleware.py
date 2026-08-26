from __future__ import annotations

import os
import time

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

_API_UNSAFE = os.environ.get('TRIAD_API_UNSAFE', '0').lower() in ('1', 'true', 'yes')
_API_TOKEN = os.environ.get('TRIAD_API_TOKEN', '')
if _API_TOKEN:
    _EXPECTED = set(t.strip() for t in _API_TOKEN.split(',') if t.strip())
else:
    _EXPECTED = None


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in ('/health', '/', '/engine'):
            return await call_next(request)
        if _EXPECTED is None:
            return await call_next(request)
        auth = request.headers.get('authorization', '')
        if auth.startswith('Bearer '):
            token = auth[7:].strip()
        else:
            token = request.query_params.get('token', '')
        if token not in _EXPECTED:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={'detail': 'invalid or missing token'},
            )
        return await call_next(request)


class UnsafeBlockMiddleware(BaseHTTPMiddleware):
    BLOCKED_PREFIXES = ('/lang/', '/adapter/')

    async def dispatch(self, request: Request, call_next):
        if not _API_UNSAFE and request.url.path.startswith(self.BLOCKED_PREFIXES):
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={'detail': 'execution endpoints disabled. set TRIAD_API_UNSAFE=1 to enable.'},
            )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 60, window: int = 60):
        super().__init__(app)
        self.max_requests = int(os.environ.get('TRIAD_API_RATE_LIMIT', max_requests))
        self.window = window
        self._history: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next):
        if self.max_requests <= 0:
            return await call_next(request)
        key = request.client.host if request.client else 'unknown'
        now = time.time()
        hist = self._history.get(key, [])
        hist = [t for t in hist if now - t < self.window]
        hist.append(now)
        self._history[key] = hist
        if len(hist) > self.max_requests:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={'detail': 'rate limit exceeded'},
            )
        return await call_next(request)
