from __future__ import annotations

import asyncio
import contextlib
import io
import os
import tempfile
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

_TIMEOUT = float(os.environ.get('TRIAD_API_TIMEOUT', '120'))

class AdapterRunRequest(BaseModel):
    source: str
    adapter: str | None = None
    vars: list[str] | None = None

class AdapterRunResult(BaseModel):
    ok: bool
    stdout: str
    stderr: str
    vars: dict[str, Any] = Field(default_factory=dict)
    detected_framework: str | None = None
    error: str | None = None

class KernelRunRequest(BaseModel):
    params: dict[str, Any] = Field(default_factory=dict)
    observables: list[str] | None = None

class KernelRunResult(BaseModel):
    ok: bool
    observables: dict[str, float] = Field(default_factory=dict)
    error: str | None = None

class FrameworkDetectRequest(BaseModel):
    source: str

class FrameworkDetectResult(BaseModel):
    detected: list[str]
    adapter: str

def _json_safe(v) -> Any:
    from triad import ntri as np
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    return str(v)

def _run_adapter_sync(source: str, adapter_name: str | None, extract_vars: list[str]) -> AdapterRunResult:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    try:
        from adapters.discovery import _FRAMEWORK_RULES, _extract_imports
        from adapters.flask import FlaskAdapter
        from adapters.headless import HeadlessAdapter

        imports = _extract_imports(source)
        detected = []
        for required, name in _FRAMEWORK_RULES:
            if any(imp in imports for imp in required):
                detected.append(name)
        framework = detected[0] if detected else 'headless'

        chosen = adapter_name or framework

        with tempfile.NamedTemporaryFile(suffix='.tri', mode='w', delete=False) as f:
            f.write(source)
            tri_path = f.name

        try:
            if chosen == 'flask':

                adapter = FlaskAdapter(tri_file=tri_path)
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    try:
                        adapter.start()
                    except RuntimeError as exc:
                        import logging
                        logging.getLogger(__name__).debug('flask adapter start stopped: %s', exc)
            else:
                adapter = HeadlessAdapter(tri_file=tri_path)
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    adapter.start()

            extracted = {}
            for name in (extract_vars or []):
                v = adapter.get(name)
                if v is not None:
                    extracted[name] = _json_safe(v)

            return AdapterRunResult(
                ok=True,
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
                vars=extracted,
                detected_framework=framework if not adapter_name else None,
            )
        finally:
            os.unlink(tri_path)

    except Exception as exc:
        return AdapterRunResult(
            ok=False,
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            vars={},
            error=str(exc),
        )

def _run_kernel_sync(params: dict, observables: list[str] | None) -> KernelRunResult:
    try:
        source_lines = ['import triad.kernel as kernel;']
        source_lines.append(f'let _p = {_dict_to_tri_literal(params)};')
        source_lines.append('let _result = kernel.run(_p);')
        obs_list = observables or ['crystallinity', 'peak', 'norm', 'k_star']
        source_lines.append(f'let _obs = kernel.extract(_result, {_list_to_tri_literal(obs_list)});')
        source = '\n'.join(source_lines)

        import os
        import tempfile

        from adapters.headless import HeadlessAdapter

        with tempfile.NamedTemporaryFile(suffix='.tri', mode='w', delete=False) as f:
            f.write(source)
            tri_path = f.name

        try:
            adapter = HeadlessAdapter(tri_file=tri_path)
            adapter.start()
            obs_raw = adapter.get('_obs')
            if obs_raw is None:
                return KernelRunResult(ok=False, error='kernel.extract returned None')
            result_obs = {k: float(v) for k, v in obs_raw.items()}
            return KernelRunResult(ok=True, observables=result_obs)
        finally:
            os.unlink(tri_path)

    except Exception as exc:
        return KernelRunResult(ok=False, error=str(exc))

def _dict_to_tri_literal(d: dict[str, object]) -> str:
    import json as _json
    pairs = []
    for k, v in d.items():
        key = _json.dumps(str(k))
        if isinstance(v, str):
            pairs.append(f'{key}: {_json.dumps(v)}')
        elif isinstance(v, bool):
            pairs.append(f'{key}: {"true" if v else "false"}')
        elif isinstance(v, (int, float)):
            pairs.append(f'{key}: {repr(v)}')
        elif v is None:
            pairs.append(f'{key}: none')
        else:
            pairs.append(f'{key}: {_json.dumps(str(v))}')
    return '{' + ', '.join(pairs) + '}'

def _list_to_tri_literal(lst: list[str]) -> str:
    import json as _json
    return '[' + ', '.join(_json.dumps(str(x)) for x in lst) + ']'

@router.post('/run', response_model=AdapterRunResult,
             summary='Execute .tri source via auto-detected or explicit adapter (headless/flask)')
async def adapter_run(req: AdapterRunRequest):
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None, _run_adapter_sync, req.source, req.adapter, req.vars or []
            ),
            timeout=_TIMEOUT,
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail=f'adapter timed out after {_TIMEOUT}s')
    return result

@router.post('/kernel/run', response_model=KernelRunResult,
             summary='Call kernel.run(params) + kernel.extract() from .tri kernel module')
async def kernel_run(req: KernelRunRequest):
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _run_kernel_sync, req.params, req.observables),
            timeout=_TIMEOUT,
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail=f'kernel timed out after {_TIMEOUT}s')
    return result

@router.post('/detect', response_model=FrameworkDetectResult,
             summary='Detect which Python framework a .tri source imports')
async def detect_framework(req: FrameworkDetectRequest):
    from adapters.discovery import _FRAMEWORK_RULES, _extract_imports
    imports = _extract_imports(req.source)
    detected = []
    for required, name in _FRAMEWORK_RULES:
        if any(imp in imports for imp in required):
            detected.append(name)
    adapter = detected[0] if detected else 'headless'
    return FrameworkDetectResult(detected=detected, adapter=adapter)
