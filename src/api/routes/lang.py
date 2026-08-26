from __future__ import annotations

import asyncio
import contextlib
import io

from fastapi import APIRouter, HTTPException

from api.models import (
    LangCheckResult,
    LangCompileResult,
    LangFormatResult,
    LangRunRequest,
    LangRunResult,
)

router = APIRouter()

_TIMEOUT = float(__import__('os').environ.get('TRIAD_API_TIMEOUT', '120'))

def _run_source_sync(source: str) -> LangRunResult:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    try:
        import os

        from embed.engine import TriadEngine
        from runtime.security import default_policy
        engine = TriadEngine(safe=True, policy=default_policy(os.getcwd()))
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            engine.run_source(source)
        return LangRunResult(stdout=stdout_buf.getvalue(), stderr=stderr_buf.getvalue(), ok=True)
    except Exception as exc:
        return LangRunResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            ok=False,
            error=str(exc),
        )

def _check_sync(source: str) -> LangCheckResult:
    try:
        from compiler.typecheck_universal import TypeCheckError, typecheck
        from frontend.parser_universal import parse
        mod = parse(source, '<api>')
        typecheck(mod)
        return LangCheckResult(ok=True, errors=[])
    except TypeCheckError as exc:
        return LangCheckResult(ok=False, errors=exc.errors)
    except Exception as exc:
        return LangCheckResult(ok=False, errors=[str(exc)])

def _compile_sync(source: str) -> LangCompileResult:
    try:
        from compiler.emit_json import emit_json
        from compiler.lower import lower_module
        from frontend.parser_universal import parse
        mod = parse(source, '<api>')
        ir = lower_module(mod)
        ir_json = emit_json(ir)
        return LangCompileResult(ir_json=ir_json, ok=True)
    except Exception as exc:
        return LangCompileResult(ir_json='', ok=False, error=str(exc))

def _format_sync(source: str) -> LangFormatResult:
    try:
        from compiler.formatter import format_universal
        from frontend.parser_universal import parse as _parse
        mod = _parse(source, '<api>')
        formatted = format_universal(mod)
        return LangFormatResult(source=formatted, ok=True)
    except Exception as exc:
        return LangFormatResult(source=source, ok=False, error=str(exc))

@router.post('/run', response_model=LangRunResult, summary='Execute TriadLang source code, return stdout/stderr')
async def lang_run(req: LangRunRequest):
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _run_source_sync, req.source),
            timeout=_TIMEOUT,
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail=f'execution timed out after {_TIMEOUT}s')
    return result

@router.post('/check', response_model=LangCheckResult, summary='Type-check TriadLang source, return error list')
async def lang_check(req: LangRunRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _check_sync, req.source)
    return result

@router.post('/compile', response_model=LangCompileResult, summary='Compile TriadLang source to IR JSON')
async def lang_compile(req: LangRunRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _compile_sync, req.source)
    return result

@router.post('/format', response_model=LangFormatResult, summary='Format TriadLang source code')
async def lang_format(req: LangRunRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _format_sync, req.source)
    return result
