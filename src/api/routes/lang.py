from __future__ import annotations
import asyncio
import contextlib
import io
import sys
from fastapi import APIRouter, HTTPException
from api.models import (
    LangRunRequest, LangRunResult, LangCheckResult,
    LangCompileResult, LangFormatResult,
)

router = APIRouter()

_TIMEOUT = float(__import__('os').environ.get('TRIAD_API_TIMEOUT', '120'))

def _run_source_sync(source: str) -> LangRunResult:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    try:
        from embed.engine import TriadEngine
        engine = TriadEngine()
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
        from frontend.parser_universal import parse
        from compiler.typecheck_universal import TypeChecker
        mod = parse(source, '<api>')
        tc = TypeChecker()
        errs = tc.check(mod)
        if errs:
            return LangCheckResult(ok=False, errors=[str(e) for e in errs])
        return LangCheckResult(ok=True, errors=[])
    except Exception as exc:
        return LangCheckResult(ok=False, errors=[str(exc)])

def _compile_sync(source: str) -> LangCompileResult:
    try:
        from frontend.parser_universal import parse
        from compiler.lower import lower_module
        from compiler.emit_json import emit_json
        mod = parse(source, '<api>')
        ir = lower_module(mod)
        ir_json = emit_json(ir)
        return LangCompileResult(ir_json=ir_json, ok=True)
    except Exception as exc:
        return LangCompileResult(ir_json='', ok=False, error=str(exc))

def _format_sync(source: str) -> LangFormatResult:
    try:
        from compiler.formatter import format_source
        formatted = format_source(source)
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
    except asyncio.TimeoutError:
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
