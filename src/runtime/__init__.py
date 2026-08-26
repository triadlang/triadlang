from __future__ import annotations

from runtime.env import load_os_environ as _load_env

_load_env()

__all__ = ['TriadCompiler', 'CompileError']

def __getattr__(name):
    if name == 'TriadCompiler':
        from runtime.compiler_runtime import TriadCompiler
        return TriadCompiler
    if name == 'CompileError':
        from runtime.compiler_runtime import CompileError
        return CompileError
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
