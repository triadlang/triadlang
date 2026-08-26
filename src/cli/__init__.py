from __future__ import annotations

__all__ = ['main', 'cmd_run', 'run_repl']

def __getattr__(name):
    if name in ('main', 'cmd_run'):
        from cli.main import cmd_run, main
        return locals()[name]
    if name == 'run_repl':
        from cli.repl import run_repl
        return run_repl
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
