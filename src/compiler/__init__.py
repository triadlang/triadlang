from __future__ import annotations

__all__ = ['TriadCompiler', 'CompileConfig', 'decode_outputs', 'main']


def __getattr__(name):
    """Load the solver-backed compiler only when it is actually requested.

    Keeping package import lightweight lets frontend-only tools such as the
    parser, formatter and type checker run without initializing the numerical
    runtime or probing GPU drivers.
    """
    if name in {'TriadCompiler', 'CompileConfig', 'decode_outputs'}:
        from compiler.triadc import CompileConfig, Compiler, decode_outputs

        values = {
            'TriadCompiler': Compiler,
            'CompileConfig': CompileConfig,
            'decode_outputs': decode_outputs,
        }
        globals().update(values)
        return values[name]
    raise AttributeError(name)

def main(argv=None):
    from cli.main import main as _main
    return _main(argv)

if __name__ == '__main__':
    import sys
    sys.exit(main())
