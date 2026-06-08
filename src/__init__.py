from __future__ import annotations
import os, sys
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from frontend.parser_universal import parse
from frontend.parser import parse as parse_legacy
from compiler.triadc import Compiler, CompileConfig, decode_outputs
from runtime.core.solver import TriadParams, integrate, integrate_2d, integrate_3d
from stdlib.regimes import resolve_regime, list_regimes

def compile_source(src: str, file_path: str = '<triad>'):
    """Parse and execute modern triadlang source.

    Uses the universal parser and the task-oriented compiler runtime, which is
    what `triad run` uses. Returns nothing: the program executes (printing its
    own output) the same way it would from the cli. For the substrate-oriented
    reg/observe/run DSL, use compile_dsl instead.
    """
    from runtime.compiler_runtime import TriadCompiler
    mod = parse(src, file_path)
    mod.file = file_path
    return TriadCompiler().compile_and_run(mod)

def compile_dsl(src: str, config: CompileConfig = None):
    """Compile the legacy reg/observe/run DSL into a runnable program.

    Pair with run()/Result to inspect substrate observables. This is the
    field-substrate pipeline; for the modern language use compile_source.
    """
    ast = parse_legacy(src)
    return Compiler(config=config).compile(ast)

def run(compiled, verbose: bool=False):
    result = compiled.runtime.run(verbose=verbose)
    return Result(compiled, result)

class Result:

    def __init__(self, compiled, run_result):
        self.compiled = compiled
        self.run_result = run_result
        self._decoded = None

    @property
    def diverged(self) -> bool:
        return bool(self.run_result.get('diverged', False))

    @property
    def decoded(self) -> dict:
        if self._decoded is None:
            self._decoded = decode_outputs(self.compiled)
        return self._decoded

    def observe(self, name: str) -> dict:
        return self.decoded.get(name, {})

    def substrate(self, name: str):
        slot = self.compiled.mm.get(name)
        return self.compiled.runtime.substrates[slot.substrate_id]
__all__ = ['parse', 'parse_legacy', 'compile_source', 'compile_dsl', 'run', 'Result', 'TriadParams', 'integrate', 'integrate_2d', 'integrate_3d', 'resolve_regime', 'list_regimes', 'CompileConfig']
