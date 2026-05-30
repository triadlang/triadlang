from __future__ import annotations
import os, sys
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)
from frontend.parser import parse
from compiler.triadc import Compiler, CompileConfig, decode_outputs
from runtime.solver import TriadParams, integrate, integrate_2d, integrate_3d
from stdlib.regimes import resolve_regime, list_regimes

def compile_source(src: str, config: CompileConfig=None):
    ast = parse(src)
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
__all__ = ['compile_source', 'run', 'Result', 'TriadParams', 'integrate', 'integrate_2d', 'integrate_3d', 'resolve_regime', 'list_regimes', 'CompileConfig']