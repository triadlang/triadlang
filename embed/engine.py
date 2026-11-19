from __future__ import annotations
import os
from frontend.parser_universal import parse
from runtime.compiler_runtime import TriadCompiler

class TriadEngine:

    def __init__(self, search_paths: list[str] | None = None,
                 backend: str = 'auto'):
        self._search_paths = search_paths or ['.']
        self._backend = backend
        self._extra_globals: dict = {}
        self._last_namespace: dict = {}

    def register(self, name: str, obj) -> None:
        self._extra_globals[name] = obj

    def get(self, name: str):
        return self._last_namespace.get(name)

    def run_source(self, text: str, file: str = '<triad>') -> dict:
        mod = parse(text, file)
        compiler = TriadCompiler()
        compiler._search_paths = self._search_paths
        code = compiler.compile_to_source(mod)
        env = compiler._make_globals(file)
        env.update(self._extra_globals)
        env['_search_paths'] = compiler._search_paths
        compiled = compile(code, file, 'exec')
        exec(compiled, env)
        self._last_namespace = {
            k: v for k, v in env.items()
            if not k.startswith('__')
        }
        return self._last_namespace

    def run_file(self, path: str) -> dict:
        path = os.path.abspath(path)
        with open(path) as f:
            src = f.read()
        return self.run_source(src, file=path)

    def eval_expr(self, text: str):
        ns = self.run_source(text)
        candidates = [v for k, v in ns.items()
                      if not k.startswith('_') and k not in
                      ('_tri_print', '_TriObj', '_TriModule',
                       '_tri_import', '_search_paths', '_math',
                       '_random', '_json', '_os', '_time_mod', '_np',
                       'len', 'range', 'enumerate', 'str', 'int',
                       'float', '_tri_type', 'abs', 'min', 'max',
                       'append', 'sorted', 'reversed', 'ccall',
                       'py_call', 'py_eval', 'py_exec', 'run_async',
                       'sleep', 'gather', 'create_task', 'async_read',
                       'async_write', 'async_fetch', '_tri_par_map',
                       'set_device', 'device', 'cuda_available',
                       '_triad_jit_hit', '_triad_jit_is_hot',
                       'gpu_map', 'gpu_fused', 'gpu_available',
                       'gpu_array', 'gpu_to_cpu', 'gpu_info',
                       '__builtins__')]
        if candidates:
            return candidates[-1]
        return None
