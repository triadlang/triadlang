from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Callable

JIT_THRESHOLD = 1
JIT_MAX_CACHE = 64
JIT_COMPILE_TIMEOUT = 30
JIT_ENABLED = True

class HotSpotTracker:

    def __init__(self):
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()
        self._compiled: set[str] = set()

    def hit(self, name: str) -> int:

        with self._lock:
            self._counts[name] = self._counts.get(name, 0) + 1
            return self._counts[name]

    def is_hot(self, name: str) -> bool:

        with self._lock:
            return (self._counts.get(name, 0) >= JIT_THRESHOLD
                    and name not in self._compiled)

    def mark_compiled(self, name: str):

        with self._lock:
            self._compiled.add(name)

    def reset(self):

        with self._lock:
            self._counts.clear()
            self._compiled.clear()

    def stats(self) -> dict:

        with self._lock:
            return {
                'counts': dict(self._counts),
                'compiled': list(self._compiled),
                'threshold': JIT_THRESHOLD,
            }

class LRUCache:

    def __init__(self, maxsize: int = JIT_MAX_CACHE):
        self._cache: OrderedDict[str, ctypes.CDLL] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key: str) -> ctypes.CDLL | None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def put(self, key: str, handle: ctypes.CDLL):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                self._cache[key] = handle
                if len(self._cache) > self._maxsize:
                    self._cache.popitem(last=False)

    def __len__(self):
        return len(self._cache)

class TieredJIT:

    def __init__(self, repo_root: str = ''):
        self.repo_root = repo_root or self._find_repo_root()
        self.tracker = HotSpotTracker()
        self._cache = LRUCache()
        self._rt_built = False
        self._compile_queue: list[tuple[str, str]] = []
        self._compile_lock = threading.Lock()
        self._temp_dir = tempfile.mkdtemp(prefix='triad_jit_')
        self._compile_count = 0

    @staticmethod
    def _find_repo_root() -> str:

        d = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if os.path.basename(d) in ('python', 'src'):
            return os.path.dirname(d)
        return d

    def ensure_runtime(self):

        if self._rt_built:
            return
        rt_dir = os.path.join(self.repo_root, 'native', 'c')
        lib_path = os.path.join(rt_dir, 'libtriad_rt.a')
        if os.path.exists(lib_path):
            self._rt_built = True
            return

    def hit(self, name: str) -> int:

        return self.tracker.hit(name)

    def is_hot(self, name: str) -> bool:

        return JIT_ENABLED and self.tracker.is_hot(name)

    def compile_function(self, name: str, c_source: str) -> ctypes.CDLL | None:

        cached = self._cache.get(name)
        if cached is not None:
            return cached

        self.ensure_runtime()
        rt_dir = os.path.join(self.repo_root, 'native', 'c')

        c_path = os.path.join(self._temp_dir, f'{name}_{self._compile_count}.c')
        so_path = os.path.join(self._temp_dir, f'{name}_{self._compile_count}.so')
        self._compile_count += 1

        with open(c_path, 'w') as f:
            f.write(c_source)

        cc = os.environ.get('CC', 'gcc')
        cflags = ['-std=c11', '-O2', '-fPIC', '-shared',
                  '-I', os.path.join(rt_dir, 'include')]
        libs = ['-L', rt_dir, '-ltriad_rt', '-lm']

        if any(os.path.isfile(h) for h in ('/usr/include/gc/gc.h',
                                           '/usr/local/include/gc/gc.h',
                                           '/opt/homebrew/include/gc/gc.h')):
            libs.append('-lgc')
        if os.path.exists('/usr/include/fftw3.h'):
            cflags.append('-DUSE_FFTW')
            libs.append('-lfftw3')

        cmd = [cc] + cflags + [c_path] + libs + ['-o', so_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=JIT_COMPILE_TIMEOUT)
            if result.returncode != 0:
                return None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

        try:
            handle = ctypes.CDLL(so_path)
            self._cache.put(name, handle)
            self.tracker.mark_compiled(name)
            return handle
        except OSError:
            return None

    def compile_and_call(self, name: str, c_source: str,
                         func_name: str, *args) -> object | None:

        handle = self.compile_function(name, c_source)
        if handle is None:
            return None
        try:
            fn = getattr(handle, func_name)
            return fn(*args)
        except (AttributeError, OSError):
            return None

    def compile_loop_body(self, loop_name: str, body_c: str,
                          param_type: str = 'int',
                          result_type: str = 'int') -> Callable | None:

        c_source = f"""#include "triad_rt.h"
{result_type} _triad_jit_{loop_name}({param_type} n) {{
    {result_type} acc = 0;
    for ({param_type} i = 0; i < n; i++) {{
        {body_c};
    }}
    return acc;
}}
"""
        handle = self.compile_function(loop_name, c_source)
        if handle is None:
            return None
        try:
            fn = getattr(handle, f'_triad_jit_{loop_name}')
            fn.argtypes = [ctypes.c_int]
            fn.restype = ctypes.c_int
            return fn
        except (AttributeError, OSError):
            return None

    def stats(self) -> dict:

        s = self.tracker.stats()
        s['cache_size'] = len(self._cache)
        s['compile_count'] = self._compile_count
        s['enabled'] = JIT_ENABLED
        s['threshold'] = JIT_THRESHOLD
        return s

    def cleanup(self):

        import shutil
        try:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except OSError:
            pass

def _triad_jit_hit(name: str) -> int:
    global _GLOBAL_JIT
    if _GLOBAL_JIT is None:
        _GLOBAL_JIT = TieredJIT()
    return _GLOBAL_JIT.hit(name)

def _triad_jit_is_hot(name: str) -> bool:

    if _GLOBAL_JIT is not None:
        return _GLOBAL_JIT.is_hot(name)
    return False

_interp_time: dict = {}
_compile_cost: list = []

def _triad_jit_time(name: str, dt: float):
    _interp_time[name] = _interp_time.get(name, 0.0) + float(dt)

def _measured_compile_cost() -> float:
    if _compile_cost:
        return _compile_cost[0]
    import time as _t
    src = ('#include <stdint.h>\n'
           'int64_t kernel(int64_t *I, double *D){ (void)I; (void)D; return 0; }\n')
    t0 = _t.perf_counter()
    lib = get_jit().compile_function('_triad_probe', src)
    cost = _t.perf_counter() - t0 if lib is not None else float('inf')
    _compile_cost.append(cost)
    return cost

def _triad_jit_ready(name: str) -> bool:
    if not JIT_ENABLED:
        return False
    return _triad_jit_is_hot(name)

_GLOBAL_JIT: TieredJIT | None = None

def get_jit(repo_root: str = '') -> TieredJIT:

    global _GLOBAL_JIT
    if _GLOBAL_JIT is None:
        _GLOBAL_JIT = TieredJIT(repo_root)
    return _GLOBAL_JIT

