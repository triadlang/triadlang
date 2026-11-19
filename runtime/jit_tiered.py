"""Tiered execution support for TriadLang.

what is active: hot-spot tracking. for-loop iterations are counted at runtime
(compiler_runtime emits _triad_jit_hit per loop), so the runtime knows which
loops are hot. these counts are available through get_jit().stats() and the
`triad jit-stats` cli command, and are useful for diagnostics and for deciding
where vectorization would pay off.

what is intentionally NOT active: automatic native compilation of hot scalar
loops. the native helpers below (compile_function, compile_loop_body) only build
int->int scalar loop kernels. speeding up a scalar loop that way is the exact
anti-pattern this project rejects: triadlang's speed comes from the vectorized
equation (FFT / NumPy / the native PDE solver), not from compiling fibonacci-style
scalar loops. so the tracker observes, but it does not trigger scalar JIT.

the native tier is meaningful only for vectorized / PDE kernels, which already run
native through the solver (runtime/core/solver.py, native/c). the helpers here
remain as building blocks for that vectorized path, not for scalar loop bodies.

the three pillars stay native where it matters:
  P1: FFT / dispersion in the solver (NumPy / CuPy / FFTW)
  P2: memory fields pass through the native solver boundary
  P3: noise / dissipation run inside the native solver path
"""
from __future__ import annotations
import os
import sys
import time
import ctypes
import tempfile
import subprocess
import threading
from typing import Optional, Callable
from collections import OrderedDict

JIT_THRESHOLD = 128        
JIT_MAX_CACHE = 64         
JIT_COMPILE_TIMEOUT = 30   
JIT_ENABLED = True         

class HotSpotTracker:
    """Tracks call counts for functions and loops to detect hot paths."""

    def __init__(self):
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()
        self._compiled: set[str] = set()

    def hit(self, name: str) -> int:
        """Record a hit for the given function/loop. Returns new count."""
        with self._lock:
            self._counts[name] = self._counts.get(name, 0) + 1
            return self._counts[name]

    def is_hot(self, name: str) -> bool:
        """Check if a function/loop has exceeded the JIT threshold."""
        with self._lock:
            return (self._counts.get(name, 0) >= JIT_THRESHOLD
                    and name not in self._compiled)

    def mark_compiled(self, name: str):
        """Mark a function as already compiled."""
        with self._lock:
            self._compiled.add(name)

    def reset(self):
        """Reset all counters."""
        with self._lock:
            self._counts.clear()
            self._compiled.clear()

    def stats(self) -> dict:
        """Return current profiling stats."""
        with self._lock:
            return {
                'counts': dict(self._counts),
                'compiled': list(self._compiled),
                'threshold': JIT_THRESHOLD,
            }

class LRUCache:
    """Simple LRU cache for JIT-compiled function handles."""

    def __init__(self, maxsize: int = JIT_MAX_CACHE):
        self._cache: OrderedDict[str, ctypes.CDLL] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[ctypes.CDLL]:
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
    """Manages tiered compilation: Python exec -> native C compilation.

    Usage:
        jit = TieredJIT(repo_root='/path/to/triad-lang')
        jit.ensure_runtime()

        # In compiled Python code, the runtime injects:
        # _triad_jit.hit('my_func')
        # if _triad_jit.is_hot('my_func'):
        #     compiled_fn = _triad_jit.compile_function('my_func', source)
    """

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
        """Walk up from cwd to find repo root."""
        d = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if os.path.basename(d) in ('python', 'src'):
            return os.path.dirname(d)
        return d

    def ensure_runtime(self):
        """Ensure libtriad_rt.a is built."""
        if self._rt_built:
            return
        rt_dir = os.path.join(self.repo_root, 'native', 'c')
        lib_path = os.path.join(rt_dir, 'libtriad_rt.a')
        if os.path.exists(lib_path):
            self._rt_built = True
            return

    def hit(self, name: str) -> int:
        """Record a call hit. Returns count."""
        return self.tracker.hit(name)

    def is_hot(self, name: str) -> bool:
        """Check if name is hot enough to compile."""
        return JIT_ENABLED and self.tracker.is_hot(name)

    def compile_function(self, name: str, c_source: str) -> Optional[ctypes.CDLL]:
        """Compile a C function to a shared library and return the handle.

        Args:
            name: function name (used for filename)
            c_source: complete C source code including the function

        Returns:
            ctypes.CDLL handle, or None on failure
        """
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
        libs = ['-L', rt_dir, '-ltriad_rt', '-lm', '-lgc']
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
                         func_name: str, *args) -> Optional[object]:
        """Compile a function and call it immediately.

        Args:
            name: cache key
            c_source: C source
            func_name: name of the function in the C source
            *args: arguments to pass to the function

        Returns:
            Return value of the compiled function, or None
        """
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
                          result_type: str = 'int') -> Optional[Callable]:
        """Compile a loop body to a native function.

        Generates a wrapper function that takes the iteration count
        and returns the accumulated result.

        Args:
            loop_name: identifier for the loop
            body_c: C code for the loop body (receives 'i' as iteration var)
            param_type: C type for the iteration variable
            result_type: C type for the return value

        Returns:
            Python-callable function, or None
        """
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
        """Return JIT statistics."""
        s = self.tracker.stats()
        s['cache_size'] = len(self._cache)
        s['compile_count'] = self._compile_count
        s['enabled'] = JIT_ENABLED
        s['threshold'] = JIT_THRESHOLD
        return s

    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        try:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            pass

def _triad_jit_hit(name: str) -> int:
    """Global JIT hit counter, injected into compiled TriadLang code."""
    if _GLOBAL_JIT is not None:
        return _GLOBAL_JIT.hit(name)
    return 0

def _triad_jit_is_hot(name: str) -> bool:
    """Global JIT hot check, injected into compiled TriadLang code."""
    if _GLOBAL_JIT is not None:
        return _GLOBAL_JIT.is_hot(name)
    return False

_GLOBAL_JIT: Optional[TieredJIT] = None

def get_jit(repo_root: str = '') -> TieredJIT:
    """Get or create the global JIT instance."""
    global _GLOBAL_JIT
    if _GLOBAL_JIT is None:
        _GLOBAL_JIT = TieredJIT(repo_root)
    return _GLOBAL_JIT
