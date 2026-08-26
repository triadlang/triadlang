from __future__ import annotations

import os
import sys
import importlib.util
import subprocess

from triad import ntri as _np

_FORCE = os.environ.get('TRIADLANG_BACKEND', 'auto').lower()
_CUDA_AVAILABLE = False
_METAL_AVAILABLE = False
_ANE_AVAILABLE = False
_cp = None
_mx = None


class _MLXRNG:
    def __init__(self, mx_mod, seed=None):
        self._mx = mx_mod
        if seed is not None:
            mx_mod.random.seed(int(seed))

    def standard_normal(self, shape):
        if isinstance(shape, int):
            shape = (shape,)
        return self._mx.random.normal(shape)


class _MLXRandom:
    def __init__(self, mx_mod):
        self._mx = mx_mod

    def default_rng(self, seed=None):
        return _MLXRNG(self._mx, seed)

    def standard_normal(self, shape):
        if isinstance(shape, int):
            shape = (shape,)
        return self._mx.random.normal(shape)


class _MLXAdapter:
    def __init__(self, mx_mod):
        self._mx = mx_mod
        self.random = _MLXRandom(mx_mod)
        self.ndarray = getattr(mx_mod, 'array')
        self.complex128 = getattr(mx_mod, 'complex64')

    def __getattr__(self, name):
        if name == 'asarray':
            return self.asarray
        if name == 'linspace':
            return self.linspace
        if name == 'float64':
            return self._mx.float32
        if name == 'pi':
            return _np.pi
        return getattr(self._mx, name)

    def asarray(self, arr, dtype=None):
        if dtype is getattr(self, 'complex128'):
            dtype = self._mx.complex64
        if dtype is self._mx.float64:
            dtype = self._mx.float32
        return self._mx.array(arr, dtype=dtype)

    def linspace(self, start, stop, num=50, endpoint=True, dtype=None):
        if endpoint:
            return self._mx.linspace(start, stop, num, dtype=dtype)
        step = (stop - start) / num
        return self._mx.linspace(start, stop - step, num, dtype=dtype)


def _metal_probe_safe() -> bool:
    if 'mlx.core' in sys.modules:
        return True
    if _FORCE in {'metal', 'mlx', 'gpu'}:
        return True
    code = (
        "import mlx.core as mx; "
        "mx.set_default_device(mx.gpu); "
        "x=mx.sum(mx.array([1.0,2.0], dtype=mx.float32)); "
        "mx.eval(x)"
    )
    try:
        proc = subprocess.run(
            [sys.executable, '-c', code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


# Importing CuPy is not enough to prove that a CUDA device is usable.  In
# particular, CuPy may be installed on CPU-only machines and its runtime probe
# then raises CUDARuntimeError.  Never probe CUDA when the user explicitly
# selected the CPU backend, and treat any ordinary probe failure as
# unavailability so importing TriadLang remains safe.
if _FORCE not in {'numpy', 'cpu'}:
    try:
        import cupy as _cp_mod

        _CUDA_AVAILABLE = _cp_mod.cuda.runtime.getDeviceCount() > 0
        if _CUDA_AVAILABLE:
            _cp = _cp_mod
    except Exception:
        _cp = None
        _CUDA_AVAILABLE = False

if (_FORCE not in {'numpy', 'cpu', 'cuda', 'ane', 'neural'}
        and sys.platform == 'darwin'
        and _metal_probe_safe()):
    try:
        import mlx.core as _mx_mod

        _mx_mod.set_default_device(_mx_mod.gpu)
        _probe = _mx_mod.sum(_mx_mod.array([1.0, 2.0], dtype=_mx_mod.float32))
        _mx_mod.eval(_probe)
        _mx = _MLXAdapter(_mx_mod)
        _METAL_AVAILABLE = True
    except Exception:
        _mx = None
        _METAL_AVAILABLE = False

if _FORCE not in {'numpy', 'cpu'} and sys.platform == 'darwin':
    _ANE_AVAILABLE = (
        'coremltools' in sys.modules
        or importlib.util.find_spec('coremltools') is not None
    )

_GLOBAL_OVERRIDE: str | None = None

def set_global_backend(name: str):

    global _GLOBAL_OVERRIDE
    _GLOBAL_OVERRIDE = name.lower()

def get_xp(backend: str='auto'):
    force = _GLOBAL_OVERRIDE if _GLOBAL_OVERRIDE is not None else _FORCE
    if backend != 'auto':
        force = backend.lower()
    if force == 'numpy' or force == 'cpu':
        return _np
    if force == 'cuda':
        if not _CUDA_AVAILABLE:
            raise RuntimeError("backend='cuda' requested but cupy/CUDA is not available")
        return _cp
    if force in {'metal', 'mlx'}:
        if not _METAL_AVAILABLE:
            raise RuntimeError("backend='metal' requested but MLX/Metal is not available")
        return _mx
    if force in {'ane', 'neural'}:
        raise RuntimeError("backend='ane' is a Core ML inference target, not an ndarray solver backend")
    if force == 'gpu':
        if _CUDA_AVAILABLE:
            return _cp
        if _METAL_AVAILABLE:
            return _mx
        raise RuntimeError("backend='gpu' requested but CUDA or Metal is not available")
    if _CUDA_AVAILABLE:
        return _cp
    if _METAL_AVAILABLE:
        return _mx
    return _np

def cuda_available() -> bool:
    return _CUDA_AVAILABLE

def metal_available() -> bool:
    return _METAL_AVAILABLE

def ane_available() -> bool:
    return _ANE_AVAILABLE

def gpu_available() -> bool:
    return _CUDA_AVAILABLE or _METAL_AVAILABLE

def accelerator_info() -> dict:
    return {
        'backend': BACKEND,
        'cuda': _CUDA_AVAILABLE,
        'metal': _METAL_AVAILABLE,
        'ane_coreml': _ANE_AVAILABLE,
        'platform': sys.platform,
    }

def asnumpy(arr):
    if _CUDA_AVAILABLE and isinstance(arr, _cp.ndarray):
        return arr.get()
    if _METAL_AVAILABLE and _is_mlx_array(arr):
        _mx._mx.eval(arr)
        return _np.asarray(arr)
    return _np.asarray(arr)

def xp_of(arr):
    if _CUDA_AVAILABLE and isinstance(arr, _cp.ndarray):
        return _cp
    if _METAL_AVAILABLE and _is_mlx_array(arr):
        return _mx
    return _np

def to_xp(arr, xp):
    if xp is _np:
        return asnumpy(arr)
    if _CUDA_AVAILABLE and xp is _cp:
        return _cp.asarray(arr)
    if _METAL_AVAILABLE and xp is _mx:
        return _mx.asarray(arr)
    return arr
try:
    xp = get_xp('auto')
except RuntimeError:
    if _FORCE in {'ane', 'neural'}:
        xp = _np
    else:
        raise
BACKEND = 'cuda' if xp is _cp else ('metal' if xp is _mx else 'numpy')

def _is_mlx_array(arr) -> bool:
    return type(arr).__module__.startswith('mlx.')

def is_gpu() -> bool:
    return BACKEND in {'cuda', 'metal'}
