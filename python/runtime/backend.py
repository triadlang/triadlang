from __future__ import annotations
import os, sys
import numpy as _np
_FORCE = os.environ.get('TRIADLANG_BACKEND', 'auto').lower()
_CUDA_AVAILABLE = False
_cp = None
try:
    import cupy as _cp_mod
    _cp_mod.cuda.runtime.getDeviceCount()
    _cp = _cp_mod
    _CUDA_AVAILABLE = True
except Exception:
    _cp = None
    _CUDA_AVAILABLE = False

def get_xp(backend: str='auto'):
    if backend == 'numpy':
        return _np
    if backend == 'cuda':
        if not _CUDA_AVAILABLE:
            raise RuntimeError("backend='cuda' requested but cupy/CUDA is not available")
        return _cp
    if _FORCE == 'numpy':
        return _np
    if _FORCE == 'cuda' and _CUDA_AVAILABLE:
        return _cp
    return _np

def cuda_available() -> bool:
    return _CUDA_AVAILABLE

def asnumpy(arr):
    if _CUDA_AVAILABLE and isinstance(arr, _cp.ndarray):
        return arr.get()
    return _np.asarray(arr)

def xp_of(arr):
    if _CUDA_AVAILABLE and isinstance(arr, _cp.ndarray):
        return _cp
    return _np

def to_xp(arr, xp):
    if xp is _np:
        return asnumpy(arr)
    if _CUDA_AVAILABLE and xp is _cp:
        return _cp.asarray(arr)
    return arr
xp = get_xp('auto')
BACKEND = 'cuda' if xp is not _np else 'numpy'

def is_gpu() -> bool:
    return BACKEND == 'cuda'