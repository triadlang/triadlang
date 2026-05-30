from __future__ import annotations
import numpy as _np
try:
    import cupy as _cp
    _cp.cuda.runtime.getDeviceCount()
    _CUDA = True
except Exception:
    _cp = None
    _CUDA = False
_state = {'xp': _np, 'fdtype': _np.float64, 'device': 'cpu'}

class _XPProxy:
    __slots__ = ()

    def __getattr__(self, name):
        return getattr(_state['xp'], name)
xp = _XPProxy()

def _resolve_fdtype(mod, dtype):
    name = None
    if isinstance(dtype, str):
        name = dtype
    else:
        try:
            name = _np.dtype(dtype).name
        except TypeError:
            name = getattr(dtype, '__name__', str(dtype))
    if '32' in name:
        return mod.float32
    if '64' in name:
        return mod.float64
    raise ValueError(f'unsupported ML float dtype: {dtype!r} (use float32 or float64)')

def set_device(device: str='cpu', dtype='float32'):
    global _state
    if device == 'cuda':
        if not _CUDA:
            raise RuntimeError("set_device('cuda') requested but cupy/CUDA is not available")
        mod = _cp
    elif device == 'cpu':
        mod = _np
    else:
        raise ValueError(f"unknown device {device!r} (use 'cpu' or 'cuda')")
    _state = {'xp': mod, 'fdtype': _resolve_fdtype(mod, dtype), 'device': device}
    return dict(_state)

def device() -> str:
    return _state['device']

def fdtype():
    return _state['fdtype']

def is_gpu() -> bool:
    return _state['device'] == 'cuda'

def cuda_available() -> bool:
    return _CUDA

def get_array_module(*arrays):
    if _CUDA:
        for a in arrays:
            if isinstance(a, _cp.ndarray):
                return _cp
    return _np

def asnumpy(arr):
    if _CUDA and isinstance(arr, _cp.ndarray):
        return arr.get()
    return _np.asarray(arr)

def coerce(data):
    mod = _state['xp']
    fd = _state['fdtype']
    if mod is _np:
        if _CUDA and isinstance(data, _cp.ndarray):
            data = data.get()
        return _np.asarray(data, dtype=fd)
    return _cp.asarray(data, dtype=fd)

def sync():
    if _CUDA and _state['device'] == 'cuda':
        _cp.cuda.runtime.deviceSynchronize()

def mem_info():
    if _CUDA and _state['device'] == 'cuda':
        free, total = _cp.cuda.runtime.memGetInfo()
        used = _cp.get_default_memory_pool().used_bytes()
        return (free, total, used)
    return (0, 0, 0)

def reset_pool():
    if _CUDA:
        _cp.get_default_memory_pool().free_all_blocks()
        _cp.get_default_pinned_memory_pool().free_all_blocks()