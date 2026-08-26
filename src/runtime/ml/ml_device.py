from __future__ import annotations

from runtime import backend as _backend

_state = {'xp': _backend.xp, 'fdtype': _backend.xp.float64, 'device': _backend.BACKEND}

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
            from triad import ntri as _np
            name = _np.dtype(dtype).name
        except TypeError:
            name = getattr(dtype, '__name__', str(dtype))
    if '32' in name:
        return mod.float32
    if '64' in name:
        return mod.float64
    raise ValueError(f'unsupported ML float dtype: {dtype!r} (use float32 or float64)')

def _sync_state():

    global _state
    _state['xp'] = _backend.get_xp('auto')
    _state['device'] = _backend.BACKEND if _backend.BACKEND in {'cuda', 'metal'} else 'cpu'
    prev = _state.get('fdtype')
    try:
        _state['fdtype'] = _resolve_fdtype(_state['xp'], prev)
    except (ValueError, TypeError, KeyError):
        _state['fdtype'] = _state['xp'].float64

def set_device(device: str='cpu', dtype='float32'):
    dev = device.lower()
    _backend.set_global_backend(dev if dev in {'cuda', 'metal', 'mlx', 'gpu'} else 'cpu')
    _sync_state()
    mod = _state['xp']
    _state['fdtype'] = _resolve_fdtype(mod, dtype)
    return dict(_state)

def device() -> str:
    _sync_state()
    return _state['device']

def fdtype():
    return _state['fdtype']

def is_gpu() -> bool:
    _sync_state()
    return _state['device'] in {'cuda', 'metal'}

def cuda_available() -> bool:
    return _backend.cuda_available()

def metal_available() -> bool:
    return _backend.metal_available()

def ane_available() -> bool:
    return _backend.ane_available()

def gpu_available() -> bool:
    return _backend.gpu_available()

def get_array_module(*arrays):
    _sync_state()
    xp_local = _state['xp']
    for a in arrays:
        if hasattr(a, 'device') and 'cuda' in str(getattr(a, 'device', '')):
            return xp_local
    return xp_local

def asnumpy(arr):
    return _backend.asnumpy(arr)

def coerce(data):
    _sync_state()
    mod = _state['xp']
    fd = _state['fdtype']
    arr = _backend.to_xp(data, mod)
    if hasattr(arr, 'astype'):
        return arr.astype(fd)
    from triad import ntri as _np
    return _np.asarray(arr, dtype=fd)

def sync():
    _sync_state()
    if _state['device'] == 'cuda':
        import cupy as _cp
        _cp.cuda.runtime.deviceSynchronize()
    elif _state['device'] == 'metal':
        _backend.xp._mx.eval()

def mem_info():
    _sync_state()
    if _state['device'] == 'cuda':
        import cupy as _cp
        free, total = _cp.cuda.runtime.memGetInfo()
        used = _cp.get_default_memory_pool().used_bytes()
        return (free, total, used)
    return (0, 0, 0)

def reset_pool():
    if _backend.cuda_available():
        import cupy as _cp
        _cp.get_default_memory_pool().free_all_blocks()
        _cp.get_default_pinned_memory_pool().free_all_blocks()
