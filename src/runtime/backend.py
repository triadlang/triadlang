from __future__ import annotations

import math
import os
import sys
import importlib.util
import subprocess

from triad import ntri as _np

_FORCE = os.environ.get('TRIADLANG_BACKEND', 'auto').lower()
_CUDA_AVAILABLE = False
_HIP_AVAILABLE = False
_TORCH_AVAILABLE = False
_TORCH_CUDA = False
_TORCH_HIP = False
_METAL_AVAILABLE = False
_ANE_AVAILABLE = False
_IS_HIP_CUPY = False
_cp = None
_mx = None
_th = None


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
        self.complex64 = getattr(mx_mod, 'complex64')
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
        if isinstance(arr, _np.ndarray):
            arr = arr.tolist()
        return self._mx.array(arr, dtype=dtype)

    def linspace(self, start, stop, num=50, endpoint=True, dtype=None):
        if endpoint:
            return self._mx.linspace(start, stop, num, dtype=dtype)
        step = (stop - start) / num
        return self._mx.linspace(start, stop - step, num, dtype=dtype)


class _TorchRNG:

    def __init__(self, th_mod, seed=None):
        self._th = th_mod
        self._gen = th_mod.Generator()
        if seed is not None:
            self._gen.manual_seed(int(seed))

    def _shape(self, size):
        if size is None:
            return ()
        if isinstance(size, int):
            return (size,)
        return tuple(size)

    def standard_normal(self, size=None, dtype=None):
        return self._th.randn(self._shape(size), generator=self._gen,
                              dtype=dtype or self._th.float32)

    def randn(self, *shape, dtype=None):
        return self._th.randn(shape, generator=self._gen, dtype=dtype or self._th.float32)

    def integers(self, low, high=None, size=None):
        if high is None:
            low, high = 0, low
        return self._th.randint(int(low), int(high), self._shape(size), generator=self._gen)

    def uniform(self, low=0.0, high=1.0, size=None):
        shape = self._shape(size)
        return self._th.rand(shape, generator=self._gen) * (high - low) + low

    def random(self, size=None):
        return self._th.rand(self._shape(size), generator=self._gen)

    def choice(self, a, size=None, replace=True, p=None):
        th = self._th
        if isinstance(a, int):
            pool = th.arange(a)
        else:
            pool = th.as_tensor(a)
        n = pool.shape[0]
        if size is None:
            count = 1
            shape = ()
        elif isinstance(size, int):
            count = size
            shape = (size,)
        else:
            count = 1
            for d in size:
                count *= d
            shape = tuple(size)
        if p is not None:
            probs = th.as_tensor(p, dtype=th.float64)
            idx = th.multinomial(probs, count, replacement=bool(replace), generator=self._gen)
        else:
            idx = th.randint(0, n, (count,), generator=self._gen)
        out = pool[idx]
        return out.reshape(shape) if shape != () else out.reshape(())

    def seed(self, s=None):
        if s is None:
            self._gen.seed()
        else:
            self._gen.manual_seed(int(s))


class _TorchRandom:

    def __init__(self, th_mod):
        self._th = th_mod
        self._fallback = _TorchRNG(th_mod)

    def default_rng(self, seed=None):
        return _TorchRNG(self._th, seed)

    def standard_normal(self, shape=None, dtype=None):
        return self._fallback.standard_normal(shape, dtype=dtype)

    def randn(self, *shape, dtype=None):
        return self._fallback.randn(*shape, dtype=dtype)

    def integers(self, low, high=None, size=None):
        return self._fallback.integers(low, high, size)

    def uniform(self, low=0.0, high=1.0, size=None):
        return self._fallback.uniform(low, high, size)

    def choice(self, a, size=None, replace=True, p=None):
        return self._fallback.choice(a, size=size, replace=replace, p=p)

    def seed(self, s=None):
        self._fallback.seed(s)


class _TorchAdapter:

    def __init__(self, th_mod):
        self._th = th_mod
        self.random = _TorchRandom(th_mod)
        self.ndarray = th_mod.Tensor
        self.complex128 = th_mod.complex128
        self.float64 = th_mod.float64

    def __getattr__(self, name):
        if name == 'asarray':
            return self.asarray
        if name == 'pi':
            return math.pi
        return getattr(self._th, name)

    def _map_dtype(self, dtype):
        th = self._th
        if isinstance(dtype, th.dtype):
            return dtype
        name = getattr(dtype, 'name', None) or getattr(dtype, '__name__', None)
        if isinstance(name, str) and hasattr(th, name):
            return getattr(th, name)
        return th.float32

    def asarray(self, arr, dtype=None):
        th = self._th
        dt = self._map_dtype(dtype) if dtype is not None else None
        if isinstance(arr, th.Tensor):
            return arr.to(dt) if dt is not None else arr
        if hasattr(arr, 'tolist'):
            data = arr.tolist()
            return th.as_tensor(data, dtype=dt) if dt is not None else th.as_tensor(data)
        if dt is not None:
            return th.as_tensor(arr, dtype=dt)
        return th.as_tensor(arr)


def _metal_usable() -> bool:
    if 'mlx.core' in sys.modules:
        return True
    if _FORCE in {'metal', 'mlx', 'gpu'}:
        return True
    try:
        have_mlx = importlib.util.find_spec('mlx.core') is not None
    except Exception:
        have_mlx = False
    if not have_mlx:
        return False
    code = (
        "import mlx.core as mx; "
        "mx.set_default_device(mx.gpu); "
        "x=mx.sum(mx.array([1.0,2.0], dtype=mx.float32)); "
        "mx.eval(x)"
    )
    try:
        proc = subprocess.run(
            [sys.executable, '-c', code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


if _FORCE not in {'numpy', 'cpu'}:
    try:
        import cupy as _cp_mod

        if _cp_mod.cuda.runtime.getDeviceCount() > 0:
            try:
                _IS_HIP_CUPY = bool(_cp_mod.cuda.runtime.is_hip)
            except Exception:
                _IS_HIP_CUPY = False
            _cp = _cp_mod
            if _IS_HIP_CUPY:
                _HIP_AVAILABLE = True
            else:
                _CUDA_AVAILABLE = True
    except Exception:
        _cp = None
        _CUDA_AVAILABLE = False
        _HIP_AVAILABLE = False
        _IS_HIP_CUPY = False

if _FORCE not in {'numpy', 'cpu'}:
    try:
        import torch as _th_mod

        _TORCH_AVAILABLE = True
        _th = _TorchAdapter(_th_mod)
        try:
            _TORCH_CUDA = bool(_th_mod.cuda.is_available())
        except Exception:
            _TORCH_CUDA = False
        try:
            _TORCH_HIP = bool(getattr(_th_mod.version, 'hip', None))
        except Exception:
            _TORCH_HIP = False
    except Exception:
        _th = None
        _TORCH_AVAILABLE = False
        _TORCH_CUDA = False
        _TORCH_HIP = False

if (_FORCE not in {'numpy', 'cpu', 'cuda', 'ane', 'neural'}
        and sys.platform == 'darwin'
        and _metal_usable()):
    try:
        import mlx.core as _mx_mod

        _mx_mod.set_default_device(_mx_mod.gpu)
        _chk = _mx_mod.sum(_mx_mod.array([1.0, 2.0], dtype=_mx_mod.float32))
        _mx_mod.eval(_chk)
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

def _torch_accel() -> bool:
    return bool(_TORCH_AVAILABLE and _th is not None and (_TORCH_CUDA or _TORCH_HIP))

def get_xp(backend: str='auto'):
    force = _GLOBAL_OVERRIDE if _GLOBAL_OVERRIDE is not None else _FORCE
    if backend != 'auto':
        force = backend.lower()
    if force == 'numpy' or force == 'cpu':
        return _np
    if force == 'cuda':
        if _CUDA_AVAILABLE and _cp is not None:
            return _cp
        if _TORCH_CUDA and _th is not None:
            return _th
        raise RuntimeError("backend='cuda' requested but neither cupy/CUDA nor torch-CUDA is available")
    if force in {'hip', 'rocm', 'amd'}:
        if _HIP_AVAILABLE and _cp is not None:
            return _cp
        if _TORCH_HIP and _th is not None:
            return _th
        raise RuntimeError("backend='hip' requested but neither cupy-ROCm nor torch-ROCm is available")
    if force == 'torch':
        if _TORCH_AVAILABLE and _th is not None:
            return _th
        raise RuntimeError("backend='torch' requested but torch is not installed")
    if force in {'metal', 'mlx'}:
        if not _METAL_AVAILABLE:
            raise RuntimeError("backend='metal' requested but MLX/Metal is not available")
        return _mx
    if force in {'ane', 'neural'}:
        raise RuntimeError("backend='ane' is a Core ML inference target, not an ndarray solver backend")
    if force == 'gpu':
        if _CUDA_AVAILABLE:
            return _cp
        if _HIP_AVAILABLE:
            return _cp
        if _METAL_AVAILABLE:
            return _mx
        if _torch_accel():
            return _th
        raise RuntimeError("backend='gpu' requested but no CUDA, ROCm/HIP or Metal device is available")
    if _CUDA_AVAILABLE:
        return _cp
    if _HIP_AVAILABLE:
        return _cp
    if _METAL_AVAILABLE:
        return _mx
    if _torch_accel():
        return _th
    return _np

def cuda_available() -> bool:
    return _CUDA_AVAILABLE

def hip_available() -> bool:
    return _HIP_AVAILABLE

def torch_available() -> bool:
    return _TORCH_AVAILABLE

def metal_available() -> bool:
    return _METAL_AVAILABLE

def ane_available() -> bool:
    return _ANE_AVAILABLE

def gpu_available() -> bool:
    return _CUDA_AVAILABLE or _HIP_AVAILABLE or _METAL_AVAILABLE

def accelerator_info() -> dict:
    return {
        'backend': BACKEND,
        'cuda': _CUDA_AVAILABLE,
        'hip_rocm': _HIP_AVAILABLE,
        'torch': _TORCH_AVAILABLE,
        'torch_cuda': _TORCH_CUDA,
        'torch_hip': _TORCH_HIP,
        'metal': _METAL_AVAILABLE,
        'ane_coreml': _ANE_AVAILABLE,
        'platform': sys.platform,
    }

def _is_torch_tensor(arr) -> bool:
    return _TORCH_AVAILABLE and isinstance(arr, _th._th.Tensor)

def copy_array(arr):
    # arrays mlx são imutáveis: cada op já devolve um array novo.
    if _METAL_AVAILABLE and _is_mlx_array(arr):
        return arr
    return arr.copy()

def select_mask(arr, mask):
    # seleção 1-D por máscara booleana em qualquer backend (mlx não indexa por máscara).
    if _METAL_AVAILABLE and (_is_mlx_array(arr) or _is_mlx_array(mask)):
        if _is_mlx_array(arr):
            _mx._mx.eval(arr)
            a = arr.tolist()
        else:
            a = _np.asarray(arr).tolist()
        if _is_mlx_array(mask):
            _mx._mx.eval(mask)
            m = mask.tolist()
        else:
            m = _np.asarray(mask).tolist()
        return _np.asarray([v for v, keep in zip(a, m) if keep])
    return arr[mask]

def asnumpy(arr):
    if _cp is not None and isinstance(arr, _cp.ndarray):
        return arr.get()
    if _METAL_AVAILABLE and _is_mlx_array(arr):
        _mx._mx.eval(arr)
        return _np.asarray(arr.tolist())
    if _is_torch_tensor(arr):
        return _np.asarray(arr.detach().cpu().numpy())
    return _np.asarray(arr)

def xp_of(arr):
    if _cp is not None and isinstance(arr, _cp.ndarray):
        return _cp
    if _METAL_AVAILABLE and _is_mlx_array(arr):
        return _mx
    if _is_torch_tensor(arr):
        return _th
    return _np

def to_xp(arr, xp):
    if xp is _np:
        return asnumpy(arr)
    if _cp is not None and xp is _cp:
        return _cp.asarray(arr)
    if _METAL_AVAILABLE and xp is _mx:
        return _mx.asarray(arr)
    if _TORCH_AVAILABLE and xp is _th:
        return _th.asarray(arr)
    return arr
try:
    xp = get_xp('auto')
except RuntimeError:
    if _FORCE in {'ane', 'neural'}:
        xp = _np
    else:
        raise
if xp is _cp and _IS_HIP_CUPY:
    BACKEND = 'hip'
else:
    BACKEND = 'cuda' if xp is _cp else ('metal' if xp is _mx else ('torch' if xp is _th else 'numpy'))

def _is_mlx_array(arr) -> bool:
    return type(arr).__module__.startswith('mlx.')

def is_gpu() -> bool:
    return BACKEND in {'cuda', 'hip', 'metal'} or (BACKEND == 'torch' and (_TORCH_CUDA or _TORCH_HIP))
