from __future__ import annotations

from collections.abc import Callable

from triad import ntri as np

try:
    import cupy as _cp
    _CUDA = True
except (ImportError, ModuleNotFoundError, AttributeError):
    _cp = None
    _CUDA = False

class GPUKernel:

    def __init__(self, name: str, fn: Callable, kernel_type: str = 'elementwise'):
        self.name = name
        self._fn = fn
        self._kernel_type = kernel_type
        self._kernel = None
        self._compiled = False

    def _compile(self):

        if self._compiled:
            return
        if not _CUDA:
            self._compiled = True
            return
        try:
            self._kernel = _cp.ElementwiseKernel(
                'T x, T y', 'T z',
                f'z = {self._body_expr}',
                name=f'triad_{self.name}'
            )
            self._compiled = True
        except (RuntimeError, ValueError, ImportError):
            self._kernel = None
            self._compiled = True

    def __call__(self, *args):
        if _CUDA and self._kernel is not None:

            gpu_args = []
            for a in args:
                if isinstance(a, np.ndarray):
                    gpu_args.append(_cp.asarray(a))
                elif hasattr(a, '__cupy_array__'):
                    gpu_args.append(a)
                else:
                    gpu_args.append(a)
            result = self._fn(*gpu_args)
            if hasattr(result, 'get'):
                return result.get()
            return result
        else:
            return self._fn(*args)

class GPUKernelBuilder:

    def __init__(self):
        self._kernels: dict[str, GPUKernel] = {}

    def register(self, name: str, fn: Callable) -> GPUKernel:

        kernel = GPUKernel(name, fn)
        self._kernels[name] = kernel
        return kernel

    def get(self, name: str) -> GPUKernel | None:
        return self._kernels.get(name)

    def map_array(self, fn: Callable, arr: np.ndarray) -> np.ndarray:

        if _CUDA:
            gpu_arr = _cp.asarray(arr)
            result = fn(gpu_arr)
            if hasattr(result, 'get'):
                return result.get()
            return np.asarray(result)
        else:
            return fn(arr)

    def fused_map(self, fn: Callable, *arrays: np.ndarray) -> np.ndarray:

        if _CUDA:
            gpu_arrays = [_cp.asarray(a) for a in arrays]
            result = fn(*gpu_arrays)
            if hasattr(result, 'get'):
                return result.get()
            return np.asarray(result)
        else:
            return fn(*arrays)

_builder = GPUKernelBuilder()

_FUSED_ADAM = None
_FUSED_ADAMW = None

if _CUDA:

    _FUSED_ADAM = _cp.ElementwiseKernel(
        'T grad, T lr, T b1, T b2, T eps, T bc1, T bc2',
        'T p, T m, T v',
        '''
        m = b1 * m + ((T)1 - b1) * grad;
        v = b2 * v + ((T)1 - b2) * grad * grad;
        T mh = m / bc1;
        T vh = v / bc2;
        p = p - lr * mh / (sqrt(vh) + eps);
        ''',
        name='triad_fused_adam')

    _FUSED_ADAMW = _cp.ElementwiseKernel(
        'T grad, T lr, T b1, T b2, T eps, T bc1, T bc2, T wd',
        'T p, T m, T v',
        '''
        T g = grad + wd * p;
        m = b1 * m + ((T)1 - b1) * g;
        v = b2 * v + ((T)1 - b2) * g * g;
        T mh = m / bc1;
        T vh = v / bc2;
        p = p - lr * mh / (sqrt(vh) + eps);
        ''',
        name='triad_fused_adamw')

def fused_adam_step(p, grad, m, v, lr, b1, b2, eps, t):

    if grad.dtype != p.dtype:
        grad = grad.astype(p.dtype)
    bc1 = 1.0 - b1 ** t
    bc2 = 1.0 - b2 ** t
    _FUSED_ADAM(grad, lr, b1, b2, eps, bc1, bc2, p, m, v)

def fused_adamw_step(p, grad, m, v, lr, b1, b2, eps, t, weight_decay):
    if grad.dtype != p.dtype:
        grad = grad.astype(p.dtype)
    bc1 = 1.0 - b1 ** t
    bc2 = 1.0 - b2 ** t
    _FUSED_ADAMW(grad, lr, b1, b2, eps, bc1, bc2, weight_decay, p, m, v)

def fused_available() -> bool:
    return _CUDA and _FUSED_ADAM is not None

def gpu_kernel(name: str, fn: Callable) -> GPUKernel:

    return _builder.register(name, fn)

def gpu_map(fn: Callable, arr: np.ndarray) -> np.ndarray:

    return _builder.map_array(fn, arr)

def gpu_fused_map(fn: Callable, *arrays: np.ndarray) -> np.ndarray:

    return _builder.fused_map(fn, *arrays)

def gpu_available() -> bool:

    return _CUDA

def gpu_array(data) -> object:

    if _CUDA:
        return _cp.asarray(data)
    return np.asarray(data)

def gpu_to_cpu(arr) -> np.ndarray:

    if _CUDA and hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)

def gpu_info() -> dict:

    if not _CUDA:
        return {'available': False, 'device': None}
    try:
        dev = _cp.cuda.Device(0)
        try:
            is_hip = bool(_cp.cuda.runtime.is_hip)
        except Exception:
            is_hip = False
        mem = getattr(dev, 'mem_info', (0, 0))
        return {
            'available': True,
            'device': dev.name.decode() if isinstance(dev.name, bytes) else dev.name,
            'hip_rocm': is_hip,
            'compute_capability': getattr(dev, 'compute_capability', None),
            'memory_total_mb': mem[1] // (1024 * 1024),
            'memory_free_mb': mem[0] // (1024 * 1024),
        }
    except Exception as e:
        return {'available': True, 'device': 'unknown', 'error': str(e)}

