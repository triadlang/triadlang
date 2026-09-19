from __future__ import annotations

import ctypes
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_LIB_PATHS = [
    os.path.join(_REPO, 'native', 'c', 'libtriad_rt.so'),
    os.path.join(_REPO, 'native', 'c', 'libtriad_rt.dylib'),
    os.path.join(_REPO, 'native', 'c', 'triad_rt.dll'),
]

def _cuda_dep_paths():
    import sys as _sys
    import glob as _glob
    if _sys.platform == 'win32':
        pf = os.environ.get('ProgramFiles', r'C:\Program Files')
        roots = [os.environ.get('CUDA_PATH')] + sorted(
            _glob.glob(os.path.join(pf, 'NVIDIA GPU Computing Toolkit', 'CUDA', 'v*')))
        for root in roots:
            if not root:
                continue
            yield os.path.join(root, 'bin', 'cublasLt64_*.dll')
            yield os.path.join(root, 'bin', 'cublas64_*.dll')
    else:
        yield '/opt/cuda/lib64/libcublasLt.so.*'
        yield '/opt/cuda/lib64/libcublas.so.*'

_lib = None

class MLNativeUnavailable(RuntimeError):
    pass

def _load():
    global _lib
    if _lib is not None:
        return _lib

    import glob as _glob
    for pat in _cuda_dep_paths():
        for dep in sorted(_glob.glob(pat)):
            try:
                ctypes.CDLL(dep, mode=ctypes.RTLD_GLOBAL)
            except OSError as exc:
                import logging
                logging.getLogger(__name__).debug('ml_native_bridge: could not load CUDA dep %s: %s', dep, exc)
    for p in _LIB_PATHS:
        if os.path.exists(p):
            _lib = ctypes.CDLL(p)
            break
    if _lib is None:
        raise MLNativeUnavailable(
            'libtriad_rt (so/dylib/dll) not found: build it with `make -C native/c` '
            'to use the ml_* native builtins in the interpreter')
    L = _lib
    i32, i64, dbl = ctypes.c_int32, ctypes.c_int64, ctypes.c_double
    P = ctypes.c_void_p
    PI32 = ctypes.POINTER(ctypes.c_int32)
    PDBL = ctypes.POINTER(ctypes.c_double)

    sigs = {
        'triad_tensor_new': (P, [i32, PI32, ctypes.c_int]),
        'triad_tensor_scalar': (P, [dbl, ctypes.c_int]),
        'triad_tensor_zeros': (P, [i32, PI32, ctypes.c_int]),
        'triad_tensor_randn': (P, [i32, PI32, ctypes.c_int]),
        'triad_tensor_backward': (None, [P, PDBL]),
        'triad_tensor_add': (P, [P, P]),
        'triad_tensor_sub': (P, [P, P]),
        'triad_tensor_mul': (P, [P, P]),
        'triad_tensor_matmul': (P, [P, P]),
        'triad_tensor_sum': (P, [P]),
        'triad_tensor_mean': (P, [P]),
        'triad_tensor_relu': (P, [P]),
        'triad_tensor_sigmoid': (P, [P]),
        'triad_tensor_tanh': (P, [P]),
        'triad_tensor_softmax': (P, [P]),
        'triad_tensor_mse_loss': (P, [P, P]),
        'triad_tensor_cross_entropy': (P, [P, P]),
        'triad_triad_new': (P, [i32, i32, ctypes.c_int]),
        'triad_triad_forward': (P, [P, P]),
        'triad_sequential_new': (P, [i32]),
        'triad_sequential_set': (None, [P, i32, i32, P]),
        'triad_sequential_forward': (P, [P, P]),
        'triad_sequential_params': (i32, [P, ctypes.POINTER(P), i32]),
        'triad_adam_new': (P, [ctypes.POINTER(P), i32, dbl, dbl, dbl, dbl]),
        'triad_adam_step': (None, [P]),
        'triad_adam_zero_grad': (None, [P]),
        'triad_sgd_new': (P, [ctypes.POINTER(P), i32, dbl, dbl]),
        'triad_sgd_step': (None, [P]),
        'triad_sgd_zero_grad': (None, [P]),
        'triad_embedding_new': (P, [i32, i32]),
        'triad_embedding_forward': (P, [P, P]),
        'triad_layer_norm_new': (P, [i32, dbl]),
        'triad_layer_norm_forward': (P, [P, P]),
        'triad_mha_new': (P, [i32, i32]),
        'triad_mha_forward': (P, [P, P]),
        'triad_transformer_new': (P, [i32, i32, i32, i32, i32]),
        'triad_transformer_forward': (P, [P, P]),
        'triad_transformer_params': (i32, [P, ctypes.POINTER(P), i32]),
        'triad_wave_triad_new': (P, [i32, i32]),
        'triad_wave_triad_forward': (P, [P, P]),
        'triad_wave_triad_params': (i32, [P, ctypes.POINTER(P), i32]),
        'triad_runtime_init': (None, []),
    }
    for name, (res, args) in sigs.items():
        fn = getattr(L, name, None)
        if fn is None:
            continue
        fn.restype = res
        fn.argtypes = args
    if hasattr(L, 'triad_runtime_init'):
        L.triad_runtime_init()
    return L

class _CTensor(ctypes.Structure):
    _fields_ = [
        ('refcount', ctypes.c_int32),
        ('ndim', ctypes.c_int32),
        ('shape', ctypes.POINTER(ctypes.c_int32)),
        ('size', ctypes.c_int64),
        ('data', ctypes.POINTER(ctypes.c_double)),
        ('grad', ctypes.POINTER(ctypes.c_double)),
        ('requires_grad', ctypes.c_int),
    ]

class _Ctriad(ctypes.Structure):
    _fields_ = [
        ('weight', ctypes.c_void_p),
        ('bias', ctypes.c_void_p),
        ('in_features', ctypes.c_int32),
        ('out_features', ctypes.c_int32),
    ]

class MLHandle:

    __slots__ = ('ptr', 'kind')

    def __init__(self, ptr, kind):
        if ptr is None:
            raise RuntimeError(
                f'native ml call failed (kind={kind!r}): the C function '
                'returned NULL (bad input shapes or OOM); refusing to '
                'continue with a null handle')
        self.ptr = ptr
        self.kind = kind

    def __repr__(self):
        return f'<ml:{self.kind} 0x{self.ptr or 0:x}>'

def _ptr(x):
    if isinstance(x, MLHandle):
        return x.ptr
    if x is None:
        return None
    raise TypeError(f'expected ml handle, got {type(x).__name__}')

def _shape_args(data):
    flat, shape = [], []
    if isinstance(data, (list, tuple)) and data and isinstance(data[0], (list, tuple)):
        rows, cols = len(data), len(data[0])
        shape = [rows, cols]
        for row in data:
            flat.extend(float(v) for v in row)
    elif isinstance(data, (list, tuple)):
        shape = [len(data)]
        flat = [float(v) for v in data]
    else:
        return None, float(data)
    return shape, flat

def ml_tensor(data, requires_grad=False):
    L = _load()
    shape, flat = _shape_args(data)
    rg = 1 if requires_grad else 0
    if shape is None:
        return MLHandle(L.triad_tensor_scalar(flat, rg), 'tensor')
    cshape = (ctypes.c_int32 * len(shape))(*shape)
    t = L.triad_tensor_new(len(shape), cshape, rg)
    ct = ctypes.cast(t, ctypes.POINTER(_CTensor)).contents
    for i, v in enumerate(flat):
        ct.data[i] = v
    return MLHandle(t, 'tensor')

def _dims(rows, cols=None):
    if cols is None:
        return (ctypes.c_int32 * 1)(int(rows)), 1
    return (ctypes.c_int32 * 2)(int(rows), int(cols)), 2

def ml_zeros(rows, cols=None, requires_grad=False):
    L = _load()
    cshape, nd = _dims(rows, cols)
    return MLHandle(L.triad_tensor_zeros(nd, cshape, 1 if requires_grad else 0), 'tensor')

def ml_randn(rows, cols=None, requires_grad=False):
    L = _load()
    cshape, nd = _dims(rows, cols)
    return MLHandle(L.triad_tensor_randn(nd, cshape, 1 if requires_grad else 0), 'tensor')

def ml_triad(in_f, out_f):
    return MLHandle(_load().triad_triad_new(int(in_f), int(out_f), 1), 'triad')

def ml_forward(layer, x):
    return MLHandle(_load().triad_triad_forward(_ptr(layer), _ptr(x)), 'tensor')

def _unary(cname):
    def fn(x):
        return MLHandle(getattr(_load(), cname)(_ptr(x)), 'tensor')
    return fn

ml_relu = _unary('triad_tensor_relu')
ml_sigmoid = _unary('triad_tensor_sigmoid')
ml_tanh_act = _unary('triad_tensor_tanh')
ml_softmax = _unary('triad_tensor_softmax')
ml_tensor_sum = _unary('triad_tensor_sum')
ml_tensor_mean = _unary('triad_tensor_mean')

def _binary(cname):
    def fn(a, b):
        return MLHandle(getattr(_load(), cname)(_ptr(a), _ptr(b)), 'tensor')
    return fn

ml_tensor_add = _binary('triad_tensor_add')
ml_tensor_sub = _binary('triad_tensor_sub')
ml_tensor_mul = _binary('triad_tensor_mul')
ml_tensor_matmul = _binary('triad_tensor_matmul')
ml_mse_loss = _binary('triad_tensor_mse_loss')
ml_cross_entropy = _binary('triad_tensor_cross_entropy')

def ml_backward(t):
    _load().triad_tensor_backward(_ptr(t), None)

def _tensor_struct(t):
    return ctypes.cast(_ptr(t), ctypes.POINTER(_CTensor)).contents

def ml_item(t):
    return float(_tensor_struct(t).data[0])

def ml_data(t, idx):
    return float(_tensor_struct(t).data[int(idx)])

def ml_print(t):
    ct = _tensor_struct(t)
    if ct.ndim == 0 or ct.size == 1:
        print(f'{ct.data[0]:.6f}')
        return
    vals = [f'{ct.data[i]:.4f}' for i in range(min(int(ct.size), 20))]
    tail = ', ...' if ct.size > 20 else ''
    print('[' + ', '.join(vals) + tail + ']')

def ml_seq_new(n):
    return MLHandle(_load().triad_sequential_new(int(n)), 'sequential')

def ml_seq_set(seq, idx, layer_type, layer=None):
    lt = int(layer_type)
    lp = _ptr(layer) if (layer is not None and lt == 0) else None
    _load().triad_sequential_set(_ptr(seq), int(idx), lt, lp)

def ml_seq_forward(seq, x):
    return MLHandle(_load().triad_sequential_forward(_ptr(seq), _ptr(x)), 'tensor')

def _collect_params(collector, handle, cap):
    arr = (ctypes.c_void_p * cap)()
    n = collector(_ptr(handle), arr, cap)
    return arr, n

def ml_adam(seq, lr):
    L = _load()
    arr, n = _collect_params(L.triad_sequential_params, seq, 256)
    return MLHandle(L.triad_adam_new(arr, n, float(lr), 0.9, 0.999, 1e-8), 'adam')

def ml_sgd(seq, lr):
    L = _load()
    arr, n = _collect_params(L.triad_sequential_params, seq, 256)
    return MLHandle(L.triad_sgd_new(arr, n, float(lr), 0.0), 'sgd')

def ml_adam_step(opt):
    _load().triad_adam_step(_ptr(opt))

def ml_adam_zero(opt):
    _load().triad_adam_zero_grad(_ptr(opt))

def ml_sgd_step(opt):
    _load().triad_sgd_step(_ptr(opt))

def ml_sgd_zero(opt):
    _load().triad_sgd_zero_grad(_ptr(opt))

def ml_embedding(vocab, dim):
    return MLHandle(_load().triad_embedding_new(int(vocab), int(dim)), 'embedding')

def ml_embedding_forward(e, idx):
    return MLHandle(_load().triad_embedding_forward(_ptr(e), _ptr(idx)), 'tensor')

def ml_layernorm(dim):
    return MLHandle(_load().triad_layer_norm_new(int(dim), 1e-5), 'layernorm')

def ml_layernorm_forward(ln, x):
    return MLHandle(_load().triad_layer_norm_forward(_ptr(ln), _ptr(x)), 'tensor')

def ml_mha(d_model, n_heads):
    return MLHandle(_load().triad_mha_new(int(d_model), int(n_heads)), 'mha')

def ml_mha_forward(m, x):
    return MLHandle(_load().triad_mha_forward(_ptr(m), _ptr(x)), 'tensor')

def ml_transformer(vocab, d_model, n_blocks, n_heads, d_ff):
    return MLHandle(_load().triad_transformer_new(
        int(vocab), int(d_model), int(n_blocks), int(n_heads), int(d_ff)), 'transformer')

def ml_transformer_forward(t, idx):
    return MLHandle(_load().triad_transformer_forward(_ptr(t), _ptr(idx)), 'tensor')

def ml_transformer_adam(t, lr):
    L = _load()
    arr, n = _collect_params(L.triad_transformer_params, t, 1024)
    return MLHandle(L.triad_adam_new(arr, n, float(lr), 0.9, 0.999, 1e-8), 'adam')

def ml_wave(in_f, out_f):
    return MLHandle(_load().triad_wave_triad_new(int(in_f), int(out_f)), 'wave')

def ml_wave_forward(w, x):
    return MLHandle(_load().triad_wave_triad_forward(_ptr(w), _ptr(x)), 'tensor')

def ml_wave_adam(w, lin, lr):
    L = _load()
    cap = 64
    arr = (ctypes.c_void_p * cap)()
    n = L.triad_wave_triad_params(_ptr(w), arr, cap)
    if lin is not None:
        cl = ctypes.cast(_ptr(lin), ctypes.POINTER(_Ctriad)).contents
        arr[n] = cl.weight
        n += 1
        if cl.bias:
            arr[n] = cl.bias
            n += 1
    return MLHandle(L.triad_adam_new(arr, n, float(lr), 0.9, 0.999, 1e-8), 'adam')

BUILTINS = {name: obj for name, obj in globals().items()
            if name.startswith('ml_') and callable(obj)}

