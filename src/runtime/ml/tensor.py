from __future__ import annotations

from collections.abc import Callable

from triad import ntri as np

if not hasattr(np, 'triad'):
    np.triad = np.full
if not hasattr(np, 'triad_like'):
    np.triad_like = np.full_like

from runtime.ml.ml_device import asnumpy

Shape = tuple[int, ...]
_GradFn = Callable[[object], None]
_ENABLE_GRAD = True

try:
    import cupy as _cp
except (ImportError, ModuleNotFoundError):
    _cp = None

_AUTO_EQ = None

def _auto_place_pair(a: TriadTensor, b: TriadTensor):

    eq = _AUTO_EQ
    if eq is None or a._data.ndim < 2 or b._data.ndim < 2:
        return a, b
    if a.device == 'cuda' or b.device == 'cuda':
        return a, b
    M = 1
    for s in a._data.shape[:-1]:
        M *= int(s)
    K = int(a._data.shape[-1])
    N = int(b._data.shape[-1])
    try:
        if eq.matmul_device(M, K, N, n_calls=1, data_at='cpu') == 'gpu':
            return a.to('cuda'), b.to('cuda')
    except (ImportError, RuntimeError, ValueError):
        pass
    return a, b

def _is_cuda_array(a) -> bool:
    return _cp is not None and isinstance(a, _cp.ndarray)

def _xp_of(a):
    return _cp if _is_cuda_array(a) else np

def _device_of(a) -> str:
    return 'cuda' if _is_cuda_array(a) else 'cpu'

def _coerce(data, device: str | None = None, dtype=None):

    if isinstance(data, TriadTensor):
        data = data._data
    from runtime.ml.ml_device import fdtype
    fd = dtype if dtype is not None else fdtype()
    if _is_cuda_array(data):
        arr = data if device in (None, 'cuda') else _cp.asnumpy(data)
    else:
        arr = np.asarray(data)
        if device == 'cuda':
            if _cp is None:
                raise RuntimeError("device='cuda' mas cupy nao esta disponivel")
            arr = _cp.asarray(arr)
    if hasattr(arr, 'astype'):
        if dtype is not None:
            if arr.dtype != fd:
                arr = arr.astype(fd)
        elif arr.dtype.kind in ('f', 'c'):
            pass
        else:
            try:
                arr = arr.astype(fd)
            except (TypeError, ValueError):
                pass
    return arr

class TriadTensor:
    __slots__ = ('_data', '_grad', '_requires_grad', '_grad_fn', '_children', '_device', '_name')

    def __init__(self, data, requires_grad: bool=False, device: str | None=None, name: str='', dtype=None):
        self._data = _coerce(data, device, dtype)
        self._grad: np.ndarray | None = None
        self._requires_grad = requires_grad
        self._grad_fn: _GradFn | None = None
        self._children: list[TriadTensor] = []
        self._device = _device_of(self._data)
        self._name = name

    @property
    def device(self) -> str:
        return _device_of(self._data)

    def to(self, device: str) -> TriadTensor:

        if device == self.device:
            return self
        out = TriadTensor(_coerce(self._data, device))
        if _ENABLE_GRAD and self._requires_grad:
            out._requires_grad = True
            out._children = [self]
            src_dev = self.device

            def _back(g):
                sg = _coerce(g, src_dev)
                self._grad = sg if self._grad is None else self._grad + sg
            out._grad_fn = _back
        return out

    def astype(self, dtype) -> TriadTensor:
        out = TriadTensor(self._data.astype(dtype))
        if _ENABLE_GRAD and self._requires_grad:
            out._requires_grad = True
            out._children = [self]
            orig_dt = self._data.dtype

            def _back(g):
                sg = g.astype(orig_dt)
                self._grad = sg if self._grad is None else self._grad + sg
            out._grad_fn = _back
        return out

    def half(self) -> TriadTensor:
        return self.astype('float16')

    def float(self) -> TriadTensor:
        return self.astype('float32')

    def double(self) -> TriadTensor:
        return self.astype('float64')

    def cuda(self) -> TriadTensor:
        return self.to('cuda')

    def cpu(self) -> TriadTensor:
        return self.to('cpu')

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, val):
        self._data = _coerce(val, self.device)

    @property
    def grad(self) -> np.ndarray | None:
        return self._grad

    @property
    def shape(self) -> Shape:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def T(self) -> TriadTensor:
        out = TriadTensor(self._data.T)
        if _ENABLE_GRAD and self._requires_grad:
            out._requires_grad = True
            out._children = [self]

            def _back(g):
                sg = g.T
                self._grad = sg if self._grad is None else self._grad + sg
            out._grad_fn = _back
        return out

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    def detach(self) -> TriadTensor:
        return TriadTensor(self._data.copy(), requires_grad=False)

    def numpy(self):
        return asnumpy(self._data)

    def item(self) -> float:
        return float(self._data)

    def reshape(self, *shape) -> TriadTensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = TriadTensor(self._data.reshape(shape))
        if _ENABLE_GRAD and self._requires_grad:
            out._requires_grad = True
            out._children = [self]
            orig = self._data.shape

            def _back(g):
                sg = g.reshape(orig)
                self._grad = sg if self._grad is None else self._grad + sg
            out._grad_fn = _back
        return out

    def view(self, *shape) -> TriadTensor:
        return self.reshape(*shape)

    def permute(self, *axes) -> TriadTensor:
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        out = TriadTensor(np.transpose(self._data, axes))
        if _ENABLE_GRAD and self._requires_grad:
            out._requires_grad = True
            out._children = [self]
            inv = [0] * len(axes)
            for i, a in enumerate(axes):
                inv[a] = i
            inv = tuple(inv)

            def _back(g):
                sg = np.transpose(g, inv)
                self._grad = sg if self._grad is None else self._grad + sg
            out._grad_fn = _back
        return out

    def squeeze(self, axis=None) -> TriadTensor:
        out = TriadTensor(np.squeeze(self._data, axis=axis)
                          if axis is not None else np.squeeze(self._data))
        if _ENABLE_GRAD and self._requires_grad:
            out._requires_grad = True
            out._children = [self]
            orig = self._data.shape

            def _back(g):
                sg = g.reshape(orig)
                self._grad = sg if self._grad is None else self._grad + sg
            out._grad_fn = _back
        return out

    def unsqueeze(self, axis: int) -> TriadTensor:
        out = TriadTensor(np.expand_dims(self._data, axis=axis))
        if _ENABLE_GRAD and self._requires_grad:
            out._requires_grad = True
            out._children = [self]
            orig = self._data.shape

            def _back(g):
                sg = g.reshape(orig)
                self._grad = sg if self._grad is None else self._grad + sg
            out._grad_fn = _back
        return out

    def expand(self, *shape) -> TriadTensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = TriadTensor(np.broadcast_to(self._data, shape).copy())
        if _ENABLE_GRAD and self._requires_grad:
            out._requires_grad = True
            out._children = [self]

            def _back(g):
                sg = _unbroadcast(g, self._data.shape)
                self._grad = sg if self._grad is None else self._grad + sg
            out._grad_fn = _back
        return out

    def flatten(self) -> TriadTensor:
        out = TriadTensor(self._data.flatten())
        if _ENABLE_GRAD and self._requires_grad:
            out._requires_grad = True
            out._children = [self]
            orig = self._data.shape

            def _back(g):
                sg = g.reshape(orig)
                self._grad = sg if self._grad is None else self._grad + sg
            out._grad_fn = _back
        return out

    def backward(self, grad: np.ndarray | None=None, free_graph: bool=True):
        if grad is None:
            grad = np.ones_like(self._data)
        self._grad = grad if self._grad is None else self._grad + grad
        topo = []
        visited = set()
        stack = [(self, False)]
        while stack:
            t, processed = stack.pop()
            if processed:
                topo.append(t)
                continue
            if id(t) in visited:
                continue
            visited.add(id(t))
            stack.append((t, True))
            for c in t._children:
                if id(c) not in visited:
                    stack.append((c, False))
        for t in reversed(topo):
            if t._grad_fn is not None and t._grad is not None:
                t._grad_fn(t._grad)
        if free_graph:

            for t in topo:
                if t._grad_fn is not None:
                    t._grad_fn = None
                    t._children = []
                    t._grad = None

    def zero_grad(self):
        self._grad = None

    def __add__(self, other):
        other = _ensure_tensor(other, like=self)
        out = TriadTensor(self._data + other._data)
        if _ENABLE_GRAD and (self._requires_grad or other._requires_grad):
            out._requires_grad = True
            out._children = [self, other]

            def _back(g):
                if self._requires_grad:
                    sg = _unbroadcast(g, self.shape)
                    self._grad = sg if self._grad is None else self._grad + sg
                if other._requires_grad:
                    og = _unbroadcast(g, other.shape)
                    other._grad = og if other._grad is None else other._grad + og
            out._grad_fn = _back
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = _ensure_tensor(other, like=self)
        out = TriadTensor(self._data - other._data)
        if _ENABLE_GRAD and (self._requires_grad or other._requires_grad):
            out._requires_grad = True
            out._children = [self, other]

            def _back(g):
                if self._requires_grad:
                    sg = _unbroadcast(g, self.shape)
                    self._grad = sg if self._grad is None else self._grad + sg
                if other._requires_grad:
                    og = _unbroadcast(-g, other.shape)
                    other._grad = og if other._grad is None else other._grad + og
            out._grad_fn = _back
        return out

    def __rsub__(self, other):
        other = _ensure_tensor(other)
        return other.__sub__(self)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            out = TriadTensor(self._data * other)
            if _ENABLE_GRAD and self._requires_grad:
                out._requires_grad = True
                out._children = [self]

                def _back(g):
                    sg = _unbroadcast(g * other, self.shape)
                    self._grad = sg if self._grad is None else self._grad + sg
                out._grad_fn = _back
            return out
        other = _ensure_tensor(other, like=self)
        out = TriadTensor(self._data * other._data)
        if _ENABLE_GRAD and (self._requires_grad or other._requires_grad):
            out._requires_grad = True
            out._children = [self, other]

            def _back(g):
                if self._requires_grad:
                    sg = _unbroadcast(g * other._data, self.shape)
                    self._grad = sg if self._grad is None else self._grad + sg
                if other._requires_grad:
                    og = _unbroadcast(g * self._data, other.shape)
                    other._grad = og if other._grad is None else other._grad + og
            out._grad_fn = _back
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = _ensure_tensor(other, like=self)
        out = TriadTensor(self._data / other._data)
        if _ENABLE_GRAD and (self._requires_grad or other._requires_grad):
            out._requires_grad = True
            out._children = [self, other]

            def _back(g):
                if self._requires_grad:
                    sg = _unbroadcast(g / other._data, self.shape)
                    self._grad = sg if self._grad is None else self._grad + sg
                if other._requires_grad:
                    og = _unbroadcast(-g * self._data / other._data ** 2, other.shape)
                    other._grad = og if other._grad is None else other._grad + og
            out._grad_fn = _back
        return out

    def __rtruediv__(self, other):
        other = _ensure_tensor(other)
        return other.__truediv__(self)

    def __neg__(self):
        out = TriadTensor(-self._data)
        if _ENABLE_GRAD and self._requires_grad:
            out._requires_grad = True
            out._children = [self]

            def _back(g):
                self._grad = -g if self._grad is None else self._grad - g
            out._grad_fn = _back
        return out

    def __pow__(self, exp):
        exp_val = exp._data if isinstance(exp, TriadTensor) else exp
        out = TriadTensor(self._data ** exp_val)
        if _ENABLE_GRAD and self._requires_grad:
            out._requires_grad = True
            out._children = [self]

            def _back(g):
                sg = _unbroadcast(g * exp_val * self._data ** (exp_val - 1), self.shape)
                self._grad = sg if self._grad is None else self._grad + sg
            out._grad_fn = _back
        return out

    def __matmul__(self, other):
        other = _ensure_tensor(other, like=self)
        if _AUTO_EQ is not None:
            self, other = _auto_place_pair(self, other)
        a_c, b_c = _amp_pair(self._data, other._data)
        out = TriadTensor(a_c @ b_c)
        if _ENABLE_GRAD and (self._requires_grad or other._requires_grad):
            out._requires_grad = True
            out._children = [self, other]

            def _back(g):
                if self._requires_grad:
                    if b_c.ndim == 1:
                        sg = np.outer(g, b_c) if g.ndim == 1 else g @ b_c
                    else:
                        sg = g @ b_c.T
                    sg = _amp_grad(sg, self._data.dtype)
                    self._grad = sg if self._grad is None else self._grad + sg
                if other._requires_grad:
                    if a_c.ndim == 1:
                        og = np.outer(a_c, g) if g.ndim == 1 else a_c.reshape(-1, 1) @ g.reshape(1, -1)
                    else:
                        og = a_c.T @ g
                    og = _amp_grad(og, other._data.dtype)
                    other._grad = og if other._grad is None else other._grad + og
            out._grad_fn = _back
        return out

    def __rmatmul__(self, other):
        other = _ensure_tensor(other)
        return other.__matmul__(self)

    def sum(self, axis=None, keepdims=False) -> TriadTensor:
        out = TriadTensor(self._data.sum(axis=axis, keepdims=keepdims))
        if _ENABLE_GRAD and self._requires_grad:
            out._requires_grad = True
            out._children = [self]

            def _back(g):
                if axis is None:
                    sg = np.triad_like(self._data, g.item() if g.size == 1 else g)
                else:
                    gg = g if keepdims else np.expand_dims(g, axis=axis)
                    sg = gg * np.ones_like(self._data)
                self._grad = sg if self._grad is None else self._grad + sg
            out._grad_fn = _back
        return out

    def mean(self, axis=None, keepdims=False) -> TriadTensor:
        n = self._data.size if axis is None else self._data.shape[axis]
        out = TriadTensor(self._data.mean(axis=axis, keepdims=keepdims))
        if _ENABLE_GRAD and self._requires_grad:
            out._requires_grad = True
            out._children = [self]

            def _back(g):
                if axis is None:
                    sg = np.triad_like(self._data, (g.item() if g.size == 1 else g) / n)
                else:
                    gg = g if keepdims else np.expand_dims(g, axis=axis)
                    sg = gg * np.ones_like(self._data) / n
                self._grad = sg if self._grad is None else self._grad + sg
            out._grad_fn = _back
        return out

    def max(self, axis=None) -> TriadTensor:
        return TriadTensor(self._data.max(axis=axis))

    def min(self, axis=None) -> TriadTensor:
        return TriadTensor(self._data.min(axis=axis))

    def __getitem__(self, idx):
        out = TriadTensor(self._data[idx])
        if _ENABLE_GRAD and self._requires_grad:
            out._requires_grad = True
            out._children = [self]

            def _back(g):
                sg = np.zeros_like(self._data)

                try:
                    np.add.at(sg, idx, g)
                except (AttributeError, TypeError):
                    import cupyx
                    cupyx.scatter_add(sg, idx, g)
                self._grad = sg if self._grad is None else self._grad + sg
            out._grad_fn = _back
        return out

    def __setitem__(self, idx, val):
        if isinstance(val, TriadTensor):
            val = val._data
        self._data[idx] = val

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for i in range(len(self._data)):
            yield TriadTensor(self._data[i])

    def __repr__(self):
        g = ', grad=True' if self._requires_grad else ''
        n = f", name='{self._name}'" if self._name else ''
        return f'TriadTensor({self._data}{g}{n})'

    def __str__(self):
        return str(self._data)

    def __dlpack__(self, *, stream=None, max_version=None, dl_device=None, copy=None):
        return self._data.__dlpack__(stream=stream, max_version=max_version,
                                      dl_device=dl_device, copy=copy)

    def __dlpack_device__(self):
        return self._data.__dlpack_device__()

    def __array_namespace__(self, /, *, api_version=None):
        import runtime.backend as _rb
        return _rb.get_xp('auto')

    def __array__(self, dtype=None, copy=None):
        return self._data.__array__(dtype=dtype, copy=copy) if hasattr(self._data, '__array__') else self._data

    def __eq__(self, other):
        other = _ensure_tensor(other, like=self)
        return TriadTensor(self._data == other._data)

    def __lt__(self, other):
        other = _ensure_tensor(other)
        return TriadTensor(self._data < other._data)

    def __gt__(self, other):
        other = _ensure_tensor(other)
        return TriadTensor(self._data > other._data)

    def __le__(self, other):
        other = _ensure_tensor(other)
        return TriadTensor(self._data <= other._data)

    def __ge__(self, other):
        other = _ensure_tensor(other)
        return TriadTensor(self._data >= other._data)

    def __float__(self):
        return float(self._data)

    def __int__(self):
        return int(self._data)

    def __bool__(self):
        if self._data.size == 1:
            return bool(self._data.item())
        raise ValueError(f"TriadTensor with shape {self._data.shape} is ambiguous for bool; use .item() or an explicit comparison")

def _amp_pair(a_data, b_data):

    from runtime.ml.amp import is_autocast
    if (is_autocast() and _is_cuda_array(a_data)
            and a_data.dtype.kind == 'f' and a_data.dtype.itemsize > 2):
        return a_data.astype('float16'), b_data.astype('float16')
    return a_data, b_data

def _amp_grad(g, ref_dtype):

    return g.astype(ref_dtype) if g.dtype != ref_dtype else g

def _ensure_tensor(x, like: TriadTensor | None = None) -> TriadTensor:
    if isinstance(x, TriadTensor):
        if like is not None and x.device != like.device:
            raise RuntimeError(
                f'tensores em devices diferentes: {x.device} vs {like.device} '
                f'— mova explicitamente com .to()')
        return x
    return TriadTensor(x, device=like.device if like is not None else None)

def _unbroadcast(g: np.ndarray, shape: Shape) -> np.ndarray:
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, (gs, s) in enumerate(zip(g.shape, shape)):
        if s == 1 and gs != 1:
            g = g.sum(axis=i, keepdims=True)
    return g

def tensor(data, requires_grad=False, name='', device=None, dtype=None) -> TriadTensor:
    return TriadTensor(data, requires_grad=requires_grad, name=name, device=device, dtype=dtype)

def zeros(*shape, requires_grad=False, device=None) -> TriadTensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    xp = _cp if device == 'cuda' else np
    return TriadTensor(xp.zeros(shape), requires_grad=requires_grad)

def ones(*shape, requires_grad=False, device=None) -> TriadTensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    xp = _cp if device == 'cuda' else np
    return TriadTensor(xp.ones(shape), requires_grad=requires_grad)

def randn(*shape, requires_grad=False, device=None) -> TriadTensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    xp = _cp if device == 'cuda' else np
    return TriadTensor(xp.random.randn(*shape), requires_grad=requires_grad)

def rand(*shape, requires_grad=False, device=None) -> TriadTensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    xp = _cp if device == 'cuda' else np
    return TriadTensor(xp.random.rand(*shape), requires_grad=requires_grad)

def arange(start, stop=None, step=1) -> TriadTensor:
    if stop is None:
        return TriadTensor(np.arange(start))
    return TriadTensor(np.arange(start, stop, step))

def linspace(start, stop, num) -> TriadTensor:
    return TriadTensor(np.linspace(start, stop, num))

def eye(n) -> TriadTensor:
    return TriadTensor(np.eye(n))

def exp(t: TriadTensor) -> TriadTensor:
    t = _ensure_tensor(t)
    out = TriadTensor(np.exp(t._data))
    if _ENABLE_GRAD and t._requires_grad:
        out._requires_grad = True
        out._children = [t]

        def _back(g):
            sg = g * out._data
            t._grad = sg if t._grad is None else t._grad + sg
        out._grad_fn = _back
    return out

def log(t: TriadTensor) -> TriadTensor:
    t = _ensure_tensor(t)
    xc = np.maximum(t._data, 1e-30)
    out = TriadTensor(np.log(xc))
    if _ENABLE_GRAD and t._requires_grad:
        out._requires_grad = True
        out._children = [t]

        def _back(g):
            sg = g / xc
            t._grad = sg if t._grad is None else t._grad + sg
        out._grad_fn = _back
    return out

def sqrt(t: TriadTensor) -> TriadTensor:
    t = _ensure_tensor(t)
    xc = np.maximum(t._data, 0.0)
    out = TriadTensor(np.sqrt(xc))
    if _ENABLE_GRAD and t._requires_grad:
        out._requires_grad = True
        out._children = [t]

        def _back(g):
            sg = g / (2 * np.maximum(out._data, 1e-30))
            t._grad = sg if t._grad is None else t._grad + sg
        out._grad_fn = _back
    return out

def tanh(t: TriadTensor) -> TriadTensor:
    t = _ensure_tensor(t)
    out = TriadTensor(np.tanh(t._data))
    if _ENABLE_GRAD and t._requires_grad:
        out._requires_grad = True
        out._children = [t]

        def _back(g):
            sg = g * (1 - out._data ** 2)
            t._grad = sg if t._grad is None else t._grad + sg
        out._grad_fn = _back
    return out

def sin(t: TriadTensor) -> TriadTensor:
    t = _ensure_tensor(t)
    out = TriadTensor(np.sin(t._data))
    if _ENABLE_GRAD and t._requires_grad:
        out._requires_grad = True
        out._children = [t]

        def _back(g):
            sg = g * np.cos(t._data)
            t._grad = sg if t._grad is None else t._grad + sg
        out._grad_fn = _back
    return out

def cos(t: TriadTensor) -> TriadTensor:
    t = _ensure_tensor(t)
    out = TriadTensor(np.cos(t._data))
    if _ENABLE_GRAD and t._requires_grad:
        out._requires_grad = True
        out._children = [t]

        def _back(g):
            sg = -g * np.sin(t._data)
            t._grad = sg if t._grad is None else t._grad + sg
        out._grad_fn = _back
    return out

def sigmoid(t: TriadTensor) -> TriadTensor:
    t = _ensure_tensor(t)
    s = 1.0 / (1.0 + np.exp(-t._data))
    out = TriadTensor(s)
    if _ENABLE_GRAD and t._requires_grad:
        out._requires_grad = True
        out._children = [t]

        def _back(g):
            sg = g * s * (1 - s)
            t._grad = sg if t._grad is None else t._grad + sg
        out._grad_fn = _back
    return out

def relu(t: TriadTensor) -> TriadTensor:
    t = _ensure_tensor(t)
    out = TriadTensor(np.maximum(0, t._data))
    if _ENABLE_GRAD and t._requires_grad:
        out._requires_grad = True
        out._children = [t]

        def _back(g):

            sg = g * (t._data > 0).astype(t._data.dtype)
            t._grad = sg if t._grad is None else t._grad + sg
        out._grad_fn = _back
    return out

def softmax(t: TriadTensor, axis=-1) -> TriadTensor:
    t = _ensure_tensor(t)
    shifted = t._data - t._data.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    s = e / e.sum(axis=axis, keepdims=True)
    out = TriadTensor(s)
    if _ENABLE_GRAD and t._requires_grad:
        out._requires_grad = True
        out._children = [t]

        def _back(g):
            sg = s * (g - (g * s).sum(axis=axis, keepdims=True))
            t._grad = sg if t._grad is None else t._grad + sg
        out._grad_fn = _back
    return out

def cross_entropy(logits: TriadTensor, targets: TriadTensor) -> TriadTensor:
    logits = _ensure_tensor(logits)
    targets = _ensure_tensor(targets)
    orig_shape = logits._data.shape
    V = orig_shape[-1]
    flat = logits._data.reshape(-1, V)
    shifted = flat - flat.max(axis=-1, keepdims=True)
    log_sum_exp = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    log_probs = shifted - log_sum_exp
    xpm = _xp_of(flat)
    target_idx = xpm.asarray(targets._data).astype(int).reshape(-1)
    n = flat.shape[0]
    loss_val = -log_probs[xpm.arange(n), target_idx].mean()
    out = TriadTensor(loss_val)
    if _ENABLE_GRAD and logits._requires_grad:
        out._requires_grad = True
        out._children = [logits]

        def _back(g):
            probs = xpm.exp(log_probs)
            sg = probs.copy()
            sg[xpm.arange(n), target_idx] -= 1
            sg = sg / n * (g.item() if isinstance(g, np.ndarray) and g.size == 1 else g)
            sg = sg.reshape(orig_shape)
            logits._grad = sg if logits._grad is None else logits._grad + sg
        out._grad_fn = _back
    return out

def mse_loss(pred: TriadTensor, target: TriadTensor) -> TriadTensor:
    diff = pred - target
    return (diff * diff).mean()

def bmm(a: TriadTensor, b: TriadTensor) -> TriadTensor:
    a = _ensure_tensor(a)
    b = _ensure_tensor(b, like=a)
    if _AUTO_EQ is not None:
        a, b = _auto_place_pair(a, b)
    a_c, b_c = _amp_pair(a._data, b._data)
    out = TriadTensor(np.matmul(a_c, b_c))
    if _ENABLE_GRAD and (a._requires_grad or b._requires_grad):
        out._requires_grad = True
        out._children = [a, b]

        def _back(g):
            if a._requires_grad:
                if b_c.ndim >= 2:
                    ag = np.matmul(g, np.swapaxes(b_c, -1, -2))
                else:
                    ag = np.outer(g, b_c) if g.ndim == 1 else g @ b_c
                ag = _amp_grad(_unbroadcast(ag, a.shape), a._data.dtype)
                a._grad = ag if a._grad is None else a._grad + ag
            if b._requires_grad:
                if a_c.ndim >= 2:
                    bg = np.matmul(np.swapaxes(a_c, -1, -2), g)
                else:
                    bg = np.outer(a_c, g) if g.ndim == 1 else a_c.reshape(-1, 1) @ g.reshape(1, -1)
                bg = _amp_grad(_unbroadcast(bg, b.shape), b._data.dtype)
                b._grad = bg if b._grad is None else b._grad + bg
        out._grad_fn = _back
    return out

def transpose(t: TriadTensor, axis1: int, axis2: int) -> TriadTensor:
    t = _ensure_tensor(t)
    out = TriadTensor(np.swapaxes(t._data, axis1, axis2))
    if _ENABLE_GRAD and t._requires_grad:
        out._requires_grad = True
        out._children = [t]

        def _back(g):
            sg = np.swapaxes(g, axis1, axis2)
            t._grad = sg if t._grad is None else t._grad + sg
        out._grad_fn = _back
    return out

def layer_norm(t: TriadTensor, gamma: TriadTensor=None, beta: TriadTensor=None, eps: float=1e-05) -> TriadTensor:
    t = _ensure_tensor(t)
    x = t._data
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mu) * inv
    g_data = gamma._data if gamma is not None else 1.0
    b_data = beta._data if beta is not None else 0.0
    out = TriadTensor(x_hat * g_data + b_data)
    needs = t._requires_grad or (gamma is not None and gamma._requires_grad) or (beta is not None and beta._requires_grad)
    if _ENABLE_GRAD and needs:
        out._requires_grad = True
        children = [t]
        if gamma is not None:
            children.append(gamma)
        if beta is not None:
            children.append(beta)
        out._children = children
        D = x.shape[-1]

        def _back(g):
            if beta is not None and beta._requires_grad:
                bg = _unbroadcast(g, beta.shape)
                beta._grad = bg if beta._grad is None else beta._grad + bg
            if gamma is not None and gamma._requires_grad:
                gg = _unbroadcast(g * x_hat, gamma.shape)
                gamma._grad = gg if gamma._grad is None else gamma._grad + gg
            if t._requires_grad:
                dxhat = g * g_data
                sg = inv / D * (D * dxhat - dxhat.sum(axis=-1, keepdims=True) - x_hat * (dxhat * x_hat).sum(axis=-1, keepdims=True))
                t._grad = sg if t._grad is None else t._grad + sg
        out._grad_fn = _back
    return out

def cat(tensors: list[TriadTensor], axis=0) -> TriadTensor:
    tensors = [_ensure_tensor(t) for t in tensors]
    out = TriadTensor(np.concatenate([t._data for t in tensors], axis=axis))
    if _ENABLE_GRAD and any(t._requires_grad for t in tensors):
        out._requires_grad = True
        out._children = list(tensors)
        sizes = [t._data.shape[axis] for t in tensors]
        pts, acc = ([], 0)
        for s in sizes[:-1]:
            acc += s
            pts.append(acc)

        def _back(g):
            parts = np.split(g, pts, axis=axis) if pts else [g]
            for t, part in zip(tensors, parts):
                if t._requires_grad:
                    t._grad = part if t._grad is None else t._grad + part
        out._grad_fn = _back
    return out

def stack(tensors: list[TriadTensor], axis=0) -> TriadTensor:
    tensors = [_ensure_tensor(t) for t in tensors]
    out = TriadTensor(np.stack([t._data for t in tensors], axis=axis))
    if _ENABLE_GRAD and any(t._requires_grad for t in tensors):
        out._requires_grad = True
        out._children = list(tensors)

        def _back(g):
            for i, t in enumerate(tensors):
                if t._requires_grad:
                    part = np.take(g, i, axis=axis)
                    t._grad = part if t._grad is None else t._grad + part
        out._grad_fn = _back
    return out

class no_grad:

    def __enter__(self):
        global _ENABLE_GRAD
        self._prev = _ENABLE_GRAD
        _ENABLE_GRAD = False

    def __exit__(self, *args):
        global _ENABLE_GRAD
        _ENABLE_GRAD = self._prev

class FunctionCtx:

    def __init__(self):
        self.saved = ()

    def save_for_backward(self, *arrays):
        self.saved = arrays

class Function:

    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError(f'{cls.__name__}.forward not implemented' if (cls := getattr(args[0], '__class__', None)) else 'Function.forward not implemented')

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError('Function.backward not implemented')

    @classmethod
    def apply(cls, *inputs):
        tensors = [_ensure_tensor(x) for x in inputs]
        ctx = FunctionCtx()
        out_data = cls.forward(ctx, *[t._data for t in tensors])
        out = TriadTensor(out_data)
        needs = [t._requires_grad for t in tensors]
        if _ENABLE_GRAD and any(needs):
            out._requires_grad = True
            out._children = list(tensors)

            def _back(g):
                grads = cls.backward(ctx, g)
                if not isinstance(grads, (tuple, list)):
                    grads = (grads,)
                if len(grads) != len(tensors):
                    raise RuntimeError(
                        f'{cls.__name__}.backward retornou {len(grads)} '
                        f'gradientes para {len(tensors)} inputs')
                for t, tg in zip(tensors, grads):
                    if t._requires_grad and tg is not None:
                        t._grad = tg if t._grad is None else t._grad + tg
            out._grad_fn = _back
        return out

def checkpoint(fn, *inputs, rng_modules=()):

    datas = [t._data for t in inputs]
    req = [t._requires_grad for t in inputs]
    rng_states = [m._rng.bit_generator.state for m in rng_modules]

    try:
        import importlib as _il
        _amp = _il.import_module('runtime.ml.amp')
        amp_state = _amp._AUTOCAST
    except (ImportError, ModuleNotFoundError):
        _amp, amp_state = None, False

    global _ENABLE_GRAD
    prev = _ENABLE_GRAD
    _ENABLE_GRAD = False
    try:
        out = fn(*inputs)
    finally:
        _ENABLE_GRAD = prev
    aux = None
    if isinstance(out, tuple):
        out, aux = out[0], out[1:]

    result = TriadTensor(out._data if isinstance(out, TriadTensor) else out)
    if _ENABLE_GRAD and any(req):
        result._requires_grad = True
        result._children = [t for t in inputs if isinstance(t, TriadTensor)]

        def _back(g):
            for m, st in zip(rng_modules, rng_states):
                m._rng.bit_generator.state = st
            ins = [TriadTensor(d, requires_grad=r) for d, r in zip(datas, req)]
            ac_prev = _amp._AUTOCAST if _amp is not None else False
            if _amp is not None:
                _amp._AUTOCAST = amp_state
            try:
                res = fn(*ins)
                if isinstance(res, tuple):
                    res = res[0]
                res.backward(g)
            finally:
                if _amp is not None:
                    _amp._AUTOCAST = ac_prev
            for t, i in zip(inputs, ins):
                if t._requires_grad and i._grad is not None:
                    t._grad = i._grad if t._grad is None else t._grad + i._grad
        result._grad_fn = _back
    if aux is not None:
        return (result,) + tuple(aux)
    return result
