from __future__ import annotations
from runtime.ml_device import xp as np
from runtime import ml_device as _D
from typing import Optional, Tuple, Union, Sequence
Shape = Tuple[int, ...]
_ENABLE_GRAD = True

class TriadTensor:
    __slots__ = ('_data', '_grad', '_requires_grad', '_grad_fn', '_children', '_device', '_name')

    def __init__(self, data, requires_grad: bool=False, device: str='cpu', name: str=''):
        if isinstance(data, TriadTensor):
            data = data._data
        self._data = _D.coerce(data)
        self._grad: Optional[np.ndarray] = None
        self._requires_grad = requires_grad
        self._grad_fn: Optional[_GradFn] = None
        self._children: list[TriadTensor] = []
        self._device = device
        self._name = name

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, val):
        self._data = _D.coerce(val)

    @property
    def grad(self) -> Optional[np.ndarray]:
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
        return TriadTensor(self._data.T)

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    def detach(self) -> TriadTensor:
        return TriadTensor(self._data.copy(), requires_grad=False)

    def numpy(self):
        return _D.asnumpy(self._data)

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

    def backward(self, grad: Optional[np.ndarray]=None):
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

    def zero_grad(self):
        self._grad = None

    def __add__(self, other):
        other = _ensure_tensor(other)
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
        other = _ensure_tensor(other)
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
        other = _ensure_tensor(other)
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
        other = _ensure_tensor(other)
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
        other = _ensure_tensor(other)
        out = TriadTensor(self._data @ other._data)
        if _ENABLE_GRAD and (self._requires_grad or other._requires_grad):
            out._requires_grad = True
            out._children = [self, other]

            def _back(g):
                if self._requires_grad:
                    if other._data.ndim == 1:
                        sg = np.outer(g, other._data) if g.ndim == 1 else g @ other._data
                    else:
                        sg = g @ other._data.T
                    self._grad = sg if self._grad is None else self._grad + sg
                if other._requires_grad:
                    if self._data.ndim == 1:
                        og = np.outer(self._data, g) if g.ndim == 1 else self._data.reshape(-1, 1) @ g.reshape(1, -1)
                    else:
                        og = self._data.T @ g
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
                    sg = np.full_like(self._data, g.item() if g.size == 1 else g)
                else:
                    sg = np.expand_dims(g, axis=axis) * np.ones_like(self._data)
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
                    sg = np.full_like(self._data, (g.item() if g.size == 1 else g) / n)
                else:
                    sg = np.expand_dims(g, axis=axis) * np.ones_like(self._data) / n
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
                sg[idx] = g
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

    def __eq__(self, other):
        other = _ensure_tensor(other)
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
        return bool(self._data.any())

def _ensure_tensor(x) -> TriadTensor:
    if isinstance(x, TriadTensor):
        return x
    return TriadTensor(x)

def _unbroadcast(g: np.ndarray, shape: Shape) -> np.ndarray:
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, (gs, s) in enumerate(zip(g.shape, shape)):
        if s == 1 and gs != 1:
            g = g.sum(axis=i, keepdims=True)
    return g

def tensor(data, requires_grad=False, name='') -> TriadTensor:
    return TriadTensor(data, requires_grad=requires_grad, name=name)

def zeros(*shape, requires_grad=False) -> TriadTensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return TriadTensor(np.zeros(shape), requires_grad=requires_grad)

def ones(*shape, requires_grad=False) -> TriadTensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return TriadTensor(np.ones(shape), requires_grad=requires_grad)

def randn(*shape, requires_grad=False) -> TriadTensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return TriadTensor(np.random.randn(*shape), requires_grad=requires_grad)

def rand(*shape, requires_grad=False) -> TriadTensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return TriadTensor(np.random.rand(*shape), requires_grad=requires_grad)

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
    out = TriadTensor(np.log(t._data))
    if _ENABLE_GRAD and t._requires_grad:
        out._requires_grad = True
        out._children = [t]

        def _back(g):
            sg = g / t._data
            t._grad = sg if t._grad is None else t._grad + sg
        out._grad_fn = _back
    return out

def sqrt(t: TriadTensor) -> TriadTensor:
    t = _ensure_tensor(t)
    out = TriadTensor(np.sqrt(t._data))
    if _ENABLE_GRAD and t._requires_grad:
        out._requires_grad = True
        out._children = [t]

        def _back(g):
            sg = g / (2 * out._data)
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
            sg = g * (t._data > 0).astype(_D.fdtype())
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
    target_idx = targets._data.astype(int).reshape(-1)
    n = flat.shape[0]
    loss_val = -log_probs[np.arange(n), target_idx].mean()
    out = TriadTensor(loss_val)
    if _ENABLE_GRAD and logits._requires_grad:
        out._requires_grad = True
        out._children = [logits]

        def _back(g):
            probs = np.exp(log_probs)
            sg = probs.copy()
            sg[np.arange(n), target_idx] -= 1
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
    b = _ensure_tensor(b)
    out = TriadTensor(np.matmul(a._data, b._data))
    if _ENABLE_GRAD and (a._requires_grad or b._requires_grad):
        out._requires_grad = True
        out._children = [a, b]

        def _back(g):
            if a._requires_grad:
                ag = np.matmul(g, np.swapaxes(b._data, -1, -2))
                ag = _unbroadcast(ag, a.shape)
                a._grad = ag if a._grad is None else a._grad + ag
            if b._requires_grad:
                bg = np.matmul(np.swapaxes(a._data, -1, -2), g)
                bg = _unbroadcast(bg, b.shape)
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
    if _ENABLE_GRAD and any((t._requires_grad for t in tensors)):
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
    if _ENABLE_GRAD and any((t._requires_grad for t in tensors)):
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