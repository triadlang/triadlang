from __future__ import annotations

from runtime.ml.tensor import (
    TriadTensor,
    bmm,
    layer_norm,
    mse_loss,
    relu,
    sigmoid,
    softmax,
    stack,
    tanh,
    tensor,
    transpose,
)
from triad import ntri as np


class Parameter(TriadTensor):

    def __init__(self, data):
        if isinstance(data, TriadTensor):
            data = data._data
        super().__init__(data, requires_grad=True)

class Module:

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: TriadTensor) -> TriadTensor:
        raise NotImplementedError(f'{type(self).__name__}.forward not implemented')

    def parameters(self) -> list[Parameter]:
        params = []
        seen: set[int] = set()

        def _collect(v):
            if isinstance(v, Parameter):
                if id(v) not in seen:
                    seen.add(id(v))
                    params.append(v)
            elif isinstance(v, Module):
                for p in v.parameters():
                    if id(p) not in seen:
                        seen.add(id(p))
                        params.append(p)
            elif isinstance(v, (list, tuple)):
                for item in v:
                    _collect(item)
            elif isinstance(v, dict):
                for item in v.values():
                    _collect(item)

        for v in self.__dict__.values():
            _collect(v)
        return params

    def zero_grad(self):
        for p in self.parameters():
            p._grad = None

    def to(self, device: str) -> Module:

        from triad import ntri as _onp

        from runtime.ml.tensor import _coerce, _is_cuda_array

        def _move(obj):
            if isinstance(obj, TriadTensor):
                obj._data = _coerce(obj._data, device)
                obj._grad = None
                obj._device = device
            elif isinstance(obj, Module):
                obj.to(device)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    _move(item)

        for k, v in list(self.__dict__.items()):
            if isinstance(v, (TriadTensor, Module, list, tuple)):
                _move(v)
            elif isinstance(v, _onp.ndarray) or _is_cuda_array(v):
                self.__dict__[k] = _coerce(v, device)
        return self

    def cuda(self) -> Module:
        return self.to('cuda')

    def cpu(self) -> Module:
        return self.to('cpu')

class triad(Module):

    def __init__(self, in_features: int, out_features: int, bias: bool=True):
        k = 1.0 / np.sqrt(in_features)
        self.weight = Parameter(tensor(np.random.uniform(-k, k, (in_features, out_features))))
        self.bias = Parameter(tensor(np.random.uniform(-k, k, (out_features,)))) if bias else None

    def forward(self, x: TriadTensor) -> TriadTensor:
        if x.ndim > 2:

            lead = x.shape[:-1]
            x2 = x.reshape(-1, x.shape[-1])
            out = bmm(x2, self.weight)
            out = out.reshape(*lead, self.weight.shape[-1])
        else:
            out = x @ self.weight if x.ndim <= 1 else bmm(x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out

class Conv1d(Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        k = 1.0 / np.sqrt(in_channels * kernel_size)
        self.weight = Parameter(tensor(np.random.uniform(-k, k, (out_channels, in_channels, kernel_size))))
        self.bias = Parameter(tensor(np.zeros(out_channels)))

    def forward(self, x: TriadTensor) -> TriadTensor:
        x_np = x._data
        if x_np.ndim == 2:
            x_np = x_np[np.newaxis, :, :]
        batch, c_in, length = x_np.shape
        if self.padding > 0:
            x_np = np.pad(x_np, ((0, 0), (0, 0), (self.padding, self.padding)))
            length = x_np.shape[2]
        out_len = (length - self.kernel_size) // self.stride + 1
        cols = np.zeros((batch, self.in_channels, self.kernel_size, out_len))
        for i in range(out_len):
            s = i * self.stride
            cols[:, :, :, i] = x_np[:, :, s:s + self.kernel_size]
        cols_flat = cols.reshape(batch, self.in_channels * self.kernel_size, out_len)
        w_flat = self.weight._data.reshape(self.out_channels, self.in_channels * self.kernel_size)
        out_np = np.einsum('oi,bil->bol', w_flat, cols_flat)
        out_np = out_np + self.bias._data.reshape(1, -1, 1)
        out = TriadTensor(out_np.squeeze(0) if x._data.ndim == 2 else out_np)
        if x._requires_grad or self.weight._requires_grad:
            out._requires_grad = True
            out._children = [x, self.weight, self.bias]
            _x_np = x_np
            _cols_flat = cols_flat
            _w_flat = w_flat
            _batch = batch
            _self = self

            def _back(g):
                if g.ndim == 2:
                    g = g[np.newaxis, :, :]
                if _self.bias._requires_grad:
                    bg = g.sum(axis=(0, 2))
                    _self.bias._grad = bg if _self.bias._grad is None else _self.bias._grad + bg
                if _self.weight._requires_grad:
                    wg = np.einsum('bol,bil->oi', g, _cols_flat)
                    wg = wg.reshape(_self.out_channels, _self.in_channels, _self.kernel_size)
                    _self.weight._grad = wg if _self.weight._grad is None else _self.weight._grad + wg
                if x._requires_grad:
                    dx_cols = np.einsum('oi,bol->bil', _w_flat, g)
                    dx_cols = dx_cols.reshape(_batch, _self.in_channels, _self.kernel_size, -1)
                    dx = np.zeros_like(_x_np)
                    out_l = dx_cols.shape[3]
                    for i in range(out_l):
                        s = i * _self.stride
                        dx[:, :, s:s + _self.kernel_size] += dx_cols[:, :, :, i]
                    if _self.padding > 0:
                        dx = dx[:, :, _self.padding:-_self.padding]
                    if x._data.ndim == 2:
                        dx = dx.squeeze(0)
                    x._grad = dx if x._grad is None else x._grad + dx
            out._grad_fn = _back
        return out

class Conv2d(Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        k = 1.0 / np.sqrt(in_channels * kernel_size * kernel_size)
        self.weight = Parameter(tensor(np.random.uniform(-k, k, (out_channels, in_channels, kernel_size, kernel_size))))
        self.bias = Parameter(tensor(np.zeros(out_channels)))

    def forward(self, x: TriadTensor) -> TriadTensor:
        x_np = x._data
        if x_np.ndim == 3:
            x_np = x_np[np.newaxis, :, :, :]
        batch, c_in, h_in, w_in = x_np.shape
        p = self.padding
        if p > 0:
            x_np = np.pad(x_np, ((0, 0), (0, 0), (p, p), (p, p)))
        _, _, h_p, w_p = x_np.shape
        ks = self.kernel_size
        s = self.stride
        h_out = (h_p - ks) // s + 1
        w_out = (w_p - ks) // s + 1
        cols = np.zeros((batch, c_in, ks, ks, h_out, w_out))
        for i in range(h_out):
            for j in range(w_out):
                cols[:, :, :, :, i, j] = x_np[:, :, i * s:i * s + ks, j * s:j * s + ks]
        cols_flat = cols.reshape(batch, c_in * ks * ks, h_out * w_out)
        w_flat = self.weight._data.reshape(self.out_channels, c_in * ks * ks)
        out_np = np.einsum('oi,bil->bol', w_flat, cols_flat).reshape(batch, self.out_channels, h_out, w_out)
        out_np = out_np + self.bias._data.reshape(1, -1, 1, 1)
        out = TriadTensor(out_np.squeeze(0) if x._data.ndim == 3 else out_np)
        if x._requires_grad or self.weight._requires_grad:
            out._requires_grad = True
            out._children = [x, self.weight, self.bias]
            _x_np = x_np
            _cols_flat = cols_flat
            _w_flat = w_flat
            _batch = batch
            _self = self

            def _back(g):
                if g.ndim == 3:
                    g = g[np.newaxis, :, :, :]
                if _self.bias._requires_grad:
                    bg = g.sum(axis=(0, 2, 3))
                    _self.bias._grad = bg if _self.bias._grad is None else _self.bias._grad + bg
                if _self.weight._requires_grad:
                    g_flat = g.reshape(_batch, _self.out_channels, -1)
                    wg = np.einsum('bol,bil->oi', g_flat, _cols_flat)
                    wg = wg.reshape(_self.out_channels, _self.in_channels, ks, ks)
                    _self.weight._grad = wg if _self.weight._grad is None else _self.weight._grad + wg
                if x._requires_grad:
                    g_flat = g.reshape(_batch, _self.out_channels, -1)
                    dx_cols = np.einsum('oi,bol->bil', _w_flat, g_flat)
                    dx_cols = dx_cols.reshape(_batch, c_in, ks, ks, h_out, w_out)
                    dx = np.zeros_like(_x_np)
                    for i in range(h_out):
                        for j in range(w_out):
                            dx[:, :, i * s:i * s + ks, j * s:j * s + ks] += dx_cols[:, :, :, :, i, j]
                    if p > 0:
                        dx = dx[:, :, p:-p, p:-p]
                    if x._data.ndim == 3:
                        dx = dx.squeeze(0)
                    x._grad = dx if x._grad is None else x._grad + dx
            out._grad_fn = _back
        return out

class ReLU(Module):

    def forward(self, x: TriadTensor) -> TriadTensor:
        return relu(x)

class Tanh(Module):

    def forward(self, x: TriadTensor) -> TriadTensor:
        return tanh(x)

class Sigmoid(Module):

    def forward(self, x: TriadTensor) -> TriadTensor:
        return sigmoid(x)

class Softmax(Module):

    def __init__(self, axis=-1):
        self.axis = axis

    def forward(self, x: TriadTensor) -> TriadTensor:
        return softmax(x, axis=self.axis)

class Sequential(Module):

    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x: TriadTensor) -> TriadTensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Parameter]:
        params = []
        for layer in self.layers:
            if isinstance(layer, Module):
                params.extend(layer.parameters())
        return params

class Flatten(Module):

    def forward(self, x: TriadTensor) -> TriadTensor:
        if x._data.ndim <= 1:
            return x
        batch = x._data.shape[0]
        return x.reshape(batch, -1)

class Dropout(Module):

    def __init__(self, p=0.5):
        self.p = p
        self.training = True

    def forward(self, x: TriadTensor) -> TriadTensor:
        if not self.training or self.p == 0:
            return x

        mask = (np.random.rand(*x.shape) > self.p).astype(x._data.dtype) / (1 - self.p)
        out = TriadTensor(x._data * mask)
        if x._requires_grad:
            out._requires_grad = True
            out._children = [x]
            _mask = mask

            def _back(g):
                sg = g * _mask
                x._grad = sg if x._grad is None else x._grad + sg
            out._grad_fn = _back
        return out

class BatchNorm1d(Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = Parameter(tensor(np.ones(num_features)))
        self.beta = Parameter(tensor(np.zeros(num_features)))
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.training = True

    def forward(self, x: TriadTensor) -> TriadTensor:
        if self.training:
            mean = x._data.mean(axis=0)
            var = x._data.var(axis=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        x_norm = TriadTensor((x._data - mean) / np.sqrt(var + self.eps))
        return x_norm * self.gamma + self.beta

class Embedding(Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(tensor(np.random.randn(num_embeddings, embedding_dim) * 0.02))

    def forward(self, idx: TriadTensor) -> TriadTensor:
        ids = idx._data.astype(int) if isinstance(idx, TriadTensor) else np.asarray(idx, dtype=int)
        out = TriadTensor(self.weight._data[ids])
        if self.weight._requires_grad:
            out._requires_grad = True
            out._children = [self.weight]
            w = self.weight

            def _back(g):
                wg = np.zeros_like(w._data)
                D = wg.shape[1] if wg.ndim > 1 else 1
                idx_f = ids.ravel() if isinstance(ids, np.ndarray) else np.asarray(ids).ravel()
                g_f = g.reshape((g.size // D, D)) if isinstance(g, np.ndarray) and g.ndim > 2 else g
                np.add.at(wg, idx_f, g_f)
                w._grad = wg if w._grad is None else w._grad + wg
            out._grad_fn = _back
        return out

class LayerNorm(Module):

    def __init__(self, normalized_shape: int, eps: float=1e-05):
        self.eps = eps
        self.gamma = Parameter(tensor(np.ones(normalized_shape)))
        self.beta = Parameter(tensor(np.zeros(normalized_shape)))

    def forward(self, x: TriadTensor) -> TriadTensor:
        return layer_norm(x, self.gamma, self.beta, self.eps)

class MultiHeadAttention(Module):

    def __init__(self, d_model: int, n_heads: int, causal: bool=True):
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.q_proj = triad(d_model, d_model)
        self.k_proj = triad(d_model, d_model)
        self.v_proj = triad(d_model, d_model)
        self.out_proj = triad(d_model, d_model)

    def _split_heads(self, x: TriadTensor, B: int, Tn: int) -> TriadTensor:
        x = x.reshape(B, Tn, self.n_heads, self.d_head)
        return transpose(x, 1, 2)

    def forward(self, x: TriadTensor) -> TriadTensor:
        B, Tn, _ = x.shape
        q = self._split_heads(self.q_proj(x), B, Tn)
        k = self._split_heads(self.k_proj(x), B, Tn)
        v = self._split_heads(self.v_proj(x), B, Tn)
        scores = bmm(q, transpose(k, -1, -2)) * (1.0 / np.sqrt(self.d_head))
        if self.causal:
            mask = np.triu(np.ones((Tn, Tn)), k=1).astype(bool)
            sd = scores._data.copy()
            sd[..., mask] = -1000000000.0
            masked = TriadTensor(sd)
            if scores._requires_grad:
                masked._requires_grad = True
                masked._children = [scores]
                keep = ~mask

                def _back(g, _src=scores):

                    sg = g * keep
                    _src._grad = sg if _src._grad is None else _src._grad + sg
                masked._grad_fn = _back
            scores = masked
        attn = softmax(scores, axis=-1)
        ctx = bmm(attn, v)
        ctx = transpose(ctx, 1, 2).reshape(B, Tn, self.d_model)
        return self.out_proj(ctx)

class FeedForward(Module):

    def __init__(self, d_model: int, d_ff: int):
        self.fc1 = triad(d_model, d_ff)
        self.fc2 = triad(d_ff, d_model)

    def forward(self, x: TriadTensor) -> TriadTensor:
        return self.fc2(relu(self.fc1(x)))

class TransformerBlock(Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, causal: bool=True):
        self.attn = MultiHeadAttention(d_model, n_heads, causal=causal)
        self.ln1 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x: TriadTensor) -> TriadTensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class Transformer(Module):

    def __init__(self, vocab_size: int, d_model: int=64, n_heads: int=4, n_layers: int=2, d_ff: int=256, max_len: int=128):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.tok_emb = Embedding(vocab_size, d_model)
        self.pos_emb = Parameter(tensor(np.random.randn(max_len, d_model) * 0.02))
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, causal=True) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model)
        self.head = triad(d_model, vocab_size, bias=False)

    def forward(self, idx: TriadTensor) -> TriadTensor:
        ids = idx._data.astype(int) if isinstance(idx, TriadTensor) else np.asarray(idx, dtype=int)
        if ids.ndim == 1:
            ids = ids[np.newaxis, :]
        B, Tn = ids.shape
        x = self.tok_emb(tensor(ids))
        pos = TriadTensor(self.pos_emb._data[:Tn])
        if self.pos_emb._requires_grad:
            pos._requires_grad = True
            pos._children = [self.pos_emb]
            pe = self.pos_emb

            def _pback(g):
                pg = np.zeros_like(pe._data)
                pg[:Tn] += g.sum(axis=0) if g.ndim == 3 else g
                pe._grad = pg if pe._grad is None else pe._grad + pg
            pos._grad_fn = _pback
        x = x + pos
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)

    def parameters(self) -> list[Parameter]:
        params = []
        for v in self.__dict__.values():
            if isinstance(v, Parameter):
                params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    elif isinstance(item, Parameter):
                        params.append(item)
        return params

class TriadSSM(Module):

    def __init__(self, d_model: int, d_state: int=64, n_memory: int=3, coupling: bool=True, noise: float=0.0, seed: int = 0):
        if n_memory < 3:
            raise ValueError(
                f"triad rule: TriadSSM P2 memory field needs at least 3 "
                f"time-scales (p1/p2/p3), got n_memory={n_memory}")
        self.d_model = d_model
        self.d_state = d_state
        self.n_memory = n_memory
        self.noise = noise
        self.coupling = coupling
        self.in_re = triad(d_model, d_state)
        self.in_im = triad(d_model, d_state)
        self.out_re = triad(d_state, d_model)
        self.out_im = triad(d_state, d_model)
        self.omega = Parameter(tensor(np.linspace(0.05, float(np.pi), d_state)))
        self.log_decay = Parameter(tensor(np.triad(d_state, 3.5)))
        self.nu = Parameter(tensor(np.linspace(-1.0, 2.0, n_memory)))
        self.lam = Parameter(tensor(np.triad((n_memory, d_state), -0.01)))
        self.Lambda = Parameter(tensor(np.triad(d_state, -0.02)))
        if coupling:
            self.couple = triad(d_state, d_state, bias=False)
            self.kappa = Parameter(tensor(np.array(0.0)))
        self.training = True
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _cmul(ar, ai, br, bi):
        return (ar * br - ai * bi, ar * bi + ai * br)

    def forward(self, x: TriadTensor) -> TriadTensor:

        if x.ndim == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])
        B, L, _ = x.shape
        ds, J = (self.d_state, self.n_memory)

        mag = sigmoid(self.log_decay)
        half = self.omega * 0.5

        custom_half = mag.data * np.exp(-1j * half.data)

        y = np.zeros((B, J, ds), dtype=np.float64)
        psi = np.zeros((B, ds), dtype=np.complex128)

        from runtime.core.solver import TriadParams
        from runtime.ml.solver_step import _build_ml_step_kernels, _solver_step_ml_batched
        base_p = TriadParams(N=ds, T=0.1, dt=0.05, mode='triad')
        from dataclasses import replace
        base_p = replace(base_p,
            nu=tuple(float(v) for v in sigmoid(self.nu).data),
            lam=tuple(float(v) for v in self.lam.data.reshape(-1)[:J]),
            Lambda=float(self.Lambda.data[0]) if self.Lambda.data.ndim else float(self.Lambda.data),
        )
        kernels = _build_ml_step_kernels(base_p)

        U_re = self.in_re(x).data
        U_im = self.in_im(x).data

        if self.coupling:
            k_val = float(self.kappa.data if np.ndim(self.kappa.data) == 0 else self.kappa.data[0])
            couple_w = self.couple.weight._data
        else:
            k_val = 0.0
            couple_w = None

        re_seq, im_seq = [], []
        for t in range(L):
            drive_re = U_re[:, t, :]
            drive_im = U_im[:, t, :]
            psi, y = _solver_step_ml_batched(
                psi, y, kernels,
                drive_re=drive_re, drive_im=drive_im,
                custom_half_lin=custom_half,
                rng=self._rng,
            )
            if k_val != 0.0 and couple_w is not None:
                coupled = (psi @ couple_w.T.conj())
                psi = psi + k_val * (coupled - psi)
            re_seq.append(np.real(psi))
            im_seq.append(np.imag(psi))
        RE = stack([TriadTensor(r) for r in re_seq], axis=1)
        IM = stack([TriadTensor(i) for i in im_seq], axis=1)
        return self.out_re(RE) + self.out_im(IM)

class TriadSSMBlock(Module):

    def __init__(self, d_model: int, d_state: int=64, d_ff: int | None=None, n_memory: int=3, coupling: bool=True, noise: float=0.0):
        self.ssm = TriadSSM(d_model, d_state=d_state, n_memory=n_memory, coupling=coupling, noise=noise)
        self.ln1 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff or 4 * d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x: TriadTensor) -> TriadTensor:
        x = x + self.ssm(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class SGD:

    def __init__(self, params: list[Parameter], lr: float=0.01, momentum: float=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p._data) for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            if p._grad is None:
                continue
            if self.momentum > 0:
                self.velocities[i] = self.momentum * self.velocities[i] - self.lr * p._grad
                p._data = p._data + self.velocities[i]
            else:
                p._data = p._data - self.lr * p._grad

    def zero_grad(self):
        for p in self.params:
            p._grad = None

class Adam:

    def __init__(self, params: list[Parameter], lr: float=0.001, betas: tuple=(0.9, 0.999), eps: float=1e-08):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p._data) for p in params]
        self.v = [np.zeros_like(p._data) for p in params]
        self.t = 0

    def step(self):
        from runtime.ml.gpu_kernel import fused_adam_step, fused_available
        from runtime.ml.tensor import _is_cuda_array
        self.t += 1
        for i, p in enumerate(self.params):
            if p._grad is None:
                continue
            if _is_cuda_array(p._data) and fused_available():

                if not _is_cuda_array(self.m[i]):
                    import cupy as _cp
                    self.m[i] = _cp.asarray(self.m[i])
                    self.v[i] = _cp.asarray(self.v[i])
                fused_adam_step(p._data, p._grad, self.m[i], self.v[i],
                                    self.lr, self.b1, self.b2, self.eps, self.t)
                continue
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * p._grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * p._grad ** 2
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p._data = p._data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p._grad = None

class AdamW:

    def __init__(self, params: list[Parameter], lr: float = 0.001,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p._data) for p in params]
        self.v = [np.zeros_like(p._data) for p in params]
        self.t = 0

    def step(self):
        from runtime.ml.gpu_kernel import fused_adamw_step, fused_available
        from runtime.ml.tensor import _is_cuda_array
        self.t += 1
        for i, p in enumerate(self.params):
            if p._grad is None:
                continue
            if _is_cuda_array(p._data) and fused_available():
                if not _is_cuda_array(self.m[i]):
                    import cupy as _cp
                    self.m[i] = _cp.asarray(self.m[i])
                    self.v[i] = _cp.asarray(self.v[i])
                fused_adamw_step(p._data, p._grad, self.m[i], self.v[i],
                                     self.lr, self.b1, self.b2, self.eps,
                                     self.t, self.weight_decay)
                continue
            grad = p._grad + self.weight_decay * p._data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * grad ** 2
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p._data = p._data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p._grad = None

class GELU(Module):

    def forward(self, x: TriadTensor) -> TriadTensor:
        cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x._data + 0.044715 * x._data ** 3)))
        out = TriadTensor(x._data * cdf)
        if x._requires_grad:
            out._requires_grad = True
            out._children = [x]

            def _back(g):
                d = cdf + x._data * 0.5 * (1 - np.tanh(
                    np.sqrt(2.0 / np.pi) * (x._data + 0.044715 * x._data ** 3)) ** 2) * np.sqrt(
                    2.0 / np.pi) * (1 + 0.134145 * x._data ** 2)
                sg = g * d
                x._grad = sg if x._grad is None else x._grad + sg
            out._grad_fn = _back
        return out

class SiLU(Module):

    def forward(self, x: TriadTensor) -> TriadTensor:
        s = 1.0 / (1.0 + np.exp(-x._data))
        out = TriadTensor(x._data * s)
        if x._requires_grad:
            out._requires_grad = True
            out._children = [x]

            def _back(g):
                sg = g * (s + x._data * s * (1 - s))
                x._grad = sg if x._grad is None else x._grad + sg
            out._grad_fn = _back
        return out

class LeakyReLU(Module):

    def __init__(self, negative_slope: float = 0.01):
        self.negative_slope = negative_slope

    def forward(self, x: TriadTensor) -> TriadTensor:
        out = TriadTensor(np.where(x._data > 0, x._data, self.negative_slope * x._data))
        if x._requires_grad:
            out._requires_grad = True
            out._children = [x]
            ns = self.negative_slope

            def _back(g):
                sg = g * np.where(x._data > 0, 1.0, ns)
                x._grad = sg if x._grad is None else x._grad + sg
            out._grad_fn = _back
        return out

class ELU(Module):

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def forward(self, x: TriadTensor) -> TriadTensor:
        out_data = np.where(x._data > 0, x._data, self.alpha * (np.exp(x._data) - 1))
        out = TriadTensor(out_data)
        if x._requires_grad:
            out._requires_grad = True
            out._children = [x]
            a = self.alpha

            def _back(g):
                sg = g * np.where(x._data > 0, 1.0, out_data + a)
                x._grad = sg if x._grad is None else x._grad + sg
            out._grad_fn = _back
        return out

class MaxPool1d(Module):

    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 0):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x: TriadTensor) -> TriadTensor:
        x_np = x._data
        if x_np.ndim == 2:
            x_np = x_np[np.newaxis, :, :]
        B, C, L = x_np.shape
        if self.padding > 0:
            x_np = np.pad(x_np, ((0, 0), (0, 0), (self.padding, self.padding)),
                          constant_values=-np.inf)
        _, _, Lp = x_np.shape
        out_len = (Lp - self.kernel_size) // self.stride + 1
        out_np = np.triad((B, C, out_len), -np.inf)
        for i in range(out_len):
            s = i * self.stride
            window = x_np[:, :, s:s + self.kernel_size]
            out_np[:, :, i] = np.max(window, axis=-1)
        out = TriadTensor(out_np.squeeze(0) if x._data.ndim == 2 else out_np)
        if x._requires_grad:
            out._requires_grad = True
            out._children = [x]
            _x_np = x_np
            _ks = self.kernel_size
            _st = self.stride

            def _back(g):
                if g.ndim == 2:
                    g = g[np.newaxis, :, :]
                dx = np.zeros_like(_x_np)
                for i in range(out_len):
                    s = i * _st
                    window = _x_np[:, :, s:s + _ks]
                    max_val = np.max(window, axis=-1, keepdims=True)
                    mask = (window == max_val)
                    dx[:, :, s:s + _ks] += mask * g[:, :, i:i + 1]
                if self.padding > 0:
                    dx = dx[:, :, self.padding:-self.padding]
                if x._data.ndim == 2:
                    dx = dx.squeeze(0)
                x._grad = dx if x._grad is None else x._grad + dx
            out._grad_fn = _back
        return out

class MaxPool2d(Module):

    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 0):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x: TriadTensor) -> TriadTensor:
        x_np = x._data
        if x_np.ndim == 3:
            x_np = x_np[np.newaxis, :, :, :]
        B, C, H, W = x_np.shape
        p = self.padding
        if p > 0:
            x_np = np.pad(x_np, ((0, 0), (0, 0), (p, p), (p, p)),
                          constant_values=-np.inf)
        _, _, Hp, Wp = x_np.shape
        ks = self.kernel_size
        s = self.stride
        H_out = (Hp - ks) // s + 1
        W_out = (Wp - ks) // s + 1
        out_np = np.triad((B, C, H_out, W_out), -np.inf)
        for i in range(H_out):
            for j in range(W_out):
                rs = i * s
                cs = j * s
                out_np[:, :, i, j] = np.max(x_np[:, :, rs:rs + ks, cs:cs + ks],
                                             axis=(-2, -1))
        out = TriadTensor(out_np.squeeze(0) if x._data.ndim == 3 else out_np)
        if x._requires_grad:
            out._requires_grad = True
            out._children = [x]
            _x_np = x_np

            def _back(g):
                if g.ndim == 3:
                    g = g[np.newaxis, :, :, :]
                dx = np.zeros_like(_x_np)
                for i in range(H_out):
                    for j in range(W_out):
                        rs = i * s
                        cs = j * s
                        window = _x_np[:, :, rs:rs + ks, cs:cs + ks]
                        max_val = np.max(window, axis=(-2, -1), keepdims=True)
                        mask = (window == max_val)
                        dx[:, :, rs:rs + ks, cs:cs + ks] += mask * g[:, :, i:i + 1, j:j + 1]
                if p > 0:
                    dx = dx[:, :, p:-p, p:-p]
                if x._data.ndim == 3:
                    dx = dx.squeeze(0)
                x._grad = dx if x._grad is None else x._grad + dx
            out._grad_fn = _back
        return out

class AvgPool1d(Module):

    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 0):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x: TriadTensor) -> TriadTensor:
        x_np = x._data
        if x_np.ndim == 2:
            x_np = x_np[np.newaxis, :, :]
        B, C, L = x_np.shape
        if self.padding > 0:
            x_np = np.pad(x_np, ((0, 0), (0, 0), (self.padding, self.padding)))
        _, _, Lp = x_np.shape
        out_len = (Lp - self.kernel_size) // self.stride + 1
        out_np = np.zeros((B, C, out_len))
        for i in range(out_len):
            s = i * self.stride
            out_np[:, :, i] = np.mean(x_np[:, :, s:s + self.kernel_size], axis=-1)
        out = TriadTensor(out_np.squeeze(0) if x._data.ndim == 2 else out_np)
        if x._requires_grad:
            out._requires_grad = True
            out._children = [x]
            ks = self.kernel_size

            def _back(g):
                if g.ndim == 2:
                    g = g[np.newaxis, :, :]
                scale = 1.0 / ks
                dx = np.zeros_like(x_np)
                for i in range(out_len):
                    s = i * self.stride
                    dx[:, :, s:s + ks] += g[:, :, i:i + 1] * scale
                if self.padding > 0:
                    dx = dx[:, :, self.padding:-self.padding]
                if x._data.ndim == 2:
                    dx = dx.squeeze(0)
                x._grad = dx if x._grad is None else x._grad + dx
            out._grad_fn = _back
        return out

class AvgPool2d(Module):

    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 0):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x: TriadTensor) -> TriadTensor:
        x_np = x._data
        if x_np.ndim == 3:
            x_np = x_np[np.newaxis, :, :, :]
        B, C, H, W = x_np.shape
        p = self.padding
        if p > 0:
            x_np = np.pad(x_np, ((0, 0), (0, 0), (p, p), (p, p)))
        _, _, Hp, Wp = x_np.shape
        ks = self.kernel_size
        s = self.stride
        H_out = (Hp - ks) // s + 1
        W_out = (Wp - ks) // s + 1
        out_np = np.zeros((B, C, H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                rs = i * s
                cs = j * s
                out_np[:, :, i, j] = np.mean(x_np[:, :, rs:rs + ks, cs:cs + ks],
                                              axis=(-2, -1))
        out = TriadTensor(out_np.squeeze(0) if x._data.ndim == 3 else out_np)
        if x._requires_grad:
            out._requires_grad = True
            out._children = [x]
            scale = 1.0 / (ks * ks)

            def _back(g):
                if g.ndim == 3:
                    g = g[np.newaxis, :, :, :]
                dx = np.zeros_like(x_np)
                for i in range(H_out):
                    for j in range(W_out):
                        rs = i * s
                        cs = j * s
                        dx[:, :, rs:rs + ks, cs:cs + ks] += g[:, :, i:i + 1, j:j + 1] * scale
                if p > 0:
                    dx = dx[:, :, p:-p, p:-p]
                if x._data.ndim == 3:
                    dx = dx.squeeze(0)
                x._grad = dx if x._grad is None else x._grad + dx
            out._grad_fn = _back
        return out

class AdaptiveAvgPool1d(Module):

    def __init__(self, output_size: int):
        self.output_size = output_size

    def forward(self, x: TriadTensor) -> TriadTensor:
        x_np = x._data
        if x_np.ndim == 2:
            x_np = x_np[np.newaxis, :, :]
        B, C, L = x_np.shape
        O = self.output_size
        out_np = np.zeros((B, C, O))
        for i in range(O):
            start = int(i * L / O)
            end = int((i + 1) * L / O)
            out_np[:, :, i] = np.mean(x_np[:, :, start:end], axis=-1)
        out = TriadTensor(out_np.squeeze(0) if x._data.ndim == 2 else out_np)
        if x._requires_grad:
            out._requires_grad = True
            out._children = [x]

            def _back(g):
                if g.ndim == 2:
                    g = g[np.newaxis, :, :]
                dx = np.zeros_like(x_np)
                for i in range(O):
                    start = int(i * L / O)
                    end = int((i + 1) * L / O)
                    seg = end - start
                    dx[:, :, start:end] += g[:, :, i:i + 1] / seg
                if x._data.ndim == 2:
                    dx = dx.squeeze(0)
                x._grad = dx if x._grad is None else x._grad + dx
            out._grad_fn = _back
        return out

class GlobalAvgPool(Module):

    def forward(self, x: TriadTensor) -> TriadTensor:
        if x._data.ndim <= 2:
            return x.mean(axis=-1)
        for ax in range(x._data.ndim - 1, 1, -1):
            x = x.mean(axis=ax)
        return x

class GroupNorm(Module):

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.gamma = Parameter(tensor(np.ones(num_channels)))
        self.beta = Parameter(tensor(np.zeros(num_channels)))

    def forward(self, x: TriadTensor) -> TriadTensor:
        x_np = x._data
        orig_shape = x_np.shape
        squeeze = False
        if x_np.ndim == 1:
            x_np = x_np[np.newaxis, :, np.newaxis]
            squeeze = True
        elif x_np.ndim == 2:
            x_np = x_np[:, :, np.newaxis]
            squeeze = True
        N, C, *spatial = x_np.shape
        G = self.num_groups
        x_grouped = x_np.reshape(N, G, C // G, *spatial)
        axes = tuple(range(2, x_grouped.ndim))
        mean = x_grouped.mean(axis=axes, keepdims=True)
        var = x_grouped.var(axis=axes, keepdims=True)
        x_norm = (x_grouped - mean) / np.sqrt(var + self.eps)
        x_norm = x_norm.reshape(N, C, *spatial)
        g = self.gamma._data.reshape(1, -1, *([1] * len(spatial)))
        b = self.beta._data.reshape(1, -1, *([1] * len(spatial)))
        out_data = x_norm * g + b
        if squeeze:
            out_data = out_data.squeeze(-1)
            if x._data.ndim == 1:
                out_data = out_data.squeeze(0)
        out = TriadTensor(out_data)
        if x._requires_grad or self.gamma._requires_grad or self.beta._requires_grad:
            out._requires_grad = True
            out._children = [x, self.gamma, self.beta]

            def _back(g_arr):
                if self.beta._requires_grad:
                    red_axes = tuple(range(g_arr.ndim))[1:]
                    bg = g_arr.sum(axis=red_axes) if red_axes else g_arr.copy()
                    self.beta._grad = bg if self.beta._grad is None else self.beta._grad + bg
                if self.gamma._requires_grad:
                    red_axes = tuple(range(g_arr.ndim))[1:]
                    gg = (g_arr * out_data).sum(axis=red_axes)
                    self.gamma._grad = gg if self.gamma._grad is None else self.gamma._grad + gg
                if x._requires_grad:
                    x._grad = g_arr if x._grad is None else x._grad + g_arr
            out._grad_fn = _back
        return out

class RNN(Module):

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 triad: str = 'tanh'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.triad = triad
        k = 1.0 / np.sqrt(hidden_size)
        self.weight_ih = Parameter(tensor(np.random.uniform(-k, k, (num_layers, input_size, hidden_size))))
        self.weight_hh = Parameter(tensor(np.random.uniform(-k, k, (num_layers, hidden_size, hidden_size))))
        self.bias_ih = Parameter(tensor(np.zeros((num_layers, hidden_size))))
        self.bias_hh = Parameter(tensor(np.zeros((num_layers, hidden_size))))

    def forward(self, x: TriadTensor, h0: TriadTensor | None = None) -> tuple[TriadTensor, TriadTensor]:
        x_np = x._data
        if x_np.ndim == 2:
            x_np = x_np[np.newaxis, :, :]
        B, T, _ = x_np.shape
        H = self.hidden_size
        h = h0._data.copy() if h0 is not None else np.zeros((self.num_layers, B, H))
        outputs = []
        for t in range(T):
            for layer in range(self.num_layers):
                inp = x_np[:, t, :] if layer == 0 else h[layer - 1]
                pre = inp @ self.weight_ih._data[layer] + h[layer] @ self.weight_hh._data[layer] + self.bias_ih._data[layer] + self.bias_hh._data[layer]
                if self.triad == 'tanh':
                    h[layer] = np.tanh(pre)
                else:
                    h[layer] = 1.0 / (1.0 + np.exp(-pre))
            outputs.append(h[-1].copy())
        out = TriadTensor(np.stack(outputs, axis=1))
        h_out = TriadTensor(h)
        if x._requires_grad:
            out._requires_grad = True
            out._children = [x]
        return out, h_out

class LSTM(Module):

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        k = 1.0 / np.sqrt(hidden_size)
        self.weight_ih = Parameter(tensor(np.random.uniform(-k, k, (num_layers, input_size, 4 * hidden_size))))
        self.weight_hh = Parameter(tensor(np.random.uniform(-k, k, (num_layers, hidden_size, 4 * hidden_size))))
        self.bias_ih = Parameter(tensor(np.zeros((num_layers, 4 * hidden_size))))
        self.bias_hh = Parameter(tensor(np.zeros((num_layers, 4 * hidden_size))))

    def forward(self, x: TriadTensor, h0: TriadTensor | None = None,
                c0: TriadTensor | None = None) -> tuple[TriadTensor, TriadTensor, TriadTensor]:
        x_np = x._data
        if x_np.ndim == 2:
            x_np = x_np[np.newaxis, :, :]
        B, T, _ = x_np.shape
        H = self.hidden_size
        h = h0._data.copy() if h0 is not None else np.zeros((self.num_layers, B, H))
        c = c0._data.copy() if c0 is not None else np.zeros((self.num_layers, B, H))
        outputs = []
        for t in range(T):
            for layer in range(self.num_layers):
                inp = x_np[:, t, :] if layer == 0 else h[layer - 1]
                gates = inp @ self.weight_ih._data[layer] + h[layer] @ self.weight_hh._data[layer] + self.bias_ih._data[layer] + self.bias_hh._data[layer]
                i_gate = 1.0 / (1.0 + np.exp(-gates[:, :H]))
                f_gate = 1.0 / (1.0 + np.exp(-gates[:, H:2 * H]))
                g_gate = np.tanh(gates[:, 2 * H:3 * H])
                o_gate = 1.0 / (1.0 + np.exp(-gates[:, 3 * H:4 * H]))
                c[layer] = f_gate * c[layer] + i_gate * g_gate
                h[layer] = o_gate * np.tanh(c[layer])
            outputs.append(h[-1].copy())
        out = TriadTensor(np.stack(outputs, axis=1))
        h_out = TriadTensor(h)
        c_out = TriadTensor(c)
        if x._requires_grad:
            out._requires_grad = True
            out._children = [x]
        return out, h_out, c_out

class GRU(Module):

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        k = 1.0 / np.sqrt(hidden_size)
        self.weight_ih = Parameter(tensor(np.random.uniform(-k, k, (num_layers, input_size, 3 * hidden_size))))
        self.weight_hh = Parameter(tensor(np.random.uniform(-k, k, (num_layers, hidden_size, 3 * hidden_size))))
        self.bias_ih = Parameter(tensor(np.zeros((num_layers, 3 * hidden_size))))
        self.bias_hh = Parameter(tensor(np.zeros((num_layers, 3 * hidden_size))))

    def forward(self, x: TriadTensor, h0: TriadTensor | None = None) -> tuple[TriadTensor, TriadTensor]:
        x_np = x._data
        if x_np.ndim == 2:
            x_np = x_np[np.newaxis, :, :]
        B, T, _ = x_np.shape
        H = self.hidden_size
        h = h0._data.copy() if h0 is not None else np.zeros((self.num_layers, B, H))
        outputs = []
        for t in range(T):
            for layer in range(self.num_layers):
                inp = x_np[:, t, :] if layer == 0 else h[layer - 1]
                gi = inp @ self.weight_ih._data[layer] + self.bias_ih._data[layer]
                gh = h[layer] @ self.weight_hh._data[layer] + self.bias_hh._data[layer]
                r_gate = 1.0 / (1.0 + np.exp(-(gi[:, :H] + gh[:, :H])))
                z_gate = 1.0 / (1.0 + np.exp(-(gi[:, H:2 * H] + gh[:, H:2 * H])))
                n_gate = np.tanh(gi[:, 2 * H:3 * H] + r_gate * gh[:, 2 * H:3 * H])
                h[layer] = (1 - z_gate) * n_gate + z_gate * h[layer]
            outputs.append(h[-1].copy())
        out = TriadTensor(np.stack(outputs, axis=1))
        h_out = TriadTensor(h)
        if x._requires_grad:
            out._requires_grad = True
            out._children = [x]
        return out, h_out

class ResidualBlock(Module):

    def __init__(self, *path_layers):
        self.path = Sequential(*path_layers)

    def forward(self, x: TriadTensor) -> TriadTensor:
        return x + self.path(x)

    def parameters(self) -> list[Parameter]:
        return self.path.parameters()

class RMSNorm(Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.gamma = Parameter(tensor(np.ones(dim)))

    def forward(self, x: TriadTensor) -> TriadTensor:
        rms = np.sqrt(np.mean(x._data ** 2, axis=-1, keepdims=True) + self.eps)
        x_norm = x._data / rms
        out = TriadTensor(x_norm * self.gamma._data)
        if x._requires_grad or self.gamma._requires_grad:
            out._requires_grad = True
            out._children = [x, self.gamma]
            _x_norm = x_norm
            _rms = rms

            def _back(g):
                if self.gamma._requires_grad:
                    gg = (g * _x_norm).sum(axis=tuple(range(g.ndim - 1)))
                    self.gamma._grad = gg if self.gamma._grad is None else self.gamma._grad + gg
                if x._requires_grad:
                    D = x._data.shape[-1]
                    dx = self.gamma._data * (g / _rms - _x_norm * (g * _x_norm * self.gamma._data).sum(axis=-1, keepdims=True) / (_rms * D))
                    x._grad = dx if x._grad is None else x._grad + dx
            out._grad_fn = _back
        return out

class CrossAttention(Module):

    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = triad(d_model, d_model)
        self.k_proj = triad(d_model, d_model)
        self.v_proj = triad(d_model, d_model)
        self.out_proj = triad(d_model, d_model)

    def forward(self, x: TriadTensor, context: TriadTensor) -> TriadTensor:
        B, Sq, _ = x.shape
        _, Sc, _ = context.shape
        q = self.q_proj(x).reshape(B, Sq, self.n_heads, self.d_head)
        k = self.k_proj(context).reshape(B, Sc, self.n_heads, self.d_head)
        v = self.v_proj(context).reshape(B, Sc, self.n_heads, self.d_head)
        q_t = q._data.transpose(0, 2, 1, 3)
        k_t = k._data.transpose(0, 2, 1, 3)
        v_t = v._data.transpose(0, 2, 1, 3)
        scores_data = q_t @ k_t.transpose(0, 1, 3, 2) * (1.0 / np.sqrt(self.d_head))
        scores = TriadTensor(scores_data)
        attn = softmax(scores, axis=-1)
        ctx_data = attn._data @ v_t
        ctx = TriadTensor(ctx_data.transpose(0, 2, 1, 3).reshape(B, Sq, self.d_model))
        return self.out_proj(ctx)

class TransformerDecoderBlock(Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, causal: bool = True):
        self.self_attn = MultiHeadAttention(d_model, n_heads, causal=causal)
        self.cross_attn = CrossAttention(d_model, n_heads)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.ln3 = LayerNorm(d_model)

    def forward(self, x: TriadTensor, context: TriadTensor) -> TriadTensor:
        x = x + self.self_attn(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), context)
        x = x + self.ff(self.ln3(x))
        return x

def xavier_uniform(weight: Parameter):
    fan_in, fan_out = weight._data.shape[:2]
    bound = np.sqrt(6.0 / (fan_in + fan_out))
    weight._data = np.random.uniform(-bound, bound, weight._data.shape)

def xavier_normal(weight: Parameter):
    fan_in, fan_out = weight._data.shape[:2]
    std = np.sqrt(2.0 / (fan_in + fan_out))
    weight._data = np.random.normal(0, std, weight._data.shape)

def kaiming_uniform(weight: Parameter, triad: str = 'relu'):
    fan_in = weight._data.shape[0]
    gain = 2.0 if triad == 'relu' else 1.0
    bound = gain * np.sqrt(3.0 / fan_in)
    weight._data = np.random.uniform(-bound, bound, weight._data.shape)

def kaiming_normal(weight: Parameter, triad: str = 'relu'):
    fan_in = weight._data.shape[0]
    gain = 2.0 if triad == 'relu' else 1.0
    std = gain / np.sqrt(fan_in)
    weight._data = np.random.normal(0, std, weight._data.shape)

def rope_apply(x: TriadTensor, positions, base: float = 10000.0) -> TriadTensor:

    from runtime.ml.tensor import cat as _cat
    xp = np
    data = x.data
    try:
        import cupy as _cp
        if isinstance(data, _cp.ndarray):
            xp = _cp
    except (ImportError, ModuleNotFoundError):
        pass
    head_dim = data.shape[-1]
    half = head_dim // 2
    freqs = 1.0 / (base ** (xp.arange(0, head_dim, 2, dtype=xp.float64) / head_dim))
    theta = xp.asarray(positions)[:, None].astype(xp.float64) * freqs[None, :]

    cos_t = xp.cos(theta).astype(xp.float64)[None, :, None, :]
    sin_t = xp.sin(theta).astype(xp.float64)[None, :, None, :]
    cos_f = TriadTensor(xp.concatenate([cos_t, cos_t], axis=-1))
    sin_f = TriadTensor(xp.concatenate([sin_t, sin_t], axis=-1))
    x1 = _slice_lastdim(x, 0, half)
    x2 = _slice_lastdim(x, half, head_dim)
    rotated = _cat([-x2, x1], axis=-1)
    return x * cos_f + rotated * sin_f

def _slice_lastdim(t: TriadTensor, a: int, b: int) -> TriadTensor:

    out = TriadTensor(t.data[..., a:b])
    import importlib
    _T = importlib.import_module('runtime.ml.tensor')
    if _T._ENABLE_GRAD and t.requires_grad:
        out._requires_grad = True
        out._children = [t]

        def _back(g):
            sg = np.zeros_like(t.data)
            sg[..., a:b] = g
            t._grad = sg if t._grad is None else t._grad + sg
        out._grad_fn = _back
    return out

class KVCache:

    def __init__(self):
        self.k = None
        self.v = None

    def append(self, k, v):
        if self.k is None:
            self.k, self.v = k, v
        else:
            xp = np
            try:
                import cupy as _cp
                if isinstance(k, _cp.ndarray):
                    xp = _cp
            except (ImportError, ModuleNotFoundError):
                pass
            self.k = xp.concatenate([self.k, k], axis=1)
            self.v = xp.concatenate([self.v, v], axis=1)
        return self.k, self.v

    @property
    def seq_len(self):
        return 0 if self.k is None else self.k.shape[1]

class CausalSelfAttention(Module):

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int = None,
                 rope_base: float = 10000.0):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv = n_kv_heads or n_heads
        assert n_heads % self.n_kv == 0
        self.head_dim = d_model // n_heads
        self.rope_base = rope_base
        self.q_proj = triad(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = triad(d_model, self.n_kv * self.head_dim, bias=False)
        self.v_proj = triad(d_model, self.n_kv * self.head_dim, bias=False)
        self.o_proj = triad(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x: TriadTensor, cache: KVCache = None) -> TriadTensor:
        B, T, D = x.shape
        past = cache.seq_len if cache is not None else 0
        positions = np.arange(past, past + T)

        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, T, self.n_kv, self.head_dim)
        v = self.v_proj(x).reshape(B, T, self.n_kv, self.head_dim)
        q = rope_apply(q, positions, self.rope_base)
        k = rope_apply(k, positions, self.rope_base)

        if cache is not None:
            k_triad_d, v_triad_d = cache.append(k.data, v.data)
            k_triad = TriadTensor(k_triad_d)
            v_triad = TriadTensor(v_triad_d)
        else:
            k_triad, v_triad = k, v
        total = k_triad.shape[1]

        n_rep = self.n_heads // self.n_kv
        if n_rep > 1:
            kr = TriadTensor(np.repeat(k_triad.data, n_rep, axis=2))
            vr = TriadTensor(np.repeat(v_triad.data, n_rep, axis=2))
            if k_triad.requires_grad:
                kr._requires_grad = True
                kr._children = [k_triad]
                _nk = self.n_kv

                def _back_k(g, _t=k_triad, _r=n_rep, _n=_nk):
                    gg = g.reshape(g.shape[0], g.shape[1], _n, _r, g.shape[-1]).sum(axis=3)
                    _t._grad = gg if _t._grad is None else _t._grad + gg
                kr._grad_fn = _back_k
            if v_triad.requires_grad:
                vr._requires_grad = True
                vr._children = [v_triad]
                _nk2 = self.n_kv

                def _back_v(g, _t=v_triad, _r=n_rep, _n=_nk2):
                    gg = g.reshape(g.shape[0], g.shape[1], _n, _r, g.shape[-1]).sum(axis=3)
                    _t._grad = gg if _t._grad is None else _t._grad + gg
                vr._grad_fn = _back_v
        else:
            kr, vr = k_triad, v_triad

        qh = q.permute(0, 2, 1, 3)
        kh = kr.permute(0, 2, 1, 3)
        vh = vr.permute(0, 2, 1, 3)

        if cache is None:

            from runtime.ml.flash_attention import flash_causal_attention
            out = flash_causal_attention(qh, kh, vh)
            out = out.permute(0, 2, 1, 3).reshape(B, T, self.n_heads * self.head_dim)
            return self.o_proj(out)

        scores = bmm(qh, kh.permute(0, 1, 3, 2)) * (self.head_dim ** -0.5)

        xp = np
        try:
            import cupy as _cp
            if isinstance(scores.data, _cp.ndarray):
                xp = _cp
        except (ImportError, ModuleNotFoundError):
            pass
        col = xp.arange(total)[None, :]
        row = xp.arange(T)[:, None]
        start = total - T

        mask = xp.where(col > start + row, -1e9, 0.0).astype(xp.float32)
        scores = scores + TriadTensor(mask[None, None])
        probs = softmax(scores, axis=-1)
        out = bmm(probs, vh)
        out = out.permute(0, 2, 1, 3).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out)

