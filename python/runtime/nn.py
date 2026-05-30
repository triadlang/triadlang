from __future__ import annotations
from runtime.ml_device import xp as np
from runtime import ml_device as _D
from runtime.tensor import TriadTensor, tensor, randn, zeros, relu, tanh, sigmoid, softmax, _ensure_tensor, bmm, transpose, layer_norm, sin, cos, stack

class Parameter(TriadTensor):

    def __init__(self, data):
        if isinstance(data, TriadTensor):
            data = data._data
        super().__init__(data, requires_grad=True)

class Module:

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: TriadTensor) -> TriadTensor:
        raise NotImplementedError

    def parameters(self) -> list[Parameter]:
        params = []
        for v in self.__dict__.values():
            if isinstance(v, Parameter):
                params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Parameter):
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p._grad = None

class Linear(Module):

    def __init__(self, in_features: int, out_features: int, bias: bool=True):
        k = 1.0 / np.sqrt(in_features)
        self.weight = Parameter(tensor(np.random.uniform(-k, k, (in_features, out_features))))
        self.bias = Parameter(tensor(np.random.uniform(-k, k, (out_features,)))) if bias else None

    def forward(self, x: TriadTensor) -> TriadTensor:
        out = x @ self.weight if x.ndim == 2 else bmm(x, self.weight)
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
        mask = (np.random.rand(*x.shape) > self.p).astype(_D.fdtype()) / (1 - self.p)
        return TriadTensor(x._data * mask, requires_grad=x._requires_grad)

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
                np.add.at(wg, ids, g)
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
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)

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

                def _back(g):
                    sg = g * keep
                    scores._grad = sg if scores._grad is None else scores._grad + sg
                masked._grad_fn = _back
            scores = masked
        attn = softmax(scores, axis=-1)
        ctx = bmm(attn, v)
        ctx = transpose(ctx, 1, 2).reshape(B, Tn, self.d_model)
        return self.out_proj(ctx)

class FeedForward(Module):

    def __init__(self, d_model: int, d_ff: int):
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)

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
        self.head = Linear(d_model, vocab_size, bias=False)

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

    def __init__(self, d_model: int, d_state: int=64, n_memory: int=3, coupling: bool=True, noise: float=0.0):
        self.d_model = d_model
        self.d_state = d_state
        self.n_memory = n_memory
        self.noise = noise
        self.coupling = coupling
        self.in_re = Linear(d_model, d_state)
        self.in_im = Linear(d_model, d_state)
        self.out_re = Linear(d_state, d_model)
        self.out_im = Linear(d_state, d_model)
        self.omega = Parameter(tensor(np.linspace(0.05, float(np.pi), d_state)))
        self.log_decay = Parameter(tensor(np.full(d_state, 3.5)))
        self.nu = Parameter(tensor(np.linspace(-1.0, 2.0, n_memory)))
        self.lam = Parameter(tensor(np.full((n_memory, d_state), -0.01)))
        self.Lambda = Parameter(tensor(np.full(d_state, -0.02)))
        if coupling:
            self.couple = Linear(d_state, d_state, bias=False)
            self.kappa = Parameter(tensor(np.array(0.0)))
        self.training = True

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
        Ah_re = mag * cos(half)
        Ah_im = mag * sin(half) * -1.0
        dmem3 = sigmoid(self.nu).reshape(1, J, 1)
        keep3 = 1.0 - dmem3
        lam3 = self.lam.reshape(1, J, ds)
        U_re = self.in_re(x)
        U_im = self.in_im(x)
        re = zeros(B, ds)
        im = zeros(B, ds)
        y = zeros(B, J, ds)
        re_seq, im_seq = ([], [])
        for t in range(L):
            re, im = self._cmul(Ah_re, Ah_im, re, im)
            re = re + U_re[:, t, :]
            im = im + U_im[:, t, :]
            rho = re * re + im * im
            y = dmem3 * y + keep3 * rho.reshape(B, 1, ds)
            V = self.Lambda * rho + (lam3 * y).sum(axis=1)
            if self.coupling:
                V = V + self.kappa * self.couple(rho)
            cV, sV = (cos(V), sin(V))
            re, im = (re * cV + im * sV, im * cV - re * sV)
            rho = re * re + im * im
            y = dmem3 * y + keep3 * rho.reshape(B, 1, ds)
            if self.training and self.noise > 0.0:
                re = re + randn(B, ds) * self.noise
                im = im + randn(B, ds) * self.noise
            re, im = self._cmul(Ah_re, Ah_im, re, im)
            re_seq.append(re)
            im_seq.append(im)
        RE = stack(re_seq, axis=1)
        IM = stack(im_seq, axis=1)
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
        self.t += 1
        for i, p in enumerate(self.params):
            if p._grad is None:
                continue
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * p._grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * p._grad ** 2
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p._data = p._data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p._grad = None