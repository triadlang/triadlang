
from __future__ import annotations

from triad import ntri as np

try:
    import cupy as _cp
except (ImportError, ModuleNotFoundError):
    _cp = None

from runtime.ml.gguf import GGUFFile, GGUFTokenizer


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

class NativeQwen35:

    def __init__(self, path: str, device: str = 'cuda', dtype='float16',
                 stream: bool = False):
        self.f = GGUFFile(path)
        kv = self.f.kv
        a = kv['general.architecture']
        assert a == 'qwen35', f'arquitetura {a} nao e qwen35'
        self.xp = _cp if (device == 'cuda' and _cp is not None) else np
        self.dtype = dtype
        self.stream = stream

        self.n_layers = int(kv[f'{a}.block_count'])
        self.d_model = int(kv[f'{a}.embedding_length'])
        self.interval = int(kv.get(f'{a}.triad_attention_interval', 4))
        self.eps = float(kv.get(f'{a}.attention.layer_norm_rms_epsilon', 1e-6))

        self.n_heads = int(kv[f'{a}.attention.head_count'])
        self.n_kv = int(kv[f'{a}.attention.head_count_kv'])
        self.head_dim = int(kv.get(f'{a}.attention.key_length',
                                   self.d_model // self.n_heads))
        self.rope_dims = int(kv.get(f'{a}.rope.dimension_count', 64))
        self.rope_base = float(kv.get(f'{a}.rope.freq_base', 10000.0))

        self.dk = int(kv[f'{a}.ssm.state_size'])
        self.n_k = int(kv[f'{a}.ssm.group_count'])
        self.d_inner = int(kv[f'{a}.ssm.inner_size'])
        self.n_v = self.d_inner // self.dk
        self.conv_k = int(kv.get(f'{a}.ssm.conv_kernel', 4))
        self.vocab = int(self.f.tensors['token_embd.weight']['ne'][1])

        self.tokenizer = GGUFTokenizer(kv)
        self._tied = 'output.weight' not in self.f.tensors
        self._head_name = ('token_embd.weight' if self._tied
                           else 'output.weight')

        self.output_norm = self._dev(self.f.get('output_norm.weight'))
        if not stream:
            self._layers = [self._load_layer(i) for i in range(self.n_layers)]
            self.embed = self._dev(self.f.get('token_embd.weight'))
            self.head_w = (self.embed if self._tied
                           else self._dev(self.f.get('output.weight')))
        else:
            self._layers = None
            self.embed = None
            self.head_w = None

    def _dev(self, arr, fp32=False):
        a = arr if fp32 else arr.astype(self.dtype)
        return self.xp.asarray(a)

    def is_attn(self, i: int) -> bool:
        return (i + 1) % self.interval == 0

    def _load_layer(self, i: int) -> dict:
        f, g = self.f, lambda n, **k: self._dev(self.f.get(f'blk.{i}.{n}'), **k)
        L = {'attn_norm': g('attn_norm.weight', fp32=True),
             'post_norm': g('post_attention_norm.weight', fp32=True),
             'ffn_gate': g('ffn_gate.weight'),
             'ffn_up': g('ffn_up.weight'),
             'ffn_down': g('ffn_down.weight')}
        if self.is_attn(i):
            L.update(q=g('attn_q.weight'), k=g('attn_k.weight'),
                     v=g('attn_v.weight'), o=g('attn_output.weight'),
                     q_norm=g('attn_q_norm.weight', fp32=True),
                     k_norm=g('attn_k_norm.weight', fp32=True))
        else:
            L.update(qkv=g('attn_qkv.weight'), z=g('attn_gate.weight'),
                     alpha=g('ssm_alpha.weight'), beta=g('ssm_beta.weight'),
                     conv=g('ssm_conv1d.weight', fp32=True),
                     A_log=g('ssm_a', fp32=True),
                     dt_bias=g('ssm_dt.bias', fp32=True),
                     ssm_norm=g('ssm_norm.weight', fp32=True),
                     out=g('ssm_out.weight'))
        return L

    def _layer(self, i: int) -> dict:
        return self._layers[i] if not self.stream else self._load_layer(i)

    def _rms(self, x, w):
        xp = self.xp
        x32 = x.astype(xp.float32)
        return ((x32 / xp.sqrt((x32 * x32).mean(-1, keepdims=True) + self.eps))
                * w.astype(xp.float32)).astype(x.dtype)

    def _rope(self, x, positions):

        xp = self.xp
        r = self.rope_dims
        xr = x[..., :r].astype(xp.float32)
        freqs = 1.0 / (self.rope_base
                       ** (xp.arange(0, r, 2, dtype=xp.float32) / r))
        th = positions[:, None].astype(xp.float32) * freqs[None]
        cos = xp.concatenate([xp.cos(th), xp.cos(th)], -1)[:, None, :]
        sin = xp.concatenate([xp.sin(th), xp.sin(th)], -1)[:, None, :]
        h = r // 2
        rot = xp.concatenate([-xr[..., h:], xr[..., :h]], -1)
        out = (xr * cos + rot * sin).astype(x.dtype)
        return xp.concatenate([out, x[..., r:]], -1)

    def _attn_block(self, x, L, positions, cache):
        xp = self.xp
        T = x.shape[0]
        h = self._rms(x, L['attn_norm'])
        qg = (h @ L['q'].T).reshape(T, self.n_heads, 2 * self.head_dim)
        q, gate = qg[..., :self.head_dim], qg[..., self.head_dim:]
        k = (h @ L['k'].T).reshape(T, self.n_kv, self.head_dim)
        v = (h @ L['v'].T).reshape(T, self.n_kv, self.head_dim)
        q = self._rms(q, L['q_norm'])
        k = self._rms(k, L['k_norm'])
        q = self._rope(q, positions)
        k = self._rope(k, positions)
        if cache['k'] is not None:
            k = xp.concatenate([cache['k'], k], 0)
            v = xp.concatenate([cache['v'], v], 0)
        cache['k'], cache['v'] = k, v
        total = k.shape[0]
        rep = self.n_heads // self.n_kv
        kr = xp.repeat(k, rep, 1).transpose(1, 2, 0).astype(xp.float32)
        vr = xp.repeat(v, rep, 1).transpose(1, 0, 2).astype(xp.float32)
        qt = q.transpose(1, 0, 2).astype(xp.float32) * self.head_dim ** -0.5
        s = xp.matmul(qt, kr)
        col = xp.arange(total)[None]
        row = xp.arange(T)[:, None]
        s = s + xp.where(col > (total - T) + row, xp.float32(-xp.inf),
                         xp.float32(0))[None]
        s -= s.max(-1, keepdims=True)
        e = xp.exp(s)
        o = xp.matmul(e / e.sum(-1, keepdims=True), vr)
        o = o.transpose(1, 0, 2)
        o = o * (1.0 / (1.0 + xp.exp(-gate.astype(xp.float32))))
        o = o.reshape(T, -1).astype(x.dtype)
        return x + o @ L['o'].T

    def _delta_block(self, x, L, cache):
        xp = self.xp
        T = x.shape[0]
        nk, nv, dk = self.n_k, self.n_v, self.dk
        h = self._rms(x, L['attn_norm'])
        qkv = (h @ L['qkv'].T).astype(xp.float32)
        z = (h @ L['z'].T).astype(xp.float32)
        a = (h @ L['alpha'].T).astype(xp.float32)
        b = (h @ L['beta'].T).astype(xp.float32)

        if cache['conv'] is None:
            cache['conv'] = xp.zeros((self.conv_k - 1, qkv.shape[1]),
                                     dtype=xp.float32)
        pad = xp.concatenate([cache['conv'], qkv], 0)
        cache['conv'] = pad[-(self.conv_k - 1):].copy()
        w = L['conv'].astype(xp.float32)
        conv = sum(pad[j:j + T] * w[:, j][None]
                   for j in range(self.conv_k))
        qkv = conv / (1.0 + xp.exp(-conv))

        q = qkv[:, :nk * dk].reshape(T, nk, dk)
        k = qkv[:, nk * dk:2 * nk * dk].reshape(T, nk, dk)
        v = qkv[:, 2 * nk * dk:].reshape(T, nv, dk)

        q = q / xp.sqrt((q * q).sum(-1, keepdims=True) + 1e-6)
        k = k / xp.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
        q = q * dk ** -0.5
        rep = nv // nk
        q = xp.tile(q, (1, rep, 1))
        k = xp.tile(k, (1, rep, 1))

        beta = 1.0 / (1.0 + xp.exp(-b))
        g = -xp.exp(L['A_log'].astype(xp.float32)) \
            * _softplus_xp(xp, a + L['dt_bias'].astype(xp.float32))

        if cache['S'] is None:
            cache['S'] = xp.zeros((nv, dk, dk), dtype=xp.float32)
        S = cache['S']
        o = xp.empty((T, nv, dk), dtype=xp.float32)
        for t in range(T):
            S = S * xp.exp(g[t])[:, None, None]
            kv = (S * k[t][:, :, None]).sum(1)
            delta = (v[t] - kv) * beta[t][:, None]
            S = S + k[t][:, :, None] * delta[:, None, :]
            o[t] = (S * q[t][:, :, None]).sum(1)
        cache['S'] = S

        o = o / xp.sqrt((o * o).mean(-1, keepdims=True) + self.eps)
        o = o * L['ssm_norm'].astype(xp.float32)[None, None]
        zz = z.reshape(T, nv, dk)
        o = o * (zz / (1.0 + xp.exp(-zz)))
        o = o.reshape(T, -1).astype(x.dtype)
        return x + o @ L['out'].T

    def _ffn(self, x, L):
        xp = self.xp
        h = self._rms(x, L['post_norm'])
        gate = h @ L['ffn_gate'].T
        up = h @ L['ffn_up'].T
        g32 = gate.astype(xp.float32)
        act = (g32 / (1.0 + xp.exp(-g32))).astype(x.dtype)
        return x + (act * up) @ L['ffn_down'].T

    def _embed_rows(self, ids_host):
        if not self.stream:
            return self.embed[self.xp.asarray(ids_host)]
        rows = self.f.get_rows('token_embd.weight', ids_host,
                               dtype=np.float32).astype(self.dtype)
        return self.xp.asarray(rows)

    def _head(self, x32):
        xp = self.xp
        if not self.stream:
            return x32 @ self.head_w.T.astype(xp.float32)
        CH = 32768
        seq = x32.shape[0]
        logits = xp.empty((seq, self.vocab), dtype=xp.float32)
        for r0 in range(0, self.vocab, CH):
            r1 = min(r0 + CH, self.vocab)
            blk = self.f.get_row_block(self._head_name, r0, r1,
                                       dtype=np.float32).astype(self.dtype)
            w = xp.asarray(blk).astype(xp.float32)
            logits[:, r0:r1] = x32 @ w.T
            del w
        return logits

    def new_state(self):
        return [({'k': None, 'v': None} if self.is_attn(i)
                 else {'S': None, 'conv': None})
                for i in range(self.n_layers)]

    def forward(self, token_ids, state=None, past: int = 0):
        xp = self.xp
        ids_host = np.asarray(token_ids)
        if state is None:
            state = self.new_state()
        x = self._embed_rows(ids_host)
        T = x.shape[0]
        positions = xp.arange(past, past + T)
        for i in range(self.n_layers):
            L = self._layer(i)
            if self.is_attn(i):
                x = self._attn_block(x, L, positions, state[i])
            else:
                x = self._delta_block(x, L, state[i])
            x = self._ffn(x, L)
        x = self._rms(x, self.output_norm)
        return self._head(x.astype(xp.float32)), state

    def generate(self, prompt: str, max_new: int = 64, verbose=False):
        tok = self.tokenizer
        ids = tok.encode(prompt)
        state = self.new_state()
        logits, state = self.forward(np.asarray(ids), state)
        out = []
        nxt = int(logits[-1].argmax())
        for _ in range(max_new):
            out.append(nxt)
            if nxt == tok.eos_id:
                break
            if verbose:
                print(tok.decode([nxt]), end='', flush=True)
            logits, state = self.forward(np.asarray([nxt]), state,
                                         past=len(ids) + len(out) - 1)
            nxt = int(logits[-1].argmax())
        return tok.decode(out)

def _softplus_xp(xp, x):
    return xp.log1p(xp.exp(-xp.abs(x))) + xp.maximum(x, 0.0)
