from __future__ import annotations

import json
import os

from triad import ntri as np

try:
    import cupy as _cp
except (ImportError, ModuleNotFoundError):
    _cp = None

_DTYPES = {
    'F64': np.float64, 'F32': np.float32, 'F16': np.float16,
    'I64': np.int64, 'I32': np.int32, 'I16': np.int16, 'I8': np.int8,
    'U8': np.uint8, 'BOOL': np.bool_, 'BF16': None,
}

def _bf16_to_f32(raw_u16: np.ndarray) -> np.ndarray:
    out = raw_u16.astype(np.uint32) << 16
    return out.view(np.float32)

class SafetensorsFile:

    def __init__(self, path: str):
        self.path = path
        with open(path, 'rb') as f:
            header_len = int.from_bytes(f.read(8), 'little')
            self.header = json.loads(f.read(header_len))
        self._data_start = 8 + header_len
        self._mm = np.memmap(path, dtype=np.uint8, mode='r')
        self.names = [k for k in self.header if k != '__metadata__']

    def get(self, name: str, dtype=np.float32) -> np.ndarray:
        info = self.header[name]
        start, end = info['data_offsets']
        raw = self._mm[self._data_start + start:self._data_start + end]
        st_dt = info['dtype']
        shape = tuple(info['shape'])
        if st_dt == 'BF16':
            arr = _bf16_to_f32(raw.view(np.uint16)).reshape(shape)
        else:
            arr = raw.view(_DTYPES[st_dt]).reshape(shape)
        return np.ascontiguousarray(arr, dtype=dtype)

    def _rows_raw(self, name: str, r0: int, r1: int):
        info = self.header[name]
        start, _ = info['data_offsets']
        st_dt = info['dtype']
        n_cols = int(info['shape'][-1])
        isz = 2 if st_dt in ('BF16', 'F16') else 4
        base = self._data_start + start
        raw = self._mm[base + r0 * n_cols * isz:base + r1 * n_cols * isz]
        if st_dt == 'BF16':
            arr = _bf16_to_f32(np.asarray(raw).view(np.uint16))
        else:
            arr = raw.view(_DTYPES[st_dt]).astype(np.float32)
        return arr.reshape(r1 - r0, n_cols)

    def get_rows(self, name: str, rows, dtype=np.float32) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        out = np.empty((rows.size, int(self.header[name]['shape'][-1])),
                       dtype=np.float32)
        for j, r in enumerate(rows):
            out[j] = self._rows_raw(name, int(r), int(r) + 1)[0]
        return out.astype(dtype, copy=False)

    def get_row_block(self, name: str, r0: int, r1: int,
                      dtype=np.float32) -> np.ndarray:
        return np.ascontiguousarray(self._rows_raw(name, r0, r1),
                                    dtype=dtype)

def _rms_norm(x, weight, eps=1e-6):
    xp = _cp if (_cp is not None and isinstance(x, _cp.ndarray)) else np
    x32 = x.astype(xp.float32)
    rms = xp.sqrt((x32 * x32).mean(axis=-1, keepdims=True) + eps)
    return ((x32 / rms) * weight.astype(xp.float32)).astype(x.dtype)

def _silu(x):
    xp = _cp if (_cp is not None and isinstance(x, _cp.ndarray)) else np
    return x / (1 + xp.exp(-x))

def _rope(x, positions, head_dim, base):
    xp = _cp if (_cp is not None and isinstance(x, _cp.ndarray)) else np
    half = head_dim // 2
    freqs = 1.0 / (base ** (xp.arange(0, head_dim, 2, dtype=xp.float32) / head_dim))
    theta = positions[:, None].astype(xp.float32) * freqs[None, :]
    cos_t = xp.cos(theta)[:, None, :]
    sin_t = xp.sin(theta)[:, None, :]
    cos_t = xp.concatenate([cos_t, cos_t], axis=-1)
    sin_t = xp.concatenate([sin_t, sin_t], axis=-1)
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = xp.concatenate([-x2, x1], axis=-1)
    return (x.astype(xp.float32) * cos_t + rotated.astype(xp.float32) * sin_t).astype(x.dtype)

class NativeQwen3:

    def __init__(self, model_dir: str, device: str = 'cuda', dtype='float16',
                 stream: bool = False):
        if model_dir.endswith('.gguf'):

            from runtime.ml.gguf import QwenGGUF
            st = QwenGGUF(model_dir)
            self.cfg = st.config
        else:
            with open(os.path.join(model_dir, 'config.json')) as f:
                self.cfg = json.load(f)
            st = SafetensorsFile(os.path.join(model_dir, 'model.safetensors'))
        self.xp = _cp if (device == 'cuda' and _cp is not None) else np
        self.dtype = dtype
        self.n_layers = self.cfg['num_hidden_layers']
        self.n_heads = self.cfg['num_attention_heads']
        self.n_kv = self.cfg['num_key_value_heads']
        self.head_dim = self.cfg['head_dim']
        self.eps = self.cfg['rms_norm_eps']
        self.rope_base = float(self.cfg['rope_theta'])

        def load(name):
            return self.xp.asarray(st.get(name, dtype=np.float32)).astype(dtype)

        self.stream = stream
        self._src = st
        self.final_norm = load('model.norm.weight')
        if stream:

            self.embed = None
            self.lm_head = None
            self.layers = None
            names = list(getattr(st, 'names', []))
            self._head_name = ('lm_head.weight' if 'lm_head.weight' in names
                               else 'model.embed_tokens.weight')
            self._vocab = int(self.cfg.get(
                'vocab_size', st.get_row_block(
                    'model.embed_tokens.weight', 0, 1).shape[0]))
            return
        self.embed = load('model.embed_tokens.weight')
        self.layers = []
        for i in range(self.n_layers):
            p = f'model.layers.{i}'
            self.layers.append({
                'in_norm': load(f'{p}.input_layernorm.weight'),
                'q': load(f'{p}.self_attn.q_proj.weight'),
                'k': load(f'{p}.self_attn.k_proj.weight'),
                'v': load(f'{p}.self_attn.v_proj.weight'),
                'o': load(f'{p}.self_attn.o_proj.weight'),
                'q_norm': load(f'{p}.self_attn.q_norm.weight'),
                'k_norm': load(f'{p}.self_attn.k_norm.weight'),
                'post_norm': load(f'{p}.post_attention_layernorm.weight'),
                'gate': load(f'{p}.mlp.gate_proj.weight'),
                'up': load(f'{p}.mlp.up_proj.weight'),
                'down': load(f'{p}.mlp.down_proj.weight'),
            })

        self.lm_head = self.embed

    def _load_layer(self, li: int):

        if not self.stream:
            return self.layers[li]
        p = f'model.layers.{li}'

        def g(n):
            w = self._src.get(f'{p}.{n}', dtype=np.float32).astype(self.dtype)
            return self.xp.asarray(w)
        return {'in_norm': g('input_layernorm.weight'),
                'q': g('self_attn.q_proj.weight'),
                'k': g('self_attn.k_proj.weight'),
                'v': g('self_attn.v_proj.weight'),
                'o': g('self_attn.o_proj.weight'),
                'q_norm': g('self_attn.q_norm.weight'),
                'k_norm': g('self_attn.k_norm.weight'),
                'post_norm': g('post_attention_layernorm.weight'),
                'gate': g('mlp.gate_proj.weight'),
                'up': g('mlp.up_proj.weight'),
                'down': g('mlp.down_proj.weight')}

    def _embed_rows(self, ids_host, ids_dev):
        if not self.stream:
            return self.embed[ids_dev]
        rows = self._src.get_rows('model.embed_tokens.weight', ids_host,
                                  dtype=np.float32).astype(self.dtype)
        return self.xp.asarray(rows)

    def _head_logits(self, x32):
        xp = self.xp
        if not self.stream:
            return x32 @ self.lm_head.T.astype(xp.float32)
        seq = x32.shape[0]

        CH = 32768
        logits = xp.empty((seq, self._vocab), dtype=xp.float32)
        for r0 in range(0, self._vocab, CH):
            r1 = min(r0 + CH, self._vocab)
            blk = self._src.get_row_block(self._head_name, r0, r1,
                                          dtype=np.float32).astype(self.dtype)
            w = xp.asarray(blk).astype(xp.float32)
            logits[:, r0:r1] = x32 @ w.T
            del w
        return logits

    def forward(self, token_ids: np.ndarray, kv_caches=None):

        xp = self.xp
        ids_host = np.asarray(token_ids)
        ids = xp.asarray(ids_host)
        x = self._embed_rows(ids_host, ids)
        seq_len = x.shape[0]
        past = 0
        if kv_caches is not None and kv_caches[0] is not None:
            past = kv_caches[0][0].shape[0]
        positions = xp.arange(past, past + seq_len)

        for li in range(self.n_layers):
            L = self._load_layer(li)
            h = _rms_norm(x, L['in_norm'], self.eps)
            q = (h @ L['q'].T).reshape(seq_len, self.n_heads, self.head_dim)
            k = (h @ L['k'].T).reshape(seq_len, self.n_kv, self.head_dim)
            v = (h @ L['v'].T).reshape(seq_len, self.n_kv, self.head_dim)

            q = _rms_norm(q, L['q_norm'], self.eps)
            k = _rms_norm(k, L['k_norm'], self.eps)
            q = _rope(q, positions, self.head_dim, self.rope_base)
            k = _rope(k, positions, self.head_dim, self.rope_base)

            if kv_caches is not None:
                if kv_caches[li] is None:
                    kv_caches[li] = [k, v]
                else:
                    kv_caches[li][0] = xp.concatenate([kv_caches[li][0], k], axis=0)
                    kv_caches[li][1] = xp.concatenate([kv_caches[li][1], v], axis=0)
                k_triad, v_triad = kv_caches[li]
            else:
                k_triad, v_triad = k, v

            n_rep = self.n_heads // self.n_kv
            if n_rep > 1:
                k_r = xp.repeat(k_triad, n_rep, axis=1)
                v_r = xp.repeat(v_triad, n_rep, axis=1)
            else:
                k_r, v_r = k_triad, v_triad

            q_t = q.transpose(1, 0, 2).astype(xp.float32)
            k_t = k_r.transpose(1, 2, 0).astype(xp.float32)
            v_t = v_r.transpose(1, 0, 2).astype(xp.float32)
            scores = xp.matmul(q_t, k_t) * (self.head_dim ** -0.5)

            total = k_triad.shape[0]
            mask = xp.zeros((seq_len, total), dtype=xp.float32)
            start = total - seq_len
            col = xp.arange(total)[None, :]
            row = xp.arange(seq_len)[:, None]
            mask = xp.where(col > start + row, xp.float32(-1e9), xp.float32(0.0))
            scores = scores + mask[None]

            scores -= scores.max(axis=-1, keepdims=True)
            e = xp.exp(scores)
            probs = e / e.sum(axis=-1, keepdims=True)
            out = xp.matmul(probs, v_t)
            out = out.transpose(1, 0, 2).reshape(seq_len, -1).astype(self.dtype)
            x = x + out @ L['o'].T

            h2 = _rms_norm(x, L['post_norm'], self.eps)
            gate = h2 @ L['gate'].T
            up = h2 @ L['up'].T
            x = x + (_silu(gate) * up) @ L['down'].T

        x = _rms_norm(x, self.final_norm, self.eps)
        return self._head_logits(x.astype(xp.float32))

    def generate(self, prompt_ids, max_new=64, eos_id=None):

        xp = self.xp
        kv = [None] * self.n_layers
        logits = self.forward(np.asarray(prompt_ids), kv)
        out = []
        next_id = int(logits[-1].argmax())
        for _ in range(max_new):
            out.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break
            logits = self.forward(np.asarray([next_id]), kv)
            next_id = int(logits[-1].argmax())
        return out
