
from __future__ import annotations

import struct

from triad import ntri as np

GGUF_MAGIC = b'GGUF'

_KV_FMT = {0: 'B', 1: 'b', 2: '<H', 3: '<h', 4: '<I', 5: '<i',
           6: '<f', 7: 'B', 10: '<Q', 11: '<q', 12: '<d'}
_KV_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
            10: 8, 11: 8, 12: 8}
T_STRING, T_ARRAY = 8, 9

F32, F16 = 0, 1
Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 = 2, 3, 6, 7, 8
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K = 10, 11, 12, 13, 14, 15
BF16 = 30

_BLOCK = {F32: (4, 1), F16: (2, 1), BF16: (2, 1),
          Q4_0: (18, 32), Q4_1: (20, 32), Q8_0: (34, 32),
          Q4_K: (144, 256), Q5_K: (176, 256), Q6_K: (210, 256)}

_TYPE_NAME = {F32: 'F32', F16: 'F16', BF16: 'BF16', Q4_0: 'Q4_0',
              Q4_1: 'Q4_1', Q5_0: 'Q5_0', Q5_1: 'Q5_1', Q8_0: 'Q8_0',
              Q2_K: 'Q2_K', Q3_K: 'Q3_K', Q4_K: 'Q4_K', Q5_K: 'Q5_K',
              Q6_K: 'Q6_K', Q8_K: 'Q8_K'}

def _bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)

def _dq_q8_0(raw: np.ndarray, n: int) -> np.ndarray:
    blocks = raw.reshape(-1, 34)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32)
    q = blocks[:, 2:].view(np.int8).astype(np.float32)
    return (d * q).reshape(-1)[:n]

def _dq_q4_0(raw: np.ndarray, n: int) -> np.ndarray:
    blocks = raw.reshape(-1, 18)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32)
    qs = blocks[:, 2:]
    lo = (qs & 0x0F).astype(np.float32) - 8.0
    hi = (qs >> 4).astype(np.float32) - 8.0
    out = np.concatenate([lo, hi], axis=1) * d
    return out.reshape(-1)[:n]

def _dq_q4_1(raw: np.ndarray, n: int) -> np.ndarray:
    blocks = raw.reshape(-1, 20)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32)
    m = blocks[:, 2:4].copy().view(np.float16).astype(np.float32)
    qs = blocks[:, 4:]
    lo = (qs & 0x0F).astype(np.float32)
    hi = (qs >> 4).astype(np.float32)
    out = np.concatenate([lo, hi], axis=1) * d + m
    return out.reshape(-1)[:n]

def _dq_q6_k(raw: np.ndarray, n: int) -> np.ndarray:

    blocks = raw.reshape(-1, 210)
    nb = blocks.shape[0]
    ql = blocks[:, :128]
    qh = blocks[:, 128:192]
    sc = blocks[:, 192:208].view(np.int8).astype(np.float32)
    d = blocks[:, 208:210].copy().view(np.float16).astype(np.float32)
    out = np.empty((nb, 256), dtype=np.float32)
    for half in (0, 1):
        qlh = ql[:, half * 64:(half + 1) * 64]
        qhh = qh[:, half * 32:(half + 1) * 32]
        base = half * 128
        q1 = (qlh[:, :32] & 0x0F) | (((qhh >> 0) & 3) << 4)
        q2 = (qlh[:, 32:] & 0x0F) | (((qhh >> 2) & 3) << 4)
        q3 = (qlh[:, :32] >> 4) | (((qhh >> 4) & 3) << 4)
        q4 = (qlh[:, 32:] >> 4) | (((qhh >> 6) & 3) << 4)
        for j, q in enumerate((q1, q2, q3, q4)):
            qf = q.astype(np.float32) - 32.0

            s_idx = half * 8 + j * 2
            s = np.concatenate([np.repeat(sc[:, s_idx:s_idx + 1], 16, axis=1),
                                np.repeat(sc[:, s_idx + 1:s_idx + 2], 16, axis=1)],
                               axis=1)
            out[:, base + j * 32: base + (j + 1) * 32] = d * s * qf
    return out.reshape(-1)[:n]

def _q4k_scales_mins(scales: np.ndarray):

    sc = np.empty((scales.shape[0], 8), dtype=np.float32)
    mn = np.empty((scales.shape[0], 8), dtype=np.float32)
    for j in range(4):
        sc[:, j] = (scales[:, j] & 63).astype(np.float32)
        mn[:, j] = (scales[:, j + 4] & 63).astype(np.float32)
    for j in range(4, 8):
        sc[:, j] = ((scales[:, j + 4] & 0x0F) | ((scales[:, j - 4] >> 6) << 4)).astype(np.float32)
        mn[:, j] = ((scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)).astype(np.float32)
    return sc, mn

def _dq_q4_k(raw: np.ndarray, n: int) -> np.ndarray:

    blocks = raw.reshape(-1, 144)
    nb = blocks.shape[0]
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32)
    dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32)
    sc, mn = _q4k_scales_mins(blocks[:, 4:16])
    qs = blocks[:, 16:]
    out = np.empty((nb, 256), dtype=np.float32)
    for j in range(4):
        b = qs[:, j * 32:(j + 1) * 32]
        lo = (b & 0x0F).astype(np.float32)
        hi = (b >> 4).astype(np.float32)
        s_lo = (d[:, 0] * sc[:, 2 * j])[:, None]
        m_lo = (dmin[:, 0] * mn[:, 2 * j])[:, None]
        s_hi = (d[:, 0] * sc[:, 2 * j + 1])[:, None]
        m_hi = (dmin[:, 0] * mn[:, 2 * j + 1])[:, None]
        out[:, j * 64: j * 64 + 32] = s_lo * lo - m_lo
        out[:, j * 64 + 32: j * 64 + 64] = s_hi * hi - m_hi
    return out.reshape(-1)[:n]

def _dq_q5_k(raw: np.ndarray, n: int) -> np.ndarray:

    blocks = raw.reshape(-1, 176)
    nb = blocks.shape[0]
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32)
    dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32)
    sc, mn = _q4k_scales_mins(blocks[:, 4:16])
    qh = blocks[:, 16:48]
    qs = blocks[:, 48:176]
    out = np.empty((nb, 256), dtype=np.float32)
    for j in range(4):
        b = qs[:, j * 32:(j + 1) * 32]
        hi_lo = ((qh >> (2 * j)) & 1).astype(np.float32) * 16.0
        hi_hi = ((qh >> (2 * j + 1)) & 1).astype(np.float32) * 16.0
        lo = (b & 0x0F).astype(np.float32) + hi_lo
        hi = (b >> 4).astype(np.float32) + hi_hi
        s_lo = (d[:, 0] * sc[:, 2 * j])[:, None]
        m_lo = (dmin[:, 0] * mn[:, 2 * j])[:, None]
        s_hi = (d[:, 0] * sc[:, 2 * j + 1])[:, None]
        m_hi = (dmin[:, 0] * mn[:, 2 * j + 1])[:, None]
        out[:, j * 64: j * 64 + 32] = s_lo * lo - m_lo
        out[:, j * 64 + 32: j * 64 + 64] = s_hi * hi - m_hi
    return out.reshape(-1)[:n]

_DEQUANT = {Q8_0: _dq_q8_0, Q4_0: _dq_q4_0, Q4_1: _dq_q4_1,
            Q4_K: _dq_q4_k, Q5_K: _dq_q5_k, Q6_K: _dq_q6_k}

class GGUFFile:

    def __init__(self, path: str):
        self.path = path
        self._mm = np.memmap(path, dtype=np.uint8, mode='r')
        buf = self._mm
        if bytes(buf[:4]) != GGUF_MAGIC:
            raise ValueError(f'{path}: magic GGUF ausente')
        self.version = int(np.frombuffer(buf[4:8], '<u4')[0])
        if self.version not in (2, 3):
            raise ValueError(f'versao GGUF {self.version} nao suportada')
        n_tensors = int(np.frombuffer(buf[8:16], '<u8')[0])
        n_kv = int(np.frombuffer(buf[16:24], '<u8')[0])
        pos = 24

        def read_str(p):
            ln = int(np.frombuffer(buf[p:p + 8], '<u8')[0])
            return bytes(buf[p + 8:p + 8 + ln]).decode('utf-8'), p + 8 + ln

        def read_val(p, vt):
            if vt == T_STRING:
                return read_str(p)
            if vt == T_ARRAY:
                et = int(np.frombuffer(buf[p:p + 4], '<u4')[0])
                ln = int(np.frombuffer(buf[p + 4:p + 12], '<u8')[0])
                p += 12
                if et == T_STRING:
                    out = []
                    for _ in range(ln):
                        s, p = read_str(p)
                        out.append(s)
                    return out, p
                sz = _KV_SIZE[et]
                arr = np.frombuffer(buf[p:p + sz * ln],
                                    _KV_FMT[et].lstrip('<')
                                    if sz == 1 else _KV_FMT[et])
                return arr, p + sz * ln
            sz = _KV_SIZE[vt]
            v = struct.unpack(_KV_FMT[vt], bytes(buf[p:p + sz]))[0]
            if vt == 7:
                v = bool(v)
            return v, p + sz

        self.kv = {}
        for _ in range(n_kv):
            key, pos = read_str(pos)
            vt = int(np.frombuffer(buf[pos:pos + 4], '<u4')[0])
            pos += 4
            self.kv[key], pos = read_val(pos, vt)

        self.tensors = {}
        order = []
        for _ in range(n_tensors):
            name, pos = read_str(pos)
            ndim = int(np.frombuffer(buf[pos:pos + 4], '<u4')[0])
            pos += 4
            ne = np.frombuffer(buf[pos:pos + 8 * ndim], '<u8').tolist()
            pos += 8 * ndim
            ttype = int(np.frombuffer(buf[pos:pos + 4], '<u4')[0])
            pos += 4
            off = int(np.frombuffer(buf[pos:pos + 8], '<u8')[0])
            pos += 8
            self.tensors[name] = {'ne': ne, 'type': ttype, 'offset': off}
            order.append(name)
        self.names = order

        align = int(self.kv.get('general.alignment', 32))
        self._data_start = (pos + align - 1) // align * align

    def info(self, name: str) -> dict:
        t = self.tensors[name]
        return {'shape': tuple(reversed(t['ne'])),
                'type': _TYPE_NAME.get(t['type'], str(t['type'])),
                'offset': t['offset']}

    def _row_layout(self, name: str):
        t = self.tensors[name]
        ne, ttype = t['ne'], t['type']
        n_cols = int(ne[0])
        bsz, bel = _BLOCK[ttype]
        if n_cols % bel != 0:
            raise ValueError(f'{name}: linhas nao alinham ao bloco '
                             f'({n_cols} % {bel})')
        return t, n_cols, n_cols // bel * bsz, ttype

    def _decode_raw(self, raw: np.ndarray, ttype: int, n: int) -> np.ndarray:
        if ttype == F32:
            return np.asarray(raw.view(np.float32))
        if ttype == F16:
            return raw.view(np.float16).astype(np.float32)
        if ttype == BF16:
            return _bf16_to_f32(np.asarray(raw).view(np.uint16))
        return _DEQUANT[ttype](np.asarray(raw), n)

    def get_rows(self, name: str, rows, dtype=np.float32) -> np.ndarray:

        t, n_cols, row_bytes, ttype = self._row_layout(name)
        start = self._data_start + t['offset']
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        out = np.empty((rows.size, n_cols), dtype=np.float32)
        for j, r in enumerate(rows):
            a = start + int(r) * row_bytes
            out[j] = self._decode_raw(self._mm[a:a + row_bytes], ttype, n_cols)
        return out.astype(dtype, copy=False)

    def get_row_block(self, name: str, r0: int, r1: int,
                      dtype=np.float32) -> np.ndarray:

        t, n_cols, row_bytes, ttype = self._row_layout(name)
        start = self._data_start + t['offset']
        raw = self._mm[start + r0 * row_bytes:start + r1 * row_bytes]
        arr = self._decode_raw(raw, ttype, (r1 - r0) * n_cols)
        return np.ascontiguousarray(arr.reshape(r1 - r0, n_cols), dtype=dtype)

    def get(self, name: str, dtype=np.float32) -> np.ndarray:
        t = self.tensors[name]
        ne, ttype = t['ne'], t['type']
        n = 1
        for d in ne:
            n *= int(d)
        if ttype not in _BLOCK:
            raise ValueError(f'{name}: tipo GGML {ttype} '
                             f'({_TYPE_NAME.get(ttype, "?")}) sem dequant')
        bsz, bel = _BLOCK[ttype]
        nbytes = (n // bel) * bsz
        start = self._data_start + t['offset']
        raw = self._mm[start:start + nbytes]
        if ttype == F32:
            arr = raw.view(np.float32)
        elif ttype == F16:
            arr = raw.view(np.float16).astype(np.float32)
        elif ttype == BF16:
            arr = _bf16_to_f32(raw.view(np.uint16))
        else:
            arr = _DEQUANT[ttype](np.asarray(raw), n)
        shape = tuple(reversed(ne))
        return np.ascontiguousarray(arr.reshape(shape), dtype=dtype)

def _q8_0_quantize(x: np.ndarray) -> np.ndarray:

    flat = x.astype(np.float32).reshape(-1, 32)
    amax = np.abs(flat).max(axis=1, keepdims=True)
    d = (amax / 127.0).astype(np.float16)
    df = d.astype(np.float32)
    df[df == 0] = 1.0
    q = np.clip(np.round(flat / df), -127, 127).astype(np.int8)
    out = np.empty((flat.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.reshape(-1)

def _q4_0_quantize(x: np.ndarray) -> np.ndarray:
    flat = x.astype(np.float32).reshape(-1, 32)
    idx = np.abs(flat).argmax(axis=1)
    maxv = flat[np.arange(flat.shape[0]), idx]
    d = (maxv / -8.0).astype(np.float16)
    df = d.astype(np.float32)
    df[df == 0] = 1.0
    q = np.clip(np.round(flat / df[:, None]) + 8, 0, 15).astype(np.uint8)
    out = np.empty((flat.shape[0], 18), dtype=np.uint8)
    out[:, :2] = d.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q[:, :16] | (q[:, 16:] << 4)
    return out.reshape(-1)

class GGUFWriter:

    def __init__(self):
        self._kv = []
        self._tensors = []

    def add_kv(self, key: str, value):
        self._kv.append((key, value))

    def add_tensor(self, name: str, arr: np.ndarray, ttype: int = F32):
        self._tensors.append((name, np.asarray(arr), ttype))

    @staticmethod
    def _pack_str(s: str) -> bytes:
        b = s.encode('utf-8')
        return struct.pack('<Q', len(b)) + b

    def _pack_val(self, v) -> bytes:
        if isinstance(v, bool):
            return struct.pack('<I', 7) + struct.pack('B', int(v))
        if isinstance(v, int):
            return struct.pack('<I', 4 if 0 <= v < 2 ** 32 else 11) + \
                (struct.pack('<I', v) if 0 <= v < 2 ** 32 else struct.pack('<q', v))
        if isinstance(v, float):
            return struct.pack('<I', 6) + struct.pack('<f', v)
        if isinstance(v, str):
            return struct.pack('<I', T_STRING) + self._pack_str(v)
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], str):
            out = struct.pack('<I', T_ARRAY) + struct.pack('<I', T_STRING) \
                + struct.pack('<Q', len(v))
            for s in v:
                out += self._pack_str(s)
            return out
        raise TypeError(f'KV nao suportado: {type(v)}')

    def write(self, path: str, align: int = 32):
        kvs = [('general.alignment', align)] + self._kv
        head = GGUF_MAGIC + struct.pack('<I', 3) \
            + struct.pack('<Q', len(self._tensors)) \
            + struct.pack('<Q', len(kvs))
        body = b''
        for k, v in kvs:
            body += self._pack_str(k) + self._pack_val(v)
        blobs, infos = [], b''
        off = 0
        for name, arr, ttype in self._tensors:
            if ttype == F32:
                raw = arr.astype(np.float32).tobytes()
            elif ttype == F16:
                raw = arr.astype(np.float16).tobytes()
            elif ttype == BF16:
                f32 = arr.astype(np.float32)
                raw = (f32.view(np.uint32) >> 16).astype(np.uint16).tobytes()
            elif ttype == Q8_0:
                raw = _q8_0_quantize(arr).tobytes()
            elif ttype == Q4_0:
                raw = _q4_0_quantize(arr).tobytes()
            else:
                raise ValueError(f'escritor nao suporta tipo {ttype}')
            ne = list(reversed(arr.shape))
            infos += self._pack_str(name) + struct.pack('<I', len(ne))
            for d in ne:
                infos += struct.pack('<Q', d)
            infos += struct.pack('<I', ttype) + struct.pack('<Q', off)
            pad = (-len(raw)) % align
            blobs.append(raw + b'\x00' * pad)
            off += len(raw) + pad
        header = head + body + infos
        pad = (-len(header)) % align
        with open(path, 'wb') as f:
            f.write(header + b'\x00' * pad)
            for b in blobs:
                f.write(b)

_HF_MAP = [
    ('token_embd.weight', 'model.embed_tokens.weight'),
    ('output_norm.weight', 'model.norm.weight'),
    ('output.weight', 'lm_head.weight'),
]
_HF_BLK = [
    ('attn_norm.weight', 'input_layernorm.weight'),
    ('attn_q.weight', 'self_attn.q_proj.weight'),
    ('attn_k.weight', 'self_attn.k_proj.weight'),
    ('attn_v.weight', 'self_attn.v_proj.weight'),
    ('attn_output.weight', 'self_attn.o_proj.weight'),
    ('attn_q_norm.weight', 'self_attn.q_norm.weight'),
    ('attn_k_norm.weight', 'self_attn.k_norm.weight'),
    ('ffn_norm.weight', 'post_attention_layernorm.weight'),
    ('ffn_gate.weight', 'mlp.gate_proj.weight'),
    ('ffn_up.weight', 'mlp.up_proj.weight'),
    ('ffn_down.weight', 'mlp.down_proj.weight'),
]

class QwenGGUF:

    def __init__(self, path: str):
        self.f = GGUFFile(path)
        self.arch = self.f.kv.get('general.architecture', 'qwen3')
        a = self.arch
        kv = self.f.kv
        emb_ne = self.f.tensors['token_embd.weight']['ne']
        d_model = int(kv.get(f'{a}.embedding_length', emb_ne[0]))
        n_heads = int(kv[f'{a}.attention.head_count'])
        head_dim = int(kv.get(f'{a}.attention.key_length',
                              d_model // n_heads))
        self.config = {
            'num_hidden_layers': int(kv[f'{a}.block_count']),
            'num_attention_heads': n_heads,
            'num_key_value_heads': int(kv.get(
                f'{a}.attention.head_count_kv', n_heads)),
            'head_dim': head_dim,
            'hidden_size': d_model,
            'rms_norm_eps': float(kv.get(
                f'{a}.attention.layer_norm_rms_epsilon', 1e-6)),
            'rope_theta': float(kv.get(f'{a}.rope.freq_base', 10000.0)),
            'vocab_size': int(emb_ne[1]),
            'tie_word_embeddings': 'output.weight' not in self.f.tensors,
        }
        self._name_map = {}
        for g, h in _HF_MAP:
            if g in self.f.tensors:
                self._name_map[h] = g
        for i in range(self.config['num_hidden_layers']):
            for g, h in _HF_BLK:
                gname = f'blk.{i}.{g}'
                if gname in self.f.tensors:
                    self._name_map[f'model.layers.{i}.{h}'] = gname

    @property
    def names(self):
        return list(self._name_map)

    def get(self, hf_name: str, dtype=np.float32) -> np.ndarray:
        return self.f.get(self._name_map[hf_name], dtype=dtype)

    def get_rows(self, hf_name: str, rows, dtype=np.float32) -> np.ndarray:
        return self.f.get_rows(self._name_map[hf_name], rows, dtype=dtype)

    def get_row_block(self, hf_name: str, r0: int, r1: int,
                      dtype=np.float32) -> np.ndarray:
        return self.f.get_row_block(self._name_map[hf_name], r0, r1,
                                    dtype=dtype)

    def __contains__(self, hf_name: str) -> bool:
        return hf_name in self._name_map

    def tokens(self):
        return self.f.kv.get('tokenizer.ggml.tokens')

def _bytes_to_unicode():

    bs = (list(range(ord('!'), ord('~') + 1))
          + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))

_PRETOK = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
           r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
           r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

class GGUFTokenizer:

    def __init__(self, kv: dict):
        self.tokens = kv['tokenizer.ggml.tokens']
        merges = kv.get('tokenizer.ggml.merges', [])
        self.rank = {tuple(m.split(' ', 1)): i for i, m in enumerate(merges)}
        self.vocab = {t: i for i, t in enumerate(self.tokens)}
        self.eos_id = int(kv.get('tokenizer.ggml.eos_token_id', -1))
        self.bos_id = int(kv.get('tokenizer.ggml.bos_token_id', -1))
        self._b2u = _bytes_to_unicode()
        self._u2b = {v: k for k, v in self._b2u.items()}
        import regex
        self._re = regex.compile(_PRETOK)

    def _bpe(self, piece: str):
        parts = list(piece)
        while len(parts) > 1:
            melhor, onde = None, -1
            for i in range(len(parts) - 1):
                r = self.rank.get((parts[i], parts[i + 1]))
                if r is not None and (melhor is None or r < melhor):
                    melhor, onde = r, i
            if melhor is None:
                break
            parts[onde:onde + 2] = [parts[onde] + parts[onde + 1]]
        return parts

    def encode(self, text: str):
        out = []
        for piece in self._re.findall(text):
            u = ''.join(self._b2u[b] for b in piece.encode('utf-8'))
            for sym in self._bpe(u):
                tid = self.vocab.get(sym)
                if tid is None:
                    out.extend(self.vocab[c] for c in sym
                               if c in self.vocab)
                else:
                    out.append(tid)
        return out

    def decode(self, ids):
        txt = ''.join(self.tokens[int(i)] for i in ids)
        bs = bytes(self._u2b.get(c, 32) for c in txt)
        return bs.decode('utf-8', errors='replace')
