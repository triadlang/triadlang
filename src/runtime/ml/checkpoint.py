
from __future__ import annotations

import json

from triad import ntri as np

try:
    import cupy as _cp
except (ImportError, ModuleNotFoundError):
    _cp = None

_DTYPE_TO_ST = {
    'float64': 'F64', 'float32': 'F32', 'float16': 'F16',
    'int64': 'I64', 'int32': 'I32', 'int16': 'I16', 'int8': 'I8',
    'uint8': 'U8', 'bool': 'BOOL',
}

def _to_numpy(a):
    if _cp is not None and isinstance(a, _cp.ndarray):
        return _cp.asnumpy(a)
    return np.asarray(a)

def save_safetensors(tensors: dict, path: str, metadata: dict = None):

    from runtime.ml.tensor import TriadTensor
    arrays = {}
    for name, t in tensors.items():
        raw = t._data if isinstance(t, TriadTensor) else t
        arrays[name] = np.ascontiguousarray(_to_numpy(raw))

    header = {}
    if metadata:
        header['__metadata__'] = {k: str(v) for k, v in metadata.items()}
    offset = 0
    order = list(arrays.keys())
    for name in order:
        a = arrays[name]
        nbytes = a.nbytes
        header[name] = {
            'dtype': _DTYPE_TO_ST[str(a.dtype)],
            'shape': list(a.shape),
            'data_offsets': [offset, offset + nbytes],
        }
        offset += nbytes

    hjson = json.dumps(header, separators=(',', ':')).encode()
    pad = (8 - len(hjson) % 8) % 8
    hjson += b' ' * pad
    with open(path, 'wb') as f:
        f.write(len(hjson).to_bytes(8, 'little'))
        f.write(hjson)
        for name in order:
            f.write(arrays[name].tobytes())

def load_safetensors(path: str) -> dict:

    from runtime.ml.native_qwen import SafetensorsFile
    st = SafetensorsFile(path)
    out = {}
    for name in st.names:
        info = st.header[name]
        np_dt = {'F64': np.float64, 'F32': np.float32, 'F16': np.float16,
                 'I64': np.int64, 'I32': np.int32, 'I16': np.int16,
                 'I8': np.int8, 'U8': np.uint8, 'BOOL': np.bool_,
                 'BF16': np.float32}[info['dtype']]
        out[name] = st.get(name, dtype=np_dt)
    return out

def state_dict(module, prefix='') -> dict:

    from runtime.ml.tensor import TriadTensor
    out = {}
    for k, v in module.__dict__.items():
        name = f'{prefix}{k}'
        if isinstance(v, TriadTensor):
            out[name] = v.data
        elif hasattr(v, '__dict__') and hasattr(v, 'forward'):
            out.update(state_dict(v, prefix=f'{name}.'))
        elif isinstance(v, (list, tuple)):
            for i, item in enumerate(v):
                if isinstance(item, TriadTensor):
                    out[f'{name}.{i}'] = item.data
                elif hasattr(item, '__dict__') and hasattr(item, 'forward'):
                    out.update(state_dict(item, prefix=f'{name}.{i}.'))
    return out

def load_state_dict(module, sd: dict, prefix=''):

    from runtime.ml.tensor import TriadTensor, _coerce
    for k, v in module.__dict__.items():
        name = f'{prefix}{k}'
        if isinstance(v, TriadTensor):
            if name in sd:
                v._data = _coerce(sd[name], v.device)
        elif hasattr(v, '__dict__') and hasattr(v, 'forward'):
            load_state_dict(v, sd, prefix=f'{name}.')
        elif isinstance(v, (list, tuple)):
            for i, item in enumerate(v):
                if isinstance(item, TriadTensor):
                    if f'{name}.{i}' in sd:
                        item._data = _coerce(sd[f'{name}.{i}'], item.device)
                elif hasattr(item, '__dict__') and hasattr(item, 'forward'):
                    load_state_dict(item, sd, prefix=f'{name}.{i}.')

def save_module(module, path: str, metadata: dict = None):
    import os
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    save_safetensors(state_dict(module), path, metadata)

def load_module(module, path: str):
    load_state_dict(module, load_safetensors(path))

def clip_grad_norm(params, max_norm: float) -> float:

    total = 0.0
    for p in params:
        if p._grad is None:
            continue
        g = p._grad
        xp = _cp if (_cp is not None and isinstance(g, _cp.ndarray)) else np
        total += float((g.astype(xp.float64) ** 2).sum())
    norm = total ** 0.5
    if norm > max_norm and norm > 0:
        scale = max_norm / norm
        for p in params:
            if p._grad is not None:
                p._grad = p._grad * scale
    return norm
