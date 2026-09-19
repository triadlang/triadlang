from __future__ import annotations

import json
from pathlib import Path

from runtime.ml.nn import Module
from triad import ntri as np


def _serialize_param(data: np.ndarray) -> dict:
    return {'shape': list(data.shape), 'dtype': str(data.dtype), 'data': data.tolist()}

_SAFE_DTYPES = ('float64', 'float32', 'float16', 'int64', 'int32')

def _deserialize_param(d: dict) -> np.ndarray:
    dtype = d.get('dtype', 'float64')
    if dtype not in _SAFE_DTYPES:
        dtype = 'float64'
    arr = np.array(d['data'], dtype=getattr(np, dtype)).reshape(d['shape'])
    return arr

def save_weights(model: Module, path: str):
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    params = model.parameters()
    state = {}
    for i, p in enumerate(params):
        state[f'param_{i}'] = _serialize_param(p._data)
    meta = {'n_params': len(params), 'format_version': 1}
    payload = {'meta': meta, 'state': state}
    Path(path).write_text(json.dumps(payload))

def load_weights(model: Module, path: str):
    payload = json.loads(Path(path).read_text())
    if payload.get('meta', {}).get('format_version') != 1:
        raise ValueError('unsupported weights format version')
    state = payload['state']
    params = model.parameters()
    if len(state) != len(params):
        raise ValueError(f'weights mismatch: file has {len(state)} params, model has {len(params)}')
    for i, p in enumerate(params):
        key = f'param_{i}'
        if key in state:
            arr = _deserialize_param(state[key])
            if arr.shape != p._data.shape:
                raise ValueError(f'weights mismatch for {key}: file {arr.shape} vs model {p._data.shape}')
            p._data = arr.astype(p._data.dtype, copy=False)

def save_checkpoint(model: Module, optimizer, path: str, extra: dict | None=None):
    params = model.parameters()
    state = {}
    for i, p in enumerate(params):
        state[f'param_{i}'] = _serialize_param(p._data)
    opt_state = {}
    if hasattr(optimizer, 'm') and hasattr(optimizer, 'v'):
        opt_state['m'] = [_serialize_param(m) for m in optimizer.m]
        opt_state['v'] = [_serialize_param(v) for v in optimizer.v]
        opt_state['t'] = optimizer.t
    payload = {'meta': {'n_params': len(params), 'format_version': 1}, 'state': state, 'optimizer': opt_state}
    if extra:
        payload['extra'] = extra
    Path(path).write_text(json.dumps(payload))

def load_checkpoint(model: Module, optimizer, path: str) -> dict | None:
    payload = json.loads(Path(path).read_text())
    state = payload['state']
    params = model.parameters()
    for i, p in enumerate(params):
        key = f'param_{i}'
        if key in state:
            p._data = _deserialize_param(state[key])
    if 'optimizer' in payload and hasattr(optimizer, 'm'):
        opt = payload['optimizer']
        if 'm' in opt:
            optimizer.m = [_deserialize_param(m) for m in opt['m']]
        if 'v' in opt:
            optimizer.v = [_deserialize_param(v) for v in opt['v']]
        if 't' in opt:
            optimizer.t = opt['t']
    return payload.get('extra')
