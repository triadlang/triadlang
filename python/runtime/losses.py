from __future__ import annotations
from runtime.ml_device import xp as np
from runtime.tensor import TriadTensor, _ensure_tensor

def bce_loss(pred: TriadTensor, target: TriadTensor) -> TriadTensor:
    pred = _ensure_tensor(pred)
    target = _ensure_tensor(target)
    p = np.clip(pred._data, 1e-07, 1 - 1e-07)
    t = target._data
    loss = -(t * np.log(p) + (1 - t) * np.log(1 - p)).mean()
    out = TriadTensor(loss)
    if pred._requires_grad:
        out._requires_grad = True
        out._children = [pred]
        _p = p
        _t = t

        def _back(g):
            n = _p.size
            sg = (_p - _t) / (_p * (1 - _p)) / n
            sg = sg.reshape(pred._data.shape)
            pred._grad = sg if pred._grad is None else pred._grad + sg
        out._grad_fn = _back
    return out

def binary_cross_entropy_with_logits(logits: TriadTensor, target: TriadTensor) -> TriadTensor:
    logits = _ensure_tensor(logits)
    target = _ensure_tensor(target)
    t = target._data
    l = logits._data
    loss = (np.maximum(l, 0) - l * t + np.log1p(np.exp(-np.abs(l)))).mean()
    out = TriadTensor(loss)
    if logits._requires_grad:
        out._requires_grad = True
        out._children = [logits]
        _l = l
        _t = t

        def _back(g):
            sig = 1.0 / (1.0 + np.exp(-_l))
            n = _l.size
            sg = (sig - _t) / n
            sg = sg.reshape(logits._data.shape)
            logits._grad = sg if logits._grad is None else logits._grad + sg
        out._grad_fn = _back
    return out

def huber_loss(pred: TriadTensor, target: TriadTensor, delta: float=1.0) -> TriadTensor:
    pred = _ensure_tensor(pred)
    target = _ensure_tensor(target)
    diff = pred._data - target._data
    abs_diff = np.abs(diff)
    quadratic = np.minimum(abs_diff, delta)
    linear = abs_diff - quadratic
    loss = (0.5 * quadratic ** 2 + delta * linear).mean()
    out = TriadTensor(loss)
    if pred._requires_grad:
        out._requires_grad = True
        out._children = [pred]
        _diff = diff
        _delta = delta

        def _back(g):
            n = _diff.size
            sg = np.where(abs_diff <= _delta, _diff, _delta * np.sign(_diff)) / n
            sg = sg.reshape(pred._data.shape)
            pred._grad = sg if pred._grad is None else pred._grad + sg
        out._grad_fn = _back
    return out

def kl_div(log_p: TriadTensor, q: TriadTensor) -> TriadTensor:
    log_p = _ensure_tensor(log_p)
    q = _ensure_tensor(q)
    loss = (q._data * (np.log(q._data + 1e-10) - log_p._data)).sum(axis=-1).mean()
    out = TriadTensor(loss)
    if log_p._requires_grad:
        out._requires_grad = True
        out._children = [log_p]
        _q = q._data

        def _back(g):
            n = log_p._data.shape[0] if log_p._data.ndim > 1 else 1
            sg = -_q / n
            sg = sg.reshape(log_p._data.shape)
            log_p._grad = sg if log_p._grad is None else log_p._grad + sg
        out._grad_fn = _back
    return out

def cosine_similarity_loss(a: TriadTensor, b: TriadTensor) -> TriadTensor:
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)
    dot = (a._data * b._data).sum(axis=-1, keepdims=True)
    norm_a = np.sqrt((a._data ** 2).sum(axis=-1, keepdims=True)) + 1e-08
    norm_b = np.sqrt((b._data ** 2).sum(axis=-1, keepdims=True)) + 1e-08
    cos_sim = dot / (norm_a * norm_b)
    loss = 1.0 - cos_sim.mean()
    out = TriadTensor(loss)
    if a._requires_grad or b._requires_grad:
        out._requires_grad = True
        out._children = [a, b]
        _a = a._data
        _b = b._data

        def _back(g):
            n = _a.shape[0] if _a.ndim > 1 else 1
            if a._requires_grad:
                sg = (-_b / (norm_a * norm_b) + cos_sim * _a / norm_a ** 2) / n
                a._grad = sg.reshape(a._data.shape) if a._grad is None else a._grad + sg.reshape(a._data.shape)
            if b._requires_grad:
                sg = (-_a / (norm_a * norm_b) + cos_sim * _b / norm_b ** 2) / n
                b._grad = sg.reshape(b._data.shape) if b._grad is None else b._grad + sg.reshape(b._data.shape)
        out._grad_fn = _back
    return out

def l1_loss(pred: TriadTensor, target: TriadTensor) -> TriadTensor:
    pred = _ensure_tensor(pred)
    target = _ensure_tensor(target)
    diff = pred._data - target._data
    loss = np.abs(diff).mean()
    out = TriadTensor(loss)
    if pred._requires_grad:
        out._requires_grad = True
        out._children = [pred]
        _diff = diff

        def _back(g):
            n = _diff.size
            sg = np.sign(_diff) / n
            sg = sg.reshape(pred._data.shape)
            pred._grad = sg if pred._grad is None else pred._grad + sg
        out._grad_fn = _back
    return out

def smooth_l1_loss(pred: TriadTensor, target: TriadTensor, beta: float=1.0) -> TriadTensor:
    return huber_loss(pred, target, delta=beta)

def nll_loss(log_probs: TriadTensor, targets: TriadTensor) -> TriadTensor:
    log_probs = _ensure_tensor(log_probs)
    targets = _ensure_tensor(targets)
    idx = targets._data.astype(int).reshape(-1)
    flat = log_probs._data.reshape(-1, log_probs._data.shape[-1])
    n = flat.shape[0]
    loss = -flat[np.arange(n), idx].mean()
    out = TriadTensor(loss)
    if log_probs._requires_grad:
        out._requires_grad = True
        out._children = [log_probs]
        _idx = idx

        def _back(g):
            sg = np.zeros_like(log_probs._data).reshape(-1, log_probs._data.shape[-1])
            sg[np.arange(n), _idx] = -1.0 / n
            sg = sg.reshape(log_probs._data.shape)
            log_probs._grad = sg if log_probs._grad is None else log_probs._grad + sg
        out._grad_fn = _back
    return out