from __future__ import annotations
import numpy as np
from runtime.tensor import TriadTensor, _ensure_tensor

def accuracy(pred: TriadTensor, target: TriadTensor) -> float:
    pred = _ensure_tensor(pred)
    target = _ensure_tensor(target)
    pred_idx = pred._data.argmax(axis=-1)
    tgt_idx = target._data.astype(int)
    if tgt_idx.ndim > 1 and tgt_idx.shape[-1] == 1:
        tgt_idx = tgt_idx.squeeze(-1)
    if pred_idx.ndim > tgt_idx.ndim:
        tgt_idx = np.broadcast_to(tgt_idx, pred_idx.shape)
    return float((pred_idx == tgt_idx).mean())

def top_k_accuracy(pred: TriadTensor, target: TriadTensor, k: int=5) -> float:
    pred = _ensure_tensor(pred)
    target = _ensure_tensor(target)
    tgt_idx = target._data.astype(int).reshape(-1)
    flat = pred._data.reshape(-1, pred._data.shape[-1])
    n = flat.shape[0]
    top_k_idx = np.argpartition(-flat, k, axis=-1)[:, :k]
    correct = sum((1 for i in range(n) if tgt_idx[i] in top_k_idx[i]))
    return correct / n

def perplexity(loss_value: float) -> float:
    return float(np.exp(loss_value))

def confusion_matrix(pred: TriadTensor, target: TriadTensor, num_classes: int | None=None) -> np.ndarray:
    pred = _ensure_tensor(pred)
    target = _ensure_tensor(target)
    pred_idx = pred._data.argmax(axis=-1).astype(int)
    tgt_idx = target._data.astype(int)
    if tgt_idx.ndim > 1:
        tgt_idx = tgt_idx.squeeze(-1)
    pred_flat = pred_idx.reshape(-1)
    tgt_flat = tgt_idx.reshape(-1)[:len(pred_flat)]
    if num_classes is None:
        num_classes = max(pred_flat.max(), tgt_flat.max()) + 1
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, t in zip(pred_flat, tgt_flat):
        cm[t, p] += 1
    return cm

def precision_recall_f1(pred: TriadTensor, target: TriadTensor, num_classes: int | None=None, average: str='macro'):
    cm = confusion_matrix(pred, target, num_classes)
    n_classes = cm.shape[0]
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    if average == 'macro':
        return (float(prec.mean()), float(rec.mean()), float(f1.mean()))
    return (prec.tolist(), rec.tolist(), f1.tolist())

def r2_score(pred: TriadTensor, target: TriadTensor) -> float:
    pred = _ensure_tensor(pred)
    target = _ensure_tensor(target)
    ss_res = ((target._data - pred._data) ** 2).sum()
    ss_tot = ((target._data - target._data.mean()) ** 2).sum()
    if ss_tot == 0:
        return 1.0
    return float(1.0 - ss_res / ss_tot)

def mean_absolute_error(pred: TriadTensor, target: TriadTensor) -> float:
    pred = _ensure_tensor(pred)
    target = _ensure_tensor(target)
    return float(np.abs(pred._data - target._data).mean())

def root_mean_squared_error(pred: TriadTensor, target: TriadTensor) -> float:
    pred = _ensure_tensor(pred)
    target = _ensure_tensor(target)
    return float(np.sqrt(((pred._data - target._data) ** 2).mean()))