"""C-vs-Python parity for native ML ops (items 5-8).

Runs ./parity_dump (C), recomputes each case with the Python runtime
(runtime/tensor.py, losses.py, metrics.py), and diffs.
"""
import subprocess, sys, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PY_ROOT = os.path.join(ROOT, "src")
sys.path.insert(0, PY_ROOT)

from runtime.ml.tensor import TriadTensor, softmax, cat, stack
from runtime import losses, metrics

out = subprocess.run([os.path.join(os.path.dirname(__file__), "parity_dump")],
                     capture_output=True, text=True, check=True).stdout
c = {}
for line in out.strip().splitlines():
    parts = line.split()
    c[parts[0]] = np.array([float(x) for x in parts[1:]])

A = np.array([[1, 5, 2], [4, 0, 6.0]])
py = {}
py["softmax_axis0"] = softmax(TriadTensor(A), axis=0)._data.reshape(-1)
py["softmax_axis1"] = softmax(TriadTensor(A), axis=1)._data.reshape(-1)
py["max_axis1"] = TriadTensor(A).max(axis=1)._data.reshape(-1)
py["min_axis0"] = TriadTensor(A).min(axis=0)._data.reshape(-1)
py["cat_axis0"] = cat([TriadTensor(A), TriadTensor([[7, 8, 9.0]])], axis=0)._data.reshape(-1)
py["stack_axis1"] = stack([TriadTensor([1, 2, 3.0]), TriadTensor([4, 5, 6.0])], axis=1)._data.reshape(-1)

prob = TriadTensor([0.2, 0.8, 0.5, 0.9])
logit = TriadTensor([-1, 2, 0, 1.5])
tgt = TriadTensor([0, 1, 0, 1.0])
py["bce"] = np.array([float(losses.bce_loss(prob, tgt)._data)])
py["bce_logits"] = np.array([float(losses.binary_cross_entropy_with_logits(logit, tgt)._data)])

L = TriadTensor([[1, 2, 0.5], [-1, 0, 3.0]])
log_sm = softmax(L, axis=1)
log_sm = TriadTensor(np.log(log_sm._data))
py["nll"] = np.array([float(losses.nll_loss(log_sm, TriadTensor([2, 0]))._data)])

pred = TriadTensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
py["accuracy"] = np.array([metrics.accuracy(pred, TriadTensor([1, 0, 0]))])
p = TriadTensor([1, 2, 3, 4.0]); t = TriadTensor([1, 2, 3, 5.0])
py["mae"] = np.array([metrics.mean_absolute_error(p, t)])
py["rmse"] = np.array([metrics.root_mean_squared_error(p, t)])

fail = 0
for k in py:
    if k not in c:
        print(f"  {k:16s} MISSING in C"); fail = 1; continue
    diff = np.max(np.abs(c[k] - py[k]))
    ok = diff < 1e-6
    print(f"  {k:16s} {'PASS' if ok else 'FAIL'}  (max|Δ|={diff:.2e})")
    if not ok:
        print(f"      C ={c[k]}\n      py={py[k]}")
        fail = 1

print("parity: FAIL" if fail else "parity: PASS")
sys.exit(fail)
