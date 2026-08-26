
from __future__ import annotations

_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
    torch = None

from runtime.backend import asnumpy, get_xp, to_xp
from runtime.ml.tensor import TriadTensor

EXTERNAL_STACK = True

def to_torch(t: TriadTensor, dtype=None, device=None) -> torch.Tensor | None:

    if not _HAS_TORCH:
        raise RuntimeError('PyTorch not installed; install torch for interop')
    arr = t._data
    if hasattr(arr, '__dlpack__'):
        return torch.from_dlpack(arr)

    np_arr = asnumpy(arr)
    tt = torch.from_numpy(np_arr)
    if dtype is not None:
        tt = tt.to(dtype)
    if device is not None:
        tt = tt.to(device)
    return tt

def from_torch(tt: torch.Tensor, requires_grad: bool = False) -> TriadTensor:

    if not _HAS_TORCH:
        raise RuntimeError('PyTorch not installed')
    if hasattr(tt, '__dlpack__'):
        xp = get_xp('auto')
        arr = to_xp(tt, xp)
    else:
        from triad import ntri as np
        arr = np.asarray(tt.detach().cpu().numpy())
    return TriadTensor(arr, requires_grad=requires_grad)

class TorchInteropError(Exception):

    pass
