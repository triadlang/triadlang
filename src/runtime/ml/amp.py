
from __future__ import annotations

from triad import ntri as np

try:
    import cupy as _cp
except (ImportError, ModuleNotFoundError):
    _cp = None

_AUTOCAST = False

def is_autocast() -> bool:
    return _AUTOCAST

class autocast:
    def __enter__(self):
        global _AUTOCAST
        self._prev = _AUTOCAST
        _AUTOCAST = True
        return self

    def __exit__(self, *args):
        global _AUTOCAST
        _AUTOCAST = self._prev
        return False

class GradScaler:
    def __init__(self, init_scale: float = 2.0 ** 16, growth_factor: float = 2.0,
                 backoff_factor: float = 0.5, growth_interval: int = 2000):
        self._scale = float(init_scale)
        self._growth = growth_factor
        self._backoff = backoff_factor
        self._interval = growth_interval
        self._good_steps = 0
        self._last_params = []

    @property
    def scale_value(self) -> float:
        return self._scale

    def scale(self, loss):

        if hasattr(loss, 'dtype') and str(loss.dtype) == 'float16':
            loss = loss.astype('float32')
        return loss * self._scale

    def step(self, optimizer) -> bool:

        inv = 1.0 / self._scale
        found_inf = False
        for p in optimizer.params:
            if p._grad is None:
                continue
            p._grad = p._grad * inv
            xp = _cp if (_cp is not None and isinstance(p._grad, _cp.ndarray)) else np
            if not bool(xp.isfinite(p._grad).all()):
                found_inf = True
        if found_inf:
            self._scale = max(self._scale * self._backoff, 1.0)
            self._good_steps = 0
            self._last_params = []
            try:
                optimizer.zero_grad()
            except AttributeError:
                for p in optimizer.params:
                    p._grad = None
            return False
        optimizer.step()
        self._last_params = list(optimizer.params)
        self._good_steps += 1
        if self._good_steps >= self._interval:
            self._scale *= self._growth
            self._good_steps = 0
        return True

    def update(self):

        if not self._last_params:
            return
        found_inf = False
        for p in self._last_params:
            if _cp is not None and isinstance(p._data, _cp.ndarray):
                if not bool(_cp.isfinite(p._data).all()):
                    found_inf = True
                    break
            else:
                if not bool(np.isfinite(p._data).all()):
                    found_inf = True
                    break
        if found_inf:
            self._scale = max(self._scale * self._backoff, 1.0)
            self._good_steps = 0
