
from __future__ import annotations

from collections.abc import Callable

from runtime.backend import asnumpy, get_xp
from runtime.ml.ml_device import fdtype
from triad import ntri as np

_IS_NATIVE_GRAD = True

def _is_field_shape(shape: tuple[int, ...]) -> bool:

    if not shape:
        return False
    last = shape[-1]
    return last >= 32 and len(shape) <= 3

def _grad_to_deltaV(g, dx: float, hbar: float = 1.0) -> np.ndarray:

    g = np.asarray(g, dtype=fdtype())

    g_norm = np.linalg.norm(g)
    if g_norm > 0:
        scale = hbar * dx / (g_norm + 1e-12)
        return g * scale
    return g

def solver_grad_step(psi_in, grad_out, dt: float = 0.01,
                     Lambda: float = 0.0, hbar: float = 1.0,
                     m: float = 1.0, backend: str = 'auto') -> np.ndarray:

    xp_local = get_xp(backend)
    psi = xp_local.asarray(psi_in)
    g = xp_local.asarray(grad_out)
    if psi.ndim == 1:
        N = psi.shape[0]
        B = 1
        psi = psi.reshape(1, N)
        g = g.reshape(1, N)
        squeeze = True
    else:
        B, N = psi.shape[:2]
        squeeze = False

    dx = 1.0
    k = 2.0 * xp_local.pi * xp_local.fft.fftfreq(N, d=dx)

    half_lin = xp_local.exp(-1j * (hbar ** 2 * k ** 2 / (2.0 * m)) * dt / (2.0 * hbar))

    deltaV = _grad_to_deltaV(g, dx, hbar)
    if deltaV.ndim > 1:
        deltaV = deltaV.reshape(B, N)
    psi_lin = psi * xp_local.exp(-1j * deltaV * dt / hbar)

    psi_back = xp_local.fft.ifft(xp_local.fft.fft(psi_lin) * xp_local.conj(half_lin))
    grad_in = xp_local.real(psi_back)
    if squeeze:
        grad_in = grad_in.reshape(N)
    return asnumpy(grad_in)

def wrap_grad_fn_with_solver(grad_fn: Callable, tensor_shape: tuple[int, ...],
                              is_field_output: bool = False) -> Callable:

    if not _IS_NATIVE_GRAD:
        return grad_fn
    if not _is_field_shape(tensor_shape):
        return grad_fn

    def _solver_back(g):

        grad_fn(g)

    return _solver_back

class SolverGradContext:

    def __enter__(self):
        global _IS_NATIVE_GRAD
        self._prev = _IS_NATIVE_GRAD
        _IS_NATIVE_GRAD = True

    def __exit__(self, *args):
        global _IS_NATIVE_GRAD
        _IS_NATIVE_GRAD = self._prev

def no_solver_grad():

    class _Ctx:
        def __enter__(self):
            global _IS_NATIVE_GRAD
            self._prev = _IS_NATIVE_GRAD
            _IS_NATIVE_GRAD = False
        def __exit__(self, *args):
            global _IS_NATIVE_GRAD
            _IS_NATIVE_GRAD = self._prev
    return _Ctx()
