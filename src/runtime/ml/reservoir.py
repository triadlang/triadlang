from __future__ import annotations

import importlib

from runtime.ml.nn import Module
from runtime.ml.tensor import TriadTensor
from triad import ntri as np


class TriadReservoir(Module):

    def __init__(self, d_in: int, N: int = 1024, steps_per_token: int = 2,
                 drive: float = 0.3, washout: int = 0, seed: int = 7,
                 Gamma: float = 30.0, normalize: bool = True,
                 backend: str = 'cpu'):

        from runtime.core.solver import TriadParams
        from runtime.ml.solver_step import _build_ml_step_kernels
        self.params = TriadParams(N=N, D=1, T=1.0, backend=backend,
                                  seed=seed, fdt_couple=True, kT=1.0,
                                  Gamma=Gamma)
        self._kernels = _build_ml_step_kernels(self.params)
        rng = np.random.default_rng(seed)

        self.W_re = rng.standard_normal((d_in, N)) * drive
        self.W_im = rng.standard_normal((d_in, N)) * drive
        self._rng = rng
        self.N = N
        self.M = len(self.params.nu)
        self.F = 3 * N + self.M * N
        self.steps_per_token = int(steps_per_token)
        self.washout = int(washout)
        self.normalize = normalize
        self._mu = None
        self._sd = None

    def parameters(self):
        return []

    def fit_normalizer(self, feats: np.ndarray):
        flat = feats.reshape(-1, feats.shape[-1])
        self._mu = flat.mean(axis=0)
        self._sd = flat.std(axis=0) + 1e-6

    def forward(self, x) -> TriadTensor:
        from runtime.ml.solver_step import _solver_step_ml_batched
        data = x.data if isinstance(x, TriadTensor) else np.asarray(x)
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 2:
            data = data[None]
        B, L, _ = data.shape
        dre_all = data @ self.W_re
        dim_all = data @ self.W_im

        psi = np.zeros((B, self.N), dtype=np.complex128)
        y = np.zeros((B, self.M, self.N))
        N = self.N
        feats = None
        for t in range(L):
            for _ in range(self.steps_per_token):
                psi, y = _solver_step_ml_batched(
                    psi, y, self._kernels,
                    drive_re=dre_all[:, t], drive_im=dim_all[:, t],
                    rng=self._rng)
            if feats is None:

                from runtime.ml.tensor import _xp_of
                xp = _xp_of(psi)
                feats = xp.empty((B, L, self.F), dtype=xp.float32)
            feats[:, t, :N] = psi.real
            feats[:, t, N:2 * N] = psi.imag
            feats[:, t, 2 * N:3 * N] = psi.real ** 2 + psi.imag ** 2
            feats[:, t, 3 * N:] = y.reshape(B, -1)

        if self.washout > 0:
            feats = feats[:, self.washout:]
        if self.normalize:
            if self._mu is None:
                self.fit_normalizer(feats)
            feats = (feats - self._mu) / self._sd
        return TriadTensor(feats)

class CrystalMemory(Module):

    def __init__(self, d_key: int, N: int = 64, L: float = 20.0,
                 D: int = 3, seed: int = 42, backend: str = 'auto',
                 crystallize_T: float = 10.0, settle_T: float = 1.0):
        _atoms = importlib.import_module('runtime.physics.atoms')
        self._phys = _atoms.CrystalMemory(N=N, L=L, D=D, seed=seed,
                                          backend=backend)
        self.n_sites = len(self._phys.sites)
        rng = np.random.default_rng(seed)
        self.R = rng.standard_normal((d_key, self.n_sites))
        self.crystallize_T = float(crystallize_T)
        self.settle_T = float(settle_T)
        self._ready = False

    def parameters(self):
        return []

    def signature(self, vec) -> np.ndarray:
        v = vec.data if isinstance(vec, TriadTensor) else np.asarray(vec)
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        return (v @ self.R) > 0

    def _ensure(self):
        if not self._ready:
            self._phys.crystallize(self.crystallize_T)
            self._ready = True

    def write(self, vec) -> np.ndarray:

        self._ensure()
        bits = self.signature(vec)
        for i in np.flatnonzero(bits):
            self._phys.write(int(i), phase=0.0)
        self._phys.settle(self.settle_T)
        return bits

    def read_bits(self) -> np.ndarray:

        self._ensure()
        info = self._phys.read()
        return np.array([s['occupied'] for s in info], dtype=bool)

    def recall(self, vec) -> float:

        self._ensure()
        q = self.signature(vec)
        r = self.read_bits()
        return 1.0 - float(np.mean(q != r))

