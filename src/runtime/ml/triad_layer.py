from __future__ import annotations

import importlib
import math

from triad import ntri as np

try:
    import cupy as _cp
except (ImportError, ModuleNotFoundError):
    _cp = None

_T = importlib.import_module('runtime.ml.tensor')
_nn = importlib.import_module('runtime.ml.nn')
TriadTensor = _T.TriadTensor

def _xp_of(a):
    return _cp if (_cp is not None and isinstance(a, _cp.ndarray)) else np

class CausalExpConv(_T.Function):

    @staticmethod
    def forward(ctx, rho, alphas, y_init):

        xp = _xp_of(rho)
        B, L, M = rho.shape
        a = alphas.astype(rho.dtype)
        t_arr = xp.arange(L, dtype=rho.dtype)[:, None]
        K = (1.0 - a)[None, :] * a[None, :] ** t_arr
        n_fft = 2 * L

        fft_dt = xp.float64 if rho.dtype == xp.float64 else xp.float32
        rho_f = xp.fft.rfft(rho.astype(fft_dt), n=n_fft, axis=1)
        K_f = xp.fft.rfft(K.astype(fft_dt), n=n_fft, axis=0)
        conv = xp.fft.irfft(rho_f * K_f[None], n=n_fft, axis=1)[:, :L, :]
        powers = a[None, :] ** xp.arange(1, L + 1, dtype=rho.dtype)[:, None]
        homog = powers[None] * y_init[:, None, :]
        ys = (conv + homog).astype(rho.dtype)
        ctx.save_for_backward(K, (B, L, M))
        return ys

    @staticmethod
    def backward(ctx, g):
        K, (B, L, M) = ctx.saved
        xp = _xp_of(g)

        g_rev = g[:, ::-1, :]
        n_fft = 2 * L
        fft_dt = xp.float64 if g.dtype == xp.float64 else xp.float32
        g_f = xp.fft.rfft(g_rev.astype(fft_dt), n=n_fft, axis=1)
        K_f = xp.fft.rfft(K.astype(fft_dt), n=n_fft, axis=0)
        conv = xp.fft.irfft(g_f * K_f[None], n=n_fft, axis=1)[:, :L, :]
        g_rho = conv[:, ::-1, :].astype(g.dtype)
        return g_rho, None, None

class TriadLayer(_nn.Module):

    def __init__(self, d_model: int, d_psi: int = None, n_heads: int = 4,
                 Lambda: float = -0.5, Sigma_lambda: float = 0.3,
                 nu_min: float = 0.01, nu_max: float = 10.0,
                 dt: float = 0.05, fast_bias: float = 3.0,
                 gamma_0: float = 0.01, T_bath: float = 0.001,
                 seed: int = 0):
        if n_heads < 3:
            raise ValueError(
                f"triad rule: TriadLayer P2 memory field needs at least 3 "
                f"time-scales (p1/p2/p3), got n_heads={n_heads}")
        self.d_model = d_model
        self.d_psi = d_psi if d_psi is not None else d_model // 2
        self.n_heads = n_heads
        self.Lambda = float(Lambda)
        self.dt = float(dt)
        self.gamma_0 = float(gamma_0)
        self.T_bath = float(T_bath)
        if self.gamma_0 <= 0.0:
            raise ValueError(f'triad rule: P3 bath requires gamma_0 > 0, got {self.gamma_0}')
        if self.T_bath <= 0.0:
            raise ValueError(f'triad rule: P3 bath requires T_bath > 0, got {self.T_bath}')
        self._rng = np.random.default_rng(seed)

        nus = [nu_max * (nu_min / nu_max) ** (j / (n_heads - 1))
               for j in range(n_heads)]
        self.nus = np.asarray(nus, dtype=np.float64)
        self.alphas = np.exp(-self.nus * dt)

        raw = [fast_bias ** (1 - j / (n_heads - 1)) for j in range(n_heads)]
        tot = sum(raw)
        lambdas = [r / tot * Sigma_lambda for r in raw]
        self.lambdas = np.asarray(lambdas, dtype=np.float64)

        self.proj_re = _nn.triad(d_model, self.d_psi)
        self.proj_im = _nn.triad(d_model, self.d_psi)
        self.out_proj = _nn.triad(2 * self.d_psi, d_model)

    def forward(self, x: TriadTensor, y_state_in=None):
        B, L, D = x.shape
        xp = _xp_of(x.data)

        re = self.proj_re(x)
        im = self.proj_im(x)

        rho = (re * re + im * im).mean(axis=-1, keepdims=True)

        rho_heads = rho.expand(B, L, self.n_heads)
        alphas_dev = xp.asarray(self.alphas)
        if y_state_in is None:
            y_init = xp.zeros((B, self.n_heads))
        else:
            y_init = xp.asarray(y_state_in)
        ys = CausalExpConv.apply(rho_heads, TriadTensor(alphas_dev),
                                 TriadTensor(y_init))
        y_state_out = ys.data[:, -1, :]

        lambdas_dev = TriadTensor(
            xp.asarray(self.lambdas)[None, None, :].astype(x.data.dtype))
        V_mem = (ys * lambdas_dev).sum(axis=-1, keepdims=True)
        V_tot = rho * self.Lambda + V_mem

        ang = V_tot * self.dt
        cos_a = _T.cos(ang)
        sin_a = _T.sin(ang)
        damp = math.exp(-self.gamma_0 * self.dt)

        new_re = (re * cos_a + im * sin_a) * damp
        new_im = (im * cos_a - re * sin_a) * damp

        if self.gamma_0 > 0 and self.T_bath > 0:
            scale = math.sqrt(2.0 * self.gamma_0 * self.T_bath * self.dt) / math.sqrt(2.0)

            seed = int(self._rng.integers(0, 2**63 - 1))
            gen = xp.random.default_rng(seed)
            nr = gen.standard_normal(new_re.shape, dtype=xp.float32) * scale
            ni = gen.standard_normal(new_im.shape, dtype=xp.float32) * scale
            nr, ni = xp.asarray(nr), xp.asarray(ni)
            new_re = new_re + TriadTensor(nr)
            new_im = new_im + TriadTensor(ni)

        psi_real = _T.cat([new_re, new_im], axis=-1)
        update = self.out_proj(psi_real)
        return x + update, y_state_out

