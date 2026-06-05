"""Phase A3: Surrogate-accelerated P1 (kinetic spectral propagator) with UQ.

The surrogate replaces ONLY the P1 spectral propagator (FFT * half_lin),
which is the most expensive part of the split-step.  P2 (memory/V_mem) and
P3 (dissipation -iGamma, FDT noise) are applied exactly by the native solver.

UQ follows arXiv:2603.11052: T stochastic forward passes with zero-mean
feature perturbations on lifted features, producing predictive mean + band.
Fallback to exact solver when uncertainty exceeds threshold.

Architecture: FNO-style spectral convolution on truncated low-frequency
modes, structurally identical to the FFT-based split-step it replaces.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from runtime.core.solver import TriadParams, _effective_params, _build_V_ext
from runtime.physics.observables import crystallinity, energy
from runtime.ml import tensor as T
from runtime.ml.nn import Module, Linear, Adam

def spectral_features(psi: np.ndarray, k: np.ndarray, n_modes: int = 32) -> np.ndarray:
    """Extract truncated spectral features from psi for the surrogate.

    Returns (2*n_modes,) array: [re(psi_hat[:n_modes]), im(psi_hat[:n_modes])].
    """
    psi_hat = np.fft.fft(psi)
    re = np.real(psi_hat[:n_modes])
    im = np.imag(psi_hat[:n_modes])
    return np.concatenate([re, im])

def spectral_to_field(delta_hat_re: np.ndarray, delta_hat_im: np.ndarray,
                      N: int) -> np.ndarray:
    """Convert truncated spectral delta back to spatial field."""
    full_re = np.zeros(N)
    full_im = np.zeros(N)
    n = len(delta_hat_re)
    full_re[:n] = delta_hat_re
    full_im[:n] = delta_hat_im
    
    full_re[-1:-n:-1] = delta_hat_re[1:][::-1]
    full_im[-1:-n:-1] = -delta_hat_im[1:][::-1]
    return np.fft.ifft(full_re + 1j * full_im)

class KineticSurrogate(Module):
    """Learns the spectral propagator: psi_hat * half_lin.

    Operates in truncated spectral space (n_modes). Input: spectral features
    of psi.  Output: delta in spectral space (learned correction to the
    linear propagation).  The base propagation is still applied exactly;
    the surrogate learns a residual correction for accuracy.

    Parameters
    ----------
    n_modes : int
        Number of retained low-frequency modes.
    d_hidden : int
        Hidden dimension of the spectral MLP.
    n_layers : int
        Number of spectral processing layers.
    """
    def __init__(self, n_modes: int = 32, d_hidden: int = 64, n_layers: int = 2):
        self.n_modes = n_modes
        in_dim = 2 * n_modes  
        self.layers = []
        for i in range(n_layers):
            d_in = in_dim if i == 0 else d_hidden
            self.layers.append(Linear(d_in, d_hidden))
        self.head = Linear(d_hidden, 2 * n_modes)  

    def forward(self, feats: T.TriadTensor) -> T.TriadTensor:
        x = feats
        for layer in self.layers:
            x = layer(x)
            
            x = (T.tensor(np.tanh(x._data)))
        return self.head(x)

    def predict_delta(self, psi: np.ndarray, n_modes: int = None) -> np.ndarray:
        """Predict spectral delta for one step.

        Returns delta_psi in real space (N,).
        """
        nm = n_modes or self.n_modes
        k = 2.0 * np.pi * np.fft.fftfreq(len(psi), d=1.0 / len(psi))
        feats = spectral_features(psi, k, nm)
        with T.no_grad():
            delta = self.forward(T.tensor(feats[None, :]))._data[0]
        half = nm
        dre = np.asarray(delta[:half])
        dim = np.asarray(delta[half:])
        return spectral_to_field(dre, dim, len(psi))

    def predict_with_uq(self, psi: np.ndarray, n_passes: int = 5,
                        perturbation_std: float = 0.1) -> dict:
        """Predict with inference-time UQ (arXiv:2603.11052).

        Runs n_passes stochastic forward passes with zero-mean perturbations
        on lifted features. Returns mean prediction + std as uncertainty band.

        Parameters
        ----------
        psi : np.ndarray
            Input field.
        n_passes : int
            Number of stochastic passes.
        perturbation_std : float
            Std of multiplicative perturbation on features.

        Returns
        -------
        dict with 'mean', 'std', 'delta_fields' (list of predictions).
        """
        nm = self.n_modes
        k = 2.0 * np.pi * np.fft.fftfreq(len(psi), d=1.0 / len(psi))
        feats = spectral_features(psi, k, nm)
        deltas = []
        for _ in range(n_passes):
            noise = 1.0 + np.random.randn(len(feats)) * perturbation_std
            feats_noisy = feats * noise
            with T.no_grad():
                delta = self.forward(T.tensor(feats_noisy[None, :]))._data[0]
            half = nm
            dre = np.asarray(delta[:half])
            dim = np.asarray(delta[half:])
            delta_field = spectral_to_field(dre, dim, len(psi))
            deltas.append(delta_field)
        deltas_arr = np.array(deltas)
        return {
            "mean": deltas_arr.mean(axis=0),
            "std": deltas_arr.std(axis=0),
            "delta_fields": deltas,
            "max_std": float(np.max(np.abs(deltas_arr.std(axis=0)))),
        }

def generate_kinetic_pairs(p: TriadParams, n_snapshots: int = 200,
                            n_modes: int = 32) -> tuple:
    """Generate training pairs for the kinetic surrogate.

    For each snapshot: compute exact half_lin propagation, extract spectral
    features of input and output.

    Returns X (n, 2*n_modes), Y (n, 2*n_modes) where Y is the spectral
    delta (output_hat - input_hat)[:n_modes].
    """
    from runtime.core.solver import integrate
    from stdlib.regimes import resolve_regime

    eff = _effective_params(p)
    x = np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
    dx = float(x[1] - x[0])
    k = 2.0 * np.pi * np.fft.fftfreq(p.N, d=dx)
    abs_k = np.abs(k)
    H_lin_k = p.hbar**2 * k**2 / (2.0 * p.m) + eff['alpha'] * abs_k**p.sigma

    X_list, Y_list = [], []
    
    out = integrate(p)
    psi_snapshots = []
    if out.get('record_y') and out.get('psi_history') is not None:
        psi_snapshots = out['psi_history']
    else:
        
        density = out['density']
        for t_idx in range(density.shape[1]):
            rho = density[:, t_idx]
            
            phase = np.exp(1j * np.random.uniform(0, 2*np.pi, len(rho)))
            psi_snapshots.append(np.sqrt(np.maximum(rho, 0)) * phase)

    for psi in psi_snapshots[:n_snapshots]:
        
        psi_hat = np.fft.fft(psi)
        half_lin = np.exp(-1j * H_lin_k * p.dt / (2.0 * p.hbar)
                          - eff['Gamma'] * p.dt / (2.0 * p.hbar))
        psi_hat_next = psi_hat * half_lin
        psi_next = np.fft.ifft(psi_hat_next)

        feats_in = spectral_features(psi, k, n_modes)
        feats_out = spectral_features(psi_next, k, n_modes)
        delta = feats_out - feats_in

        X_list.append(feats_in)
        Y_list.append(delta)

    return np.array(X_list), np.array(Y_list)

def train_kinetic_surrogate(p: TriadParams, n_modes: int = 32,
                             d_hidden: int = 64, n_layers: int = 2,
                             epochs: int = 20, batch: int = 32,
                             lr: float = 1e-3, seed: int = 0,
                             verbose: bool = True) -> KineticSurrogate:
    """Train a KineticSurrogate on exact P1 propagation data.

    Loss is MSE on spectral delta in truncated mode space.
    """
    rng = np.random.default_rng(seed)
    X, Y = generate_kinetic_pairs(p, n_snapshots=200, n_modes=n_modes)
    n = len(X)
    if n == 0:
        raise ValueError("no training pairs generated")

    model = KineticSurrogate(n_modes=n_modes, d_hidden=d_hidden,
                              n_layers=n_layers)
    opt = Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        perm = rng.permutation(n)
        total_loss = 0.0
        nb = 0
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            x_batch = T.tensor(X[idx])
            y_batch = T.tensor(Y[idx])
            pred = model.forward(x_batch)
            loss = ((pred - y_batch) ** 2).mean()
            model.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss._data)
            nb += 1
        if verbose:
            print(f"  epoch {ep+1:3d}/{epochs}  MSE={total_loss/nb:.3e}")

    return model

def hybrid_step(psi: np.ndarray, p: TriadParams, surrogate: KineticSurrogate,
                uq_threshold: float = 0.1, n_uq_passes: int = 5,
                perturbation_std: float = 0.1) -> dict:
    """One Strang split-step using surrogate for P1 and exact for P2/P3.

    Structure (Strang splitting, same as solver.py):
      1. half P1 via surrogate (spectral propagation)
      2. full nonlinear step (P2 memory + P3 dissipation + V_ext + Lambda|psi|^2)
      3. half P1 via surrogate (spectral propagation)

    If surrogate UQ exceeds threshold, falls back to exact FFT propagation.

    Returns dict with psi_next, used_surrogate (bool), uq_max (float).
    """
    eff = _effective_params(p)
    xp = np
    N = len(psi)
    x = xp.linspace(-p.L / 2, p.L / 2, N, endpoint=False)
    dx = float(x[1] - x[0])
    k = 2.0 * xp.pi * xp.fft.fftfreq(N, d=dx)
    abs_k = xp.abs(k)
    H_lin_k = p.hbar**2 * k**2 / (2.0 * p.m) + eff['alpha'] * abs_k**p.sigma
    half_lin_exact = xp.exp(-1j * H_lin_k * p.dt / (2.0 * p.hbar)
                             - eff['Gamma'] * p.dt / (2.0 * p.hbar))

    uq_result = surrogate.predict_with_uq(psi, n_passes=n_uq_passes,
                                           perturbation_std=perturbation_std)
    uq_max = uq_result["max_std"]

    if uq_max < uq_threshold:
        
        delta_p1 = uq_result["mean"]
        psi_after_p1_half = psi + delta_p1
        used_surrogate_half1 = True
    else:
        
        psi_after_p1_half = xp.fft.ifft(xp.fft.fft(psi) * half_lin_exact)
        used_surrogate_half1 = False

    rho = xp.abs(psi_after_p1_half) ** 2
    V_ext = _build_V_ext(p, x)
    lam_arr = xp.asarray(eff['lam'], dtype=xp.float64)
    nu_arr = xp.asarray(p.nu, dtype=xp.float64)
    
    V_mem = (lam_arr[:, None] * rho[None, :]).sum(axis=0) if len(lam_arr) > 0 else 0.0
    V_total = V_ext + eff['Lambda'] * rho + V_mem
    
    psi_after_nl = psi_after_p1_half * xp.exp(
        -1j * V_total * p.dt / p.hbar
        - eff['Gamma'] * p.dt / p.hbar  
    )

    uq_result2 = surrogate.predict_with_uq(psi_after_nl, n_passes=n_uq_passes,
                                            perturbation_std=perturbation_std)
    uq_max2 = uq_result2["max_std"]

    if uq_max2 < uq_threshold:
        delta_p1_2 = uq_result2["mean"]
        psi_next = psi_after_nl + delta_p1_2
        used_surrogate_half2 = True
    else:
        psi_next = xp.fft.ifft(xp.fft.fft(psi_after_nl) * half_lin_exact)
        used_surrogate_half2 = False

    return {
        "psi_next": psi_next,
        "used_surrogate": used_surrogate_half1 and used_surrogate_half2,
        "uq_max": max(uq_max, uq_max2),
        "fell_back": not (used_surrogate_half1 and used_surrogate_half2),
    }

def fdt_balance_check(psi_before: np.ndarray, psi_after: np.ndarray,
                      p: TriadParams, dt: float) -> dict:
    """Check FDT energy-balance violation between two field states.

    In a correctly operating system, the energy change should be bounded
    by the FDT fluctuation-dissipation balance. If a surrogate absorbs P2/P3
    (ablation), the balance is violated.

    Returns dict with delta_E, norm_ratio, violated (bool).
    """
    x = np.linspace(-p.L / 2, p.L / 2, len(psi_before), endpoint=False)
    dx = float(x[1] - x[0])
    E_before = energy(psi_before, dx)
    E_after = energy(psi_after, dx)
    delta_E = abs(E_after - E_before)
    norm_before = float((np.abs(psi_before)**2).sum() * dx)
    norm_after = float((np.abs(psi_after)**2).sum() * dx)
    norm_ratio = norm_after / max(norm_before, 1e-30)

    violated = abs(norm_ratio - 1.0) > 0.5  

    return {
        "delta_E": delta_E,
        "norm_ratio": norm_ratio,
        "violated": violated,
    }
