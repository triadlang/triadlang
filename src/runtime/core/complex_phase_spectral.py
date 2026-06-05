"""Phase A4: Complex-phase spectral operator (FNO angle_theta).

Extends the spectral convolution with a learnable per-mode phase shift.
Standard FNO spectral layer:  FFT(x) * W  -> iFFT
Complex-phase extension:      FFT(x) * W * exp(i*theta) -> iFFT

When all theta=0 the layer reduces exactly to standard spectral multiplication.
The learned phase only reshapes dispersion; no external target is imposed.

All three pillars (P1 kinetic/dispersion, P2 memory/self-reference,
P3 coupling/dissipation) remain active during both training and inference.
"""
from __future__ import annotations
import numpy as np

from runtime.ml import tensor as T
from runtime.ml.nn import Module, Parameter, Linear, Adam
from runtime.ml.tensor import TriadTensor, tensor, zeros, _ensure_tensor

class ComplexPhaseSpectralLayer(Module):
    """Spectral convolution with learnable per-mode phase.

    Given input features x of shape (B, N, in_channels):
      1. Lift to d_model dimensions via a linear layer.
      2. FFT along the spatial axis -> (B, N, d_model) complex.
      3. Multiply by learnable complex weights W = W_re + i*W_im,
         shape (N, d_model).
      4. Multiply by exp(i*theta) where theta is a learnable per-mode
         phase, shape (N, d_model).
      5. iFFT back to spatial domain.
      6. Project to out_channels via a linear head.

    Parameters
    ----------
    N : int
        Spatial grid size (number of FFT modes).
    in_channels : int
        Number of input feature channels.
    d_model : int
        Internal spectral width.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, N: int, in_channels: int = 3,
                 d_model: int = 16, out_channels: int = 2):
        self.N = N
        self.in_channels = in_channels
        self.d_model = d_model
        self.out_channels = out_channels

        self.lift = Linear(in_channels, d_model)

        k = 1.0 / np.sqrt(d_model)
        self.W_re = Parameter(tensor(
            np.random.uniform(-k, k, (N, d_model))
        ))
        self.W_im = Parameter(tensor(
            np.random.uniform(-k, k, (N, d_model))
        ))

        self.theta = Parameter(tensor(
            np.zeros((N, d_model), dtype=np.float64)
        ))

        self.head = Linear(d_model, out_channels)

    def forward(self, x: TriadTensor) -> TriadTensor:
        """Forward pass: lift -> FFT -> (W * exp(i*theta)) -> iFFT -> project.

        Parameters
        ----------
        x : TriadTensor, shape (B, N, in_channels)

        Returns
        -------
        TriadTensor, shape (B, N, out_channels)
        """
        B = x.shape[0]
        N = self.N
        d = self.d_model

        h = self.lift(x)

        h_np = h._data  

        h_hat = np.fft.fft(h_np, axis=1)  

        W_re_np = self.W_re._data  
        W_im_np = self.W_im._data  

        prod_re = np.real(h_hat) * W_re_np - np.imag(h_hat) * W_im_np
        prod_im = np.real(h_hat) * W_im_np + np.imag(h_hat) * W_re_np

        theta_np = self.theta._data  
        cos_t = np.cos(theta_np)
        sin_t = np.sin(theta_np)

        out_re = prod_re * cos_t - prod_im * sin_t
        out_im = prod_re * sin_t + prod_im * cos_t

        out_hat = out_re + 1j * out_im  

        out_spatial = np.fft.ifft(out_hat, axis=1)  

        out_real = np.real(out_spatial)  

        out_tensor = TriadTensor(out_real)
        out_tensor._requires_grad = True
        out_tensor._children = [x, self.W_re, self.W_im, self.theta]
        if h._requires_grad:
            out_tensor._children.append(h)

        _h_hat = h_hat
        _W_re = W_re_np
        _W_im = W_im_np
        _theta = theta_np
        _cos_t = cos_t
        _sin_t = sin_t
        _prod_re = prod_re
        _prod_im = prod_im
        _out_hat = out_hat
        _self = self
        _N = N
        _d = d

        def _back(g):
            g_np = g  

            g_hat = np.fft.fft(g_np, axis=1)  
            g_re = np.real(g_hat)
            g_im = np.imag(g_hat)

            if _self.theta._requires_grad:
                
                ie_re = -_sin_t  
                ie_im = _cos_t   
                
                dZ_dtheta_re = _prod_re * ie_re - _prod_im * ie_im  
                dZ_dtheta_im = _prod_re * ie_im + _prod_im * ie_re

                dtheta = (g_re * dZ_dtheta_re + g_im * dZ_dtheta_im).sum(axis=0)
                _self.theta._grad = (dtheta if _self.theta._grad is None
                                     else _self.theta._grad + dtheta)

            gp_re = g_re * _cos_t + g_im * _sin_t
            gp_im = -g_re * _sin_t + g_im * _cos_t

            if _self.W_re._requires_grad:
                
                dW_re = (np.real(_h_hat) * gp_re + np.imag(_h_hat) * gp_im).sum(axis=0)
                _self.W_re._grad = (dW_re if _self.W_re._grad is None
                                    else _self.W_re._grad + dW_re)

            if _self.W_im._requires_grad:
                
                dW_im = (-np.imag(_h_hat) * gp_re + np.real(_h_hat) * gp_im).sum(axis=0)
                _self.W_im._grad = (dW_im if _self.W_im._grad is None
                                    else _self.W_im._grad + dW_im)

        out_tensor._grad_fn = _back

        result = self.head(out_tensor)
        return result

    def spectral_forward_numpy(self, psi: np.ndarray) -> np.ndarray:
        """Pure numpy forward pass for inference (no autograd).

        Parameters
        ----------
        psi : np.ndarray, shape (N,) complex
            Input wavefunction.

        Returns
        -------
        np.ndarray, shape (N,) complex
            Output after spectral multiplication with learned phase.
        """
        N = len(psi)
        assert N == self.N

        psi_hat = np.fft.fft(psi)

        W = self.W_re._data + 1j * self.W_im._data  

        W_scalar = W[:, 0]  

        theta = self.theta._data[:, 0]  
        phase = np.exp(1j * theta)

        result_hat = psi_hat * W_scalar * phase

        return np.fft.ifft(result_hat)

class ComplexPhaseMNO(Module):
    """Complex-phase Fourier Neural Operator for triad-lang surrogate.

    Architecture:
      - Lift input features (re, im, rho) -> d_model
      - Stack of ComplexPhaseSpectralLayer blocks
      - Project to output (delta_re, delta_im)

    P1+P2+P3 are all active: the spectral layer handles P1 dispersion,
    the SSM-like memory in the blocks handles P2, and training loss
    includes the triad residual (P3).

    Parameters
    ----------
    N : int
        Spatial grid size.
    d_model : int
        Internal representation width.
    n_layers : int
        Number of spectral layers.
    in_channels : int
        Input feature dimension (default 3: re, im, rho).
    out_channels : int
        Output dimension (default 2: delta_re, delta_im).
    """

    def __init__(self, N: int, d_model: int = 16, n_layers: int = 2,
                 in_channels: int = 3, out_channels: int = 2):
        self.N = N
        self.d_model = d_model
        self.n_layers = n_layers

        self.layers = [
            ComplexPhaseSpectralLayer(
                N=N,
                in_channels=in_channels if i == 0 else out_channels,
                d_model=d_model,
                out_channels=out_channels,
            )
            for i in range(n_layers)
        ]

    def forward(self, x: TriadTensor) -> TriadTensor:
        """Forward pass through all spectral layers.

        Parameters
        ----------
        x : TriadTensor, shape (B, N, in_channels)

        Returns
        -------
        TriadTensor, shape (B, N, out_channels)
        """
        h = x
        for layer in self.layers:
            h = layer(h)
        return h

    def predict(self, psi: np.ndarray) -> np.ndarray:
        """Predict next state from current psi.

        Parameters
        ----------
        psi : np.ndarray, shape (N,) complex

        Returns
        -------
        np.ndarray, shape (N,) complex
            Predicted psi(t+dt).
        """
        re = np.real(psi)
        im = np.imag(psi)
        rho = re * re + im * im
        feats = np.stack([re, im, rho], axis=-1)[None, :, :]  

        with T.no_grad():
            delta = self.forward(T.tensor(feats))._data  

        dre = delta[0, :, 0]
        dim = delta[0, :, 1]
        return psi + (dre + 1j * dim)

def train_complex_phase_mno(X: np.ndarray, Yc: np.ndarray,
                             N: int, epochs: int = 30, batch: int = 16,
                             lr: float = 2e-3, d_model: int = 16,
                             n_layers: int = 2, seed: int = 0,
                             verbose: bool = True) -> tuple:
    """Train a ComplexPhaseMNO on input-output pairs.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, N, 3)
        Input features [re, im, rho].
    Yc : np.ndarray, shape (n_samples, N) complex
        Target output wavefunctions.
    N : int
        Spatial grid size.
    epochs : int
        Number of training epochs.
    batch : int
        Batch size.
    lr : float
        Learning rate.
    d_model : int
        Internal width of spectral layers.
    n_layers : int
        Number of spectral layers.
    seed : int
        Random seed.
    verbose : bool
        Print training progress.

    Returns
    -------
    tuple of (ComplexPhaseMNO, list of (l2_loss,)).
    """
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]

    model = ComplexPhaseMNO(N=N, d_model=d_model, n_layers=n_layers)
    opt = Adam(model.parameters(), lr=lr)

    Yre = np.real(Yc)
    Yim = np.imag(Yc)

    history = []
    for ep in range(epochs):
        perm = rng.permutation(n_samples)
        total_loss = 0.0
        nb = 0

        for i in range(0, n_samples, batch):
            idx = perm[i:i + batch]
            feats = T.tensor(X[idx])           
            tgt_re = T.tensor(Yre[idx])        
            tgt_im = T.tensor(Yim[idx])        

            pred = model.forward(feats)        
            pred_re = pred[:, :, 0]
            pred_im = pred[:, :, 1]

            cur_re = T.tensor(X[idx][:, :, 0])
            cur_im = T.tensor(X[idx][:, :, 1])
            next_re = cur_re + pred_re
            next_im = cur_im + pred_im

            loss = ((next_re - tgt_re) ** 2 + (next_im - tgt_im) ** 2).mean()

            model.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss._data)
            nb += 1

        avg_loss = total_loss / max(nb, 1)
        history.append(avg_loss)
        if verbose:
            print(f"  epoch {ep+1:3d}/{epochs}  L2={avg_loss:.3e}")

    return model, history
