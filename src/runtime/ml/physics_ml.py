
from __future__ import annotations

from runtime.ml.nn import LayerNorm, Module, Parameter, triad
from runtime.ml.tensor import TriadTensor, no_grad, tensor
from runtime.ml.tensor import cat as t_cat
from runtime.ml.tensor import cos as t_cos
from runtime.ml.tensor import sin as t_sin
from triad import ntri as np


class FourierFeatures(Module):

    def __init__(self, in_dim: int, n_features: int = 64, sigma: float = 1.0):
        self.n_features = n_features
        self.B = tensor(np.random.randn(in_dim, n_features // 2) * sigma)
        self.B._requires_grad = False

    def forward(self, x: TriadTensor) -> TriadTensor:
        x_proj = x @ self.B._data * 2.0 * np.pi
        return t_cat([t_sin(TriadTensor(x_proj)), t_cos(TriadTensor(x_proj))], axis=-1)

class SpectralConv1d(Module):

    def __init__(self, in_channels: int, out_channels: int, n_modes: int):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        scale = 1.0 / (in_channels * out_channels)
        self.weight = Parameter(tensor(np.random.randn(in_channels, out_channels, n_modes) * scale))

    def forward(self, x: TriadTensor) -> TriadTensor:
        x_np = x._data
        if x_np.ndim == 2:
            x_np = x_np[np.newaxis, :, :]
        B, C, N = x_np.shape
        x_hat = np.fft.fft(x_np, axis=-1)
        out_hat = np.zeros((B, self.out_channels, N), dtype=complex)
        w = self.weight._data
        for b in range(B):
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    out_hat[b, oc, :self.n_modes] += w[ic, oc, :] * x_hat[b, ic, :self.n_modes]
        out = np.fft.ifft(out_hat, axis=-1).real
        return TriadTensor(out.squeeze(0) if x._data.ndim == 2 else out)

class FourierNeuralOperator(Module):

    def __init__(self, in_features: int, out_features: int, d_model: int = 64,
                 n_modes: int = 32, n_layers: int = 4):
        self.lift = triad(in_features, d_model)
        self.spectral_layers = [SpectralConv1d(d_model, d_model, n_modes)
                                for _ in range(n_layers)]
        self.local_layers = [triad(d_model, d_model) for _ in range(n_layers)]
        self.norms = [LayerNorm(d_model) for _ in range(n_layers)]
        self.head = triad(d_model, out_features)

    def forward(self, x: TriadTensor) -> TriadTensor:
        x = self.lift(x)
        for sp, loc, norm in zip(self.spectral_layers, self.local_layers, self.norms):
            x_n = norm(x)
            x_sp = sp(TriadTensor(x_n._data.transpose(0, 2, 1) if x_n._data.ndim == 3 else x_n._data[np.newaxis].transpose(0, 2, 1)))
            if x_sp._data.ndim == 3 and x_n._data.ndim == 3:
                x_sp = TriadTensor(x_sp._data.transpose(0, 2, 1))
            x = x + x_sp + loc(x_n)
        return self.head(x)

class HamiltonianLayer(Module):

    def __init__(self, d_model: int, d_state: int = 64):
        self.d_model = d_model
        self.d_state = d_state
        self.in_re = triad(d_model, d_state)
        self.in_im = triad(d_model, d_state)
        self.out_re = triad(d_state, d_model)
        self.out_im = triad(d_state, d_model)
        self.log_H_diag = Parameter(tensor(np.random.randn(d_state) * 0.1))
        self.dt = 0.01

    def forward(self, x: TriadTensor) -> TriadTensor:
        re = self.in_re(x)
        im = self.in_im(x)
        omega = np.exp(self.log_H_diag._data) * self.dt
        cos_w = np.cos(omega)
        sin_w = np.sin(omega)
        re_data = re._data * cos_w + im._data * sin_w
        im_data = im._data * cos_w - re._data * sin_w
        return self.out_re(TriadTensor(re_data)) + self.out_im(TriadTensor(im_data))

class TriadResidualLoss:

    def __init__(self, hbar: float = 1.0, m: float = 1.0, Lambda: float = -1.0,
                 Gamma: float = 0.0, dx: float = 0.1, dt: float = 0.01):
        self.hbar = hbar
        self.m = m
        self.Lambda = Lambda
        self.Gamma = Gamma
        self.dx = dx
        self.dt = dt

    def __call__(self, psi_re: TriadTensor, psi_im: TriadTensor) -> TriadTensor:
        re = psi_re._data
        im = psi_im._data
        N = re.shape[-1]
        k = 2.0 * np.pi * np.fft.fftfreq(N, d=self.dx)
        kin_coeff = self.hbar ** 2 * k ** 2 / (2.0 * self.m)
        re_hat = np.fft.fft(re, axis=-1)
        im_hat = np.fft.fft(im, axis=-1)
        kin_re = np.fft.ifft(kin_coeff * re_hat, axis=-1).real
        kin_im = np.fft.ifft(kin_coeff * im_hat, axis=-1).real
        rho = re ** 2 + im ** 2
        V_nl = self.Lambda * rho
        H_re = kin_re + V_nl * re + self.Gamma * im
        H_im = kin_im + V_nl * im - self.Gamma * re
        res_re = H_re
        res_im = H_im
        loss = TriadTensor(np.mean(res_re ** 2 + res_im ** 2))
        if psi_re._requires_grad or psi_im._requires_grad:
            loss._requires_grad = True
            loss._children = [psi_re, psi_im]

            def _back(g):

                val = float(g) if getattr(g, 'ndim', 0) == 0 else float(g.item())
                n = res_re.size
                scale = 2.0 * val / n

                def K(v):
                    return np.fft.ifft(kin_coeff * np.fft.fft(v, axis=-1),
                                       axis=-1).real

                cross = 2.0 * self.Lambda * re * im
                d_re = scale * (K(res_re)
                                + self.Lambda * (3.0 * re ** 2 + im ** 2) * res_re
                                + (cross - self.Gamma) * res_im)
                d_im = scale * (K(res_im)
                                + self.Lambda * (re ** 2 + 3.0 * im ** 2) * res_im
                                + (cross + self.Gamma) * res_re)
                if psi_re._requires_grad:
                    psi_re._grad = d_re if psi_re._grad is None \
                        else psi_re._grad + d_re
                if psi_im._requires_grad:
                    psi_im._grad = d_im if psi_im._grad is None \
                        else psi_im._grad + d_im
            loss._grad_fn = _back
        return loss

class ConservationLoss:

    def __init__(self, dx: float = 0.1):
        self.dx = dx

    def norm_loss(self, psi_re: TriadTensor, psi_im: TriadTensor,
                  target_norm: float = 1.0) -> TriadTensor:
        rho = psi_re * psi_re + psi_im * psi_im
        norm = rho.sum() * self.dx
        diff = norm - target_norm
        return diff * diff

    def energy_loss(self, psi_re: TriadTensor, psi_im: TriadTensor,
                    target_energy: float | None = None) -> TriadTensor:
        rho = psi_re * psi_re + psi_im * psi_im
        E = rho.sum() * self.dx
        if target_energy is not None:
            diff = E - target_energy
            return diff * diff
        return E

class BoundaryLoss:

    def __init__(self, mode: str = 'periodic'):
        self.mode = mode

    def __call__(self, field: TriadTensor) -> TriadTensor:
        if self.mode == 'periodic':
            diff = field._data[..., 0] - field._data[..., -1]
            return TriadTensor(np.mean(diff ** 2))
        elif self.mode == 'dirichlet':
            return TriadTensor(np.mean(field._data[..., 0] ** 2 + field._data[..., -1] ** 2))
        return TriadTensor(0.0)

class PhysicsInformedMLP(Module):

    def __init__(self, spatial_dim: int = 1, hidden_dim: int = 64, n_layers: int = 4,
                 fourier_features: int = 64, fourier_sigma: float = 2.0):
        self.fourier = FourierFeatures(spatial_dim, fourier_features, fourier_sigma)
        layers = []
        d_in = fourier_features
        for _ in range(n_layers):
            layers.append(triad(d_in, hidden_dim))
            d_in = hidden_dim
        self.layers = layers
        self.head = triad(hidden_dim, 2)
        self.n_layers = n_layers

    def forward(self, x: TriadTensor) -> TriadTensor:
        x = self.fourier(x)
        for layer in self.layers:
            x = layer(x)
            x = TriadTensor(np.tanh(x._data))
        return self.head(x)

    def predict_field(self, coords: np.ndarray) -> np.ndarray:
        with no_grad():
            out = self.forward(tensor(coords))
        re = out._data[..., 0]
        im = out._data[..., 1]
        return re + 1j * im

class TriadNeuralOperator(Module):

    def __init__(self, N: int = 64, d_model: int = 32, n_modes: int = 32,
                 n_layers: int = 4, n_memory: int = 3):
        self.N = N
        self.lift = triad(3, d_model)
        self.fno_layers = [SpectralConv1d(d_model, d_model, n_modes)
                           for _ in range(n_layers)]
        self.local_layers = [triad(d_model, d_model) for _ in range(n_layers)]
        self.norms = [LayerNorm(d_model) for _ in range(n_layers)]
        self.n_memory = n_memory
        self.log_nu = Parameter(tensor(np.linspace(0.5, 2.0, n_memory)))
        self.lam = Parameter(tensor(np.random.uniform(-0.3, -0.05, (n_memory, N)) * 0.02))
        self.head = triad(d_model, 2)

    def forward(self, x: TriadTensor) -> TriadTensor:
        x = self.lift(x)
        for fno, loc, norm in zip(self.fno_layers, self.local_layers, self.norms):
            x_n = norm(x)
            x_sp = fno(TriadTensor(x_n._data.transpose(0, 2, 1) if x_n._data.ndim == 3 else x_n._data[np.newaxis].transpose(0, 2, 1)))
            if x_sp._data.ndim == 3:
                x_sp = TriadTensor(x_sp._data.transpose(0, 2, 1))
            x = x + x_sp + loc(x_n)
        return self.head(x)

    def predict_step(self, psi: np.ndarray, y_memory: np.ndarray | None = None,
                     ctx: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
        N = len(psi)
        re = np.real(psi)
        im = np.imag(psi)
        rho = re * re + im * im
        feats = np.stack([re, im, rho], axis=-1)[None, :, :]
        with no_grad():
            delta = self.forward(tensor(feats))._data
        dre = delta[0, :, 0]
        dim = delta[0, :, 1]
        psi_next = psi + (dre + 1j * dim)
        if y_memory is None:
            y_memory = np.zeros((self.n_memory, N))
        nu = np.exp(self.log_nu._data)
        lam = self.lam._data
        rho_next = np.abs(psi_next) ** 2
        y_next = np.exp(-nu[:, None]) * y_memory + (1 - np.exp(-nu[:, None])) * rho_next[None, :]
        return psi_next, y_next
