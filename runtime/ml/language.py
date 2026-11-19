from __future__ import annotations
import numpy as np
from runtime.ml import ml_device as _D
from runtime.ml.ml_device import xp
from runtime.ml.tensor import TriadTensor, tensor, softmax, _ensure_tensor, no_grad
from runtime.ml.nn import Module, Parameter, Linear, LayerNorm, Embedding
from runtime.ml.serialization import save_weights, load_weights

class CharTokenizer:
    def __init__(self, text: str = ''):
        chars = sorted(set(text))
        self.vocab_size = len(chars) + 1
        self.pad_id = 0
        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        self.itos = {i + 1: ch for i, ch in enumerate(chars)}
        self.itos[0] = ''

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, self.pad_id) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.itos.get(i, '') for i in ids)

def _solver_step(psi_re, psi_im, params, dt, dx, rng=None):
    _xp = xp
    N = psi_re.shape[-1]
    k = 2.0 * _xp.pi * _xp.fft.fftfreq(N, d=dx)
    hbar = params['hbar']
    m = params['m']
    Lambda = params['Lambda']
    Gamma = params['Gamma']
    alpha = params['alpha']
    sigma = params['sigma']
    kT = params['kT']
    lam = _xp.asarray(params['lam'])
    nu = _xp.asarray(params['nu'])
    V_ext = _xp.asarray(params['V_ext'])

    abs_k = _xp.abs(k)
    H_lin = hbar ** 2 * k ** 2 / (2.0 * m) + alpha * abs_k ** sigma
    decay = _xp.exp(-Gamma * dt / (2.0 * hbar))
    half_re = _xp.cos(H_lin * dt / (2.0 * hbar)) * decay
    half_im = -_xp.sin(H_lin * dt / (2.0 * hbar)) * decay

    M = len(nu)
    ou_decay = _xp.exp(-nu * dt * 0.5) if M else None

    def _fft_mul(re, im, fr, fi):
        f_re = _xp.fft.fft(re) * fr - _xp.fft.fft(im) * fi
        f_im = _xp.fft.fft(re) * fi + _xp.fft.fft(im) * fr
        return _xp.fft.ifft(f_re).real, _xp.fft.ifft(f_im).real

    re1, im1 = _fft_mul(psi_re, psi_im, half_re, half_im)
    rho = re1 * re1 + im1 * im1

    y_list = []
    for j in range(M):
        yj_prev = params.get(f'y_{j}', _xp.zeros(N))
        if ou_decay is not None:
            yj = ou_decay[j] * yj_prev + (1.0 - ou_decay[j]) * rho
        else:
            yj = rho
        y_list.append(yj)
    V_mem = _xp.zeros(N) if M == 0 else sum(lam[j] * y_list[j] for j in range(M))
    V_tot = V_ext + Lambda * rho + V_mem

    phase = -V_tot * dt / hbar
    re2 = re1 * _xp.cos(phase) - im1 * _xp.sin(phase)
    im2 = im1 * _xp.cos(phase) + re1 * _xp.sin(phase)

    rho2 = re2 * re2 + im2 * im2
    y_list2 = []
    for j in range(M):
        if ou_decay is not None:
            yj2 = ou_decay[j] * y_list[j] + (1.0 - ou_decay[j]) * rho2
        else:
            yj2 = rho2
        y_list2.append(yj2)

    f_FDT = 2.0 * Gamma * dx * kT / hbar if Gamma > 0 else 0.0
    noise_amp = float(_xp.sqrt(_xp.asarray(f_FDT * dt / dx))) if f_FDT > 0 else 0.0
    if noise_amp > 0:
        if rng is None:
            rng = _xp.random.default_rng()
        re2 = re2 + float(noise_amp) * rng.standard_normal(re2.shape) / 1.4142135623730951
        im2 = im2 + float(noise_amp) * rng.standard_normal(im2.shape) / 1.4142135623730951

    re3, im3 = _fft_mul(re2, im2, half_re, half_im)

    new_params = dict(params)
    for j in range(M):
        new_params[f'y_{j}'] = y_list2[j]

    return re3, im3, new_params

class TriadFullBlock(Module):
    def __init__(self, d_model: int, N: int = 64, n_memory: int = 3):
        self.d_model = d_model
        self.N = N
        self.n_memory = n_memory
        self.n_memory = n_memory
        scale = 0.02
        self.in_proj = Linear(d_model, 2 * N, bias=True)
        self.out_proj = Linear(N, d_model, bias=True)
        self.ln = LayerNorm(d_model)
        self.p_omega = Parameter(tensor(np.full(N, 0.05)))
        self.p_Lambda = Parameter(tensor(np.full(N, -0.5)))
        self.p_Gamma = Parameter(tensor(np.full(N, 0.05)))
        self.p_alpha = Parameter(tensor(np.full(N, 0.15)))
        self.p_sigma = Parameter(tensor(np.full(N, 1.5)))
        self.p_lam = Parameter(tensor(np.random.uniform(-0.3, -0.05, (n_memory, N)) * scale))
        self.p_nu = Parameter(tensor(np.linspace(0.5, 2.0, n_memory)))
        self.p_hbar = Parameter(tensor(np.array(1.0)))
        self.p_m = Parameter(tensor(np.array(1.0)))
        self.p_kT = Parameter(tensor(np.array(1.0)))
        self.dt = 0.005
        self.dx = 1.0 / N
        self.training = True
        self._y_state = None

    def _get_params(self):
        _xp = xp
        return {
            'omega': float(self.p_omega._data.mean()),
            'Lambda': float(self.p_Lambda._data.mean()),
            'Gamma': float(abs(self.p_Gamma._data.mean())),
            'alpha': float(self.p_alpha._data.mean()),
            'sigma': float(abs(self.p_sigma._data.mean()) + 0.5),
            'hbar': float(abs(self.p_hbar._data) + 0.1),
            'm': float(abs(self.p_m._data) + 0.1),
            'kT': float(abs(self.p_kT._data)),
            'lam': self.p_lam._data,
            'nu': self.p_nu._data,
            'V_ext': _xp.zeros(self.N),
        }

    def forward(self, x: TriadTensor) -> TriadTensor:
        residual = x
        x = self.ln(x)

        if x.ndim == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])
        B, T, D = x.shape

        proj = self.in_proj(x)
        psi_re_data = proj._data[:, :, :self.N]
        psi_im_data = proj._data[:, :, self.N:]

        params_dev = self._get_params()

        re_out = []
        im_out = []
        batch_params = [dict(params_dev) for _ in range(B)]
        block_rng = _D.xp.random.default_rng() if self.training else _D.xp.random.default_rng(42)

        for b in range(B):
            bp = batch_params[b]
            re_seq = []
            im_seq = []
            for t in range(T):
                re_t = psi_re_data[b, t]
                im_t = psi_im_data[b, t]
                re_new, im_new, bp = _solver_step(re_t, im_t, bp, self.dt, self.dx, block_rng)
                re_seq.append(re_new)
                im_seq.append(im_new)
            re_out.append(_D.xp.stack(re_seq))
            im_out.append(_D.xp.stack(im_seq))
            batch_params[b] = bp

        re_arr = _D.xp.stack(re_out)
        im_arr = _D.xp.stack(im_out)
        rho_arr = _D.xp.sqrt(re_arr * re_arr + im_arr * im_arr + 1e-12)

        rho_t = TriadTensor(rho_arr)
        rho_t._requires_grad = proj._requires_grad
        rho_t._children = [proj]

        def _back(g):
            d_re = re_arr / (rho_arr + 1e-12)
            d_im = im_arr / (rho_arr + 1e-12)
            g_proj = _D.xp.concatenate([g * d_re, g * d_im], axis=-1)
            proj._grad = g_proj if proj._grad is None else proj._grad + g_proj
        rho_t._grad_fn = _back

        out = self.out_proj(rho_t)
        out = out + residual
        return out

class TriadFullLM(Module):
    def __init__(self, vocab_size: int, d_model: int = 64, N_solver: int = 64,
                 n_blocks: int = 4, n_memory: int = 3):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tok_emb = Embedding(vocab_size, d_model)
        self.blocks = [
            TriadFullBlock(d_model, N=N_solver, n_memory=n_memory)
            for _ in range(n_blocks)
        ]
        self.ln_f = LayerNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size, bias=False)
        self.training = True

    def forward(self, idx: TriadTensor) -> TriadTensor:
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def loss(self, idx: TriadTensor, targets: TriadTensor) -> TriadTensor:
        logits = self.forward(idx)
        B, T, C = logits.shape
        logits_2d = logits.reshape(B * T, C)
        targets_np = targets._data.astype(int).reshape(B * T)
        shifted = logits_2d._data - logits_2d._data.max(axis=1, keepdims=True)
        exp_s = np.exp(shifted)
        probs = exp_s / exp_s.sum(axis=1, keepdims=True)
        log_probs = np.log(probs + 1e-12)
        nll = -log_probs[np.arange(B * T), targets_np]
        loss_val = np.mean(nll)
        out = TriadTensor(loss_val)
        out._requires_grad = True
        out._children = [logits_2d]

        def _back(g):
            grad = probs.copy()
            grad[np.arange(B * T), targets_np] -= 1.0
            grad /= (B * T)
            grad *= g
            logits_2d._grad = grad if logits_2d._grad is None else logits_2d._grad + grad
        out._grad_fn = _back
        return out

    def generate(self, idx: list[int], max_new: int = 50,
                 temperature: float = 1.0, top_k: int = 0) -> list[int]:
        was_training = self.training
        self.training = False
        for _ in range(max_new):
            ctx = idx[-256:]
            x = tensor(np.array([ctx], dtype=np.int64))
            with no_grad():
                logits = self.forward(x)
            logits_np = logits._data[0, -1] / max(temperature, 1e-8)
            if top_k > 0:
                top_vals = np.sort(logits_np)[-top_k:]
                logits_np = np.where(logits_np >= top_vals[0], logits_np, -1e9)
            pr = np.exp(logits_np - logits_np.max())
            pr = pr / pr.sum()
            next_id = int(np.random.choice(len(pr), p=pr))
            idx.append(next_id)
        self.training = was_training
        return idx

class TriadLM(Module):
    def __init__(self, vocab_size: int, d_model: int = 64, d_state: int = 64,
                 n_blocks: int = 4, n_memory: int = 3, coupling: bool = True):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tok_emb = Embedding(vocab_size, d_model)
        self.blocks = [
            TriadFullBlock(d_model, N=d_state, n_memory=n_memory)
            for _ in range(n_blocks)
        ]
        self.ln_f = LayerNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size, bias=False)
        self.training = True

    def forward(self, idx: TriadTensor) -> TriadTensor:
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def loss(self, idx: TriadTensor, targets: TriadTensor) -> TriadTensor:
        logits = self.forward(idx)
        B, T, C = logits.shape
        logits_2d = logits.reshape(B * T, C)
        targets_np = targets._data.astype(int).reshape(B * T)
        shifted = logits_2d._data - logits_2d._data.max(axis=1, keepdims=True)
        exp_s = np.exp(shifted)
        probs = exp_s / exp_s.sum(axis=1, keepdims=True)
        log_probs = np.log(probs + 1e-12)
        nll = -log_probs[np.arange(B * T), targets_np]
        loss_val = np.mean(nll)
        out = TriadTensor(loss_val)
        out._requires_grad = True
        out._children = [logits_2d]

        def _back(g):
            grad = probs.copy()
            grad[np.arange(B * T), targets_np] -= 1.0
            grad /= (B * T)
            grad *= g
            logits_2d._grad = grad if logits_2d._grad is None else logits_2d._grad + grad
        out._grad_fn = _back
        return out

    def generate(self, idx: list[int], max_new: int = 50,
                 temperature: float = 1.0, top_k: int = 0) -> list[int]:
        was_training = self.training
        self.training = False
        for _ in range(max_new):
            ctx = idx[-256:]
            x = tensor(np.array([ctx], dtype=np.int64))
            with no_grad():
                logits = self.forward(x)
            logits_np = logits._data[0, -1] / max(temperature, 1e-8)
            if top_k > 0:
                top_vals = np.sort(logits_np)[-top_k:]
                logits_np = np.where(logits_np >= top_vals[0], logits_np, -1e9)
            pr = np.exp(logits_np - logits_np.max())
            pr = pr / pr.sum()
            next_id = int(np.random.choice(len(pr), p=pr))
            idx.append(next_id)
        self.training = was_training
        return idx

class TextDataset:
    def __init__(self, tokens: list[int], seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len - 1)

    def batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        n = len(self)
        if n < batch_size:
            batch_size = n
        ix = np.random.randint(0, n, size=batch_size)
        x = np.array([self.tokens[i:i + self.seq_len] for i in ix], dtype=np.int64)
        y = np.array([self.tokens[i + 1:i + self.seq_len + 1] for i in ix], dtype=np.int64)
        return x, y
