from __future__ import annotations

import copy
from dataclasses import dataclass

from runtime.backend import asnumpy as _asnumpy
from runtime.core.solver import TriadParams, integrate
from runtime.ml.nn import Adam, Module, relu, triad
from runtime.ml.tensor import tensor
from runtime.physics.observables import crystallinity, peak_density
from triad import ntri as np


class SpectralExtractor:

    def __init__(self, N: int, n_modes: int):
        self.N = N
        self.n_modes = min(n_modes, N)

    def extract(self, psi: np.ndarray) -> np.ndarray:

        psi_k = np.fft.fft(psi)
        amp = np.abs(psi_k[:self.n_modes])
        phase = np.angle(psi_k[:self.n_modes])
        return np.concatenate([amp, phase]).astype(np.float64)

    def extract_batch(self, psi_batch: np.ndarray) -> np.ndarray:

        if psi_batch.ndim != 2:
            raise ValueError(f'extract_batch needs 2D (B, N), got shape {psi_batch.shape}')
        B = psi_batch.shape[0]
        feats = np.empty((B, 2 * self.n_modes), dtype=np.float64)
        for i in range(B):
            feats[i] = self.extract(psi_batch[i])
        return feats

    @property
    def dim(self) -> int:
        return 2 * self.n_modes

class ConsistencyMap(Module):

    def __init__(self, extractor: SpectralExtractor, hidden_dim: int = 64,
                 n_layers: int = 3):
        super().__init__()
        self.extractor = extractor
        self.n_modes = extractor.n_modes
        self.N = extractor.N
        self.in_dim = extractor.dim
        self.out_dim = 2 * self.n_modes

        layers = []
        dim_in = self.in_dim
        for _ in range(n_layers - 1):
            layers.append(triad(dim_in, hidden_dim))
            dim_in = hidden_dim
        layers.append(triad(dim_in, self.out_dim))
        self.layers = layers

    def forward(self, feats: tensor) -> tensor:

        x = feats
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = relu(x)
        return x

    def predict_delta(self, psi):

        feats = self.extractor.extract(psi)[np.newaxis, :]
        from runtime.ml.tensor import no_grad
        with no_grad():
            delta_spec = _asnumpy(self.forward(tensor(feats))._data[0])
        dre = delta_spec[:self.n_modes]
        dim_ = delta_spec[self.n_modes:]
        delta_k = dre + 1j * dim_
        delta_triad = np.zeros(self.N, dtype=np.complex128)
        delta_triad[:self.n_modes] = delta_k
        if self.n_modes > 1:
            delta_triad[-(self.n_modes - 1):] = np.conj(delta_k[1:])[::-1]
        delta_x = np.fft.ifft(delta_triad)
        return delta_x

    def predict(self, psi: np.ndarray) -> np.ndarray:

        return psi + self.predict_delta(psi)

@dataclass
class TrajectorySample:

    psi_t: np.ndarray
    psi_final: np.ndarray
    t_frac: float

class TrajectoryTeacher:

    def __init__(self, base_params: TriadParams, n_snapshots: int = 8,
                 record_every: int | None = None,
                 force_mode: str | None = None):
        if force_mode is not None and force_mode != 'triad':
            raise ValueError(
                f"triad rule: mode={force_mode!r} does not exist; "
                f"the Triad Triad is 'triad' only. P1+P2+P3 are inseparable.")
        self.base_params = base_params
        self.n_snapshots = n_snapshots
        self.record_every = record_every
        self.force_mode = force_mode

    def _compute_record_every(self, n_steps: int) -> int:
        if self.record_every is not None:
            return self.record_every
        return max(1, n_steps // self.n_snapshots)

    def _apply_mode(self, p: TriadParams) -> TriadParams:
        if self.force_mode is not None and self.force_mode != 'triad':
            raise ValueError(
                f"triad rule: mode={self.force_mode!r} does not exist; "
                f"the Triad Triad is 'triad' only. P1+P2+P3 are inseparable.")
        from dataclasses import replace
        return replace(p, mode='triad')

    def generate_trajectory(self, seed: int = 0,
                            psi0: np.ndarray | None = None,
                            y0: np.ndarray | None = None) -> list[TrajectorySample]:

        p = copy.deepcopy(self.base_params)
        p.seed = seed
        p = self._apply_mode(p)
        n_steps = int(round(p.T / p.dt))
        p.record_every = self._compute_record_every(n_steps)

        result = integrate(p, psi0=psi0, y0=y0, auto_halve_dt=False)

        psi_final = result['psi_final']
        t_arr = result['t']
        density = result['density']

        samples = []
        n_snaps = density.shape[1] if density.ndim > 1 else 0

        if n_snaps < 2:
            return [TrajectorySample(psi_final.copy(), psi_final.copy(), 1.0)]

        phase_final = np.angle(psi_final)

        for i in range(n_snaps):
            rho = density[:, i]
            amp = np.sqrt(np.maximum(rho, 0.0))

            frac = t_arr[i] / t_arr[-1] if t_arr[-1] > 0 else 1.0
            psi_t = amp * np.exp(1j * phase_final)
            samples.append(TrajectorySample(psi_t, psi_final, frac))

        return samples

    def generate_trajectory_exact(self, seed: int = 0,
                                  psi0: np.ndarray | None = None,
                                  y0: np.ndarray | None = None) -> list[TrajectorySample]:

        from runtime.core.solver import _integrate_steps
        p = copy.deepcopy(self.base_params)
        p.seed = seed
        p = self._apply_mode(p)
        n_steps = int(round(p.T / p.dt))
        rec_every = self._compute_record_every(n_steps)

        psi_final = None
        checkpoints = []
        for ckpt in _integrate_steps(p, psi0=psi0, y0=y0,
                                     auto_halve_dt=False,
                                     record_every=rec_every):
            checkpoints.append(ckpt)
            psi_final = ckpt['psi']

        if psi_final is None:
            return []

        samples = []
        total_steps = checkpoints[-1]['step'] if checkpoints else 1
        for ckpt in checkpoints:
            frac = ckpt['step'] / max(total_steps, 1)
            samples.append(TrajectorySample(
                psi_t=ckpt['psi'],
                psi_final=psi_final.copy(),
                t_frac=frac,
            ))
        return samples

    def generate_samples(self, n_trajectories: int = 10,
                         base_seed: int = 0,
                         exact: bool = True) -> list[TrajectorySample]:

        all_samples = []
        gen_fn = self.generate_trajectory_exact if exact else self.generate_trajectory
        for i in range(n_trajectories):
            samples = gen_fn(seed=base_seed + i * 137)
            all_samples.extend(samples)
        return all_samples

def _spectral_to_position_delta(delta_spec: np.ndarray, n_modes: int,
                                 N: int) -> np.ndarray:

    dre = delta_spec[:n_modes]
    dim_ = delta_spec[n_modes:]
    delta_k = dre + 1j * dim_
    delta_triad = np.zeros(N, dtype=np.complex128)
    delta_triad[:n_modes] = delta_k
    if n_modes > 1:
        delta_triad[-(n_modes - 1):] = np.conj(delta_k[1:])[::-1]
    return np.fft.ifft(delta_triad)

def global_consistency_loss(delta_spec: tensor, feats: tensor,
                             psi_t_real: tensor, psi_t_imag: tensor,
                             target_real: tensor, target_imag: tensor,
                             n_modes: int, N: int) -> tensor:

    pred_re_k = delta_spec[:, :n_modes]
    pred_im_k = delta_spec[:, n_modes:]

    target_delta_real_k = target_real - psi_t_real
    target_delta_imag_k = target_imag - psi_t_imag

    loss_re = ((pred_re_k - target_real[:, :n_modes]) ** 2).mean()
    loss_im = ((pred_im_k - target_imag[:, :n_modes]) ** 2).mean()
    return loss_re + loss_im

def local_consistency_loss(delta_spec_t: tensor,
                            delta_spec_t1: tensor) -> tensor:

    diff = delta_spec_t - delta_spec_t1
    return (diff * diff).mean()

@dataclass
class ConsistencyTrainConfig:

    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-3
    w_global: float = 1.0
    w_local: float = 0.1
    n_modes: int = 32
    hidden_dim: int = 64
    n_layers: int = 3
    n_trajectories: int = 10
    base_seed: int = 0
    verbose: bool = True

def _prepare_spectral_targets(samples: list[TrajectorySample],
                               extractor: SpectralExtractor,
                               n_modes: int) -> tuple:

    n = len(samples)
    N = samples[0].psi_t.shape[0]
    feats = np.zeros((n, 2 * n_modes), dtype=np.float64)
    target_delta_k = np.zeros((n, 2 * n_modes), dtype=np.float64)
    fracs = np.zeros(n, dtype=np.float64)
    traj_ids = np.zeros(n, dtype=np.int64)

    for i, s in enumerate(samples):
        feats[i] = extractor.extract(s.psi_t)
        delta = s.psi_final - s.psi_t
        delta_k = np.fft.fft(delta)[:n_modes]
        target_delta_k[i, :n_modes] = np.real(delta_k)
        target_delta_k[i, n_modes:] = np.imag(delta_k)
        fracs[i] = s.t_frac

    return feats, target_delta_k, fracs, traj_ids

def train_consistency(
    params: TriadParams,
    config: ConsistencyTrainConfig | None = None,
) -> tuple[ConsistencyMap, list[dict]]:

    if config is None:
        config = ConsistencyTrainConfig()

    p = copy.deepcopy(params)
    from dataclasses import replace
    p = replace(p, mode='triad')
    N = p.N

    extractor = SpectralExtractor(N, config.n_modes)
    model = ConsistencyMap(extractor, hidden_dim=config.hidden_dim,
                           n_layers=config.n_layers)

    teacher = TrajectoryTeacher(p, n_snapshots=8)
    samples = teacher.generate_samples(
        n_trajectories=config.n_trajectories,
        base_seed=config.base_seed,
        exact=True,
    )

    if len(samples) < 2:
        if config.verbose:
            print("warning: fewer than 2 samples generated, cannot train local loss")

    feats, target_k, fracs, _ = _prepare_spectral_targets(
        samples, extractor, config.n_modes)

    order = np.argsort(fracs)
    feats = feats[order]
    target_k = target_k[order]
    fracs = fracs[order]

    n_samples = len(feats)
    n_local_pairs = max(0, n_samples - 1)

    opt = Adam(model.parameters(), lr=config.lr)
    rng = np.random.default_rng(config.base_seed + 42)
    history = []

    for ep in range(config.epochs):
        perm = rng.permutation(n_samples)
        tot_global = 0.0
        tot_local = 0.0
        nb = 0

        for i in range(0, n_samples, config.batch_size):
            idx = perm[i:i + config.batch_size]
            batch_feats = tensor(feats[idx])
            batch_target = tensor(target_k[idx])

            delta_pred = model.forward(batch_feats)

            diff = delta_pred - batch_target
            loss_global = (diff * diff).mean()

            loss_local = tensor(np.array(0.0))
            if n_local_pairs > 0:

                local_pairs = []
                for j in range(len(idx) - 1):

                    pos_a = np.searchsorted(order, idx[j])
                    pos_b = np.searchsorted(order, idx[j + 1])
                    if abs(pos_b - pos_a) == 1:
                        local_pairs.append((j, j + 1))

                if local_pairs:
                    for a, b in local_pairs:
                        d = delta_pred[a] - delta_pred[b]
                        loss_local = loss_local + (d * d).mean()
                    loss_local = loss_local * (1.0 / max(len(local_pairs), 1))

            loss = config.w_global * loss_global + config.w_local * loss_local

            model.zero_grad()
            loss.backward()
            opt.step()

            tot_global += float(loss_global._data)
            tot_local += float(loss_local._data)
            nb += 1

        avg_global = tot_global / max(nb, 1)
        avg_local = tot_local / max(nb, 1)
        history.append({
            'epoch': ep + 1,
            'global_loss': avg_global,
            'local_loss': avg_local,
        })
        if config.verbose:
            print(f'  epoch {ep + 1:3d}/{config.epochs}  '
                  f'global={avg_global:.3e}  local={avg_local:.3e}')

    return model, history

def consistency_distill(
    params: TriadParams,
    n_modes: int = 32,
    epochs: int = 50,
    n_trajectories: int = 10,
    verbose: bool = True,
) -> tuple[ConsistencyMap, list[dict]]:

    config = ConsistencyTrainConfig(
        epochs=epochs,
        n_modes=n_modes,
        n_trajectories=n_trajectories,
        verbose=verbose,
    )
    return train_consistency(params, config)

def evaluate_consistency(model: ConsistencyMap, params: TriadParams,
                         n_test: int = 5, seed: int = 9999) -> dict:

    self_errors = []
    pred_errors = []
    cryst_diffs = []
    last = {}
    for i in range(max(1, n_test)):
        p = copy.deepcopy(params)
        from dataclasses import replace
        p = replace(p, mode='triad')
        p.seed = seed + i

        result = integrate(p, auto_halve_dt=False)
        psi_final = result['psi_final']
        dx = result['dx']

        delta_identity = model.predict_delta(psi_final)
        self_errors.append(float(np.sqrt(np.mean(np.abs(delta_identity) ** 2))))

        x = np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
        dx_val = float(x[1] - x[0])
        psi0 = np.exp(-x ** 2 / 8.0).astype(np.complex128)
        psi0 /= np.sqrt((np.abs(psi0) ** 2).sum() * dx_val)

        psi_pred = model.predict(psi0)
        pred_errors.append(float(np.sqrt(np.mean(np.abs(psi_pred - psi_final) ** 2))))

        cryst_pred = crystallinity(psi_pred, dx)
        cryst_true = crystallinity(psi_final, dx)
        cryst_diffs.append(abs(cryst_pred - cryst_true))
        last = {
            'crystallinity_predicted': cryst_pred,
            'crystallinity_true': cryst_true,
            'peak_density_predicted': float(peak_density(psi_pred)),
            'peak_density_true': float(peak_density(psi_final)),
        }

    return {
        'n_test': max(1, n_test),
        'self_error': float(np.mean(self_errors)),
        'prediction_error': float(np.mean(pred_errors)),
        'crystallinity_diff': float(np.mean(cryst_diffs)),
        **last,
    }

