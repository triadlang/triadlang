from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np
from runtime.solver import TriadParams, integrate, _effective_params
from runtime.multi_runtime import MultiRuntime, CouplingEdge, Segment, Substrate
from runtime.observables import crystallinity, dominant_wavenumber, peak_density, ipr, participation_ratio, norm as field_norm, fwhm
from runtime.backend import get_xp, asnumpy

@dataclass
class ReadoutConfig:
    what: str = 'full'
    fields: tuple = ('crystallinity', 'k_star', 'peak', 'ipr', 'participation', 'fwhm', 'norm', 'density_pca')

class EquationRuntime:

    def __init__(self, n_substrates: int=4, regime: str='B0', N: int=128, coupling: str='ring', kappa: float=-3.0, seed: int=0, backend: str='auto', readout: Optional[ReadoutConfig]=None):
        self.n_substrates = n_substrates
        self.regime = regime
        self.N = N
        self.coupling = coupling
        self.kappa = kappa
        self.seed = seed
        self.backend = backend
        self.readout = readout or ReadoutConfig()
        self._substrate_ids = []
        self._mr = None
        self._input_buffer = []
        self._results = None

    def inject(self, signal: np.ndarray, substrate_idx: int=0):
        self._input_buffer.append((substrate_idx, np.asarray(signal)))

    def inject_batch(self, signals: list[np.ndarray]):
        for i, s in enumerate(signals):
            self.inject(s, substrate_idx=i % self.n_substrates)

    def run(self, T: float=5.0, dt: float=0.005, verbose: bool=False) -> dict:
        import time as _time
        t0 = _time.perf_counter()
        mr = MultiRuntime(dt=dt, record_every=9999)
        xp = get_xp(self.backend)
        subs = []
        for i in range(self.n_substrates):
            from stdlib.regimes import resolve_regime
            p = resolve_regime(self.regime, seed=self.seed + i)
            p = TriadParams(**{**p.__dict__, 'N': self.N, 'T': T, 'backend': self.backend})
            sub = mr.add_substrate(f's{i}', p)
            subs.append(sub.id)
            if self._input_buffer:
                for buf_idx, buf_signal in self._input_buffer:
                    if buf_idx == i:
                        perturbation = xp.asarray(buf_signal, dtype=xp.complex128)
                        sub.psi = sub.psi + perturbation
                        norm_val = xp.sqrt((xp.abs(sub.psi) ** 2).sum() * sub.dx)
                        sub.psi = sub.psi / norm_val
        edges = []
        if self.coupling == 'ring':
            edges = [CouplingEdge(src_id=subs[i], dst_id=subs[(i + 1) % self.n_substrates], kappa=self.kappa) for i in range(self.n_substrates)]
        elif self.coupling == 'full':
            edges = [CouplingEdge(src_id=subs[i], dst_id=subs[j], kappa=self.kappa / self.n_substrates) for i in range(self.n_substrates) for j in range(self.n_substrates) if i != j]
        elif self.coupling == 'chain':
            edges = [CouplingEdge(src_id=subs[i], dst_id=subs[i + 1], kappa=self.kappa) for i in range(self.n_substrates - 1)]
        mr.add_segment(Segment(t_start=0.0, t_end=T, edges=edges))
        mr.run(verbose=verbose)
        per_substrate = []
        density_features = []
        for sid, sub in mr.substrates.items():
            psi = asnumpy(sub.psi)
            dx = sub.dx
            obs = {'crystallinity': crystallinity(psi, dx), 'k_star': dominant_wavenumber(psi, dx, k_min=2 * np.pi / sub.params.L), 'peak': peak_density(psi), 'ipr': ipr(psi, dx), 'participation': participation_ratio(psi, dx), 'fwhm': fwhm(psi, dx), 'norm': field_norm(psi, dx)}
            rho = np.abs(psi) ** 2
            obs['density_pca'] = float(np.var(rho))
            obs['memory_mean'] = float(np.mean(asnumpy(sub.y)))
            obs['memory_var'] = float(np.var(asnumpy(sub.y)))
            obs['memory_energy'] = float(np.sum(asnumpy(sub.y) ** 2) * dx)
            per_substrate.append(obs)
            rho_norm = rho / (rho.max() + 1e-10)
            density_features.append(rho_norm)
        total_state = []
        for obs in per_substrate:
            for f in self.readout.fields:
                if f == 'density_pca':
                    total_state.append(obs.get('density_pca', 0.0))
                elif f == 'memory_energy':
                    total_state.append(obs.get('memory_energy', 0.0))
                else:
                    total_state.append(obs.get(f, 0.0))
        total_state = np.array(total_state)
        coupling_map = {}
        for e in edges:
            coupling_map[f's{e.src_id}->s{e.dst_id}'] = e.kappa
        n_bytes = sum((sub.psi.nbytes + sub.y.nbytes + sub.half_lin.nbytes for sub in mr.substrates.values()))
        elapsed = _time.perf_counter() - t0
        self._mr = mr
        self._results = {'per_substrate': per_substrate, 'coupling_map': coupling_map, 'total_state': total_state, 'density_features': density_features, 'state_dim': len(total_state), 'memory_kb': n_bytes / 1024, 'elapsed': elapsed}
        self._input_buffer.clear()
        return self._results

    @staticmethod
    def ridge_readout(states: np.ndarray, targets: np.ndarray, train_frac: float=0.7, lam: float=0.01):
        X = np.hstack([states, np.ones((len(states), 1))])
        n = len(X)
        ntr = int(n * train_frac)
        Xtr, Xte = (X[:ntr], X[ntr:])
        ytr, yte = (targets[:ntr], targets[ntr:])
        mu, sd = (Xtr.mean(0), Xtr.std(0) + 1e-09)
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        Xtr[:, -1] = 1
        Xte[:, -1] = 1
        W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
        pred = Xte @ W
        ss_res = ((yte - pred) ** 2).sum()
        ss_tot = ((yte - yte.mean()) ** 2).sum() + 1e-12
        return (1 - ss_res / ss_tot, pred)

    def state_size(self) -> int:
        return self.n_substrates * len(self.readout.fields)

    def memory_kb(self) -> float:
        if self._results:
            return self._results['memory_kb']
        return self.n_substrates * self.N * (16 + 8 * 3 + 16) / 1024