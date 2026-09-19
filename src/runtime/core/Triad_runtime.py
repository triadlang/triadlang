from __future__ import annotations

from dataclasses import asdict, dataclass

from runtime.core.equilibrium_solver import DriveSpec as _DriveSpec
from runtime.core.equilibrium_solver import FixedPointSolver as _FixedPointSolver
from runtime.core.fast_solver import equilibrate as _equilibrate
from runtime.core.fast_solver import fast_integrate as _fast_integrate
from runtime.core.multi_runtime import CouplingLink, MultiRuntime, Segment
from runtime.core.solver import TriadParams, integrate as _integrate
from runtime.core.solver import integrate_adaptive

SOLVERS = ('integrate', 'fast', 'fixedpoint')

_FIXEDPOINT_CTOR_KEYS = ('m_anderson', 'mixing', 'tol', 'max_iter')


def solve_with(p: TriadParams, solver: str = 'integrate', **kw) -> dict:
    if solver == 'integrate':
        return _integrate(p, **kw)
    if solver == 'fast':
        return _fast_integrate(p, **kw)
    if solver == 'fixedpoint':
        ctor = {k: kw.pop(k) for k in _FIXEDPOINT_CTOR_KEYS if k in kw}
        drive = _as_drive(kw.pop('drive', None), p)
        n_substeps = kw.pop('n_substeps', 50)
        verbose = kw.pop('verbose', False)
        if kw:
            raise TypeError(f'fixedpoint got unexpected keywords: {sorted(kw)}')
        res = _FixedPointSolver(**ctor).solve_equilibrium(
            p, drive=drive, n_substeps=n_substeps, verbose=verbose)
        out = asdict(res)
        out['psi_final'] = res.psi
        return out
    raise ValueError(f'solver must be one of {SOLVERS}, got {solver!r}')
from runtime.observers import ConvergenceObserver, SelfVerifier
from runtime.physics.observables import (
    crystallinity,
    dominant_wavenumber,
    energy,
    fwhm,
    ipr,
    participation_ratio,
    peak_density,
)
from runtime.physics.observables import norm as field_norm
from triad import ntri as np


def _as_drive(drive, p):
    if drive is None or isinstance(drive, _DriveSpec):
        return drive
    if isinstance(drive, dict):
        return _DriveSpec(**drive)
    arr = np.asarray(drive, dtype=float).reshape(-1)
    if arr.shape[0] != p.N:
        raise ValueError(f'array drive needs {p.N} samples, got {arr.shape[0]}')
    x_src = np.linspace(-p.L / 2, p.L / 2, arr.shape[0])
    return _DriveSpec(drive_type='custom',
                      custom_fn=lambda xa: np.interp(np.asarray(xa, dtype=float), x_src, arr))

def _solve_equilibrium(p, drive=None, tol=0.01, max_iter=200,
                       anderson_m=5, anderson_beta=0.8,
                       substeps=10, verbose=False):
    solver = _FixedPointSolver(m_anderson=anderson_m, mixing=anderson_beta, tol=tol)
    return solver.solve_equilibrium(p, drive=_as_drive(drive, p), n_substeps=substeps, verbose=verbose)
from runtime.backend import asnumpy, get_xp


@dataclass
class ReadoutConfig:
    what: str = 'triad'
    fields: tuple = ('crystallinity', 'k_star', 'peak', 'ipr', 'participation', 'fwhm', 'norm', 'density_pca')

class TriadRuntime:

    def __init__(self, n_substrates: int=4, regime: str='B0', N: int=128, coupling: str='ring', kappa: float=-3.0, seed: int=0, backend: str='auto', readout: ReadoutConfig | None=None, solver: str='integrate'):
        if solver not in SOLVERS:
            raise ValueError(f'solver must be one of {SOLVERS}, got {solver!r}')
        self.n_substrates = n_substrates
        self.regime = regime
        self.N = N
        self.coupling = coupling
        self.kappa = kappa
        self.seed = seed
        self.backend = backend
        self.readout = readout or ReadoutConfig()
        self.solver = solver
        self._substrate_ids = []
        self._mr = None
        self._input_buffer = []
        self._results = None

    def inject(self, signal: np.ndarray, substrate_idx: int=0):
        self._input_buffer.append((substrate_idx, np.asarray(signal)))

    def inject_batch(self, signals: list[np.ndarray]):
        for i, s in enumerate(signals):
            self.inject(s, substrate_idx=i % self.n_substrates)

    def solve(self, p: TriadParams, **kw) -> dict:
        return solve_with(p, self.solver, **kw)

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
                        sub.psi = sub.psi / (norm_val + 1e-15)
        links = []
        if self.coupling == 'ring':
            links = [CouplingLink(src_id=subs[i], dst_id=subs[(i + 1) % self.n_substrates], kappa=self.kappa) for i in range(self.n_substrates)]
        elif self.coupling == 'triad':
            links = [CouplingLink(src_id=subs[i], dst_id=subs[j], kappa=self.kappa / self.n_substrates) for i in range(self.n_substrates) for j in range(self.n_substrates) if i != j]
        elif self.coupling == 'chain':
            links = [CouplingLink(src_id=subs[i], dst_id=subs[i + 1], kappa=self.kappa) for i in range(self.n_substrates - 1)]
        mr.add_segment(Segment(t_start=0.0, t_end=T, links=links))
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
        for e in links:
            coupling_map[f's{e.src_id}->s{e.dst_id}'] = e.kappa
        n_bytes = sum(sub.psi.nbytes + sub.y.nbytes + sub.half_lin.nbytes for sub in mr.substrates.values())
        elapsed = _time.perf_counter() - t0
        self._mr = mr
        self._results = {'per_substrate': per_substrate, 'coupling_map': coupling_map, 'total_state': total_state, 'density_features': density_features, 'state_dim': len(total_state), 'memory_kb': n_bytes / 1024, 'elapsed': elapsed}
        self._input_buffer.clear()
        return self._results

    def run_adaptive(self, T: float = 30.0, dt: float = 0.005, stop: str = "adaptive",
                     risk: float = 0.05, verbose: bool = False) -> dict:

        import time as _time
        t0 = _time.perf_counter()

        from stdlib.regimes import resolve_regime
        p = resolve_regime(self.regime, seed=self.seed)
        p = TriadParams(**{**p.__dict__, 'N': self.N, 'T': T, 'dt': dt,
                           'backend': self.backend})

        if stop == "adaptive":
            obs = ConvergenceObserver(
                metrics=["crystallinity", "energy"],
                tol=risk,
                window=5,
                min_checkpoints=8,
            )
            result = integrate_adaptive(
                p,
                stop_fn=obs.as_stop_fn(),
                max_T=T,
                checkpoint_every=100,
            )
            psi = result["psi_final"]
            dx = result["dx"]
            elapsed = _time.perf_counter() - t0
            return {
                "stopped_at_T": result["stopped_at_T"],
                "steps_taken": result["steps_taken"],
                "crystallinity": crystallinity(psi, dx),
                "k_star": dominant_wavenumber(psi, dx),
                "ipr": ipr(psi, dx),
                "participation": participation_ratio(psi, dx),
                "energy": energy(psi, dx, hbar=p.hbar, m=p.m, Lambda=p.Lambda),
                "certified": obs.certified,
                "n_checkpoints": obs.n_checked,
                "elapsed": elapsed,
                "psi_final": psi,
                "params": p,
            }
        else:
            return self.run(T=T, dt=dt, verbose=verbose)

    def run_verified(self, T: float = 10.0, dt: float = 0.005, ensemble: int = 5,
                      verbose: bool = False) -> dict:

        import time as _time

        from stdlib.regimes import resolve_regime
        t0 = _time.perf_counter()

        sv = SelfVerifier(ensemble=ensemble)
        base_p = resolve_regime(self.regime, seed=self.seed)
        base_p = TriadParams(**{**base_p.__dict__, 'N': self.N, 'T': T,
                                 'dt': dt, 'backend': self.backend})
        vresult = sv.run_ensemble(base_p, T=T)
        elapsed = _time.perf_counter() - t0
        vresult["elapsed"] = elapsed
        vresult["regime"] = self.regime
        return vresult

    def solve_equilibrium(self, drive: np.ndarray | None = None,
                          tol: float = 0.01, max_iter: int = 200,
                          anderson_m: int = 5, anderson_beta: float = 0.8,
                          substeps: int = 10, verbose: bool = False) -> dict:

        import time as _time

        from stdlib.regimes import resolve_regime
        t0 = _time.perf_counter()

        base_p = resolve_regime(self.regime, seed=self.seed)
        p = TriadParams(**{**base_p.__dict__, 'N': self.N,
                           'backend': self.backend})
        result = _solve_equilibrium(
            p, drive=drive, tol=tol, max_iter=max_iter,
            anderson_m=anderson_m, anderson_beta=anderson_beta,
            substeps=substeps, verbose=verbose,
        )
        elapsed = _time.perf_counter() - t0

        if not isinstance(result, dict):
            try:
                result = asdict(result)
            except (TypeError, ValueError):
                result = {k: getattr(result, k) for k in result.__dataclass_fields__}
        result["elapsed"] = elapsed
        result["regime"] = self.regime
        return result

    def equilibrate(self, n_super: int = 50, max_supersteps: int = 60,
                     tol_obs: float = 0.02, verbose: bool = False) -> dict:

        import time as _time

        from stdlib.regimes import resolve_regime
        t0 = _time.perf_counter()

        base_p = resolve_regime(self.regime, seed=self.seed)
        p = TriadParams(**{**base_p.__dict__, 'N': self.N,
                           'backend': self.backend})
        result = _equilibrate(p, n_super=n_super,
                              max_supersteps=max_supersteps,
                              tol_obs=tol_obs, verbose=verbose)
        elapsed = _time.perf_counter() - t0
        result["elapsed"] = elapsed
        result["regime"] = self.regime
        return result

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
