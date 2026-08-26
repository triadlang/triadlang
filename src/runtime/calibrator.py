
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from runtime.core.solver import TriadParams, integrate
from runtime.physics.observables import (
    crystallinity,
    dominant_wavenumber,
    ipr,
    participation_ratio,
)
from triad import ntri as np

_OBS_FNS: dict[str, Callable] = {
    "crystallinity": lambda r, dx: crystallinity(r["psi_final"], dx),
    "dominant_k": lambda r, dx: dominant_wavenumber(r["psi_final"], dx),
    "ipr": lambda r, dx: ipr(r["psi_final"], dx),
    "participation": lambda r, dx: participation_ratio(r["psi_final"], dx),
}

DEFAULT_OBSERVABLES = ["crystallinity", "dominant_k"]

def _measure_obs(params: TriadParams,
                 obs_names: Sequence[str]) -> dict[str, float]:

    r = integrate(params)
    dx = r["dx"]
    out: dict[str, float] = {}
    for name in obs_names:
        fn = _OBS_FNS.get(name)
        if fn is None:
            raise ValueError(f"Unknown observable {name!r}")
        out[name] = float(fn(r, dx))
    return out

_TUNABLE_NAMES = ("Lambda", "Gamma", "alpha", "sigma", "kT", "f_FDT")

def _params_to_vec(params: TriadParams,
                   names: Sequence[str] = _TUNABLE_NAMES) -> np.ndarray:

    d = params.__dict__
    return np.array([float(d[n]) for n in names], dtype=np.float64)

def _vec_to_params(params: TriadParams,
                   vec: np.ndarray,
                   names: Sequence[str] = _TUNABLE_NAMES) -> TriadParams:

    d = dict(params.__dict__)
    for i, n in enumerate(names):
        d[n] = float(vec[i])

    return TriadParams(**d)

def _perturbation_scale(params: TriadParams,
                        names: Sequence[str] = _TUNABLE_NAMES) -> np.ndarray:

    v = _params_to_vec(params, names)
    return np.maximum(np.abs(v) * 1e-3, 1e-6)

@dataclass
class CalibrationConfig:

    observables: list[str] = None
    max_iter: int = 15
    damping: float = 1e-2
    jacobi_scale: bool = True
    perturbation_frac: float = 1e-3
    tol_residual: float = 0.02
    integrate_T: float = 10.0
    verbose: bool = True

    def __post_init__(self):
        if self.observables is None:
            self.observables = list(DEFAULT_OBSERVABLES)

class GaussNewtonCalibrator:

    def __init__(self, config: CalibrationConfig = None):
        self.config = config or CalibrationConfig()
        self.obs_history: list[dict[str, float]] = []

    def _compute_residual(self, obs: dict[str, float]) -> np.ndarray:

        self.obs_history.append(obs)
        if len(self.obs_history) < 2:
            return np.zeros(len(self.config.observables))

        n_obs = len(self.config.observables)
        residual = np.zeros(n_obs)
        prev = self.obs_history[-2]
        for i, name in enumerate(self.config.observables):

            old_val = prev.get(name, 0.0)
            new_val = obs[name]
            if abs(old_val) > 1e-10:
                residual[i] = (new_val - old_val) / abs(old_val)
            else:
                residual[i] = new_val
        return residual

    def _compute_jacobian(self,
                          params: TriadParams,
                          base_obs: dict[str, float],
                          ) -> np.ndarray:

        names = _TUNABLE_NAMES
        n_obs = len(self.config.observables)
        n_params = len(names)
        J = np.zeros((n_obs, n_params))

        scales = _perturbation_scale(params, names)

        for j in range(n_params):

            p_plus = _vec_to_params(params,
                                    _params_to_vec(params, names) + scales[j] * np.eye(n_params)[j],
                                    names)
            p_plus = replace(p_plus, T=self.config.integrate_T)
            obs_plus = _measure_obs(p_plus, self.config.observables)

            for i, name in enumerate(self.config.observables):
                J[i, j] = (obs_plus[name] - base_obs[name]) / scales[j]

        return J

    def calibrate(self, params: TriadParams) -> dict:

        cfg = self.config
        names = _TUNABLE_NAMES
        self.obs_history = []

        p = replace(params, T=cfg.integrate_T)
        param_vec = _params_to_vec(p, names)

        history = []
        converged = False

        for it in range(cfg.max_iter):

            p = replace(p, seed=it)

            obs = _measure_obs(p, cfg.observables)
            residual = self._compute_residual(obs)

            if len(self.obs_history) < 2:
                history.append({
                    "iter": it,
                    "params": dict(zip(names, param_vec.tolist())),
                    "observables": obs,
                    "residual_norm": 0.0,
                })
                continue

            res_norm = float(np.linalg.norm(residual))
            if cfg.verbose:
                obs_str = ", ".join(f"{k}={v:.4f}" for k, v in obs.items())
                print(f"  iter {it+1}/{cfg.max_iter}: residual={res_norm:.5f}  {obs_str}")

            history.append({
                "iter": it,
                "params": dict(zip(names, param_vec.tolist())),
                "observables": obs,
                "residual_norm": res_norm,
            })

            if res_norm < cfg.tol_residual:
                converged = True
                break

            J = self._compute_jacobian(p, obs)

            JtJ = J.T @ J

            if cfg.jacobi_scale:
                diag = np.diag(JtJ).copy()
                diag = np.where(diag > 1e-12, diag, 1.0)
                D_inv_sqrt = np.diag(1.0 / np.sqrt(diag))
                JtJ_scaled = D_inv_sqrt @ JtJ @ D_inv_sqrt
                JtJ_reg = JtJ_scaled + cfg.damping * np.eye(len(param_vec))
                step_scaled = np.linalg.solve(JtJ_reg, D_inv_sqrt @ J.T @ residual)
                step = -D_inv_sqrt @ step_scaled
            else:
                JtJ_reg = JtJ + cfg.damping * np.eye(len(param_vec))
                step = -np.linalg.solve(JtJ_reg, J.T @ residual)

            max_step = 0.1 * np.maximum(np.abs(param_vec), 1e-6)
            step = np.clip(step, -max_step, max_step)

            param_vec = param_vec + step
            p = _vec_to_params(p, param_vec, names)

        return {
            "converged": converged,
            "n_iters": len(history),
            "history": history,
            "final_params": dict(zip(names, param_vec.tolist())),
            "final_observables": history[-1]["observables"] if history else {},
        }

def calibrate(params: TriadParams, observables: list[str] = None,
              max_iter: int = 15, verbose: bool = False) -> dict:

    cfg = CalibrationConfig(max_iter=max_iter, verbose=verbose)
    if observables is not None:
        cfg.observables = list(observables)
    return GaussNewtonCalibrator(cfg).calibrate(params)
