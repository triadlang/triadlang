
from __future__ import annotations

from dataclasses import dataclass, field

from runtime.core.solver import TriadParams, integrate
from runtime.physics.observables import crystallinity, dominant_wavenumber, energy
from triad import ntri as np


@dataclass
class PIDConfig:

    kp: float = 1.0
    ki: float = 0.1
    kd: float = 0.05
    max_bias: float = 2.0
    max_integral: float = 5.0
    target: float = 0.8
    observable: str = "crystallinity"
    well_center: float = 0.0
    well_sigma: float = 2.0

@dataclass
class PIDState:

    integral: float = 0.0
    prev_error: float = 0.0
    prev_bias: float = 0.0
    history: list = field(default_factory=list)

def _measure_observable(psi: np.ndarray, dx: float, p: TriadParams,
                         name: str) -> float:

    if name == "crystallinity":
        return crystallinity(psi, dx)
    elif name == "energy":
        return energy(psi, dx, hbar=p.hbar, m=p.m, Lambda=p.Lambda)
    elif name == "k_star":
        return dominant_wavenumber(psi, dx)
    else:
        raise ValueError(f"unknown observable: {name}")

def _pid_step(state: PIDState, error: float, dt_ctrl: float,
              cfg: PIDConfig) -> float:

    p_term = cfg.kp * error

    state.integral += error * dt_ctrl
    state.integral = np.clip(state.integral, -cfg.max_integral, cfg.max_integral)
    i_term = cfg.ki * state.integral

    d_term = cfg.kd * (error - state.prev_error) / max(dt_ctrl, 1e-10)

    bias = np.clip(p_term + i_term + d_term, -cfg.max_bias, cfg.max_bias)

    state.prev_error = error
    state.prev_bias = bias
    state.history.append(bias)

    return bias

def steer(p: TriadParams, cfg: PIDConfig | None = None,
          n_rounds: int = 20, T_per_round: float = 2.0,
          verbose: bool = False) -> dict:

    if cfg is None:
        cfg = PIDConfig()

    pid = PIDState()
    obs_history = []
    bias_history = []
    converged = False
    final_rnd = n_rounds

    x = np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
    dx = float(x[1] - x[0])
    psi = np.exp(-x**2 / 8.0).astype(np.complex128)
    psi /= np.sqrt((np.abs(psi)**2).sum() * dx)
    y = None

    for rnd in range(n_rounds):

        p_round = TriadParams(**{**p.__dict__, 'T': T_per_round,
                                  'seed': p.seed + rnd * 13})
        r = integrate(p_round, psi0=psi, y0=y)
        psi = r['psi_final']
        y = r['y_final']

        obs_val = _measure_observable(psi, dx, p, cfg.observable)

        error = cfg.target - obs_val

        bias = _pid_step(pid, error, dt_ctrl=1.0, cfg=cfg)

        sigma = cfg.well_sigma
        center = cfg.well_center
        bias_cap = bias

        def _make_vext(b, sig, cen):
            def vext_fn(x_arr):
                return -b * np.exp(-(x_arr - cen)**2 / (2 * sig**2))
            return vext_fn

        p = TriadParams(**{**p.__dict__,
                            'V_ext': _make_vext(bias_cap, sigma, center),
                            'seed': p.seed})

        obs_history.append(obs_val)
        bias_history.append(bias)
        final_rnd = rnd + 1

        if verbose and (rnd % 5 == 0 or rnd < 3):
            print(f"  round {rnd:3d}  obs={obs_val:.4f}  err={error:.4f}  "
                  f"bias={bias:.4f}")

        if len(obs_history) >= 3:
            recent_errors = [abs(cfg.target - o) for o in obs_history[-3:]]
            if all(e < 0.05 * max(cfg.target, 0.01) for e in recent_errors):
                converged = True
                if verbose:
                    print(f"  STEERED at round {rnd}: obs={obs_val:.4f}")
                final_rnd = rnd + 1
                break

    final_obs = {
        "crystallinity": crystallinity(psi, dx),
        "energy": energy(psi, dx, hbar=p.hbar, m=p.m, Lambda=p.Lambda),
        "k_star": dominant_wavenumber(psi, dx),
    }

    return {
        "psi_final": psi,
        "y_final": y,
        "converged": converged,
        "n_rounds": final_rnd,
        "observables_history": obs_history,
        "bias_history": bias_history,
        "observables": final_obs,
        "target": cfg.target,
        "observable": cfg.observable,
    }
