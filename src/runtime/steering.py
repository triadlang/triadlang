"""Phase B4: V_ext steering via bounded PID controller.

Steers the field toward a target observable (e.g. crystallinity=0.8) by
modulating the external potential V_ext(x).  Uses a PID controller that
adjusts a potential bias parameter (amplitude, center, or width of a
Gaussian well) based on the observable error.

Constraints:
  - V_ext bias is BOUNDED (max_bias parameter) to prevent imposing structure.
  - Steering only via V_ext, never clamping psi.
  - The PID integral term is anti-windup limited.
  - All three pillars remain active throughout (P1+P2+P3).
  - The controller converges to the target EMERGENTLY from the PDE dynamics.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np

from runtime.core.solver import integrate, TriadParams
from runtime.physics.observables import crystallinity, energy, dominant_wavenumber
from runtime.backend import get_xp, asnumpy

@dataclass
class PIDConfig:
    """PID controller configuration for V_ext steering."""
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
    """Internal state of the PID controller."""
    integral: float = 0.0
    prev_error: float = 0.0
    prev_bias: float = 0.0
    history: list = field(default_factory=list)

def _measure_observable(psi: np.ndarray, dx: float, p: TriadParams,
                         name: str) -> float:
    """Measure the specified observable from the field."""
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
    """One PID update.  Returns the new bias amplitude (bounded)."""
    
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

def steer(p: TriadParams, cfg: Optional[PIDConfig] = None,
          n_rounds: int = 20, T_per_round: float = 2.0,
          verbose: bool = False) -> dict:
    """Run V_ext steering via PID control over multiple integration rounds.

    Each round:
      1. Measure the target observable from the current field.
      2. Compute error vs target.
      3. PID updates the V_ext bias amplitude.
      4. Re-integrate with the new V_ext.

    The V_ext is a Gaussian well: V_ext(x) = -bias * exp(-x^2 / (2*sigma^2))
    The bias is bounded to prevent imposing structure.

    Parameters
    ----------
    p : TriadParams
        Base parameters (will be modified each round with new V_ext).
    cfg : PIDConfig, optional
        PID configuration.  Uses defaults if None.
    n_rounds : int
        Number of steering rounds.
    T_per_round : float
        Integration time per round.
    verbose : bool

    Returns
    -------
    dict with final psi, observables history, bias history, converged.
    """
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
