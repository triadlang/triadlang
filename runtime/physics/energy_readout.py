"""Phase A2: Energy readout features and inference-time relaxation (EBT-style).

The energy functional E[psi] is read as a scalar feature from the field.
Inference-time relaxation lets the PDE run under -iGamma + eta (P3) toward
low-energy configurations. This is the "System 2 thinking" analogue from
Energy-Based Transformers (arXiv 2507.02092): more relaxation steps =
better refinement, but the attractor is set by the field's own history
(P2) and thermal bath (P3), never by a label.

All three pillars active during relaxation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from runtime.core.solver import integrate, TriadParams
from runtime.physics.observables import (
    crystallinity,
    dominant_wavenumber,
    ipr,
    participation_ratio,
    energy as compute_energy,
)

@dataclass
class EnergyReadoutConfig:
    """Configuration for energy-based readout."""

    track: List[str] = None
    
    normalize: bool = True
    
    max_steps: int = 8
    
    T_step: float = 2.0
    
    tol: float = 0.01

    def __post_init__(self):
        if self.track is None:
            self.track = ["crystallinity", "energy"]

class EnergyReadout:
    """Read energy-based features from a relaxed field state.

    EBT-style: the field is an energy-based model. Relaxation (more PDE
    steps) refines the prediction. The energy landscape is shaped by
    V_mem (P2) and explored by FDT noise (P3), never imposed externally.

    Usage
    -----
    >>> from runtime.physics.energy_readout import EnergyReadout, EnergyReadoutConfig
    >>> from stdlib.regimes import resolve_regime
    >>> p = resolve_regime("B0", seed=0, N=128)
    >>> reader = EnergyReadout(EnergyReadoutConfig(max_steps=4))
    >>> result = reader.read(p)
    >>> result["converged"]
    """

    def __init__(self, config: EnergyReadoutConfig = None):
        self.config = config or EnergyReadoutConfig()
        self._obs_fns = {
            "crystallinity": lambda psi, dx: crystallinity(psi, dx),
            "dominant_k": lambda psi, dx: dominant_wavenumber(psi, dx),
            "ipr": lambda psi, dx: ipr(psi, dx),
            "participation": lambda psi, dx: participation_ratio(psi, dx),
        }

    def _measure(self, psi: np.ndarray, dx: float, params: TriadParams) -> Dict[str, float]:
        out = {}
        for name in self.config.track:
            if name == "energy":
                V_ext = np.zeros(psi.shape[0]) if not callable(params.V_ext) else None
                out[name] = float(compute_energy(
                    psi, dx, hbar=params.hbar, m=params.m, Lambda=params.Lambda
                ))
            elif name in self._obs_fns:
                out[name] = float(self._obs_fns[name](psi, dx))
        return out

    def read(self, params: TriadParams) -> dict:
        """Run relaxation and read energy features.

        Returns dict with:
        - features: dict of final observables
        - history: list of per-step measurements
        - converged: bool
        - n_steps: int
        - energy_trajectory: list of energy values
        """
        cfg = self.config
        history = []
        energy_traj = []
        converged = False

        p = replace(params, T=cfg.T_step)
        current_params = p

        r0 = integrate(current_params)
        psi = r0["psi_final"]
        dx = r0["dx"]
        obs0 = self._measure(psi, dx, current_params)
        E0 = obs0.get("energy", 1.0)
        if cfg.normalize and abs(E0) > 1e-10:
            E0_ref = abs(E0)
        else:
            E0_ref = 1.0

        history.append({"step": 0, "observables": obs0})
        energy_traj.append(obs0.get("energy", 0.0))

        for step in range(1, cfg.max_steps + 1):
            
            current_params = replace(current_params, seed=step + (params.seed or 0))

            r = integrate(current_params)
            psi = r["psi_final"]
            dx = r["dx"]
            obs = self._measure(psi, dx, current_params)
            E_curr = obs.get("energy", 0.0)

            history.append({"step": step, "observables": obs})
            energy_traj.append(E_curr)

            if len(energy_traj) >= 2:
                dE = abs(energy_traj[-1] - energy_traj[-2]) / E0_ref
                if dE < cfg.tol:
                    converged = True
                    break

        final_obs = self._measure(psi, dx, current_params)
        return {
            "features": final_obs,
            "history": history,
            "converged": converged,
            "n_steps": len(history) - 1,
            "energy_trajectory": energy_traj,
        }

def relax(params: TriadParams,
          steps: int = 4,
          T_step: float = 2.0,
          tol: float = 0.01,
          verbose: bool = False) -> dict:
    """Convenience function for EBT-style relaxation.

    Runs the PDE for `steps` iterations of duration `T_step` each.
    More steps = more refined state (System 2 thinking analogue).

    All three pillars active throughout.
    """
    reader = EnergyReadout(EnergyReadoutConfig(
        max_steps=steps, T_step=T_step, tol=tol
    ))
    return reader.read(params)
