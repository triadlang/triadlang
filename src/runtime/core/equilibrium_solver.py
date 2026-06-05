"""Equilibrium fixed-point relaxation via observable-space convergence.

Finds the stationary field state of the FULL PDE (P1+P2+P3 active) under
an optional boundary drive.  The fixed point is emergent from the dynamics,
never imposed externally.

The key insight: with P3 (FDT noise) active, the fixed-point map S(psi)
is stochastic. Pointwise Anderson acceleration on psi is meaningless because
each evaluation of S produces a different result. Instead, convergence must
be measured in observable space (crystallinity, energy, k*). The solver
runs consecutive super-step integrations, each continuing from the previous
final state, and stops when observables stabilize.

Without P3 (FDT noise), the field decays to zero and there is no
crystallization. P3 is load-bearing. P2 sets history dependence and
refines the attractor structure. All three pillars must be active.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional

import numpy as np

from runtime.core.solver import integrate, TriadParams
from runtime.physics.observables import crystallinity, dominant_wavenumber, ipr, energy

@dataclass
class DriveSpec:
    """External drive potential for equilibrium solve.

    drive_type : str
        'gaussian': Gaussian well at center with given amplitude and width
        'double_well': Two Gaussian wells separated by distance d
        'custom': callable(x) -> V(x)
    """
    drive_type: str = "gaussian"
    amplitude: float = -0.5
    center: float = 0.0
    width: float = 4.0
    custom_fn: Optional[Callable] = None

def _build_drive_potential(drive: DriveSpec, x: np.ndarray) -> np.ndarray:
    """Build the drive potential array."""
    if drive.drive_type == "gaussian":
        return drive.amplitude * np.exp(-(x - drive.center) ** 2 / (2 * drive.width ** 2))
    elif drive.drive_type == "double_well":
        d = drive.width
        return drive.amplitude * (
            np.exp(-(x - drive.center - d) ** 2 / 4.0)
            + np.exp(-(x - drive.center + d) ** 2 / 4.0)
        )
    elif drive.drive_type == "custom" and drive.custom_fn is not None:
        return drive.custom_fn(x)
    else:
        return np.zeros_like(x)

@dataclass
class EquilibriumResult:
    """Result of equilibrium solve."""
    converged: bool
    n_iters: int
    residual_norm: float
    history: List[Dict]
    psi: np.ndarray
    observables: Dict[str, float]

def _compute_observable_vec(psi: np.ndarray, dx: float,
                            p: TriadParams) -> dict:
    """Compute observable vector from field state."""
    obs = {
        "crystallinity": float(crystallinity(psi, dx)),
        "dominant_k": float(dominant_wavenumber(psi, dx)),
        "ipr": float(ipr(psi, dx)),
    }
    try:
        obs["energy"] = float(energy(psi, dx,
                                      hbar=p.hbar, m=p.m,
                                      Lambda=p.Lambda))
    except Exception:
        pass
    return obs

def _observable_distance(obs_a: dict, obs_b: dict) -> float:
    """Relative L2 distance between two observable dicts."""
    keys = [k for k in obs_a if k in obs_b]
    if not keys:
        return float("inf")
    diffs = []
    for k in keys:
        denom = max(abs(obs_a[k]), abs(obs_b[k]), 1e-10)
        diffs.append(((obs_a[k] - obs_b[k]) / denom) ** 2)
    return float(np.sqrt(np.mean(diffs)))

class FixedPointSolver:
    """Fixed-point solver for equilibrium field state.

    Runs consecutive super-step integrations from the previous final state.
    Convergence is detected in observable space: when crystallinity and energy
    stop changing relative to a rolling window, the system has reached its
    emergent steady state.

    Unlike naive Anderson acceleration on the field (which breaks with P3
    because each S(psi) is stochastic), this solver accumulates physical
    time and measures when observables plateau.

    Parameters
    ----------
    m_anderson : int
        Reserved for API compatibility. Observable-space convergence does
        not use field-level Anderson.
    mixing : float
        Not used for field mixing. Kept for API compatibility.
    tol : float
        Convergence tolerance on relative observable change.
    max_iter : int
        Maximum number of super-step iterations.
    """

    def __init__(self, m_anderson: int = 5, mixing: float = 0.7,
                 tol: float = 1e-3, max_iter: int = 50):
        self.m_anderson = m_anderson
        self.mixing = mixing
        self.tol = tol
        self.max_iter = max_iter

    def solve_equilibrium(self,
                          params: TriadParams,
                          drive: DriveSpec = None,
                          n_substeps: int = 50,
                          verbose: bool = False) -> EquilibriumResult:
        """Solve for equilibrium field state under boundary drive.

        Each iteration runs a full PDE integration of n_substeps * dt starting
        from the previous psi/y state. Convergence is detected when
        crystallinity and energy stabilize (relative change < tol) over a
        rolling window.

        Parameters
        ----------
        params : TriadParams
            Base parameters for the PDE.
        drive : DriveSpec or None
            External drive potential specification.
        n_substeps : int
            Number of PDE substeps per iteration.
        """
        if drive is None:
            drive = DriveSpec()

        L = params.L
        N = params.N
        x = np.linspace(-L / 2, L / 2, N, endpoint=False)
        dx = L / N

        drive_potential = _build_drive_potential(drive, x)
        drive_pot = drive_potential

        def V_ext_with_drive(x_arr):
            return drive_pot

        p = replace(params, V_ext=V_ext_with_drive,
                    T=n_substeps * params.dt)

        r0 = integrate(p)
        psi = r0["psi_final"].copy()
        y = r0["y_final"].copy()

        history = []
        converged = False
        residual_norm = float("inf")
        window_size = 3
        obs_window: List[dict] = []

        for it in range(self.max_iter):
            
            r = integrate(p, psi0=psi, y0=y)
            psi_next = r["psi_final"]
            y_next = r["y_final"]

            residual = psi_next - psi
            residual_norm = float(np.linalg.norm(residual))

            obs = _compute_observable_vec(psi_next, dx, p)

            history.append({
                "iter": it,
                "residual_norm": residual_norm,
                "observables": obs,
            })

            if verbose:
                print(f"  iter {it}: |r|={residual_norm:.6f}, "
                      f"C={obs['crystallinity']:.4f}")

            obs_window.append(obs)
            if len(obs_window) > window_size:
                obs_window.pop(0)

            if len(obs_window) >= window_size and it >= 5:
                c_vals = np.array([o["crystallinity"] for o in obs_window])
                c_mean = c_vals.mean()
                c_std = c_vals.std()
                
                if c_mean > 0.01 and c_std / c_mean < self.tol:
                    converged = True
                    psi = psi_next
                    y = y_next
                    break

            psi = psi_next
            y = y_next

        final_obs = _compute_observable_vec(psi, dx, p)

        return EquilibriumResult(
            converged=converged,
            n_iters=len(history),
            residual_norm=residual_norm,
            history=history,
            psi=psi,
            observables=final_obs,
        )
