from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from runtime.core.solver import TriadParams, _build_V_ext, integrate
from runtime.physics.observables import crystallinity, dominant_wavenumber, energy, ipr
from triad import ntri as np


@dataclass
class DriveSpec:

    drive_type: str = "gaussian"
    amplitude: float = -0.5
    center: float = 0.0
    width: float = 4.0
    custom_fn: Callable | None = None

def _build_drive_potential(drive: DriveSpec, x: np.ndarray) -> np.ndarray:

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

    converged: bool
    n_iters: int
    residual_norm: float
    history: list[dict]
    psi: np.ndarray
    observables: dict[str, float]

def _compute_observable_vec(psi: np.ndarray, dx: float,
                            p: TriadParams) -> dict:

    obs = {
        "crystallinity": float(crystallinity(psi, dx)),
        "dominant_k": float(dominant_wavenumber(psi, dx)),
        "ipr": float(ipr(psi, dx)),
    }
    try:
        obs["energy"] = float(energy(psi, dx,
                                      hbar=p.hbar, m=p.m,
                                      Lambda=p.Lambda))
    except (ValueError, ArithmeticError, RuntimeError):
        obs["energy"] = float('nan')
    return obs

def _observable_distance(obs_a: dict, obs_b: dict) -> float:

    keys = [k for k in obs_a if k in obs_b]
    if not keys:
        return float("inf")
    diffs = []
    for k in keys:
        denom = max(abs(obs_a[k]), abs(obs_b[k]), 1e-10)
        diffs.append(((obs_a[k] - obs_b[k]) / denom) ** 2)
    return float(np.sqrt(np.mean(diffs)))

class FixedPointSolver:

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

        if drive is None:
            drive = DriveSpec()

        L = params.L
        N = params.N
        x = np.linspace(-L / 2, L / 2, N, endpoint=False)
        dx = L / N

        drive_potential = _build_drive_potential(drive, x)
        drive_pot = drive_potential
        base_v_ext_arr = _build_V_ext(params, x)

        def V_ext_with_drive(x_arr):
            return base_v_ext_arr + drive_pot

        p = replace(params, V_ext=V_ext_with_drive,
                    T=n_substeps * params.dt)

        r0 = integrate(p)
        psi = r0["psi_final"].copy()
        y = r0["y_final"].copy()

        history = []
        converged = False
        residual_norm = float("inf")
        window_size = 3
        obs_window: list[dict] = []

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
