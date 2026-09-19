
from __future__ import annotations

from dataclasses import dataclass, replace

from runtime.core.solver import TriadParams, integrate
from runtime.physics.observables import crystallinity, dominant_wavenumber
from triad import ntri as np


def turing_wavelength(psi: np.ndarray, dx: float) -> float:

    rho = np.abs(psi) ** 2
    rho_hat = np.fft.rfft(rho)
    power = np.abs(rho_hat) ** 2
    N = len(rho)
    L = N * dx
    k_vals = 2.0 * np.pi * np.arange(len(power)) / L

    if len(power) < 2:
        return 0.0
    k_star = k_vals[1 + np.argmax(power[1:])]
    if k_star < 1e-10:
        return float("inf")
    return 2.0 * np.pi / k_star

def pattern_entropy(psi: np.ndarray, dx: float) -> float:

    rho = np.abs(psi) ** 2
    rho_hat = np.fft.rfft(rho)
    power = np.abs(rho_hat) ** 2

    total = power.sum()
    if total < 1e-20:
        return 0.0
    prob = power / total

    prob = prob[prob > 1e-20]
    return float(-np.sum(prob * np.log(prob)))

def pattern_metrics(psi: np.ndarray, dx: float) -> dict[str, float]:

    return {
        "turing_wavelength": turing_wavelength(psi, dx),
        "pattern_entropy": pattern_entropy(psi, dx),
    }

@dataclass
class SigmaSchedule:

    sigma_start: float = 0.5
    sigma_end: float = 2.0
    mode: str = "triad"
    n_steps: int = 10
    crystallinity_target: float = 0.5

    def sigma_at_step(self, step: int) -> float:

        if self.mode == "triad":
            frac = step / max(self.n_steps - 1, 1)
            return self.sigma_start + frac * (self.sigma_end - self.sigma_start)
        raise ValueError(f"unknown SigmaSchedule mode {self.mode!r}; use 'triad'")

@dataclass
class CurriculumResult:

    converged: bool
    n_steps: int
    history: list[dict]
    final_params: TriadParams
    final_observables: dict[str, float]

def run_curriculum(params: TriadParams,
                   schedule: SigmaSchedule = None,
                   T_per_step: float = 3.0,
                   verbose: bool = False) -> CurriculumResult:

    if schedule is None:
        schedule = SigmaSchedule()

    history = []
    p = params
    converged = False

    for step in range(schedule.n_steps):
        sigma = schedule.sigma_at_step(step)
        p = replace(p, sigma=sigma, T=T_per_step, seed=(params.seed or 0) + step)

        r = integrate(p)
        dx = r["dx"]
        psi = r["psi_final"]

        obs = {
            "crystallinity": float(crystallinity(psi, dx)),
            "dominant_k": float(dominant_wavenumber(psi, dx)),
            "sigma": sigma,
        }
        obs.update(pattern_metrics(psi, dx))

        history.append({
            "step": step,
            "sigma": sigma,
            "observables": obs,
        })

        if verbose:
            print(f"  step {step}: sigma={sigma:.2f}, C={obs['crystallinity']:.4f}, "
                  f"entropy={obs['pattern_entropy']:.2f}")

        if schedule.mode == "emergent":
            if obs["crystallinity"] >= schedule.crystallinity_target:

                schedule.sigma_start = sigma
                if sigma >= schedule.sigma_end:
                    converged = True
                    break
        else:
            if step == schedule.n_steps - 1:
                converged = obs["crystallinity"] > 0.3

    final_obs = history[-1]["observables"] if history else {}
    return CurriculumResult(
        converged=converged,
        n_steps=len(history),
        history=history,
        final_params=p,
        final_observables=final_obs,
    )
