from __future__ import annotations

from dataclasses import dataclass

from runtime.physics.observables import crystallinity, dominant_wavenumber, energy, ipr
from triad import ntri as np


@dataclass
class SubstrateObservables:

    substrate_id: int
    energy: float
    crystallinity: float
    fdt_precision: float
    k_star: float
    ipr_val: float

def compute_fdt_precision(sub) -> float:

    f_fdt = sub.f_FDT_e
    if f_fdt <= 0:

        return 1e12
    dx = sub.dx
    D = sub.D
    variance = f_fdt / (dx ** D)
    if variance <= 0:
        return 1e12
    return 1.0 / variance

def gather_observables(substrates: dict) -> dict[int, SubstrateObservables]:

    from runtime.backend import asnumpy
    result = {}
    for sid, sub in substrates.items():
        if not sub.active:
            continue
        psi_np = np.asarray(asnumpy(sub.psi))
        dx = float(sub.dx)
        result[sid] = SubstrateObservables(
            substrate_id=sid,
            energy=float(energy(psi_np, dx)),
            crystallinity=float(crystallinity(psi_np, dx)),
            fdt_precision=compute_fdt_precision(sub),
            k_star=float(dominant_wavenumber(psi_np, dx)),
            ipr_val=float(ipr(psi_np, dx)),
        )
    return result

class FieldRouter:

    def __init__(self, method: str = 'precision_weighted',
                 temperature: float = 0.0):
        self.method = method
        self.temperature = temperature

    def compute_gates(self, obs: dict[int, SubstrateObservables]
                      ) -> dict[int, float]:

        if not obs:
            return {}

        ids = list(obs.keys())
        precisions = np.array([obs[sid].fdt_precision for sid in ids])

        if self.method == 'precision_weighted':
            gates = self._precision_weighted(precisions)
        else:

            gates = np.ones(len(ids)) / len(ids)

        return {sid: float(gates[i]) for i, sid in enumerate(ids)}

    def _precision_weighted(self, precisions: np.ndarray) -> np.ndarray:

        prec = np.maximum(precisions, 1e-30)

        if self.temperature > 0:
            log_prec = np.log(prec)

            scaled = log_prec / self.temperature
            shifted = scaled - scaled.max()
            exp_s = np.exp(shifted)
            gates = exp_s / exp_s.sum()
        else:
            gates = prec / prec.sum()

        gates = gates / gates.sum()
        return gates

    def route(self, coupling_links: list,
              substrates_obs: dict[int, SubstrateObservables]
              ) -> dict[tuple, float]:

        gates = self.compute_gates(substrates_obs)
        result = {}
        for e in coupling_links:
            src_gate = gates.get(e.src_id, 0.0)
            dst_gate = gates.get(e.dst_id, 0.0)

            weight = np.sqrt(src_gate * dst_gate) if (src_gate > 0 and dst_gate > 0) else 0.0
            effective_kappa = float(e.kappa * weight)
            result[(e.src_id, e.dst_id)] = effective_kappa
        return result
