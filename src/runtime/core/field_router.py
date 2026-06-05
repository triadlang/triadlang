"""Precision-weighted FieldRouter for multi-substrate MoE coupling.

The FieldRouter reads per-substrate observables (energy, crystallinity,
FDT precision = 1/variance of the substrate's thermal bath) and computes
gating weights that determine how strongly each substrate's field couples
to its neighbours through CouplingEdges.

All substrates run full mode (P1+P2+P3 active).  Gating is derived from
each substrate's own thermal/memory state, not from an external target.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from runtime.physics.observables import crystallinity, energy, dominant_wavenumber, ipr

@dataclass
class SubstrateObservables:
    """Snapshot of observables for a single substrate."""
    substrate_id: int
    energy: float
    crystallinity: float
    fdt_precision: float  
    k_star: float
    ipr_val: float

def compute_fdt_precision(sub) -> float:
    """Compute FDT precision = 1/variance for a substrate.

    The FDT bath variance is proportional to f_FDT_e * dx^D (the noise
    amplitude squared integrated over the grid).  Precision is its inverse.

    A substrate with low FDT variance (tightly constrained thermal bath)
    has high precision and should dominate routing.
    """
    f_fdt = sub.f_FDT_e
    if f_fdt <= 0:
        
        return 1e12
    dx = sub.dx
    D = getattr(sub, 'D', 1)
    variance = f_fdt / (dx ** D)
    if variance <= 0:
        return 1e12
    return 1.0 / variance

def gather_observables(substrates: dict) -> Dict[int, SubstrateObservables]:
    """Gather observables from all active substrates.

    Parameters
    ----------
    substrates : dict[int, Substrate]
        Map from substrate id to Substrate object.

    Returns
    -------
    dict[int, SubstrateObservables]
    """
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
    """Precision-weighted router for multi-substrate MoE coupling.

    The router assigns a gating weight to each substrate based on the
    inverse variance (precision) of its FDT thermal bath.  Substrates
    with higher precision (lower thermal noise) receive more routing
    weight, since their fields carry more reliable information.

    This is the MoE (mixture of experts) gating function for triad-lang's
    multi-substrate coupling.  Every substrate runs P1+P2+P3 in full mode;
    the router only modulates coupling strengths between them.

    Parameters
    ----------
    method : str
        Routing method.  'precision_weighted' (default) uses
        gate_i = precision_i / sum(precision_j).
    temperature : float
        Softmax temperature for soft routing.  When 0, hard precision
        weighting is used.  Higher values flatten the distribution.
    """

    def __init__(self, method: str = 'precision_weighted',
                 temperature: float = 0.0):
        self.method = method
        self.temperature = temperature

    def compute_gates(self, obs: Dict[int, SubstrateObservables]
                      ) -> Dict[int, float]:
        """Compute gating weights for all observed substrates.

        Parameters
        ----------
        obs : dict[int, SubstrateObservables]
            Per-substrate observable snapshots.

        Returns
        -------
        dict[int, float]
            Per-substrate gate weight.  All weights sum to 1.0.
        """
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
        """gate_i = precision_i / sum(precision_j).

        With temperature > 0, applies softmax over log-precisions:
            gate_i = exp(log(prec_i) / T) / sum(exp(log(prec_j) / T))
        """
        
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

    def route(self, coupling_edges: list,
              substrates_obs: Dict[int, SubstrateObservables]
              ) -> Dict[tuple, float]:
        """Compute weighted coupling strengths for all edges.

        For each CouplingEdge, the base kappa is scaled by the source
        substrate's gate weight and the destination substrate's gate
        weight.  This implements precision-weighted MoE routing:
        information flows preferentially from high-precision sources
        to high-precision destinations.

        Parameters
        ----------
        coupling_edges : list[CouplingEdge]
            Edges defining the coupling topology.
        substrates_obs : dict[int, SubstrateObservables]
            Current per-substrate observables.

        Returns
        -------
        dict[(src_id, dst_id), float]
            Effective coupling strength for each edge.
        """
        gates = self.compute_gates(substrates_obs)
        result = {}
        for e in coupling_edges:
            src_gate = gates.get(e.src_id, 0.0)
            dst_gate = gates.get(e.dst_id, 0.0)
            
            weight = np.sqrt(src_gate * dst_gate) if (src_gate > 0 and dst_gate > 0) else 0.0
            effective_kappa = float(e.kappa * weight)
            result[(e.src_id, e.dst_id)] = effective_kappa
        return result
