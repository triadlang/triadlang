"""Fase 4  Camada 4: co-design com reservoir fisico (spin-wave/fotonico), exploratorio.

Principios (do relatorio):
  - O reservoir fisico e ele proprio um sistema aberto nao-linear na borda do caos: a mesma
    fenomenologia de P3. Ele faz READOUT de observaveis emergentes OU fornece o ruido eta
    fisico em tempo real (hardware-in-the-loop).
  - Guardrail: o ruido fisico injetado DEVE satisfazer a relacao FDT; caso contrario o ruido
    introduz vies nao-fisico e a borda do caos vira artefato de calibracao. A borda do caos
    e verificada (Lyapunov maximo ~ 0), onde a capacidade de memoria tem pico.

Sem hardware aqui: um emulador em software com a MESMA interface que um adaptador de
dispositivo teria. O seam de hardware-in-the-loop ja existe no solver: integrate(...,
noise_provider=...). A fonte FDT abaixo honra o f_FDT que o solver calculou (acoplado a
Gamma quando fdt_couple=True na Fase 1), entao o ruido fisico fica FDT-consistente por
construcao; a verificacao confirma variancia e brancura.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from runtime.core.solver import TriadParams, integrate, _effective_params
from runtime.physics.observables import reservoir_memory_capacity

class PhysicalReservoirEmulator:
    """Reservoir fisico emulado. 'gain' e o raio espectral efetivo: varre a borda do caos."""

    def __init__(self, size: int = 300, gain: float = 1.0, leak: float = 0.9,
                 in_scale: float = 0.3, seed: int = 0):
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((size, size)) / np.sqrt(size)
        ev = float(np.max(np.abs(np.linalg.eigvals(W))))
        self.W = W / (ev + 1e-12)          
        self.W_in = in_scale * rng.standard_normal(size)
        self.size = size
        self.gain = gain
        self.leak = leak
        self.rng = rng

    def _step(self, x: np.ndarray, u: float) -> np.ndarray:
        pre = self.gain * (self.W @ x + self.W_in * u)
        return (1.0 - self.leak) * x + self.leak * np.tanh(pre)

    def run(self, u_series, washout: int = 100, x0: Optional[np.ndarray] = None):
        x = np.zeros(self.size) if x0 is None else x0.copy()
        states = []
        for t, u in enumerate(u_series):
            x = self._step(x, float(u))
            if t >= washout:
                states.append(x.copy())
        return np.asarray(states), x

    def maximal_lyapunov(self, u_series, washout: int = 200, eps: float = 1e-8) -> float:
        """Benettin (twin trajectory). ~0 indica a borda do caos; >0 caos; <0 ordenado."""
        u_series = np.asarray(u_series, dtype=float)
        x = np.zeros(self.size)
        for u in u_series[:washout]:
            x = self._step(x, float(u))
        xp = x + eps * self.rng.standard_normal(self.size)
        d0 = np.linalg.norm(xp - x)
        xp = x + (xp - x) * (eps / (d0 + 1e-30))
        s = 0.0
        n = 0
        for u in u_series[washout:]:
            x = self._step(x, float(u))
            xp = self._step(xp, float(u))
            d = np.linalg.norm(xp - x)
            if d > 0:
                s += np.log(d / eps)
                n += 1
                xp = x + (xp - x) * (eps / d)
        return float(s / max(n, 1))

    def memory_capacity(self, length: int = 1500, washout: int = 100,
                        max_delay: Optional[int] = None, seed: int = 1) -> dict:
        rng = np.random.default_rng(seed)
        u = rng.uniform(-1.0, 1.0, length)
        states, _ = self.run(u, washout=washout)
        return reservoir_memory_capacity(states, u[washout:], max_delay=max_delay)

@dataclass
class FDTNoiseSource:
    """Fonte de eta hardware-in-the-loop, compativel com solver.integrate(noise_provider=...).

    mode='white_fdt'        ruido branco com variancia FDT (consistente).
    mode='reservoir_colored'  1-polo no tempo: preserva a variancia mas correlaciona (viola a
                              FDT branca). E o ARTEFATO que o guardrail manda evitar; incluido
                              para a verificacao conseguir distinguir consistente de artefato.
    """
    mode: str = 'white_fdt'
    color_rho: float = 0.7
    seed: int = 0
    _rng: object = field(default=None, repr=False)
    _state: object = field(default=None, repr=False)
    record: list = field(default_factory=list)

    def reset(self, N: int):
        self._rng = np.random.default_rng(self.seed)
        self._state = np.zeros(N, dtype=np.complex128)
        self.record = []

    def provider(self, step: int, dt: float, N: int, dx: float, f_FDT: float) -> np.ndarray:
        if self._rng is None or self._state is None or np.shape(self._state)[0] != N:
            self.reset(N)
        noise_amp = float(np.sqrt(f_FDT * dt / dx))   
        a = self._rng.standard_normal(N)
        b = self._rng.standard_normal(N)
        white = noise_amp * (a + 1j * b) / np.sqrt(2.0)
        if self.mode == 'white_fdt':
            eta = white
        else:
            r = self.color_rho
            self._state = r * self._state + np.sqrt(1.0 - r * r) * white
            eta = self._state
        self.record.append(np.asarray(eta).copy())
        return eta

def verify_fdt(source: FDTNoiseSource, p: TriadParams, T: float = 20.0) -> dict:
    """Roda o solver com a fonte fisica e confere a FDT: variancia realizada vs exigida e
    brancura temporal (autocorrelacao de lag 1). FDT-consistente se ambos baterem."""
    pl = TriadParams(**{**p.__dict__, 'T': T})
    source.reset(pl.N)
    out = integrate(pl, noise_provider=source.provider, auto_halve_dt=False)
    rec = np.asarray(source.record)        
    dx = out['dx']
    dt = pl.dt
    eff = _effective_params(pl)
    f_FDT_e = eff['f_FDT']
    if getattr(pl, 'fdt_couple', False) and eff['Gamma'] > 0:
        f_FDT_e = 2.0 * eff['Gamma'] * dx * pl.kT / pl.hbar
    required_var = f_FDT_e * dt / dx
    realized_var = float(np.mean(np.abs(rec) ** 2)) if rec.size else 0.0
    if rec.shape[0] > 2:
        a0, a1 = rec[:-1], rec[1:]
        lag1 = float(np.mean(np.real(a0 * np.conj(a1))) / (np.mean(np.abs(rec) ** 2) + 1e-30))
    else:
        lag1 = 0.0
    var_ratio = realized_var / (required_var + 1e-30)
    fdt_consistent = abs(var_ratio - 1.0) < 0.1 and abs(lag1) < 0.1
    return {'required_var': required_var, 'realized_var': realized_var,
            'var_ratio': var_ratio, 'lag1_autocorr': lag1,
            'fdt_consistent': bool(fdt_consistent),
            'final_norm': float((np.abs(out['psi_final']) ** 2).sum() * dx)}

def edge_of_chaos_sweep(gains, size: int = 300, leak: float = 0.9,
                        lyap_len: int = 1200, mc_len: int = 1500, seed: int = 0) -> list:
    """Para cada gain: Lyapunov maximo e capacidade de memoria. O pico de MC deve coincidir
    com Lyapunov ~ 0 (borda do caos), como nos dispositivos spin-wave/fotonicos citados."""
    rng = np.random.default_rng(seed)
    drive = rng.uniform(-1.0, 1.0, lyap_len)
    rows = []
    for g in gains:
        res = PhysicalReservoirEmulator(size=size, gain=float(g), leak=leak, seed=seed)
        lyap = res.maximal_lyapunov(drive)
        mc = res.memory_capacity(length=mc_len)['memory_capacity']
        rows.append({'gain': float(g), 'lyapunov': lyap, 'memory_capacity': mc})
    return rows
