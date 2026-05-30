from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from runtime.vm import TriadVM, VMState, Instruction, create_vm_state
from runtime.solver import TriadParams
from runtime.multi_runtime import CouplingEdge, Segment
from stdlib.regimes import resolve_regime

@dataclass
class VMSubstrate:
    name: str
    vm: TriadVM
    params: dict
    id: int = -1
    psi: Optional[np.ndarray] = None
    dx: float = 0.25

class VMRuntime:

    def __init__(self, dt: float=0.005, record_every: int=4):
        self.dt = dt
        self.record_every = record_every
        self.substrates: dict[str, VMSubstrate] = {}
        self._next_id = 0
        self._segments: list[tuple[float, float, list]] = []
        self.global_t: float = 0.0
        self._coupling: dict[str, list[tuple[str, float]]] = {}

    def add_substrate(self, name: str, params: TriadParams) -> VMSubstrate:
        p = params
        N = p.N
        L = p.L
        dx = L / N
        k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
        from runtime.solver import _build_V_ext
        x = np.linspace(-L / 2, L / 2, N, endpoint=False)
        V_ext = _build_V_ext(p, x)
        alpha = p.alpha
        sigma = p.sigma
        hbar = p.hbar
        Gamma = p.Gamma
        H_lin = hbar ** 2 * k ** 2 / 2.0 + alpha * np.abs(k) ** sigma
        half_lin = np.exp(-1j * H_lin * self.dt / (2 * hbar) - Gamma * self.dt / (2 * hbar))
        psi = np.exp(-x ** 2 / 8.0).astype(np.complex128)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
        nu = np.array(p.nu)
        lam = np.array(p.lam)
        M = len(nu)
        noise_amp = np.sqrt(p.f_FDT * self.dt / dx) if p.f_FDT > 0 else 0.0
        state = VMState(psi=psi, y=np.zeros((M, N)), V_ext=V_ext, half_lin=half_lin, noise_amp=noise_amp, Gamma=Gamma, Lambda=p.Lambda, dt=self.dt, hbar=hbar, nu=nu, lam=lam, rng=np.random.default_rng(p.seed))
        vm = TriadVM(state)
        vm.state.id = self._next_id
        self._next_id += 1
        sub = VMSubstrate(name=name, vm=vm, params=p, id=self._next_id - 1)
        sub.psi = state.psi
        sub.dx = dx
        self.substrates[name] = sub
        return sub

    def add_segment(self, seg):
        self._segments.append((seg.t_start, seg.t_end, seg.edges))
        for edge in seg.edges:
            src_name = self._id_to_name(edge.src_id)
            dst_name = self._id_to_name(edge.dst_id)
            if dst_name not in self._coupling:
                self._coupling[dst_name] = []
            self._coupling[dst_name].append((src_name, edge.kappa))

    def _id_to_name(self, sid: int) -> str:
        for name, sub in self.substrates.items():
            if sub.vm.state.id == sid:
                return name
        return ''

    def run(self, verbose: bool=False):
        for seg_start, seg_end, edges in self._segments:
            T = seg_end - seg_start
            n_steps = max(1, int(round(T / self.dt)))
            self._setup_coupling()
            for step in range(n_steps):
                for name, sub in self.substrates.items():
                    vm = sub.vm
                    s = vm.state
                    coupled_rho = self._get_coupled_rho(name)
                    s.coupled_rho = coupled_rho
                    prog = vm.load_strang_step()
                    vm.load(prog)
                    vm.run()
            for name, sub in self.substrates.items():
                sub.psi = sub.vm.state.psi
        self._ran = True

    def _setup_coupling(self):
        pass

    def _get_coupled_rho(self, dst_name: str) -> Optional[np.ndarray]:
        if dst_name not in self._coupling:
            return None
        total = None
        for src_name, kappa in self._coupling[dst_name]:
            src = self.substrates.get(src_name)
            if src is None:
                continue
            rho = np.abs(src.psi) ** 2
            if total is None:
                total = kappa * rho
            else:
                total = total + kappa * rho
        return total

def _patch_substrate(sub):
    sub.psi = sub.vm.state.psi
    sub.dx = sub.vm.state.V_ext.shape[0] and sub.params.get('L', 32.0) / sub.vm.state.psi.shape[0]