from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
OPCODES = {'FWD': 1, 'INV': 2, 'NLF': 3, 'POT': 4, 'MEM': 5, 'CPL': 6, 'DSP': 7, 'NOI': 8, 'HALT': 255}

@dataclass
class Instruction:
    op: str
    args: tuple = ()

@dataclass
class VMState:
    psi: np.ndarray
    y: np.ndarray
    V_ext: np.ndarray
    half_lin: np.ndarray
    noise_amp: float = 0.0
    Gamma: float = 0.0
    Lambda: float = -0.5
    dt: float = 0.005
    hbar: float = 1.0
    nu: np.ndarray = field(default_factory=lambda: np.array([2.0, 0.5, 0.1]))
    lam: np.ndarray = field(default_factory=lambda: np.array([-0.3, -0.2, -0.1]))
    kappa: float = -3.0
    coupled_rho: Optional[np.ndarray] = None
    rng: Optional[np.random.Generator] = None

class TriadVM:

    def __init__(self, state: VMState):
        self.state = state
        if state.rng is None:
            state.rng = np.random.default_rng(0)
        self._ip = 0
        self._program = []

    def load(self, instructions: list[Instruction]):
        self._program = instructions
        self._ip = 0

    def load_strang_step(self) -> list[Instruction]:
        return [Instruction('FWD'), Instruction('NLF', (0.5,)), Instruction('INV'), Instruction('POT'), Instruction('MEM'), Instruction('CPL'), Instruction('NOI'), Instruction('FWD'), Instruction('NLF', (0.5,)), Instruction('INV')]

    def load_full_run(self, n_steps: int) -> list[Instruction]:
        prog = []
        for _ in range(n_steps):
            prog.extend(self.load_strang_step())
        prog.append(Instruction('HALT'))
        return prog

    def execute(self, n_steps: int=1) -> VMState:
        prog = self.load_full_run(n_steps)
        self.load(prog)
        return self.run()

    def run(self, verbose: bool=False) -> VMState:
        while self._ip < len(self._program):
            instr = self._program[self._ip]
            if instr.op == 'HALT':
                break
            self._exec(instr, verbose)
            self._ip += 1
        return self.state

    def _exec(self, instr: Instruction, verbose: bool=False):
        s = self.state
        op = instr.op
        if op == 'FWD':
            s.psi = np.fft.fft(s.psi)
        elif op == 'INV':
            s.psi = np.fft.ifft(s.psi)
        elif op == 'NLF':
            frac = instr.args[0] if instr.args else 1.0
            s.psi = s.psi * s.half_lin ** frac
        elif op == 'POT':
            rho = np.abs(s.psi) ** 2
            M = len(s.nu)
            V_mem = (s.lam[:, None] * s.y).sum(axis=0) if M > 0 else 0.0
            V_couple = s.kappa * s.coupled_rho if s.coupled_rho is not None else 0.0
            V_tot = s.V_ext + s.Lambda * rho + V_mem + V_couple
            s.psi = s.psi * np.exp(-1j * V_tot * s.dt / s.hbar)
        elif op == 'MEM':
            rho = np.abs(s.psi) ** 2
            M = len(s.nu)
            if M > 0:
                decay = np.exp(-s.nu * s.dt)
                s.y = decay[:, None] * s.y + (1 - decay[:, None]) * rho
        elif op == 'CPL':
            pass
        elif op == 'DSP':
            s.psi = s.psi * np.exp(-s.Gamma * s.dt / s.hbar)
        elif op == 'NOI':
            if s.noise_amp > 0:
                xi = s.rng.standard_normal(len(s.psi))
                xip = s.rng.standard_normal(len(s.psi))
                s.psi = s.psi + s.noise_amp * (xi + 1j * xip) / np.sqrt(2.0)
        if verbose:
            norm = float(np.sum(np.abs(s.psi) ** 2))
            print(f'  [{self._ip:>4}] {op:4s} |ψ|²={norm:.4f}')

def create_vm_state(N: int=128, L: float=32.0, Lambda: float=-0.5, Gamma: float=0.05, f_FDT: float=0.002, dt: float=0.005, seed: int=0, **kwargs) -> TriadVM:
    rng = np.random.default_rng(seed)
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    dx = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    psi = np.exp(-x ** 2 / 8.0).astype(np.complex128)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    nu = np.array(kwargs.get('nu', (2.0, 0.5, 0.1)))
    lam = np.array(kwargs.get('lam', (-0.3, -0.2, -0.1)))
    M = len(nu)
    alpha = kwargs.get('alpha', 0.15)
    sigma = kwargs.get('sigma', 1.5)
    hbar = 1.0
    H_lin = hbar ** 2 * k ** 2 / 2.0 + alpha * np.abs(k) ** sigma
    half_lin = np.exp(-1j * H_lin * dt / (2 * hbar) - Gamma * dt / (2 * hbar))
    V_ext = 0.5 * 1.0 * 0.05 ** 2 * x ** 2
    noise_amp = np.sqrt(f_FDT * dt / dx) if f_FDT > 0 else 0.0
    state = VMState(psi=psi, y=np.zeros((M, N)), V_ext=V_ext, half_lin=half_lin, noise_amp=noise_amp, Gamma=Gamma, Lambda=Lambda, dt=dt, hbar=hbar, nu=nu, lam=lam, rng=rng)
    return TriadVM(state)