from __future__ import annotations

from dataclasses import dataclass

from runtime.core.solver import TriadParams
from runtime.vm import TriadVM, VMState
from triad import ntri as np


@dataclass
class VMSubstrate:
    name: str
    vm: TriadVM
    params: TriadParams
    id: int = -1
    psi: np.ndarray | None = None
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
        self._ran: bool = False

    def add_substrate(self, name: str, params: TriadParams) -> VMSubstrate:
        p = params
        N = p.N
        L = p.L
        dx = L / N
        k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
        from runtime.core.solver import _build_V_ext
        x = np.linspace(-L / 2, L / 2, N, endpoint=False)
        V_ext = _build_V_ext(p, x)
        alpha = p.alpha
        sigma = p.sigma
        hbar = p.hbar
        Gamma = p.Gamma
        H_lin = hbar ** 2 * k ** 2 / 2.0 + alpha * np.abs(k) ** sigma

        half_lin = np.exp(-1j * H_lin * self.dt / hbar)
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
        self._segments.append((seg.t_start, seg.t_end, seg.links))
        for link in seg.links:
            src_name = self._id_to_name(link.src_id)
            dst_name = self._id_to_name(link.dst_id)
            if dst_name not in self._coupling:
                self._coupling[dst_name] = []
            self._coupling[dst_name].append((src_name, link.kappa))

    def _id_to_name(self, sid: int) -> str:
        for name, sub in self.substrates.items():
            if sub.vm.state.id == sid:
                return name
        return ''

    def run(self, verbose: bool=False):
        for seg_start, seg_end, links in self._segments:
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
        for dst_name, links in list(self._coupling.items()):
            if dst_name not in self.substrates:
                continue
            valid_links = []
            for src_name, kappa in links:
                if src_name in self.substrates:
                    valid_links.append((src_name, kappa))
            if valid_links:
                self._coupling[dst_name] = valid_links
            else:
                del self._coupling[dst_name]

        for name, sub in self.substrates.items():
            if sub.psi is None:
                sub.psi = sub.vm.state.psi.copy()
            if name in self._coupling:
                first_kappa = self._coupling[name][0][1] if self._coupling[name] else -3.0
                sub.vm.state.kappa = first_kappa

    def _get_coupled_rho(self, dst_name: str) -> np.ndarray | None:
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

