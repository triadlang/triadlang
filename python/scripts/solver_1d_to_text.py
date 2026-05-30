import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from runtime.solver import TriadParams, integrate

def _f(x):
    return repr(float(x))

def _emit_tuple(xs):
    parts = [_f(v) for v in xs]
    return '(' + ' '.join(parts) + ')'

def build_initial_psi(N: int, L: float) -> np.ndarray:
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    dx = x[1] - x[0]
    psi = np.exp(-x ** 2 / 8.0).astype(np.complex128)
    psi /= np.sqrt((np.abs(psi) ** 2).sum() * dx)
    return psi

def dump_run(name: str, p: TriadParams, lines: list) -> None:
    psi0 = build_initial_psi(p.N, p.L)
    dx = p.L / p.N
    norm0 = float((np.abs(psi0) ** 2).sum() * dx)
    out = integrate(p, psi0)
    psi_f = out['psi_final']
    rho = (np.abs(psi_f) ** 2).astype(np.float64)
    normT = float(rho.sum() * dx)
    lines.append(f'fixture {name}')
    lines.append(f'N {p.N} L {_f(p.L)} dt {_f(p.dt)} T {_f(p.T)} mode {p.mode}')
    lines.append(f'Lambda {_f(p.Lambda)} alpha {_f(p.alpha)} sigma {_f(p.sigma)} Gamma {_f(p.Gamma)} f_FDT {_f(p.f_FDT)}')
    lines.append(f'nu {_emit_tuple(p.nu)} lam {_emit_tuple(p.lam)}')
    v_ext = 'None' if p.V_ext is None else p.V_ext if isinstance(p.V_ext, str) else '<callable>'
    lines.append(f'V_ext {v_ext} omega {_f(p.omega)}')
    lines.append(f'norm_initial {_f(norm0)} norm_final {_f(normT)}')
    lines.append('density ' + ' '.join((_f(v) for v in rho)))

def run() -> str:
    lines: list[str] = []
    p_lin = TriadParams(N=64, L=32.0, dt=0.005, T=0.5, hbar=1.0, m=1.0, Lambda=0.0, alpha=0.0, sigma=2.0, Gamma=0.0, f_FDT=0.0, nu=np.array([]), lam=np.array([]), mode='linear', seed=42, V_ext=None, omega=0.0, D=1)
    dump_run('linear', p_lin, lines)
    p_full = TriadParams(N=64, L=32.0, dt=0.005, T=0.5, hbar=1.0, m=1.0, Lambda=-0.5, alpha=0.15, sigma=1.5, Gamma=0.05, f_FDT=0.0, nu=np.array([2.0, 0.5, 0.1]), lam=np.array([-0.3, -0.2, -0.1]), mode='full', seed=42, V_ext='harmonic', omega=0.05, D=1)
    dump_run('full_noiseless', p_full, lines)
    return '\n'.join(lines) + '\n'
if __name__ == '__main__':
    sys.stdout.write(run())