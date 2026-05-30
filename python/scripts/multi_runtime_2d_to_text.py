import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from runtime.solver import TriadParams
from runtime.multi_runtime import MultiRuntime, CouplingEdge, Segment
NUM_SAMPLES = 4

def _f(x: float) -> str:
    return repr(float(x))

def _emit_psi_flat(psi: np.ndarray) -> str:
    a = np.asarray(psi).ravel(order='C')
    n = a.shape[0]
    indices = list(range(min(NUM_SAMPLES, n)))
    if n > NUM_SAMPLES:
        mid = n // 2
        indices += list(range(mid, min(mid + NUM_SAMPLES, n)))
    parts = []
    for i in indices:
        parts.append(_f(a[i].real))
        parts.append(_f(a[i].imag))
    return ' '.join(parts)

def _checksum(arr: np.ndarray) -> tuple[float, float, float, float]:
    a = np.asarray(arr).ravel(order='C')
    s = 0.0
    sq = 0.0
    mx = -np.inf
    mn = np.inf
    for v in a:
        v = float(v)
        s += v
        sq += v * v
        if v > mx:
            mx = v
        if v < mn:
            mn = v
    return (s, sq, mx, mn)

def build_program() -> MultiRuntime:
    L = 16.0
    N = 32
    dt = 0.01
    T_settle = 0.3
    T_couple = 0.5

    def make_params(seed: int) -> TriadParams:
        return TriadParams(N=N, L=L, dt=dt, T=T_settle + T_couple, hbar=1.0, m=1.0, omega=1.0, Lambda=-0.3, alpha=0.0, sigma=2.0, Gamma=0.05, f_FDT=0.0, lam=np.array([0.03]), nu=np.array([0.5]), mode='full', seed=seed, V_ext='harmonic', D=2)
    rt = MultiRuntime(dt=dt, record_every=4)
    rt.add_substrate('A', make_params(seed=11), psi=None)
    rt.add_substrate('B', make_params(seed=22), psi=None)
    rt.segments = [Segment(t_start=0.0, t_end=T_settle, edges=[], active_ids=None), Segment(t_start=T_settle, t_end=T_settle + T_couple, edges=[CouplingEdge(src_id=0, dst_id=1, kappa=-0.2, coupling_mode='density'), CouplingEdge(src_id=1, dst_id=0, kappa=-0.15, coupling_mode='density')], active_ids=None)]
    rt.global_t = 0.0
    return rt

def run() -> str:
    rt = build_program()
    res = rt.run()
    out: list[str] = []
    out.append(f"diverged {int(bool(res.get('diverged')))}")
    out.append(f'global_t {_f(rt.global_t)}')
    for sid in sorted(rt.substrates.keys()):
        sub = rt.substrates[sid]
        psi = np.asarray(sub.psi)
        dxD = sub.dx ** sub.D
        norm = float((np.abs(psi) ** 2).sum() * dxD)
        out.append(f'sub {sub.name} D {sub.D} N {sub.params.N} dx {_f(sub.dx)}')
        out.append(f'sub {sub.name} norm {_f(norm)}')
        out.append(f'sub {sub.name} psi {_emit_psi_flat(psi)}')
        nrec = sub.density_traj.shape[-1] if sub.density_traj is not None else 0
        out.append(f'sub {sub.name} n_records {nrec}')
        if sub.density_traj is not None:
            s, sq, mx, mn = _checksum(sub.density_traj)
            out.append(f'sub {sub.name} dens_sum {_f(s)} dens_sumsq {_f(sq)} dens_max {_f(mx)} dens_min {_f(mn)}')
        else:
            out.append(f'sub {sub.name} dens_sum 0.0 dens_sumsq 0.0 dens_max 0.0 dens_min 0.0')
    return '\n'.join(out) + '\n'

def main() -> None:
    sys.stdout.write(run())
if __name__ == '__main__':
    main()