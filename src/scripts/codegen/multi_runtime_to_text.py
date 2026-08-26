import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.codec.codec import IntCalib, encode_int
from runtime.core.multi_runtime import CouplingLink, MultiRuntime, Segment
from runtime.core.solver import TriadParams
from triad import ntri as np

NUM_SAMPLES = 4

def _f(x: float) -> str:
    return repr(float(x))

def _emit_samples(arr: np.ndarray) -> str:
    a = np.asarray(arr).ravel()
    N = a.shape[0]
    indices = list(range(min(NUM_SAMPLES, N)))
    if N > NUM_SAMPLES:
        mid = N // 2
        indices += list(range(mid, min(mid + NUM_SAMPLES, N)))
    parts = []
    for i in indices:
        v = a[i]
        if np.iscomplexobj(a):
            parts.append(_f(v.real))
            parts.append(_f(v.imag))
        else:
            parts.append(_f(float(v)))
    return ' '.join(parts)

def _checksum(arr: np.ndarray) -> tuple[float, float, float, float]:
    a = np.asarray(arr).ravel()
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

def build_program() -> tuple[MultiRuntime, list[str]]:
    L = 32.0
    N = 64
    dt = 0.01
    T_settle = 0.5
    T_couple = 1.0
    int_calib = IntCalib()

    def make_params(seed: int) -> TriadParams:

        return TriadParams(N=N, L=L, dt=dt, T=T_settle + T_couple, hbar=1.0, m=1.0, omega=1.0, Lambda=-0.5, alpha=0.15, sigma=1.5, Gamma=0.05, f_FDT=0.002, fdt_couple=True, kT=1.0, lam=np.array([-0.05, -0.03, -0.01]), nu=np.array([0.5, 0.1, 0.02]), mode='triad', seed=seed, V_ext='harmonic', D=1)
    rt = MultiRuntime(dt=dt, record_every=4)
    pA = make_params(seed=11)
    pB = make_params(seed=22)
    psiA = encode_int(3, int_calib, L, N)
    psiB = encode_int(5, int_calib, L, N)
    rt.add_substrate('A', pA, psi=psiA)
    rt.add_substrate('B', pB, psi=psiB)
    seg1 = Segment(t_start=0.0, t_end=T_settle, links=[], active_ids=None)
    rt.add_segment(seg1)
    rt.global_t = T_settle
    rt.global_t = 0.0
    links = [CouplingLink(src_id=0, dst_id=1, kappa=-0.3, coupling_mode='density'), CouplingLink(src_id=1, dst_id=0, kappa=-0.2, coupling_mode='dc_subtracted')]
    seg2 = Segment(t_start=T_settle, t_end=T_settle + T_couple, links=links, active_ids=None)
    rt.add_segment(seg2)
    rt.segments = [Segment(t_start=0.0, t_end=T_settle, links=[], active_ids=None), seg2]
    return (rt, ['A', 'B'])

def run() -> str:
    rt, names = build_program()
    res = rt.run()
    out: list[str] = []
    out.append(f"diverged {int(bool(res.get('diverged')))}")
    out.append(f'global_t {_f(rt.global_t)}')
    for n in names:
        sub = next(s for s in rt.substrates.values() if s.name == n)
        psi = np.asarray(sub.psi)
        rho = np.abs(psi) ** 2
        norm = float(rho.sum() * sub.dx)
        out.append(f'sub {n} N {sub.params.N} dx {_f(sub.dx)}')
        out.append(f'sub {n} norm {_f(norm)}')
        out.append(f'sub {n} psi {_emit_samples(psi)}')
        out.append(f'sub {n} n_records {(sub.density_traj.shape[1] if sub.density_traj is not None else 0)}')
        if sub.density_traj is not None:
            s, sq, mx, mn = _checksum(sub.density_traj)
            out.append(f'sub {n} dens_sum {_f(s)} dens_sumsq {_f(sq)} dens_max {_f(mx)} dens_min {_f(mn)}')
        else:
            out.append(f'sub {n} dens_sum 0.0 dens_sumsq 0.0 dens_max 0.0 dens_min 0.0')
    return '\n'.join(out) + '\n'

def main() -> None:
    sys.stdout.write(run())
if __name__ == '__main__':
    main()
