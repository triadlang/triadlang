"""End-to-end parity: C solver (triad_solver.c) vs Python solver
(runtime/core/solver.py).

Triad rule: nothing is isolated. Every case runs the FULL coupled P1.P2.P3
dynamics in both engines and compares physical observables of the evolved
state (norm, peak density, participation, per-scale P2 memory means).

Noise-on cases are compared as ensembles over seeds: the C engine draws the
FDT bath from xorshift64+Box-Muller and the Python engine from PCG64, so
individual realizations differ by construction while the coupled dynamics
must agree statistically. The eta->0 limit keeps Gamma dissipation, the P2
memory feedback and the P1 spectral operator fully coupled and must match
to FFT round-off.
"""
import dataclasses
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

from runtime.core import solver as S  # noqa: E402

FAILURES = []


def make_params(**kw):
    base = dict(N=256, T=2.0, dt=0.005, L=32.0, backend='numpy', record_every=4)
    base.update(kw)
    return S.TriadParams(**base)


def run_c(p):
    res, ok = S._try_native_c(p)
    if not ok:
        raise RuntimeError('native C path refused to delegate (check libtriad_rt.so)')
    return res


def run_python(p):
    saved = (S._try_native_c, S._try_gpu_fused)
    S._try_native_c = lambda q, **kw: (None, False)
    S._try_gpu_fused = lambda q, **kw: (None, False)
    try:
        return S.integrate(p)
    finally:
        S._try_native_c, S._try_gpu_fused = saved


def observables(res, p):
    rho = np.abs(res['psi_final']) ** 2
    dV = res['dx'] ** p.D
    norm = float(rho.sum() * dV)
    part = norm * norm / max(float((rho ** 2).sum() * dV), 1e-300)
    y = np.asarray(res['y_final'])
    out = {'norm': norm, 'peak': float(rho.max()), 'part': float(part)}
    for j in range(len(p.nu)):
        out[f'y{j}'] = float(y[j].mean())
    return out


def check(name, ok, detail):
    tag = 'PASS' if ok else 'FAIL'
    print(f'  [{tag}] {name}: {detail}')
    if not ok:
        FAILURES.append(name)


def ensemble_case(name, p, seeds):
    print(f'case: {name} (full P1.P2.P3, FDT bath on, {len(seeds)} seeds)')
    cs, pys = [], []
    for s in seeds:
        ps = dataclasses.replace(p, seed=s)
        cs.append(observables(run_c(ps), ps))
        pys.append(observables(run_python(ps), ps))
    for key in cs[0]:
        a = np.array([o[key] for o in cs])
        b = np.array([o[key] for o in pys])
        ma, mb = a.mean(), b.mean()
        sea = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
        seb = b.std(ddof=1) / np.sqrt(len(b)) if len(b) > 1 else 0.0
        z = abs(ma - mb) / np.sqrt(sea ** 2 + seb ** 2 + 1e-300)
        rel = abs(ma - mb) / max(abs(mb), 1e-300)
        ok = (z <= 3.0) or (rel <= 0.05)
        check(f'{key}', ok,
              f'C={ma:.6g}+/-{sea:.2g}  py={mb:.6g}+/-{seb:.2g}  z={z:.2f} rel={rel:.3%}')


def deterministic_case(name, **kw):
    print(f'case: {name} (eta->0 limit; Gamma, P2 memory and P1 operator still coupled)')
    p = make_params(fdt_couple=False, f_FDT=0.002, **kw)
    p.f_FDT = 0.0
    rc = run_c(p)
    saved_v = S._validate_triad_params
    S._validate_triad_params = lambda q: None
    try:
        rp = run_python(p)
    finally:
        S._validate_triad_params = saved_v
    scale_psi = max(float(np.abs(rp['psi_final']).max()), 1e-300)
    scale_y = max(float(np.abs(rp['y_final']).max()), 1e-300)
    dpsi = float(np.abs(rc['psi_final'] - rp['psi_final']).max()) / scale_psi
    dy = float(np.abs(rc['y_final'] - rp['y_final']).max()) / scale_y
    nc = float((np.abs(rc['psi_final']) ** 2).sum() * rc['dx'] ** p.D)
    npy = float((np.abs(rp['psi_final']) ** 2).sum() * rp['dx'] ** p.D)
    check('psi_final', dpsi <= 1e-5, f'max rel diff={dpsi:.3g}')
    check('y_final', dy <= 1e-5, f'max rel diff={dy:.3g}')
    check('norm', abs(nc - npy) / max(npy, 1e-300) <= 1e-8,
          f'C={nc:.12g} py={npy:.12g}')
    for key, tol in (('t', 1e-12), ('peak_t', 1e-8), ('participation_t', 1e-8),
                     ('density', 1e-8)):
        if key not in rc or key not in rp:
            continue
        a, b = np.asarray(rc[key]), np.asarray(rp[key])
        check(f'{key} shape', a.shape == b.shape, f'C{a.shape} py{b.shape}')
        if a.shape != b.shape:
            continue
        scale = max(float(np.abs(b).max()), 1e-300)
        d = float(np.abs(a - b).max()) / scale
        check(key, d <= tol, f'max rel diff={d:.3g}')


def main():
    seeds = list(range(8))
    ensemble_case('D=1 periodic gaussian', make_params(D=1), seeds)
    ensemble_case('D=2 periodic gaussian', make_params(D=2, N=48, T=1.0), seeds)
    ensemble_case('D=3 periodic gaussian', make_params(D=3, N=24, T=0.5), seeds)
    ensemble_case('D=1 absorbing gaussian', make_params(D=1, bc='absorbing'), seeds)
    ensemble_case('D=1 periodic chaos', make_params(D=1, init='chaos'), seeds)
    ensemble_case('D=1 record_every=7', make_params(D=1, record_every=7), seeds)
    deterministic_case('D=1', D=1, N=128, T=1.0)
    deterministic_case('D=1 record_every=7 trajectory', D=1, N=128, T=1.0,
                       record_every=7)
    deterministic_case('D=1 record_every=3 k0', D=1, N=128, T=1.0,
                       record_every=3, init_k0=(0.5, 0.0, 0.0))
    deterministic_case('D=2', D=2, N=32, T=0.5)
    deterministic_case('D=3', D=3, N=16, T=0.25)
    deterministic_case('D=3 record_every=7', D=3, N=16, T=0.25, record_every=7)
    deterministic_case('D=3 k0 record_every=6', D=3, N=16, T=0.25,
                       record_every=6, init_k0=(0.4, -0.3, 0.2))
    deterministic_case('D=4 k0 4-axis record_every=7', D=4, N=8, T=0.25,
                       record_every=7, init_k0=(0.3, -0.2, 0.1, 0.25))
    deterministic_case('D=5 k0 5-axis', D=5, N=6, T=0.1,
                       init_k0=(0.3, -0.2, 0.1, 0.25, -0.15))

    print()
    if FAILURES:
        print(f'solver parity e2e: FAIL ({len(FAILURES)} checks)')
        return 1
    print('solver parity e2e: PASS (full coupled P1.P2.P3 in both engines)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
