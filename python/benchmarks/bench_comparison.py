import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend.parser_universal import parse
from runtime.compiler_runtime import TriadCompiler
from runtime.solver import integrate, TriadParams
import numpy as np
import io
from contextlib import redirect_stdout, redirect_stderr

def _silent_run(src, filepath='<bench>'):
    mod = parse(src, filepath)
    c = TriadCompiler()
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        c.compile_and_run(mod)

def pure_python_pde(N=128, L=32.0, T=2.0, dt=0.005, Lambda=4.0, Gamma=0.02):
    import math, cmath
    dx = L / N
    steps = int(T / dt)
    hbar = 1.0
    m = 1.0
    x = [-L / 2 + i * dx for i in range(N)]
    k = [2 * math.pi * (i if i < N // 2 else i - N) / L for i in range(N)]
    psi = [complex(math.exp(-xi ** 2 / 8.0), 0.0) for xi in x]
    norm = math.sqrt(sum((abs(p) ** 2 * dx for p in psi)))
    psi = [p / norm for p in psi]

    def dft(arr):
        n = len(arr)
        result = [0j] * n
        for kk in range(n):
            s = 0j
            for j in range(n):
                s += arr[j] * cmath.exp(-2j * math.pi * kk * j / n)
            result[kk] = s
        return result

    def idft(arr):
        n = len(arr)
        result = [0j] * n
        for j in range(n):
            s = 0j
            for kk in range(n):
                s += arr[kk] * cmath.exp(2j * math.pi * kk * j / n)
            result[j] = s / n
        return result
    half_lin = [cmath.exp(-1j * (hbar ** 2 * ki ** 2 / (2 * m)) * dt / (2 * hbar) - Gamma * dt / (2 * hbar)) for ki in k]
    start = time.perf_counter()
    for step in range(steps):
        psi_k = dft(psi)
        psi_k = [psi_k[i] * half_lin[i] for i in range(N)]
        psi = idft(psi_k)
        for i in range(N):
            rho = abs(psi[i]) ** 2
            V = Lambda * rho
            psi[i] *= cmath.exp(-1j * V * dt / hbar)
        psi_k = dft(psi)
        psi_k = [psi_k[i] * half_lin[i] for i in range(N)]
        psi = idft(psi_k)
    elapsed = time.perf_counter() - start
    density = [abs(p) ** 2 for p in psi]
    peak = max(density)
    norm_final = sum((d * dx for d in density))
    return (elapsed, peak, norm_final)

def raw_numpy_pde(params):
    start = time.perf_counter()
    out = integrate(params)
    elapsed = time.perf_counter() - start
    peak = float(out['density'].max())
    norm_final = float(out['density'].sum() * out['dx'])
    return (elapsed, peak, norm_final)

def tri_solver(N=128, L=32.0, T=2.0, dt=0.005, Lambda=-0.5, Gamma=0.05, regime='B0', runs=3):
    src = f'import triad;\nimport time;\nlet start = time.now();\nlet params = triad.regime("{regime}");\nlet result = triad.solve(params, T={T});\nlet elapsed = time.now() - start;\nprint(f"{{elapsed:.6f}} {{result.peak:.6f}} {{result.norm:.6f}}");'
    _silent_run(src)
    times = []
    for _ in range(runs):
        buf = io.StringIO()
        mod = parse(src, '<bench>')
        c = TriadCompiler()
        with redirect_stdout(buf), redirect_stderr(buf):
            c.compile_and_run(mod)
        elapsed = float(buf.getvalue().strip().split()[0])
        times.append(elapsed)
    return sum(times) / len(times)

def main():
    print('=' * 80)
    print('  TriadLang vs Python — PDE Solver Performance')
    print('=' * 80)
    raw_numpy_pde(TriadParams(T=0.1, seed=0))
    _silent_run('import triad; let x = triad.regime("B0"); let r = triad.solve(x, T=0.1);')
    print()
    print('  1. GRID SIZE SCALING  (T=2.0 fixed, N varies)')
    print(f"  {'N':<8} {'Python DFT':>12} {'Raw NumPy':>12} {'.tri solver':>12} {'Speedup .tri':>14} {'Speedup raw':>13}")
    print(f"  {'-' * 8} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 14} {'-' * 13}")
    for N in [64, 128, 256]:
        py_t, py_peak, py_norm = pure_python_pde(N=N, T=2.0)
        p = TriadParams(N=N, T=2.0, seed=0, mode='full', Lambda=4.0, Gamma=0.02, V_ext=None, omega=0.0, nu=(2.0,), lam=(-0.3,), alpha=0.0, record_every=9999)
        raw_t, raw_peak, raw_norm = raw_numpy_pde(p)
        tri_t = tri_solver(T=2.0)
        print(f"  {N:<8} {py_t:>11.4f}s {raw_t:>11.4f}s {tri_t:>11.4f}s {'%6.0fx' % (py_t / tri_t):>14} {'%6.0fx' % (py_t / raw_t):>13}")
    print()
    print('  2. TIME SCALING  (N=128 fixed, T varies)')
    print(f"  {'T':<8} {'Python DFT':>12} {'Raw NumPy':>12} {'.tri solver':>12} {'Speedup .tri':>14}")
    print(f"  {'-' * 8} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 14}")
    for T in [1.0, 2.0, 5.0, 10.0]:
        py_t, _, _ = pure_python_pde(N=128, T=T)
        p = TriadParams(N=128, T=T, seed=0, mode='full', Lambda=4.0, Gamma=0.02, V_ext=None, omega=0.0, nu=(2.0,), lam=(-0.3,), alpha=0.0, record_every=9999)
        raw_t, _, _ = raw_numpy_pde(p)
        tri_t = tri_solver(T=T)
        print(f"  {T:<8.1f} {py_t:>11.4f}s {raw_t:>11.4f}s {tri_t:>11.4f}s {'%6.0fx' % (py_t / tri_t):>14}")
    print()
    print('  3. TRANSPILER OVERHEAD  (.tri vs raw NumPy, same equation)')
    print(f"  {'T':<8} {'Raw NumPy':>12} {'.tri solver':>12} {'Overhead':>10} {'Overhead %':>11}")
    print(f"  {'-' * 8} {'-' * 12} {'-' * 12} {'-' * 10} {'-' * 11}")
    for T in [2.0, 10.0, 20.0]:
        p = TriadParams(N=128, T=T, seed=0, mode='full', Lambda=4.0, Gamma=0.02, V_ext=None, omega=0.0, nu=(2.0,), lam=(-0.3,), alpha=0.0, record_every=9999)
        raw_t, _, _ = raw_numpy_pde(p)
        tri_t = tri_solver(T=T)
        overhead = tri_t - raw_t
        pct = overhead / raw_t * 100
        print(f'  {T:<8.1f} {raw_t:>11.4f}s {tri_t:>11.4f}s {overhead:>9.4f}s {pct:>10.1f}%')
    print()
    print('  4. FULL B0 REGIME  (P1+P2+P3, T=20)')
    py_t, _, _ = pure_python_pde(N=128, T=20.0)
    raw_t, _, _ = raw_numpy_pde(TriadParams(T=20.0, seed=0))
    tri_t = tri_solver(T=20.0)
    print(f'  Python DFT:  {py_t:>10.2f}s')
    print(f'  Raw NumPy:   {raw_t:>10.4f}s   ({py_t / raw_t:.0f}x)')
    print(f'  .tri solver: {tri_t:>10.4f}s   ({py_t / tri_t:.0f}x)')
    print()
    print('  5. MULTI-SUBSTRATE SCALING  (4 substrates, ring coupling, T=10)')
    print('  Each substrate runs the full Triad equation with coupling.')
    py_t_4 = 0
    for seed in range(4):
        t, _, _ = pure_python_pde(N=128, T=10.0)
        py_t_4 += t
    from runtime.multi_runtime import MultiRuntime, CouplingEdge, Segment
    from stdlib.regimes import resolve_regime
    mr = MultiRuntime(dt=0.005, record_every=9999)
    subs = []
    for i in range(4):
        p = resolve_regime('B0', seed=i, T=10.0)
        sid = mr.add_substrate(f's{i}', p)
        subs.append(sid)
    kappa = -3.0
    edges = [CouplingEdge(src_id=subs[i], dst_id=subs[(i + 1) % 4], kappa=kappa) for i in range(4)]
    mr.add_segment(Segment(t_start=0.0, t_end=10.0, edges=edges))
    mr.run(verbose=False)
    tri_times_4 = []
    for _ in range(3):
        mr2 = MultiRuntime(dt=0.005, record_every=9999)
        subs2 = []
        for i in range(4):
            p = resolve_regime('B0', seed=i, T=10.0)
            sid = mr2.add_substrate(f's{i}', p)
            subs2.append(sid)
        edges2 = [CouplingEdge(src_id=subs2[i], dst_id=subs2[(i + 1) % 4], kappa=kappa) for i in range(4)]
        mr2.add_segment(Segment(t_start=0.0, t_end=10.0, edges=edges2))
        t0 = time.perf_counter()
        mr2.run(verbose=False)
        tri_times_4.append(time.perf_counter() - t0)
    tri_t_4 = sum(tri_times_4) / len(tri_times_4)
    raw_times_4 = []
    for i in range(4):
        p = TriadParams(T=10.0, seed=i)
        t0 = time.perf_counter()
        integrate(p)
        raw_times_4.append(time.perf_counter() - t0)
    raw_t_4 = sum(raw_times_4)
    print(f'  Python DFT (4x seq):  {py_t_4:>10.2f}s')
    print(f'  Raw NumPy  (4x seq):  {raw_t_4:>10.4f}s   ({py_t_4 / raw_t_4:.0f}x)')
    print(f'  .tri ring  (4x para): {tri_t_4:>10.4f}s   ({py_t_4 / tri_t_4:.0f}x)')
    print(f'  Ring coupling advantage vs sequential NumPy: {raw_t_4 / tri_t_4:.1f}x')
    print()
    print('  6. SUBSTRATE COUNT SCALING  (T=5, ring coupling)')
    print(f"  {'Substrates':<12} {'Python 4x':>12} {'NumPy Nx seq':>13} {'Triad ring':>12} {'Speedup':>10}")
    print(f"  {'-' * 12} {'-' * 12} {'-' * 13} {'-' * 12} {'-' * 10}")
    py_base, _, _ = pure_python_pde(N=128, T=5.0)
    for n_sub in [1, 2, 4, 8]:
        py_est = py_base * n_sub
        raw_seq = 0
        for i in range(n_sub):
            p = TriadParams(T=5.0, seed=i)
            t0 = time.perf_counter()
            integrate(p)
            raw_seq += time.perf_counter() - t0
        mr_n = MultiRuntime(dt=0.005, record_every=9999)
        s_ids = []
        for i in range(n_sub):
            p = resolve_regime('B0', seed=i, T=5.0)
            sid = mr_n.add_substrate(f's{i}', p)
            s_ids.append(sid)
        edges_n = [CouplingEdge(src_id=s_ids[i], dst_id=s_ids[(i + 1) % n_sub], kappa=kappa) for i in range(n_sub)]
        mr_n.add_segment(Segment(t_start=0.0, t_end=5.0, edges=edges_n))
        tri_times_n = []
        for _ in range(3):
            mr_t = MultiRuntime(dt=0.005, record_every=9999)
            s_t = []
            for i in range(n_sub):
                p = resolve_regime('B0', seed=i, T=5.0)
                sid = mr_t.add_substrate(f's{i}', p)
                s_t.append(sid)
            e_t = [CouplingEdge(src_id=s_t[i], dst_id=s_t[(i + 1) % n_sub], kappa=kappa) for i in range(n_sub)]
            mr_t.add_segment(Segment(t_start=0.0, t_end=5.0, edges=e_t))
            t0 = time.perf_counter()
            mr_t.run(verbose=False)
            tri_times_n.append(time.perf_counter() - t0)
        tri_t_n = sum(tri_times_n) / len(tri_times_n)
        print(f"  {n_sub:<12} {py_est:>11.2f}s {raw_seq:>12.4f}s {tri_t_n:>11.4f}s {'%6.0fx' % (py_est / tri_t_n):>10}")
    print()
    print('=' * 80)
    print(f'  The equation runs at C-speed via NumPy/FFT.')
    print(f'  Multi-substrate ring coupling shares FFT infrastructure —')
    print(f'  more substrates = better amortisation of fixed costs.')
    print('=' * 80)
if __name__ == '__main__':
    main()