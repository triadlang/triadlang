import sys, os, time, io
from contextlib import redirect_stdout, redirect_stderr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.solver import integrate, TriadParams
from runtime.multi_runtime import MultiRuntime, CouplingEdge, Segment
from runtime.backend import cuda_available
from stdlib.regimes import resolve_regime
if cuda_available():
    import cupy as cp

def bench(label, fn, runs=3):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)

def pure_python_pde(N=128, T=2.0):
    import math, cmath
    L, dx, dt = (32.0, 32.0 / N, 0.005)
    hbar, m, Lambda, Gamma = (1.0, 1.0, 4.0, 0.02)
    steps = int(T / dt)
    x = [-L / 2 + i * dx for i in range(N)]
    k = [2 * math.pi * (i if i < N // 2 else i - N) / L for i in range(N)]
    psi = [complex(math.exp(-xi ** 2 / 8.0), 0.0) for xi in x]
    norm = math.sqrt(sum((abs(p) ** 2 * dx for p in psi)))
    psi = [p / norm for p in psi]

    def dft(a):
        n = len(a)
        r = [0j] * n
        for kk in range(n):
            s = 0j
            for j in range(n):
                s += a[j] * cmath.exp(-2j * math.pi * kk * j / n)
            r[kk] = s
        return r

    def idft(a):
        n = len(a)
        r = [0j] * n
        for j in range(n):
            s = 0j
            for kk in range(n):
                s += a[kk] * cmath.exp(2j * math.pi * kk * j / n)
            r[j] = s / n
        return r
    hl = [cmath.exp(-1j * (hbar ** 2 * ki ** 2 / (2 * m)) * dt / (2 * hbar) - Gamma * dt / (2 * hbar)) for ki in k]
    t0 = time.perf_counter()
    for _ in range(steps):
        pk = dft(psi)
        pk = [pk[i] * hl[i] for i in range(N)]
        psi = idft(pk)
        for i in range(N):
            rho = abs(psi[i]) ** 2
            psi[i] *= cmath.exp(-1j * Lambda * rho * dt / hbar)
        pk = dft(psi)
        pk = [pk[i] * hl[i] for i in range(N)]
        psi = idft(pk)
    return time.perf_counter() - t0

def run_ring(backend, T=5.0, n_sub=4):
    mr = MultiRuntime(dt=0.005, record_every=9999)
    subs = []
    for i in range(n_sub):
        p = resolve_regime('B0', seed=i, T=T)
        p = TriadParams(**{**p.__dict__, 'backend': backend})
        sub = mr.add_substrate(f's{i}', p)
        subs.append(sub.id)
    edges = [CouplingEdge(src_id=subs[i], dst_id=subs[(i + 1) % n_sub], kappa=-3.0) for i in range(n_sub)]
    mr.add_segment(Segment(t_start=0.0, t_end=T, edges=edges))
    mr.run(verbose=False)

def tri_inprocess(T=2.0):
    from frontend.parser_universal import parse
    from runtime.compiler_runtime import TriadCompiler
    src = f'import triad; import time;\nlet start = time.now();\nlet params = triad.regime("B0");\nlet result = triad.solve(params, T={T});\nlet elapsed = time.now() - start;\nprint(f"{{elapsed:.6f}}");'
    buf = io.StringIO()
    mod = parse(src, '<bench>')
    c = TriadCompiler()
    with redirect_stdout(buf), redirect_stderr(buf):
        c.compile_and_run(mod)
    return float(buf.getvalue().strip().split()[0])

def main():
    has_gpu = cuda_available()
    print('=' * 80)
    print(f"  TriadLang Benchmark Suite  |  GPU: {('RTX 4060 (CuPy)' if has_gpu else 'N/A')}")
    print('=' * 80)
    print('\n  1. 1D PDE SOLVER — Grid Scaling (T=2.0)')
    print(f"  {'N':<8} {'Python DFT':>12} {'Triad CPU':>12} {'.tri lang':>12} {'vs Python':>10}")
    print(f"  {'-' * 8} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 10}")
    for N in [64, 128, 256]:
        py_t = bench(f'Python N={N}', lambda N=N: pure_python_pde(N=N, T=2.0), runs=1)
        cpu_t = bench(f'CPU N={N}', lambda N=N: integrate(TriadParams(N=N, T=2.0, seed=0, backend='numpy', record_every=9999)))
        tri_t = bench(f'.tri N={N}', lambda N=N: tri_inprocess(T=2.0), runs=1)
        print(f'  {N:<8} {py_t:>11.3f}s {cpu_t:>11.4f}s {tri_t:>11.4f}s {py_t / tri_t:>9.0f}x')
    if has_gpu:
        print('\n  2. 3D PDE SOLVER — GPU vs CPU (T=2.0)')
        print(f"  {'N':<8} {'Elements':>12} {'CPU':>12} {'GPU':>12} {'GPU speedup':>12}")
        print(f"  {'-' * 8} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12}")
        for N in [32, 48, 64]:
            cpu_t = bench(f'CPU 3D N={N}', lambda N=N: integrate(TriadParams(N=N, T=2.0, seed=0, D=3, backend='numpy', record_every=9999)))
            integrate(TriadParams(N=N, T=2.0, seed=0, D=3, backend='cuda', record_every=9999))
            cp.cuda.Stream.null.synchronize()
            gpu_t = bench(f'GPU 3D N={N}', lambda N=N: (integrate(TriadParams(N=N, T=2.0, seed=0, D=3, backend='cuda', record_every=9999)), cp.cuda.Stream.null.synchronize()))
            print(f'  {N:<8} {N ** 3:>11,}  {cpu_t:>11.4f}s {gpu_t:>11.4f}s {cpu_t / gpu_t:>11.2f}x')
    print('\n  3. MULTI-SUBSTRATE RING COUPLING (4 substrates, T=5.0)')
    cpu_ring = bench('CPU ring', lambda: run_ring('numpy'))
    print(f'  CPU 4-substrate ring: {cpu_ring:.4f}s')
    if has_gpu:
        run_ring('cuda')
        cp.cuda.Stream.null.synchronize()
        gpu_ring = bench('GPU ring', lambda: (run_ring('cuda'), cp.cuda.Stream.null.synchronize()))
        print(f'  GPU 4-substrate ring: {gpu_ring:.4f}s  ({cpu_ring / gpu_ring:.2f}x)')
    print('\n  4. SUBSTRATE COUNT SCALING (T=5.0, ring coupling)')
    print(f"  {'Substrates':<12} {'CPU time':>12} {'CPU per sub':>12}")
    print(f"  {'-' * 12} {'-' * 12} {'-' * 12}")
    for n_sub in [1, 2, 4, 8]:
        t = bench(f'{n_sub} subs', lambda n=n_sub: run_ring('numpy', n_sub=n))
        print(f'  {n_sub:<12} {t:>11.4f}s {t / n_sub:>11.4f}s')
    print('\n' + '=' * 80)
    print('  Architecture:')
    print('    - Pure Python: O(N²) DFT loops — exponentially slow')
    print('    - TriadLang:   O(N log N) NumPy/FFT — C-speed via the Triad equation')
    print("    - GPU (CuPy):  CUDA FFTs — available for 3D/large grids via backend='cuda'")
    print(f"    - Auto-detect: backend='auto' uses CPU (optimal for 1D/2D N<512)")
    print("    - Force GPU:   backend='cuda' or TRIADLANG_BACKEND=cuda")
    print('=' * 80)
if __name__ == '__main__':
    main()