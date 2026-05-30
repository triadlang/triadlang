import sys, os, time, math, cmath
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.solver import integrate, TriadParams
from runtime.multi_runtime import MultiRuntime, CouplingEdge, Segment
from runtime.backend import cuda_available
from stdlib.regimes import resolve_regime
import numpy as np
if cuda_available():
    import cupy as cp

def bench(fn, runs=3):
    for _ in range(runs + 1):
        t0 = time.perf_counter()
        fn()
        t = time.perf_counter() - t0
    return t

def pure_python_1d(N, T, dt=0.005):
    L, dx = (32.0, 32.0 / N)
    hbar, m, Lam, Gam = (1.0, 1.0, -0.5, 0.05)
    steps = int(T / dt)
    x = [-L / 2 + i * dx for i in range(N)]
    k = [2 * math.pi * (i if i < N // 2 else i - N) / L for i in range(N)]
    psi = [complex(math.exp(-xi ** 2 / 8.0), 0.0) for xi in x]
    norm = math.sqrt(sum((abs(p) ** 2 * dx for p in psi)))
    psi = [p / norm for p in psi]
    V_ext = [0.5 * m * 0.05 ** 2 * xi ** 2 for xi in x]

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
    hl = [cmath.exp(-1j * (hbar ** 2 * ki ** 2 / (2 * m)) * dt / (2 * hbar) - Gam * dt / (2 * hbar)) for ki in k]
    nu = [2.0, 0.5, 0.1]
    lam = [-0.3, -0.2, -0.1]
    y = [[0.0] * N for _ in nu]
    f_fdt = 0.002
    noise_amp = math.sqrt(f_fdt * dt / (dx * hbar ** 2)) if f_fdt > 0 else 0.0
    import random
    rng = random.Random(0)
    t0 = time.perf_counter()
    for step in range(steps):
        pk = dft(psi)
        pk = [pk[i] * hl[i] for i in range(N)]
        psi = idft(pk)
        rho = [abs(p) ** 2 for p in psi]
        for j in range(len(nu)):
            decay = math.exp(-nu[j] * dt * 0.5)
            y[j] = [decay * y[j][i] + (1 - decay) * rho[i] for i in range(N)]
        V_mem = [sum((lam[j] * y[j][i] for j in range(len(nu)))) for i in range(N)]
        V_tot = [V_ext[i] + Lam * rho[i] + V_mem[i] for i in range(N)]
        psi = [psi[i] * cmath.exp(-1j * V_tot[i] * dt / hbar) for i in range(N)]
        rho = [abs(p) ** 2 for p in psi]
        for j in range(len(nu)):
            decay = math.exp(-nu[j] * dt * 0.5)
            y[j] = [decay * y[j][i] + (1 - decay) * rho[i] for i in range(N)]
        if noise_amp > 0:
            psi = [psi[i] + noise_amp * (rng.gauss(0, 1) + 1j * rng.gauss(0, 1)) / math.sqrt(2) for i in range(N)]
        pk = dft(psi)
        pk = [pk[i] * hl[i] for i in range(N)]
        psi = idft(pk)
    return time.perf_counter() - t0

def pure_python_coupled(N, T, n_sub=4, dt=0.005):
    L, dx = (32.0, 32.0 / N)
    hbar, m = (1.0, 1.0)
    steps = int(T / dt)
    kappa = -3.0
    configs = [(-0.5, 0.05, 0.002, [2.0, 0.5, 0.1], [-0.3, -0.2, -0.1])] * n_sub

    def make_sub(cfg, seed):
        Lam, Gam, f_fdt, nu, lam = cfg
        x = [-L / 2 + i * dx for i in range(N)]
        k = [2 * math.pi * (i if i < N // 2 else i - N) / L for i in range(N)]
        psi = [complex(math.exp(-xi ** 2 / 8.0), 0.0) for xi in x]
        norm = math.sqrt(sum((abs(p) ** 2 * dx for p in psi)))
        psi = [p / norm for p in psi]
        hl = [cmath.exp(-1j * (hbar ** 2 * ki ** 2 / (2 * m)) * dt / (2 * hbar) - Gam * dt / (2 * hbar)) for ki in k]
        V_ext = [0.5 * m * 0.05 ** 2 * xi ** 2 for xi in x]
        y = [[0.0] * N for _ in nu]
        na = math.sqrt(f_fdt * dt / (dx * hbar ** 2)) if f_fdt > 0 else 0.0
        import random
        rng = random.Random(seed)
        return [psi, hl, V_ext, y, nu, lam, Lam, Gam, na, rng, x]

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
    subs = [make_sub(configs[i], i) for i in range(n_sub)]
    t0 = time.perf_counter()
    for step in range(steps):
        densities = [[abs(s[0][i]) ** 2 for i in range(N)] for s in subs]
        for i, s in enumerate(subs):
            psi, hl, V_ext, y, nu, lam, Lam, Gam, na, rng, x = s
            prev = (i - 1) % n_sub
            V_couple = [kappa * densities[prev][j] * dx for j in range(N)]
            pk = dft(psi)
            pk = [pk[m] * hl[m] for m in range(N)]
            psi = idft(pk)
            rho = [abs(p) ** 2 for p in psi]
            for j in range(len(nu)):
                decay = math.exp(-nu[j] * dt * 0.5)
                y[j] = [decay * y[j][m] + (1 - decay) * rho[m] for m in range(N)]
            V_mem = [sum((lam[jj] * y[jj][m] for jj in range(len(nu)))) for m in range(N)]
            V_tot = [V_ext[m] + Lam * rho[m] + V_mem[m] + V_couple[m] for m in range(N)]
            psi = [psi[m] * cmath.exp(-1j * V_tot[m] * dt / hbar) for m in range(N)]
            rho = [abs(p) ** 2 for p in psi]
            for j in range(len(nu)):
                decay = math.exp(-nu[j] * dt * 0.5)
                y[j] = [decay * y[j][m] + (1 - decay) * rho[m] for m in range(N)]
            if na > 0:
                psi = [psi[m] + na * (rng.gauss(0, 1) + 1j * rng.gauss(0, 1)) / math.sqrt(2) for m in range(N)]
            pk = dft(psi)
            pk = [pk[m] * hl[m] for m in range(N)]
            psi = idft(pk)
            s[0] = psi
    return time.perf_counter() - t0

def triad_gpu(N, T, D=1):
    p = TriadParams(N=N, T=T, seed=0, backend='cuda', D=D, record_every=9999)
    integrate(p)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    integrate(p)
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter() - t0

def triad_cpu(N, T, D=1):
    p = TriadParams(N=N, T=T, seed=0, backend='numpy', D=D, record_every=9999)
    return bench(lambda: integrate(p))

def triad_gpu_ring(N, T, n_sub=4):
    mr = MultiRuntime(dt=0.005, record_every=9999)
    subs = []
    for i in range(n_sub):
        p = resolve_regime('B0', seed=i, T=T)
        p = TriadParams(**{**p.__dict__, 'backend': 'cuda', 'N': N})
        sub = mr.add_substrate(f's{i}', p)
        subs.append(sub.id)
    edges = [CouplingEdge(src_id=subs[i], dst_id=subs[(i + 1) % n_sub], kappa=-3.0) for i in range(n_sub)]
    mr.add_segment(Segment(t_start=0.0, t_end=T, edges=edges))
    mr.run(verbose=False)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    mr2 = MultiRuntime(dt=0.005, record_every=9999)
    subs2 = []
    for i in range(n_sub):
        p = resolve_regime('B0', seed=i, T=T)
        p = TriadParams(**{**p.__dict__, 'backend': 'cuda', 'N': N})
        sub = mr2.add_substrate(f's{i}', p)
        subs2.append(sub.id)
    edges2 = [CouplingEdge(src_id=subs2[i], dst_id=subs2[(i + 1) % n_sub], kappa=-3.0) for i in range(n_sub)]
    mr2.add_segment(Segment(t_start=0.0, t_end=T, edges=edges2))
    mr2.run(verbose=False)
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter() - t0

def triad_cpu_ring(N, T, n_sub=4):
    mr = MultiRuntime(dt=0.005, record_every=9999)
    subs = []
    for i in range(n_sub):
        p = resolve_regime('B0', seed=i, T=T)
        p = TriadParams(**{**p.__dict__, 'backend': 'numpy', 'N': N})
        sub = mr.add_substrate(f's{i}', p)
        subs.append(sub.id)
    edges = [CouplingEdge(src_id=subs[i], dst_id=subs[(i + 1) % n_sub], kappa=-3.0) for i in range(n_sub)]
    mr.add_segment(Segment(t_start=0.0, t_end=T, edges=edges))
    return bench(lambda: mr.run(verbose=False))

def main():
    has_gpu = cuda_available()
    print('=' * 78)
    print('  TriadLang GPU vs Python')
    print(f"  GPU: {('RTX 4060 (CuPy 14.1)' if has_gpu else 'N/A')}")
    print('=' * 78)
    print('\n  1. SINGLE SUBSTRATE 1D — full Triad equation (P1+P2+P3)')
    print(f"  {'N':<8} {'Python':>10} {'Triad CPU':>10} {'Triad GPU':>10} {'CPU vs Py':>10} {'GPU vs Py':>10}")
    print(f"  {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    T = 2.0
    for N in [64, 128]:
        py_t = pure_python_1d(N, T)
        cpu_t = triad_cpu(N, T)
        if has_gpu:
            gpu_t = triad_gpu(N, T)
            print(f'  {N:<8} {py_t:>9.2f}s {cpu_t:>9.4f}s {gpu_t:>9.4f}s {py_t / cpu_t:>9.0f}x {py_t / gpu_t:>9.0f}x')
        else:
            print(f"  {N:<8} {py_t:>9.2f}s {cpu_t:>9.4f}s {'N/A':>10} {py_t / cpu_t:>9.0f}x {'N/A':>10}")
    print('\n  2. FOUR SUBSTRATES — ring coupling (T=5.0)')
    print(f"  {'N':<8} {'Python':>10} {'Triad CPU':>10} {'Triad GPU':>10} {'CPU vs Py':>10} {'GPU vs Py':>10}")
    print(f"  {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for N in [64, 128]:
        py_t = pure_python_coupled(N, T=5.0, n_sub=4)
        cpu_t = triad_cpu_ring(N, T=5.0)
        if has_gpu:
            gpu_t = triad_gpu_ring(N, T=5.0)
            print(f'  {N:<8} {py_t:>9.2f}s {cpu_t:>9.4f}s {gpu_t:>9.4f}s {py_t / cpu_t:>9.0f}x {py_t / gpu_t:>9.0f}x')
        else:
            print(f"  {N:<8} {py_t:>9.2f}s {cpu_t:>9.4f}s {'N/A':>10} {py_t / cpu_t:>9.0f}x {'N/A':>10}")
    print('\n  3. SUBSTRATE COUNT — ring coupling (N=128, T=5.0)')
    print(f"  {'Subs':<6} {'Python':>12} {'Triad CPU':>12} {'Triad GPU':>12} {'CPU vs Py':>10} {'GPU vs Py':>10}")
    print(f"  {'-' * 6} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 10} {'-' * 10}")
    for n_sub in [1, 2, 4]:
        py_t = pure_python_coupled(128, T=5.0, n_sub=n_sub)
        cpu_t = triad_cpu_ring(128, T=5.0, n_sub=n_sub)
        if has_gpu:
            gpu_t = triad_gpu_ring(128, T=5.0, n_sub=n_sub)
            print(f'  {n_sub:<6} {py_t:>11.2f}s {cpu_t:>11.4f}s {gpu_t:>11.4f}s {py_t / cpu_t:>9.0f}x {py_t / gpu_t:>9.0f}x')
        else:
            print(f"  {n_sub:<6} {py_t:>11.2f}s {cpu_t:>11.4f}s {'N/A':>12} {py_t / cpu_t:>9.0f}x {'N/A':>10}")
    print('\n' + '=' * 78)
if __name__ == '__main__':
    main()