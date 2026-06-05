"""TriadLang scale benchmark - pushes N to the limit.

Usage:
    python src/benchmarks/bench_scale.py          # run all
    python src/benchmarks/bench_scale.py --1d     # 1D only
    python src/benchmarks/bench_scale.py --2d     # 2D only
    python src/benchmarks/bench_scale.py --3d     # 3D only
    python src/benchmarks/bench_scale.py --max-n 2**30   # set max N for 1D
"""
from __future__ import annotations
import sys, os, time, gc, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from runtime.core.solver import TriadParams, integrate, integrate_2d, integrate_3d
from runtime.physics.observables import crystallinity

COMMON = dict(
    T=2.0, seed=0, mode='full', Lambda=4.0, Gamma=0.02,
    V_ext=None, omega=0.0, nu=(2.0,),
    lam=(-0.3,), alpha=0.0, sigma=1.5,
    fdt_couple=True, record_every=9999,
)

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        'docs', 'benchmark.log')

def log(msg: str):
    print(msg, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')

def bench_1d(max_n: int):
    log('\n=== 1D SCALE BENCHMARK (T=2.0, full P1+P2+P3) ===')
    log(f'{"N":>12} {"CPU(s)":>12} {"GPU(s)":>12} {"CPU C":>8} {"GPU C":>8} {"speedup":>10} {"err"}')
    log('-' * 90)

    integrate(TriadParams(N=256, T=0.1, backend='auto', seed=0))
    integrate(TriadParams(N=256, T=0.1, backend='cpu', seed=0))

    for exp in range(7, 30):
        n = 2 ** exp
        if n > max_n:
            break
        gc.collect()

        cpu_t, cpu_c = None, None
        try:
            p = TriadParams(N=n, backend='cpu', **COMMON)
            t0 = time.perf_counter()
            r = integrate(p)
            cpu_t = time.perf_counter() - t0
            cpu_c = crystallinity(r['psi_final'], r['dx'])
        except Exception as e:
            cpu_t = f'ERR'
            cpu_c = '-'

        gc.collect()
        gpu_t, gpu_c = None, None
        try:
            p = TriadParams(N=n, backend='auto', **COMMON)
            t0 = time.perf_counter()
            r = integrate(p)
            gpu_t = time.perf_counter() - t0
            gpu_c = crystallinity(r['psi_final'], r['dx'])
        except Exception as e:
            gpu_t = f'ERR'
            gpu_c = '-'

        speedup = ''
        if isinstance(cpu_t, float) and isinstance(gpu_t, float) and gpu_t > 0:
            speedup = f'{cpu_t/gpu_t:.1f}x'
        elif gpu_t == 'ERR':
            speedup = 'cpu wins'
        elif cpu_t == 'ERR':
            speedup = 'gpu wins'

        cpu_s = f'{cpu_t:.4f}' if isinstance(cpu_t, float) else str(cpu_t)
        gpu_s = f'{gpu_t:.4f}' if isinstance(gpu_t, float) else str(gpu_t)
        c_s = f'{cpu_c:.4f}' if isinstance(cpu_c, float) else str(cpu_c)
        g_s = f'{gpu_c:.4f}' if isinstance(gpu_c, float) else str(gpu_c)

        log(f'{n:>12} {cpu_s:>12} {gpu_s:>12} {c_s:>8} {g_s:>8} {speedup:>10}')

def bench_2d(max_n: int):
    log('\n=== 2D SCALE BENCHMARK (T=5.0, full P1+P2+P3) ===')
    log(f'{"N":>8} {"N^2":>12} {"GPU(s)":>12} {"cryst":>8} {"err"}')
    log('-' * 80)

    for exp in range(4, 16):
        n = 2 ** exp
        total = n * n
        if total > max_n:
            break
        gc.collect()
        try:
            p = TriadParams(N=n, backend='auto', D=2, **COMMON)
            t0 = time.perf_counter()
            r = integrate_2d(p)
            elapsed = time.perf_counter() - t0
            c = crystallinity(r['psi_final'].flatten(), r['dx'])
            log(f'{n:>8} {total:>12} {elapsed:>12.4f} {c:>8.4f}')
        except Exception as e:
            log(f'{n:>8} {total:>12} {"ERR":>12} {"":>8} {type(e).__name__}: {e}')
            break

def bench_3d(max_n: int):
    log('\n=== 3D SCALE BENCHMARK (T=5.0, full P1+P2+P3) ===')
    log(f'{"N":>6} {"N^3":>12} {"GPU(s)":>12} {"cryst":>8} {"err"}')
    log('-' * 80)

    for exp in range(3, 12):
        n = 2 ** exp
        total = n * n * n
        if total > max_n:
            break
        gc.collect()
        try:
            p = TriadParams(N=n, backend='auto', D=3, **COMMON)
            t0 = time.perf_counter()
            r = integrate_3d(p)
            elapsed = time.perf_counter() - t0
            c = crystallinity(r['psi_final'].flatten(), r['dx'])
            log(f'{n:>6} {total:>12} {elapsed:>12.4f} {c:>8.4f}')
        except Exception as e:
            log(f'{n:>6} {total:>12} {"ERR":>12} {"":>8} {type(e).__name__}: {e}')
            break

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--1d', dest='d1', action='store_true', help='1D only')
    ap.add_argument('--2d', dest='d2', action='store_true', help='2D only')
    ap.add_argument('--3d', dest='d3', action='store_true', help='3D only')
    ap.add_argument('--max-n', type=int, default=2**28, help='max total elements')
    args = ap.parse_args()

    with open(LOG_PATH, 'w') as f:
        f.write('')

    log('TriadLang Scale Benchmark')
    log(f'date: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    log(f'max total elements: {args.max_n}')

    do_1d = not (args.d2 or args.d3) or args.d1
    do_2d = args.d2
    do_3d = args.d3
    if not args.d1 and not args.d2 and not args.d3:
        do_1d = do_2d = do_3d = True

    if do_1d:
        bench_1d(args.max_n)
    if do_2d:
        bench_2d(args.max_n)
    if do_3d:
        bench_3d(args.max_n)

    log('\ndone.')

if __name__ == '__main__':
    main()
