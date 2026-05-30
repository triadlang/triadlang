import time
import numpy as np
from runtime.fast_solver import fast_integrate, TriadParams

def stress_solver():
    nu = [2.0, 0.5, 0.1]
    lam = [-0.3, -0.2, -0.1]
    total = 0.0
    for i in range(10):
        p = TriadParams(N=128, L=32.0, dt=0.005, T=1.0, hbar=1.0, m=1.0, omega=0.05, Lambda=-0.5, alpha=0.15, sigma=1.5, Gamma=0.05, f_FDT=0.002, nu=nu, lam=lam, mode='full', seed=i, V_ext='harmonic')
        r = fast_integrate(p)
        total += np.sum(r['density_final']) * r['dx']
    return total

def stress_language():
    total = 0
    for i in range(10000):
        x = i * i + i
        if x % 7 == 0:
            total = total + x
    return total

def stress_closure():

    def make_acc(init):

        def add(n):
            nonlocal init
            init = init + n
            return init
        return add
    a = make_acc(0)
    total = 0
    for i in range(1000):
        total = a(i)
    return total

def stress_class():

    class Vec2:

        def __init__(self, x, y):
            self.x = x
            self.y = y

        def add_1(self):
            self.x += 1
            self.y += 1

        def mag(self):
            return (self.x ** 2 + self.y ** 2) ** 0.5
    v = Vec2(0, 0)
    for i in range(1000):
        v.add_1()
    return v.mag()

def stress_generator():

    def gen_squares(n):
        i = 0
        while i < n:
            yield (i * i)
            i = i + 1
    total = 0
    for x in gen_squares(1000):
        total = total + x
    return total
print('=== PYTHON stress ===')
t0 = time.perf_counter()
s = stress_solver()
t1 = time.perf_counter()
print(f'solver: {s:.4f} ({t1 - t0:.4f}s)')
t0 = time.perf_counter()
s = stress_language()
t1 = time.perf_counter()
print(f'language: {s} ({t1 - t0:.6f}s)')
t0 = time.perf_counter()
s = stress_closure()
t1 = time.perf_counter()
print(f'closure: {s} ({t1 - t0:.6f}s)')
t0 = time.perf_counter()
s = stress_class()
t1 = time.perf_counter()
print(f'class: {s:.2f} ({t1 - t0:.6f}s)')
t0 = time.perf_counter()
s = stress_generator()
t1 = time.perf_counter()
print(f'generator: {s} ({t1 - t0:.6f}s)')