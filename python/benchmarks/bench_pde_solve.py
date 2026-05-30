import time, math, cmath
N = 128
L = 32.0
dx = L / N
dt = 0.005
T = 2.0
steps = int(T / dt)
hbar = 1.0
m = 1.0
Lambda = 4.0
Gamma = 0.02
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
print(f'Pure Python DFT-loop solver: {elapsed:.4f}s')
print(f'  steps={steps}, N={N}, peak={peak:.6f}, norm={norm_final:.6f}')