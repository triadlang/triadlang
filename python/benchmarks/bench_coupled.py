import time, math, cmath
N = 128
L = 32.0
dx = L / N
dt = 0.005
T = 5.0
steps = int(T / dt)
hbar = 1.0
m = 1.0
kappa = -2.5

def make_substrate(Lambda, Gamma, seed):
    x = [-L / 2 + i * dx for i in range(N)]
    k = [2 * math.pi * (i if i < N // 2 else i - N) / L for i in range(N)]
    psi = [complex(math.exp(-xi ** 2 / 8.0), 0.0) for xi in x]
    norm = math.sqrt(sum((abs(p) ** 2 * dx for p in psi)))
    psi = [p / norm for p in psi]
    half_lin = [cmath.exp(-1j * (hbar ** 2 * ki ** 2 / (2 * m)) * dt / (2 * hbar) - Gamma * dt / (2 * hbar)) for ki in k]
    return (psi, half_lin, k)

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
substrates = [make_substrate(4.0, 0.02, 0), make_substrate(4.0, 0.02, 1), make_substrate(-0.5, 0.05, 2), make_substrate(-0.5, 0.05, 3)]
start = time.perf_counter()
for step in range(steps):
    densities = []
    for psi, half_lin, k in substrates:
        d = [abs(p) ** 2 for p in psi]
        densities.append(d)
    for i, (psi, half_lin, k) in enumerate(substrates):
        j_prev = (i - 1) % 4
        coupling = [kappa * densities[j_prev][idx] * dx for idx in range(N)]
        psi_k = dft(psi)
        psi_k = [psi_k[m] * half_lin[m] for m in range(N)]
        psi = idft(psi_k)
        for idx in range(N):
            rho = abs(psi[idx]) ** 2
            V = substrates[i][1][0]
            psi[idx] *= cmath.exp(-1j * (4.0 * rho + coupling[idx]) * dt / hbar)
        psi_k = dft(psi)
        psi_k = [psi_k[m] * half_lin[m] for m in range(N)]
        psi = idft(psi_k)
        substrates[i] = (psi, half_lin, k)
elapsed = time.perf_counter() - start
print(f'Pure Python 4-substrate ring coupling: {elapsed:.4f}s')