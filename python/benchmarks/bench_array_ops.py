import time, math
N = 1000000
start = time.perf_counter()
x = [-16.0 + 32.0 * i / N for i in range(N)]
y = [math.sin(xi) for xi in x]
z = [math.exp(-xi ** 2 / 8.0) for xi in x]
w = [y[i] * z[i] + 0.5 * x[i] for i in range(N)]
total = sum(w)
peak = max((abs(wi) for wi in w))
elapsed = time.perf_counter() - start
print(f'Pure Python array ops: {elapsed:.4f}s')
print(f'  N={N}, sum={total:.6f}, peak={peak:.6f}')