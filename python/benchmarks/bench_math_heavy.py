import math

def integrate(a, b, n):
    h = (b - a) / n
    total = 0.0
    for i in range(n + 1):
        x = a + i * h
        y = math.sin(x) * math.exp(-0.1 * x) + math.cos(x * 0.5)
        if i == 0 or i == n:
            total += y / 2.0
        else:
            total += y
    return total * h
result = integrate(0.0, 100.0, 50000)
print(f'integral = {result}')