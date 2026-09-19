from __future__ import annotations

import math
import random
from dataclasses import dataclass

from triad import ntri as np


@dataclass
class ReservoirConfig:
    n_units: int
    coupling: float
    spectral_radius: float
    leaking_rate: float = 1.0

class PhysicalReservoir:
    def __init__(self, config: ReservoirConfig):
        self.config = config
        self.state = np.zeros(config.n_units)
        self.weights = np.random.randn(config.n_units, config.n_units)
        scale = config.spectral_radius / max(np.abs(np.linalg.eigvals(self.weights)))
        self.weights *= scale

    def update(self, input_signal: np.ndarray) -> np.ndarray:
        self.state = np.tanh(
            self.config.coupling * (self.weights @ self.state) + input_signal
        )
        return self.state

    def reset(self):
        self.state = np.zeros(self.config.n_units)

def reservoir_capacity(states: np.ndarray, inputs: np.ndarray) -> float:
    if states.shape[0] < inputs.shape[0]:
        return 0.0
    return float(np.linalg.det(states.T @ states))

def reservoir_memory(reservoir: PhysicalReservoir, input_signal: np.ndarray) -> float:
    reservoir.reset()
    outputs = []
    for inp in input_signal:
        out = reservoir.update(inp)
        outputs.append(out)
    outputs = np.array(outputs)
    return float(np.mean(outputs))

def reservoir_dynamics(psi, dt: float = 0.005) -> dict:
    density = np.abs(psi) ** 2
    return {
        'mean': float(np.mean(density)),
        'variance': float(np.var(density)),
        'autocorrelation': float(np.corrcoef(density[:-1], density[1:])[0, 1])
    }

class FDTNoiseSource:
    def __init__(self, mode: str = 'white_fdt', color_rho: float = 0.7, seed: int = 0):
        if mode not in ('white_fdt', 'reservoir_colored'):
            raise ValueError(f'unknown FDTNoiseSource mode: {mode!r}')
        self.mode = mode
        self.color_rho = color_rho
        self.seed = seed

    def sample(self, n: int, variance: float) -> np.ndarray:
        rng = random.Random(self.seed)
        if self.mode == 'reservoir_colored':
            rho = self.color_rho
            se = math.sqrt(variance * (1.0 - rho * rho))
            x = 0.0
            for _ in range(200):
                x = rho * x + se * rng.gauss(0.0, 1.0)
            out = []
            for _ in range(n):
                x = rho * x + se * rng.gauss(0.0, 1.0)
                out.append(x)
            return np.array(out)
        sd = math.sqrt(variance)
        return np.array([sd * rng.gauss(0.0, 1.0) for _ in range(n)])

def verify_fdt(source: FDTNoiseSource, params, T: float = 5.0) -> dict:
    gamma = float(getattr(params, 'Gamma', 0.05))
    kT = float(getattr(params, 'kT', 1.0))
    dt = float(getattr(params, 'dt', 0.005))
    n = max(int(T / dt), 16)
    target = 2.0 * gamma * kT / dt
    x = source.sample(n, target)
    ratio = float(np.var(x)) / target
    lag1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    ok = abs(ratio - 1.0) < 0.25 and abs(lag1) < 0.15
    return {'var_ratio': ratio, 'lag1_autocorr': lag1, 'fdt_consistent': bool(ok)}

def chaos_threshold_sweep(gains, size: int = 150, leak: float = 0.9,
                          lyap_len: int = 1500, mc_len: int = 1500,
                          seed: int = 0, max_delay: int = 10) -> list:
    rng = random.Random(seed)
    W0 = np.array([[rng.gauss(0.0, 1.0) for _ in range(size)] for _ in range(size)])
    rho0 = max(float(v) for v in np.abs(np.linalg.eigvals(W0)).tolist())
    W0 = W0 * (1.0 / rho0)
    Win = np.array([[rng.gauss(0.0, 0.5)] for _ in range(size)])
    steps = max(1, max(lyap_len, mc_len))
    rows = []
    for g in gains:
        gW = W0 * float(g)
        x = np.zeros((size, 1))
        d = np.array([[rng.gauss(0.0, 1.0)] for _ in range(size)])
        dn = math.sqrt(float(np.sum(d * d)))
        d = d * (1e-8 / dn)
        for _ in range(200):
            u = rng.gauss(0.0, 1.0)
            x = (1.0 - leak) * x + leak * np.tanh(gW @ x + Win * u)
        acc = 0.0
        states = []
        inputs = []
        for _ in range(steps):
            u = rng.gauss(0.0, 1.0)
            tn = np.tanh(gW @ x + Win * u)
            x = (1.0 - leak) * x + leak * tn
            d = (1.0 - leak) * d + leak * ((1.0 - tn * tn) * (gW @ d))
            sep = math.sqrt(float(np.sum(d * d)))
            acc += math.log(sep / 1e-8)
            d = d * (1e-8 / sep)
            states.append([float(r[0]) for r in x.tolist()] + [1.0])
            inputs.append(u)
        lyap = acc / steps
        mc = 0.0
        p = size + 1
        if len(states) > max_delay and max_delay >= 1:
            stride = max(1, len(states) // 600)
            idx = list(range(max_delay, len(states)))[::stride]
            Xr = [states[i] for i in idx]
            X = np.array(Xr)
            XtX = X.T @ X
            scale = float(np.sum(np.abs(XtX))) / (p * p)
            A = XtX + np.eye(p) * (1e-6 * (scale + 1e-30))
            for dl in range(1, max_delay + 1):
                ys = [inputs[i - dl] for i in idx]
                my = sum(ys) / len(ys)
                sst = sum((y - my) ** 2 for y in ys)
                if sst == 0.0:
                    continue
                xty = [0.0] * p
                for r, y in zip(Xr, ys):
                    for j in range(p):
                        xty[j] += r[j] * y
                w = np.linalg.solve(A, np.array(xty)).tolist()
                sse = 0.0
                for r, y in zip(Xr, ys):
                    yh = sum(r[j] * w[j] for j in range(p))
                    sse += (y - yh) ** 2
                r2 = 1.0 - sse / sst
                if r2 > 0.0:
                    mc += r2
        rows.append({'gain': float(g), 'lyapunov': lyap, 'memory_capacity': mc})
    return rows
