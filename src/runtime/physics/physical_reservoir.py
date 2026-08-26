from __future__ import annotations

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
