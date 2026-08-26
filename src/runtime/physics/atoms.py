from __future__ import annotations

from dataclasses import dataclass

from triad import ntri as np


@dataclass
class Atom:
    id: int
    position: tuple[float, float, float]
    element: str
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

def detect_atoms(psi, threshold: float = 0.5) -> list[Atom]:
    xp = np.array(psi)
    density = np.abs(psi) ** 2
    peaks = density > threshold
    atoms = []
    for idx in np.argwhere(peaks):
        atoms.append(Atom(
            id=len(atoms),
            position=(float(idx[0]), float(idx[1]), 0.0),
            element='H'
        ))
    return atoms

class AtomTracker:
    def __init__(self):
        self.atoms: list[Atom] = []

    def update(self, psi):
        self.atoms = detect_atoms(psi)

def inject_atom(psi, position: tuple[float, float], sigma: float = 1.0) -> np.ndarray:
    x = np.linspace(-10, 10, len(psi))
    y = np.linspace(-10, 10, len(psi))
    X, Y = np.meshgrid(x, y)
    gaussian = np.exp(-((X - position[0])**2 + (Y - position[1])**2) / (2 * sigma**2))
    return psi + gaussian * 0.1

def remove_atom(psi, position: tuple[float, float], radius: float = 2.0) -> np.ndarray:
    x = np.linspace(-10, 10, len(psi))
    y = np.linspace(-10, 10, len(psi))
    X, Y = np.meshgrid(x, y)
    mask = ((X - position[0])**2 + (Y - position[1])**2) > radius**2
    return psi * mask

def bcc_sites(L: float, N: int) -> list[tuple[float, float, float]]:
    sites = []
    a = L / 4
    for i in range(2):
        for j in range(2):
            for k in range(2):
                sites.append((i * a, j * a, k * a))
                sites.append((i * a + a/2, j * a + a/2, k * a + a/2))
    return sites

def read_lattice(file: str) -> list[tuple[float, float, float]]:
    return []

class CrystalMemory:
    def __init__(self, N: int = 32, L: float = 20.0, D: int = 3, lattice_const: float = 4.62, seed: int = 0):
        self.N = N
        self.L = L
        self.D = D
        self.lattice_const = lattice_const
        self.seed = seed
        self.sites: list[tuple[float, float, float]] = []
        self.occupancy: list[float] = []
        self._rng = np.random.default_rng(seed)

    def crystallize(self, T: float) -> list[dict]:
        n = int(self.N ** (1/3)) + 1
        self.sites = []
        self.occupancy = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    x = (i - n/2) * self.lattice_const
                    y = (j - n/2) * self.lattice_const
                    z = (k - n/2) * self.lattice_const
                    if x**2 + y**2 + z**2 < (self.L/2)**2:
                        self.sites.append((x, y, z))
                        self.occupancy.append(1.0 if self._rng.random() > 0.3 else 0.0)
        return [{'position': s, 'occupied': bool(o > 0.5)} for s, o in zip(self.sites, self.occupancy)]

    def read(self) -> list[dict]:
        return [{'position': s, 'occupied': bool(o > 0.5), 'phase': float(o)} for s, o in zip(self.sites, self.occupancy)]

    def write(self, index: int, phase: float = 1.0):
        if 0 <= index < len(self.occupancy):
            self.occupancy[index] = phase

    def settle(self, T: float):
        pass
