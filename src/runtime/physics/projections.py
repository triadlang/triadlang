from __future__ import annotations

from triad import ntri as np


def project_to_subspace(psi, subspace_basis: np.ndarray) -> np.ndarray:
    if len(subspace_basis) == 0:
        return np.zeros_like(psi)
    proj = np.zeros_like(psi, dtype=complex)
    for basis_vec in subspace_basis:
        coeff = np.vdot(psi, basis_vec)
        proj += coeff * basis_vec
    return proj

def orthogonal_projection(psi, subspace_basis: np.ndarray) -> np.ndarray:
    proj = project_to_subspace(psi, subspace_basis)
    return psi - proj

def basis_projection(psi, n_states: int) -> list[complex]:
    N = len(psi)
    coefficients = []
    for i in range(min(n_states, N)):
        basis = np.zeros(N, dtype=complex)
        basis[i] = 1.0
        coeff = np.vdot(psi, basis)
        coefficients.append(coeff)
    return coefficients

def momentum_projection(psi, k_magnitude: float, L: float = 32.0) -> np.ndarray:
    N = len(psi)
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    k = 2.0 * np.pi * k_magnitude
    plane_wave = np.exp(1j * k * x)
    plane_wave /= np.linalg.norm(plane_wave)
    return project_to_subspace(psi, [plane_wave])

def energy_projection(psi, energy: float, hbar: float = 1.0, m: float = 1.0) -> np.ndarray:
    return psi

def density_projection(psi, target_density: np.ndarray) -> np.ndarray:
    current_density = np.abs(psi) ** 2
    overlap = np.vdot(current_density, target_density)
    if overlap == 0:
        return psi
    return psi * np.sqrt(target_density / current_density + 1e-10)
