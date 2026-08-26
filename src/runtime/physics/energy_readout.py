from __future__ import annotations

from triad import ntri as np


def energy_readout(psi, dx: float = 0.1, hbar: float = 1.0, m: float = 1.0) -> float:
    N = len(psi)
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    psi_k = np.fft.fft(psi)
    kinetic = np.sum((hbar**2 * k**2 / (2*m)) * np.abs(psi_k)**2) * dx / N
    density = np.abs(psi)**2
    potential = np.sum(density) * dx
    return float(kinetic + potential)

def kinetic_energy(psi, dx: float = 0.1, hbar: float = 1.0, m: float = 1.0) -> float:
    N = len(psi)
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    psi_k = np.fft.fft(psi)
    kinetic = np.sum((hbar**2 * k**2 / (2*m)) * np.abs(psi_k)**2) * dx / N
    return float(kinetic)

def potential_energy(psi, dx: float = 0.1) -> float:
    density = np.abs(psi)**2
    potential = np.sum(density) * dx
    return float(potential)

def interaction_energy(psi, Lambda: float = -0.5, dx: float = 0.1) -> float:
    density = np.abs(psi)**2
    interaction = 0.5 * Lambda * np.sum(density**2) * dx
    return float(interaction)

def total_energy(psi, dx: float = 0.1, hbar: float = 1.0, m: float = 1.0, Lambda: float = -0.5) -> dict:
    kinetic = kinetic_energy(psi, dx, hbar, m)
    potential = potential_energy(psi, dx)
    interaction = interaction_energy(psi, Lambda, dx)
    return {
        'kinetic': kinetic,
        'potential': potential,
        'interaction': interaction,
        'total': kinetic + potential + interaction
    }

def energy_spectrum(psi, dx: float = 0.1, hbar: float = 1.0, m: float = 1.0) -> np.ndarray:
    N = len(psi)
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    psi_k = np.fft.fft(psi)
    E_k = (hbar**2 * k**2 / (2*m)) * np.abs(psi_k)**2
    return E_k
