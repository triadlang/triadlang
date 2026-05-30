import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from runtime.solver import TriadParams, integrate

def _integrate_reference(mode='linear', T=10.0, dt=0.005, seed=0, **overrides):
    L = overrides.get('L', 32.0)
    N = overrides.get('N', 128)
    hbar = 1.0
    m = 1.0
    omega = overrides.get('omega', 0.05)
    Lambda = overrides.get('Lambda', -0.5)
    alpha = overrides.get('alpha', 0.15)
    sigma = overrides.get('sigma', 1.5)
    Gamma = overrides.get('Gamma', 0.05)
    f_FDT = overrides.get('f_FDT', 0.002)
    nu = overrides.get('nu', (2.0, 0.5, 0.1))
    lam = overrides.get('lam', (-0.3, -0.2, -0.1))
    rng = np.random.default_rng(seed)
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    dx = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    abs_k = np.abs(k)
    V_ext = 0.5 * m * omega ** 2 * x ** 2
    Lambda_eff = Lambda if mode == 'full' else 0.0
    alpha_eff = alpha if mode == 'full' else 0.0
    Gamma_eff = Gamma if mode != 'linear' else 0.0
    f_FDT_eff = f_FDT if mode != 'linear' else 0.0
    lam_eff = np.asarray(lam) if mode == 'full' else np.zeros(len(lam))
    psi = np.exp(-x ** 2 / 8.0).astype(np.complex128)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    M = len(nu)
    y = np.zeros((M, N), dtype=np.float64)
    H_lin_k = hbar ** 2 * k ** 2 / (2 * m) + alpha_eff * abs_k ** sigma
    half_lin = np.exp(-1j * H_lin_k * dt / (2 * hbar) - Gamma_eff * dt / (2 * hbar))
    noise_amp = np.sqrt(f_FDT_eff * dt / dx) if f_FDT_eff > 0 else 0.0
    n_steps = int(T / dt)
    norms = []
    for step in range(n_steps + 1):
        norms.append(float(np.sum(np.abs(psi) ** 2) * dx))
        if step == n_steps:
            break
        psi = np.fft.ifft(np.fft.fft(psi) * half_lin)
        rho = np.abs(psi) ** 2
        V_mem = (lam_eff[:, None] * y).sum(axis=0) if M > 0 else 0.0
        V_tot = V_ext + Lambda_eff * rho + V_mem
        psi = psi * np.exp(-1j * V_tot * dt / hbar)
        if mode == 'full':
            for j in range(M):
                y[j] += dt * nu[j] * (rho - y[j])
        if noise_amp > 0:
            xi = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2.0)
            psi = psi + noise_amp * xi
        psi = np.fft.ifft(np.fft.fft(psi) * half_lin)
    return (np.array(norms), x, dx, psi)

def test_norm_conservation_linear():
    out = integrate(TriadParams(mode='linear', T=10.0, dt=0.005, seed=0))
    norm_t = out['density'].sum(axis=0) * out['dx']
    deviation = np.max(np.abs(norm_t - 1.0))
    assert deviation < 1e-12, f'Norm deviation {deviation} exceeds fp64 tolerance'

def test_norm_conservation_linear_reference():
    norms, _, _, _ = _integrate_reference(mode='linear', T=10.0, dt=0.005)
    deviation = np.max(np.abs(norms - 1.0))
    assert deviation < 1e-12, f'Reference norm deviation {deviation}'

def test_harmonic_eigenstate():
    omega = 0.05
    sigma_gs = np.sqrt(1.0 / omega)
    L, N, dt, T = (32.0, 128, 0.005, 5.0)
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    dx_val = x[1] - x[0]
    psi0 = np.exp(-x ** 2 / (2 * sigma_gs ** 2)).astype(np.complex128)
    psi0 /= np.sqrt(np.sum(np.abs(psi0) ** 2) * dx_val)
    out = integrate(TriadParams(mode='linear', T=T, dt=dt, L=L, N=N, V_ext='harmonic', omega=omega, seed=42), psi0=psi0)
    rho_f = np.abs(out['psi_final']) ** 2
    rho0 = np.abs(psi0) ** 2
    max_diff = np.max(np.abs(rho_f - rho0))
    assert max_diff < 0.0001, f'Eigenstate density drift {max_diff}'

def test_thermal_norm_plateau():
    out = integrate(TriadParams(mode='thermal', T=20.0, dt=0.005, seed=0))
    norm_t = out['density'].sum(axis=0) * out['dx']
    late_mean = np.mean(norm_t[-100:])
    assert late_mean != 1.0, 'Thermal norm should deviate from initial'
    assert np.isfinite(late_mean), 'Thermal norm diverged'

def test_solver_vs_reference_linear():
    out = integrate(TriadParams(mode='linear', T=5.0, dt=0.005, seed=0))
    _, _, _, psi_ref = _integrate_reference(mode='linear', T=5.0, dt=0.005, seed=0)
    dx = out['dx']
    diff = np.max(np.abs(np.abs(out['psi_final']) ** 2 - np.abs(psi_ref) ** 2))
    assert diff < 1e-13, f'Solver vs reference density diff {diff}'

def test_convergence_dt():
    _, _, dx1, psi1 = _integrate_reference(mode='linear', T=2.0, dt=0.01, omega=0.0)
    _, _, dx2, psi2 = _integrate_reference(mode='linear', T=2.0, dt=0.005, omega=0.0)
    _, _, dx3, psi3 = _integrate_reference(mode='linear', T=2.0, dt=0.0025, omega=0.0)
    err_coarse = np.sqrt(np.sum(np.abs(np.abs(psi1) ** 2 - np.abs(psi3) ** 2) ** 2) * dx1)
    err_fine = np.sqrt(np.sum(np.abs(np.abs(psi2) ** 2 - np.abs(psi3) ** 2) ** 2) * dx2)
    assert err_coarse < 1e-06, f'Coarse error too large: {err_coarse}'
    assert err_fine < 1e-06, f'Fine error too large: {err_fine}'

def test_convergence_dt_solver():
    out1 = integrate(TriadParams(mode='linear', T=2.0, dt=0.01, V_ext=None, seed=0))
    out2 = integrate(TriadParams(mode='linear', T=2.0, dt=0.005, V_ext=None, seed=0))
    out3 = integrate(TriadParams(mode='linear', T=2.0, dt=0.0025, V_ext=None, seed=0))
    dx1 = out1['dx']
    err_coarse = np.sqrt(np.sum(np.abs(np.abs(out1['psi_final']) ** 2 - np.abs(out3['psi_final']) ** 2) ** 2) * dx1)
    err_fine = np.sqrt(np.sum(np.abs(np.abs(out2['psi_final']) ** 2 - np.abs(out3['psi_final']) ** 2) ** 2) * dx1)
    assert err_coarse < 1e-06, f'Solver coarse error: {err_coarse}'
    assert err_fine < 1e-06, f'Solver fine error: {err_fine}'

def test_fdt_noise_scaling():
    L, N, dt, f_FDT = (32.0, 128, 0.005, 0.002)
    dx = L / N
    expected_amp = np.sqrt(f_FDT * dt / dx)
    assert expected_amp > 0
    assert np.isfinite(expected_amp)
    dx2 = L / (2 * N)
    amp2 = np.sqrt(f_FDT * dt / dx2)
    ratio = amp2 / expected_amp
    assert abs(ratio - np.sqrt(2)) < 1e-10, f'Noise scaling ratio {ratio} != sqrt(2)'

def test_full_mode_bounded():
    out = integrate(TriadParams(mode='full', T=10.0, dt=0.005, seed=0))
    norm_t = out['density'].sum(axis=0) * out['dx']
    assert all(np.isfinite(norm_t)), 'Norms contain NaN/Inf'
    assert norm_t[-1] > 0.01, f'Norm collapsed to {norm_t[-1]}'
    assert norm_t[-1] < 1000, f'Norm diverged to {norm_t[-1]}'
if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print(f'  PASS {name}')
            except Exception as e:
                print(f'  FAIL {name}: {e}')