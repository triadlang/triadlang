# Triad Equation — Mathematical and Algorithmic Reference

**Version 1.0.** A self-contained technical specification of the equation, the numerical scheme used to integrate it, the implementation, the ablation modes used as controls, the validation protocol, the observables, and the parameter regimes. Reproducible end-to-end from this document alone.

---

## How to read this document

This is a reference specification, not an exposition. Sections are independent and cross-reference each other by number. The intended reader is someone implementing the equation, validating an existing implementation, or interpreting outputs.

Sections 1–4 are the minimum to reproduce. Sections 5–13 cover extensions, observables, and operational details. Appendices contain conventions and a working code skeleton.

There is no claim made in this document beyond what is stated. The equation is defined, its terms are sourced, the algorithm is specified, the controls are listed, and the diagnostics are enumerated. What the outputs mean for any specific scientific question is outside the scope of this reference.

---

## 0. Notation and conventions

### 0.1 Variables

| Symbol | Type | Meaning |
|---|---|---|
| $\Psi(t, \mathbf{x})$ | complex scalar field | primary state |
| $\rho(t, \mathbf{x}) = \|\Psi\|^2$ | real non-negative | density |
| $y_j(t, \mathbf{x})$ | real, $j = 1, \dots, M$ | auxiliary memory fields |
| $V_{\text{ext}}(\mathbf{x})$ | real, time-independent | external potential |
| $V_{\text{mem}}(t, \mathbf{x})$ | real | memory potential, derived from $y_j$ |
| $\eta(t, \mathbf{x})$ | complex stochastic | FDT-locked noise |

### 0.2 Parameters (canonical names)

| Symbol | Range / type | Role |
|---|---|---|
| $\hbar$ | $> 0$ | natural units, default $1$ |
| $m$ | $> 0$ | mass scale, default $1$ |
| $\Lambda$ | real, sign matters | cubic self-interaction strength |
| $\alpha$ | $\geq 0$ | fractional Laplacian strength |
| $\sigma$ | $\in (0, 2]$ | fractional exponent ($\sigma = 2$ recovers standard Laplacian) |
| $\Gamma$ | $\geq 0$ | dissipation rate |
| $\nu_j$ | $> 0$, $j = 1, \dots, M$ | memory relaxation rates (inverse timescales) |
| $\lambda_j$ | real, sign matters, $j = 1, \dots, M$ | memory coupling strengths |
| $f_{\text{FDT}} = 2\gamma_0 k_B T$ | $\geq 0$ | FDT noise amplitude |
| $M$ | positive integer | number of memory modes |

### 0.3 Sign conventions

The signs of $\Lambda$ and $\lambda_j$ define the dynamical regime. The conventions used throughout this document:

- $\Lambda < 0$: attractive self-interaction (focusing). $\Lambda > 0$: repulsive (defocusing).
- $\lambda_j < 0$: attractive memory wells (memory enhances $|\Psi|^2$ where it has already been high). $\lambda_j > 0$: repulsive memory (memory suppresses returning to previously occupied regions).
- The sign of $\lambda_j$ relative to $\Lambda$ determines whether memory acts with or against the cubic. **Anti-collapse regime**: $\Lambda < 0$ with $\lambda_j > 0$ (memory opposes self-trapping). **Structure-forming regime**: $\Lambda < 0$ with $\lambda_j < 0$ in an external well (memory deepens carved channels).

The sign choice is constitutive of the regime under study; defaults in §A.3 specify both options.

### 0.4 Spatial dimension

The equation is written below in $D$-dimensional form ($D \in \{1, 2, 3\}$). The 1D form is the canonical reference for validation. 2D and 3D extensions are tensor-product factorizations of the 1D scheme; see §5.

### 0.5 FFT convention

The discrete Fourier transform uses the NumPy/SciPy convention:

$$\hat{f}_k = \sum_{n=0}^{N-1} f_n \, e^{-2\pi i k n / N}, \qquad f_n = \frac{1}{N} \sum_{k=0}^{N-1} \hat{f}_k \, e^{+2\pi i k n / N}.$$

Wavenumbers are constructed as `k = 2*pi*np.fft.fftfreq(N, d=dx)`, so $k \in [-\pi/dx, \pi/dx)$ with the standard NumPy ordering (positive frequencies first, then negative).

### 0.6 Domain and boundary conditions

Periodic boundary conditions throughout. Domain: $[-L/2, L/2)^D$, with $N$ grid points per axis, $dx = L/N$. Periodic BCs are imposed by the FFT-based propagator; no additional boundary treatment is needed.

For non-periodic physics (e.g. localised states in a harmonic well), $L$ must be chosen large enough that $|\Psi|^2$ at the box boundary remains negligible throughout the run. The harmonic potential confines naturally; this is checked by monitoring $|\Psi(\pm L/2, t)|^2 / \max |\Psi|^2 < 10^{-6}$ for the duration of the run.

---

## 1. The equation

### 1.1 Primary equation

$$i \hbar \, \partial_t \Psi = \left[ -\frac{\hbar^2}{2m} \nabla^2 + V_{\text{ext}}(\mathbf{x}) + \Lambda |\Psi|^2 + V_{\text{mem}}(t, \mathbf{x}) + \alpha (-\Delta)^{\sigma/2} - i\Gamma \right] \Psi + \eta(t, \mathbf{x}). \tag{1}$$

### 1.2 Auxiliary memory subsystem

$$V_{\text{mem}}(t, \mathbf{x}) = \sum_{j=1}^M \lambda_j \, y_j(t, \mathbf{x}), \tag{2a}$$

$$\partial_t y_j(t, \mathbf{x}) = \nu_j \bigl( |\Psi(t, \mathbf{x})|^2 - y_j(t, \mathbf{x}) \bigr), \quad j = 1, \dots, M. \tag{2b}$$

Each $y_j$ is a low-pass filter of the local density $|\Psi|^2$ with time constant $\tau_j = 1/\nu_j$. The collection $\{(\nu_j, \lambda_j)\}_{j=1}^M$ is a Prony series approximation to a memory kernel; see §2.3.

### 1.3 FDT-locked noise

$$\langle \eta(t, \mathbf{x}) \, \eta^*(t', \mathbf{x}') \rangle = 2 \gamma_0 k_B T \, \delta(t - t') \, \delta^{(D)}(\mathbf{x} - \mathbf{x}'), \tag{3a}$$

$$\langle \eta(t, \mathbf{x}) \, \eta(t', \mathbf{x}') \rangle = 0. \tag{3b}$$

The composite quantity $f_{\text{FDT}} = 2\gamma_0 k_B T$ is exposed as a single parameter. The "lock" is the requirement that this amplitude is not independently tunable from $\Gamma$ when both are derived from the same bath; in practice this means $f_{\text{FDT}}$ and $\Gamma$ are set together when modelling a specific physical bath, and $f_{\text{FDT}}$ acts as the temperature dial when $\Gamma$ is fixed.

### 1.4 Discretisation of the noise

On the grid, the continuum $\delta^{(D)}(\mathbf{x} - \mathbf{x}')$ becomes $\delta_{ij} / dx^D$. The per-step stochastic increment to $\Psi$ at grid point $i$ is

$$\Psi_i \leftarrow \Psi_i + \sqrt{\frac{f_{\text{FDT}} \, dt}{dx^D}} \cdot \frac{\xi_i + i \, \xi'_i}{\sqrt{2}}, \tag{4}$$

with $\xi_i, \xi'_i \sim \mathcal{N}(0, 1)$ independent. The factor $\sqrt{f_{\text{FDT}} dt / dx^D}$ implements equation (3a) in the discretised regime. Failure to scale by $dx^{-D/2}$ produces a noise floor that depends on the grid resolution, which is a frequent implementation error.

### 1.5 Initial conditions

Equation (1) is supplemented by initial conditions

$$\Psi(0, \mathbf{x}) = \Psi_0(\mathbf{x}), \qquad y_j(0, \mathbf{x}) = 0 \;\; \forall j. \tag{5}$$

$\Psi_0$ is normalised to $\int |\Psi_0|^2 \, d^D x = 1$. The convention $y_j(0) = 0$ corresponds to "no prior history at $t = 0$"; alternative pre-loaded conditions ($y_j(0) = |\Psi_0|^2$ or steady-state-of-some-other-input) are admissible but must be declared.

---

## 2. Term-by-term breakdown

Each term in equation (1) has a specific origin in established mathematical physics. This section documents the origin and the operational role.

### 2.1 The kinetic term $-\hbar^2/(2m) \nabla^2$

Standard Schrödinger kinetic operator. Diagonal in Fourier space: $-\hbar^2 / (2m) \, k^2$. Drives dispersion. In the absence of all other terms, equation (1) reduces to the free Schrödinger equation, and any localised initial state disperses as $|\Psi(t)|^2 \sim t^{-D/2}$ at large $t$.

### 2.2 The cubic term $\Lambda |\Psi|^2$

Local self-interaction (Gross–Pitaevskii / Kerr nonlinearity). Origin: short-range mean-field interaction in the second-quantised theory of bosons (Gross 1961, Pitaevskii 1961). In the Mori–Zwanzig framework (§2.3), this term emerges as the Markovian limit of a short-time, short-range component of the projected interaction.

Sign determines the regime:
- $\Lambda > 0$: defocusing. Globally well-posed in any dimension.
- $\Lambda < 0$: focusing. In 1D, gives bright solitons. In 2D, $L^2$-critical: blowup for sufficiently concentrated initial data. In 3D, $L^2$-supercritical: blowup for any sufficiently concentrated initial data above a finite norm threshold (Sulem & Sulem 1999).

The focusing regime in 3D is the canonical setting for the anti-collapse demonstrations (see §10.3).

### 2.3 The memory potential $V_{\text{mem}} = \sum_j \lambda_j y_j$

Originates from the projection-operator formalism of statistical mechanics (Mori 1965, Zwanzig 1960). The general Mori–Zwanzig equation for a projected observable $A(t)$ takes the form

$$\dot{A}(t) = i \Omega A(t) - \int_0^t K(t - s) A(s) \, ds + f(t),$$

where $K(t)$ is a memory kernel and $f(t)$ is a fluctuating force orthogonal to the projected subspace. For completely monotonic kernels $K(t)$ (the class admissible under fluctuation–dissipation constraints), $K(t)$ can be approximated to arbitrary accuracy by a Prony series:

$$K(t) \approx \sum_{j=1}^M \lambda_j \nu_j \, e^{-\nu_j t}.$$

This Prony decomposition is exactly Markovian-embedded by introducing auxiliary fields $y_j$ satisfying equation (2b). The convolution against $K$ becomes $V_{\text{mem}} = \sum_j \lambda_j y_j$ as in (2a). The equivalence between continuous-bath memory and the discrete-mode Markovian embedding has been proven non-perturbatively (Tamascelli, Smirne, Huelga & Plenio 2018, Phys. Rev. Lett. 120:030402).

Multi-scale separation in $\{\nu_j\}$ (typically several decades) implements a hierarchy of memory timescales: fast modes ($\nu_j \gg 1/dt$) track the instantaneous density, slow modes ($\nu_j \ll 1/T_{\text{run}}$) accumulate history over the whole simulation.

### 2.4 The fractional Laplacian $\alpha (-\Delta)^{\sigma/2}$

Non-local dispersion operator. Defined in Fourier space as

$$\widehat{(-\Delta)^{\sigma/2} \Psi}(k) = |k|^\sigma \hat{\Psi}(k). \tag{6}$$

Origin: Lévy-flight path integral over the kinetic term (Laskin 2000, Phys. Rev. E 62:3135). When the bath dynamics integrated out via Mori–Zwanzig have heavy-tailed (non-Gaussian) spatial statistics, the resulting effective kinetic operator is $(-\Delta)^{\sigma/2}$ with $\sigma < 2$.

$\sigma = 2$ recovers the standard Laplacian $-\Delta$; the fractional operator adds an additional dispersive channel. Set $\alpha = 0$ to disable.

### 2.5 The dissipation $-i\Gamma$

Linear non-Hermitian decay. Adding $-i\Gamma$ to the Hamiltonian gives $\partial_t |\Psi|^2 \supset -2\Gamma/\hbar \cdot |\Psi|^2$, i.e. uniform exponential decay of the norm in the absence of noise.

In equation (1), this decay is balanced by the stochastic forcing $\eta$ via the fluctuation–dissipation relation (3a); the equilibrium norm is finite and set by $f_{\text{FDT}}$ and $\Gamma$ together. With $\Gamma > 0$ and $\eta = 0$ the field decays to zero; with $\eta \neq 0$ and $\Gamma = 0$ the norm diverges; with both at their FDT-locked values the system reaches a finite stationary distribution.

### 2.6 The external potential $V_{\text{ext}}(\mathbf{x})$

Time-independent confining or driving potential. Standard choices:

- **Harmonic well**: $V_{\text{ext}}(\mathbf{x}) = \frac{1}{2} m \omega^2 |\mathbf{x}|^2$. Confines the field; required for the structure-forming regime (§10.3) and for any non-trivial steady state under FDT noise.
- **None**: $V_{\text{ext}} = 0$. Required for the anti-collapse 3D demonstration (§10.4), since the confinement would itself prevent dispersion.
- **Gaussian bump (input)**: $V_{\text{ext}}(\mathbf{x}, t) = A \exp(-|\mathbf{x} - \mathbf{x}_0|^2 / 2w^2)$, used as a driven input in multi-substrate protocols (§11).

### 2.7 The stochastic forcing $\eta$

Complex white Gaussian noise with the FDT correlator (3a). Generated as in (4). The noise is the inhomogeneous term in the stochastic PDE; it is not absorbed into the Hamiltonian.

The "FDT lock" is the constraint $\langle \eta \eta^* \rangle \propto \Gamma$. In implementations where $\Gamma$ and $f_{\text{FDT}}$ are independent dials, the lock is the physical condition that they be set together when both originate from the same bath. The numerical machinery does not enforce the lock automatically; it is a modelling choice on the user.

---

## 3. Numerical scheme

### 3.1 Operator splitting

Equation (1) is integrated via Strang split-step. Over one time step $dt$, the propagator is decomposed as

$$U(dt) \approx U_{\text{lin}}(dt/2) \cdot U_{\text{pot}}(dt) \cdot U_{\text{lin}}(dt/2), \tag{7}$$

where $U_{\text{lin}}$ contains the linear, diagonal-in-Fourier operators (kinetic, fractional, dissipation), and $U_{\text{pot}}$ contains the diagonal-in-real-space operators (external potential, cubic, memory). The auxiliary memory fields and the stochastic forcing are updated inside the real-space step.

Strang splitting is second-order accurate in $dt$ for the deterministic part. For the stochastic part, the order is $1/2$ in $dt$ for strong convergence, $1$ for weak convergence; refining $dt$ improves both.

### 3.2 Linear half-step (Fourier space)

In Fourier space, the linear operators are diagonal:

$$H_{\text{lin}}(k) = \frac{\hbar^2 k^2}{2m} + \alpha |k|^\sigma. \tag{8}$$

The dissipation contributes $-i\Gamma$ as a uniform multiplier in real space (since it is local) but is conventionally absorbed into the Fourier-space propagator for cleanliness. The half-step propagator is

$$\tilde{U}_{\text{lin}}(dt/2; k) = \exp\!\left( -\frac{i \, H_{\text{lin}}(k) \, dt}{2\hbar} - \frac{\Gamma \, dt}{2\hbar} \right). \tag{9}$$

The propagator is precomputed once at the start of the run (it depends only on $k$, $dt$, and fixed parameters). The half-step is then

$$\Psi \leftarrow \text{IFFT}\bigl[ \tilde{U}_{\text{lin}}(dt/2) \cdot \text{FFT}[\Psi] \bigr]. \tag{10}$$

### 3.3 Real-space full step (potential + memory)

The total real-space potential at step $n$:

$$V_{\text{tot}}^n(\mathbf{x}) = V_{\text{ext}}(\mathbf{x}) + \Lambda \, |\Psi^n(\mathbf{x})|^2 + \sum_{j=1}^M \lambda_j \, y_j^n(\mathbf{x}). \tag{11}$$

The propagator (diagonal in real space):

$$\Psi \leftarrow \Psi \cdot \exp\!\left( -\frac{i \, V_{\text{tot}}^n \, dt}{\hbar} \right). \tag{12}$$

### 3.4 Memory field update

For each $j$, the auxiliary field follows

$$\partial_t y_j = \nu_j (\rho - y_j),$$

with $\rho = |\Psi|^2$. Two update schemes are admissible:

**Forward Euler** (used in the reference solver, fine when $\nu_j \, dt \ll 1$):

$$y_j^{n+1} = y_j^n + dt \cdot \nu_j \cdot (\rho^n - y_j^n). \tag{13a}$$

**Exact Ornstein–Uhlenbeck step** (used when $\nu_j \, dt$ is not small):

$$y_j^{n+1} = e^{-\nu_j dt} \, y_j^n + (1 - e^{-\nu_j dt}) \, \rho^n. \tag{13b}$$

The OU form is unconditionally stable; the Euler form requires $\nu_j \, dt < 2$ for stability and $\nu_j \, dt < 0.1$ for accuracy. Default: use Euler when $\max_j \nu_j \, dt < 0.05$, otherwise OU.

### 3.5 Stochastic forcing step

The noise increment is applied once per step, as specified by (4):

$$\xi_i, \xi'_i \sim \mathcal{N}(0, 1) \text{ iid}, \tag{14a}$$

$$\Psi_i^{n+1} \leftarrow \Psi_i^{n+1} + \sqrt{\frac{f_{\text{FDT}} \, dt}{dx^D}} \cdot \frac{\xi_i + i \xi'_i}{\sqrt{2}}. \tag{14b}$$

The noise is applied **before** the second Fourier half-step; this ordering is consistent with the Strang composition and avoids spurious correlations.

### 3.6 Full step composition

The complete update at step $n$:

```
1.  ψ ← FFT(ψ);  ψ ← Ũ_lin(dt/2) · ψ;  ψ ← IFFT(ψ)
2.  ρ ← |ψ|²
3.  V_tot ← V_ext + Λρ + Σⱼ λⱼ yⱼ
4.  ψ ← exp(-i V_tot dt / ℏ) · ψ
5.  yⱼ ← update(yⱼ, ρ, νⱼ, dt)   [for all j; eq. (13a) or (13b)]
6.  ψ ← ψ + (noise increment)   [eq. (14)]
7.  ψ ← FFT(ψ);  ψ ← Ũ_lin(dt/2) · ψ;  ψ ← IFFT(ψ)
```

Steps 1 and 7 are the two halves of Strang. Steps 2–6 are the real-space block. The auxiliary memory $y_j$ uses the density $\rho$ computed in step 2 (i.e. the post-Fourier-half-step density, pre-potential-step density).

### 3.7 Stability conditions

For numerical stability:

- **CFL-like for kinetic term**: $\hbar dt / (2m \, dx^2) < \pi$ (Nyquist limit on the kinetic propagator phase). In natural units with $\hbar = m = 1$ and $dx = L/N$, this requires $dt < 2\pi (L/N)^2$.

- **Nonlinear stability**: $|\Lambda| \cdot \max_t \max_x |\Psi|^2 \cdot dt < \pi$. In the focusing regime this can become binding near peaks; halve $dt$ if peak amplitude exceeds expected range.

- **Memory stability** (Euler): $\max_j \nu_j \cdot dt < 2$, with $< 0.1$ recommended for accuracy.

- **Fractional Laplacian**: $\alpha \cdot (k_{\max})^\sigma \cdot dt / \hbar < \pi$, where $k_{\max} = \pi/dx$.

In practice, for the canonical 1D config ($L = 32$, $N = 128$, $dx = 0.25$), $dt = 0.005$ satisfies all bounds with margin. For 3D ($N = 128$ per axis, $L = 20$), $dt = 0.0025$ is the standard choice.

### 3.8 Auto-halving heuristic

For strong nonlinearity ($|\Lambda| \geq 4$), the reference solver halves $dt$ automatically: if `dt > 0.0025`, set `dt = 0.0025`. This is a safety rule for the focusing regime where the time step would otherwise be marginal. The user can override by setting `auto_halve_dt = False`.

---

## 4. Implementation: 1D reference

### 4.1 Variables and types

| Variable | Type | Shape | Description |
|---|---|---|---|
| `psi` | complex128 | `(N,)` | wavefunction |
| `y` | float64 | `(M, N)` | memory fields, $M$ modes |
| `x` | float64 | `(N,)` | spatial grid |
| `k` | float64 | `(N,)` | wavenumber grid |
| `V_ext` | float64 | `(N,)` | external potential (precomputed) |
| `half_lin` | complex128 | `(N,)` | precomputed half-step propagator (9) |
| `nu`, `lam` | float64 | `(M,)` | memory parameters |

### 4.2 Initialisation

```python
import numpy as np

# grid
x  = np.linspace(-L/2, L/2, N, endpoint=False)
dx = x[1] - x[0]
k  = 2 * np.pi * np.fft.fftfreq(N, d=dx)

# external potential (harmonic well example)
V_ext = 0.5 * m * omega**2 * x**2

# precomputed linear half-step propagator (eq. 9)
H_lin_k = hbar * k**2 / (2*m) + alpha * np.abs(k)**sigma
half_lin = np.exp(-1j * H_lin_k * dt / (2*hbar)
                  - Gamma * dt / (2*hbar))

# initial state (Gaussian, normalised)
psi = np.exp(-x**2 / 8.0).astype(np.complex128)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)

# memory fields, initially zero
M = len(nu)
y = np.zeros((M, N), dtype=np.float64)

# noise amplitude per step
noise_amp = np.sqrt(f_FDT * dt / dx) if f_FDT > 0 else 0.0

# RNG
rng = np.random.default_rng(seed)
```

### 4.3 Main loop

```python
n_steps = int(T_total / dt)

for step in range(n_steps):
    # 1. linear half-step in Fourier space
    psi_k = np.fft.fft(psi)
    psi_k *= half_lin
    psi   = np.fft.ifft(psi_k)
    
    # 2. real-space potential + memory
    rho   = np.abs(psi)**2
    V_mem = (lam[:, None] * y).sum(axis=0) if M > 0 else 0.0
    V_tot = V_ext + Lambda * rho + V_mem
    psi   = psi * np.exp(-1j * V_tot * dt / hbar)
    
    # 3. memory field update (Euler form; switch to OU if needed)
    if M > 0:
        for j in range(M):
            y[j] += dt * nu[j] * (rho - y[j])
    
    # 4. stochastic kick
    if noise_amp > 0:
        xi = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2.0)
        psi = psi + noise_amp * xi
    
    # 5. linear half-step
    psi_k = np.fft.fft(psi)
    psi_k *= half_lin
    psi   = np.fft.ifft(psi_k)
    
    # (record at the requested frequency; see §8)
```

This is the exact algorithm of the reference solver. It is correct under the specifications of §3.

### 4.4 Memory layout

For 1D with $N = 128$, $M = 3$: $\Psi$ is 2 KB (`complex128`), $y$ is 3 KB (`float64`), propagator 2 KB, $V_{\text{ext}}$ 1 KB. Total working set well under any cache. Per-step cost is dominated by two FFTs at $\mathcal{O}(N \log N)$.

### 4.5 Performance

For 1D / $N = 128$ / $T = 20$ / $dt = 0.005$ (4000 steps): on a single CPU core, ~1 s wall time with NumPy FFTs. On GPU (CuPy), ~0.1 s wall time but the dispatch overhead dominates; use 1D on CPU.

---

## 5. Implementation: 2D and 3D extensions

The scheme is dimension-agnostic. Every operator is either diagonal in Fourier space (kinetic, fractional, dissipation) or diagonal in real space (external potential, cubic, memory, noise). FFT is replaced by ND-FFT.

### 5.1 Grid construction (3D example)

```python
x  = np.linspace(-L/2, L/2, N, endpoint=False)
dx = x[1] - x[0]
X, Y, Z = np.meshgrid(x, x, x, indexing="ij")  # shape (N, N, N)

kvec  = 2 * np.pi * np.fft.fftfreq(N, d=dx)
kx, ky, kz = np.meshgrid(kvec, kvec, kvec, indexing="ij")
k2    = kx**2 + ky**2 + kz**2
k_mag = np.sqrt(k2)
```

### 5.2 Propagator and operators

The 3D propagator is

$$\tilde{U}_{\text{lin}}(dt/2; \mathbf{k}) = \exp\!\left( -\frac{i \, (\hbar^2 |\mathbf{k}|^2 / 2m + \alpha |\mathbf{k}|^\sigma) \, dt}{2\hbar} - \frac{\Gamma dt}{2\hbar} \right).$$

The auxiliary fields are 3D arrays: `y` has shape `(M, N, N, N)`. The potential and memory steps are identical in form to 1D; broadcasting handles the dimension automatically.

### 5.3 Noise scaling in $D$ dimensions

The noise amplitude must use $dx^D$ in the denominator:

```python
noise_amp = np.sqrt(f_FDT * dt / dx**D)
```

This is the only line that changes from 1D. Failure to scale produces a grid-dependent noise floor.

### 5.4 Memory and compute cost

For 3D with $N$ per axis and $M$ memory modes:

| Quantity | Cost |
|---|---|
| Per-step compute | $\mathcal{O}(N^3 \log N)$ (two 3D FFTs) |
| $\Psi$ memory | $16 N^3$ bytes (complex128) |
| $y$ memory | $8 M N^3$ bytes |
| Propagator | $16 N^3$ bytes |

For $N = 128$, $M = 3$: $\Psi$ is 32 MB, $y$ is 24 MB, propagator 32 MB. With workspace, $\sim 200$ MB total. A 4060 8 GB GPU fits $N = 256$ comfortably, $N = 320$ tightly. CPU runs are feasible up to $N = 128$.

### 5.5 GPU backend (CuPy)

The reference 3D solver supports a CuPy backend via a thin abstraction layer. The interface is `xp = numpy` or `xp = cupy`; the algorithm is identical. The only added concern is precision: cuFFT supports fp32 and fp64; fp32 gives ~7-digit accuracy with norm conservation $\sim 10^{-7}$ over $10^4$ steps. Mixed-precision (fp32 storage, fp64 accumulation) is unnecessary for the regimes used here.

### 5.6 Batched runs (multiple seeds in parallel)

For seed-variance analysis, run multiple instances in parallel by adding a leading batch dimension. The FFTs accept batched inputs natively. The auxiliary memory and the noise increments do likewise. Memory cost scales linearly with batch size; on an 8 GB GPU with $N = 128$, batches of 8–16 are practical.

---

## 6. Ablation modes

The reference solver exposes three ablation modes used as controls. Each mode disables one or more terms.

### 6.1 `mode = "linear"`

All non-linear, dissipative, and stochastic terms disabled:

| Parameter | Effective value |
|---|---|
| $\Lambda$ | 0 |
| $\lambda_j$ | 0 (all $j$) |
| $\alpha$ | 0 |
| $\Gamma$ | 0 |
| $f_{\text{FDT}}$ | 0 |

The dynamics reduce to the standard Schrödinger equation in $V_{\text{ext}}$. Unitary, deterministic, norm-conserving. Used to test the solver against the analytic propagator of $V_{\text{ext}}$ (e.g. harmonic oscillator eigenstates).

### 6.2 `mode = "thermal"`

The cubic, memory, and fractional terms disabled; dissipation and noise on:

| Parameter | Effective value |
|---|---|
| $\Lambda$ | 0 |
| $\lambda_j$ | 0 |
| $\alpha$ | 0 |
| $\Gamma$ | nominal |
| $f_{\text{FDT}}$ | nominal |

The field thermalises to a Gaussian distribution determined by $V_{\text{ext}}$ and $f_{\text{FDT}}/\Gamma$. Used as a control for "pure thermal bath" behaviour.

### 6.3 `mode = "full"`

All terms active at their nominal values. This is the equation as specified in §1.

### 6.4 Individual-term ablations

For finer-grained controls, individual terms can be disabled by zeroing the corresponding parameter:

| Ablation name | Parameters disabled |
|---|---|
| `no_memory` | $\lambda_j = 0$ for all $j$ |
| `no_cubic` | $\Lambda = 0$ |
| `no_fractional` | $\alpha = 0$ |
| `no_noise` | $f_{\text{FDT}} = 0$ |
| `no_dissipation` | $\Gamma = 0$, also $f_{\text{FDT}} = 0$ (FDT requires both or neither) |
| `no_vext` | $V_{\text{ext}}(\mathbf{x}) = 0$ |

The structural reading: each individual-term ablation tests what role one operator plays. Removing memory and observing what changes tells you what memory was doing. Removing the cubic and observing tells you what the cubic was doing. The interpretation of these ablations is the user's; the numerical machinery only reports the results.

---

## 7. Validation protocol

Any implementation of the equation must be validated before any results are reported. The minimum protocol:

### 7.1 Norm conservation (unitary limit)

Run `mode = "linear"` with $V_{\text{ext}} = 0$ (free Schrödinger). The norm $\int |\Psi|^2 \, d^D x$ must be conserved to machine precision over the run duration. Expected tolerance:

| Precision | Norm conservation over $10^4$ steps |
|---|---|
| fp64 | $\leq 10^{-12}$ |
| fp32 | $\leq 10^{-6}$ |
| fp16 | $\leq 10^{-3}$ (not recommended) |

Deviation larger than this indicates a bug in the Fourier propagator or the FFT normalisation.

### 7.2 Harmonic oscillator eigenstates

Run `mode = "linear"` with $V_{\text{ext}} = \frac{1}{2} m \omega^2 x^2$ and $\Psi_0 = \phi_0$, the ground state. The field must remain a (complex-rotating) Gaussian with $\sigma = \sqrt{\hbar/(m\omega)}$, with the only time dependence being the global phase $e^{-i \omega t / 2}$. Numerically: $\max_x ||\Psi(t, x)|^2 - |\Psi(0, x)|^2|$ stays below the precision floor of §7.1.

### 7.3 Thermal limit

Run `mode = "thermal"` with $V_{\text{ext}}$ harmonic. The stationary density $\langle |\Psi(t, x)|^2 \rangle_t$ at late times must be Gaussian, with width determined by the equipartition $\langle E \rangle = k_B T / 2$ (per quadratic degree of freedom). The numerical width must agree with the analytic prediction within the seed-variance.

### 7.4 Reference solver comparison

If a reference implementation is available (e.g. `psi_equation.py`), run the two side-by-side with identical parameters, identical seed, identical initial state. The comparison metrics:

| Mode | Quantity | Tolerance |
|---|---|---|
| linear | $\max_t \max_x ||\Psi_A|^2 - |\Psi_B|^2|$ | $\leq 10^{-13}$ (fp64) |
| full, noise off | relative L2 error in $|\Psi|^2$ at $t = T$ | $\leq 10^{-3}$ |
| full, with noise | distribution comparison via KS test, $p > 0.05$ | over multiple seeds |

The deterministic-mode tolerance ($10^{-13}$ for fp64) is the strictest. Any larger discrepancy indicates a bug somewhere; track it down before proceeding.

### 7.5 Convergence check

For one canonical configuration, run with $dt$ and $dt/2$ (same seed, doubled step count for the smaller $dt$). The final density should match to within $\mathcal{O}(dt^2)$ for the deterministic part. This validates the second-order Strang splitting.

Similarly, halve $dx$ (double $N$ at fixed $L$): the result should be insensitive at the precision floor. Sensitivity indicates a spatial-resolution problem (likely the noise scaling or the cubic peak resolution).

### 7.6 Acceptance criteria

A solver is considered validated when:

1. Norm conserved to precision floor in linear mode (§7.1).
2. Harmonic ground state preserved (§7.2).
3. Thermal distribution matches analytic prediction (§7.3).
4. Match against reference solver within the tolerances of §7.4.
5. Convergence behaves as $\mathcal{O}(dt^2)$ in the deterministic part (§7.5).

Until all five hold, do not report results. The most common bug is incorrect noise scaling (item 7.3 fails); the second most common is incorrect FFT normalisation (item 7.1 fails); the third is incorrect Strang ordering (item 7.5 fails).

---

## 8. Observables

The state vector $(\Psi, y_1, \dots, y_M)$ produces many derived quantities. This section enumerates the standard ones, with their definitions and intended uses.

### 8.1 Field-level scalar observables

**Norm**: $\|\Psi\|^2 = \int |\Psi|^2 d^D x$. Conserved in unitary limit; reaches a finite plateau under FDT.

**Peak density**: $\rho_{\max}(t) = \max_{\mathbf{x}} |\Psi(t, \mathbf{x})|^2$.

**FWHM**: full width at half maximum of $|\Psi(t, \cdot)|^2$. Approximate measure of spatial extent.

**Inverse participation ratio (IPR)**:

$$\text{IPR}(t) = \frac{\int |\Psi|^4 \, d^D x}{\bigl( \int |\Psi|^2 \, d^D x \bigr)^2}. \tag{15}$$

High IPR (close to $1/V_{\text{box}}^{-1}$) means localised; low IPR means delocalised. Together with FWHM, gives a complete picture of spatial concentration.

**Participation ratio** (inverse of IPR, in units of effective support):

$$P(t) = \frac{(\int |\Psi|^2)^2}{\int |\Psi|^4}.$$

Reports as "effective number of grid points containing the field".

### 8.2 Memory observables

**Memory persistence** (per mode):

$$\mathcal{M}_j(t) = \frac{\int y_j^2 \, d^D x}{\int |\Psi|^4 \, d^D x}. \tag{16}$$

Ratio of memory amplitude to current density. Close to 1: memory tracks current density. Larger: memory holds onto density that has decayed. Smaller: memory has not yet built up.

**Aggregate memory persistence**: same form, with $y_j$ replaced by $V_{\text{mem}} / (\sum_j |\lambda_j|)$.

### 8.3 Spectral observables

**Radial power spectrum**:

$$P(|\mathbf{k}|) = \langle |\hat{\Psi}(\mathbf{k})|^2 \rangle_{\text{shell}}, \tag{17}$$

with the average taken over the spherical shell of radius $|\mathbf{k}|$ in 3D, circular in 2D, or just the symmetric pair in 1D.

**Dominant wavenumber** $k_*$: location of the peak of $P(|\mathbf{k}|)$ above some lower cutoff (typically $k > 2\pi/L$ to exclude the DC component).

**Crystallinity**: fraction of total power at $|\mathbf{k}| > k_{\text{cutoff}}$ for some structural cutoff $k_{\text{cutoff}}$. Reports how much of the field's energy is in spatially-structured modes vs. in the DC/long-wavelength component.

$$C(t) = \frac{\sum_{|\mathbf{k}| > k_{\text{cutoff}}} |\hat{\Psi}|^2}{\sum_{\mathbf{k}} |\hat{\Psi}|^2}. \tag{18}$$

Above 0.5: structurally developed. Below 0.2: dispersive.

### 8.4 Lattice family detection (3D)

For 3D crystalline phases, fit the angular distribution of $|\hat{\Psi}|^2$ on the dominant shell ($|\mathbf{k}| = k_*$) to the canonical Bravais signatures:

- **SC** (simple cubic): peaks at $\pm \hat{x}, \pm \hat{y}, \pm \hat{z}$, 6 peaks total.
- **BCC** (body-centred cubic): peaks at $(1, 1, 0)/\sqrt{2}$-type permutations, 12 peaks.
- **FCC** (face-centred cubic): peaks at $(1, 1, 1)/\sqrt{3}$-type permutations, 8 peaks.
- **HCP** (hexagonal close-packed): 6 in-plane + 2 axial.

The detector reports a score per family (fraction of shell power aligned with the family's canonical directions) and the best-matching family. See `observables.py` in the 3D reference for implementation; the cosine-window threshold is 0.15.

### 8.5 Time-aggregated metrics

For comparison across runs, time-aggregated metrics:

**`stabilization_score`**: normalised decrease of the relative variance of $\|\Psi\|^2$ over time. Quantifies how much the global observable settles from initial transient to late-time plateau.

**`structure_persistence`**: late-time time-average of the crystallinity $C(t)$, normalised against the early-time average. Reports whether spatial structure that builds up early survives to late times.

**`time_to_stabilize`**: first time at which $\|\Psi\|^2$ enters and stays within $\pm 10\%$ of its late-time mean.

**`cluster_persistence`**: number of distinct local maxima in $|\Psi|^2$, time-averaged over late window.

### 8.6 SOTA-metric proxies (probe-level)

For comparison against external "complexity / integration / consciousness" measures, the field can be probed at $N_{\text{probes}}$ spatial locations, returning $N_{\text{probes}}$ time series. The following are computed on this multivariate time series:

- **LZc** (Lempel–Ziv complexity, binarised)
- **Φ-like proxy** (integrated information, via Tononi's $\phi$ or a Φ-ID proxy)
- **PCIst** (perturbational complexity index, requires injecting a perturbation and measuring the response)
- **Workspace / ignition**: fraction of probes simultaneously active above threshold
- **Causal density**: total Granger causality between probe pairs
- **Metastability**: variance of the Kuramoto order parameter

These probe-level metrics see only the time series at the probe locations; they do not see the spatial field structure. A field with strong spatial structure but no temporal probe-level variability will read as "low complexity" on these metrics. This is a known limitation of probe-based measurement; the field-level observables (§8.1–8.4) capture what the probes miss.

### 8.7 Recording conventions

The reference solver records every `record_every` steps (default 4) for a sampling frequency of $1/(record\_every \cdot dt)$. Stored quantities:

| Field | Shape | Dtype |
|---|---|---|
| `t` | `(n_records,)` | float64 |
| `full_density` | `(N, n_records)` (1D) or higher | float64 |
| `density_channels` | `(n_channels, n_records)` | float64 |
| `phase` | `(n_channels, n_records)` | float64 |
| `y` | `(M, N, n_records)` | float64 |
| `params` | dict | — |

Channel binning: the spatial domain is split into `n_channels` equal bins; the per-channel density is the integral of $|\Psi|^2$ over the bin. Probes are placed at bin centres.

---

## 9. Initial conditions

### 9.1 Single Gaussian (canonical)

$$\Psi_0(\mathbf{x}) = \mathcal{N} \exp\!\left( -\frac{|\mathbf{x}|^2}{2 s^2} \right) \exp(i \mathbf{k}_0 \cdot \mathbf{x}), \tag{19}$$

with $\mathcal{N}$ chosen to normalise $\int |\Psi_0|^2 d^D x = 1$. Default $s = 0.5$–$1.0$ in box units, $\mathbf{k}_0 = 0$ (no momentum) or $\mathbf{k}_0 = (0.5, 0, 0)$ (momentum kick along $x$).

### 9.2 Two Gaussians

$$\Psi_0(\mathbf{x}) = \mathcal{N} \bigl[ G(\mathbf{x} - \mathbf{x}_+) + G(\mathbf{x} - \mathbf{x}_-) \bigr], \tag{20}$$

with $G(\mathbf{x}) = \exp(-|\mathbf{x}|^2 / 2s^2)$. Default separation $|\mathbf{x}_+ - \mathbf{x}_-| = 4$–$6$ in box units, $s = 1.0$. Subdivide into:

- **Attractive** ($\Lambda < 0$, $\lambda_j < 0$): default; two regions form a shared memory basin.
- **Repulsive** ($\Lambda < 0$, $\lambda_j > 0$): memory pushes the two regions apart.
- **Neutral** ($\lambda_j = 0$): pure cubic, no memory channel between regions.

### 9.3 Random field

$$\Psi_0(\mathbf{x}) = \mathcal{N} \sum_{\mathbf{k}: |\mathbf{k}| < k_{\text{cut}}} a_{\mathbf{k}} e^{i \mathbf{k} \cdot \mathbf{x}}, \tag{21}$$

with $a_{\mathbf{k}}$ complex Gaussian iid. The cutoff $k_{\text{cut}}$ prevents grid-scale features; default $k_{\text{cut}} = k_{\max}/4$.

### 9.4 Perturbed Gaussian

Single Gaussian plus a low-amplitude random perturbation:

$$\Psi_0(\mathbf{x}) = G(\mathbf{x}) + \epsilon \, \zeta(\mathbf{x}), \tag{22}$$

with $\zeta$ a normalised random field (§9.3) and $\epsilon \sim 0.01$–$0.05$. Used to seed instability without a fully random IC.

### 9.5 Pre-loaded memory

For testing whether the system retains past states, set $y_j(0) = \rho_{\text{prior}}(\mathbf{x})$ for some chosen prior density. The reference convention is $y_j(0) = 0$ (no history); deviations from this must be explicit.

---

## 10. Parameter regimes

The equation supports several qualitatively distinct regimes. The regime is determined by signs and relative magnitudes of $\Lambda$, $\lambda_j$, $\Gamma$, $f_{\text{FDT}}$, $\alpha$, and the presence or absence of $V_{\text{ext}}$.

### 10.1 Default B0 (structure-forming)

The canonical configuration for studying emergent structure under FDT:

| Parameter | Value |
|---|---|
| $\Lambda$ | $-0.5$ (attractive) |
| $\lambda_j$ | $(-0.3, -0.2, -0.1)$ (attractive memory) |
| $\nu_j$ | $(2.0, 0.5, 0.1)$ |
| $\alpha$ | $0.15$ |
| $\sigma$ | $1.5$ |
| $\Gamma$ | $0.05$ |
| $f_{\text{FDT}}$ | $0.002$ ($= 2\gamma_0 k_B T$) |
| $V_{\text{ext}}$ | harmonic, $\omega = 0.05$ |
| Grid (1D) | $L = 32$, $N = 128$, $dt = 0.005$ |
| Grid (3D) | $L = 20$, $N = 128$, $dt = 0.0025$ |
| Time | $T = 20$ (1D), $T = 6$ (3D) |

In this regime, an initial Gaussian under FDT noise undergoes: noisy/chaotic transient → self-stabilisation onto an FDT plateau → emergence of a persistent localised filament carved by the memory feedback.

### 10.2 Dispersive (kinetic dominates)

$\Lambda = 0$, $\lambda_j = 0$, $V_{\text{ext}} = 0$, full $\Gamma$ and $f_{\text{FDT}}$: kinetic dispersion plus thermal bath, no self-interaction. The field thermalises in space to a wide Gaussian. Used as a "no nonlinearity" baseline.

### 10.3 Anti-collapse (3D supercritical NLS with memory)

$\Lambda < 0$ with $|\Lambda| \gg 1$ (strongly attractive, blow-up regime), $\lambda_j > 0$ (repulsive memory), no $V_{\text{ext}}$, with or without FDT:

| Parameter | Value |
|---|---|
| $\Lambda$ | $-10$ |
| $\lambda_j$ | $(3.0, 1.0)$ (repulsive memory) |
| $\nu_j$ | $(10.0, 0.5)$ |
| $V_{\text{ext}}$ | none |
| Grid | 3D, $N = 128$, $L = 20$, $dt = 0.0025$ |
| Time | $T = 6$ |

The 3D cubic NLS is $L^2$-supercritical; a localised initial state with $\Lambda < 0$ would blow up in finite time (peak density diverges, support shrinks to a point). With the memory mechanism active, the trajectory diverges from the no-memory trajectory after $t \sim 1$: peak density falls by 4–5 orders of magnitude, effective spatial support rises by 4–5 orders of magnitude, and the field stabilises in a bounded distributed configuration. This is the structural anti-collapse demonstration.

The matched control is the same configuration with $\lambda_j = 0$, which exhibits the expected collapse trajectory.

### 10.4 Crystallisation (R5 series)

$\Lambda = -8$, hierarchical memory with $\Sigma_\lambda$-split structure, no $V_{\text{ext}}$, unitary ($\Gamma = 0$, $f_{\text{FDT}} = 0$) or with FDT:

| Parameter | Value |
|---|---|
| $\Lambda$ | $-8$ |
| $\lambda_j$ split | $(0.75 \Sigma_\lambda, 0.25 \Sigma_\lambda)$ with $\Sigma_\lambda = 1.5$ |
| $\nu_j$ | $(10.0, 0.5)$ |
| Grid | 3D, $N = 128$, $L = 20$, $dt = 0.0025$ |
| Time | $T = 15$ |

From a Gaussian initial state, modulational instability selects a periodic lattice; the dominant wavenumber satisfies $k_* L \approx 16.3$ across multiple calibrations (the "robust attractor"), and the Bravais family depends on the initial momentum and on FDT coupling. Used as the substrate for the R5–R10 progression.

### 10.5 Failure modes

The regimes that **don't** persist as bounded structure (used as controls):

- **No memory** ($\lambda_j = 0$): in the focusing regime, blow-up or quasi-blow-up. In the dispersive regime, decay to spatial uniformity.
- **No noise** ($f_{\text{FDT}} = 0$, $\Gamma > 0$): pure decay to zero.
- **No dissipation** ($\Gamma = 0$, $f_{\text{FDT}} > 0$): unbounded growth of the norm.
- **No cubic and no memory**: pure linear Schrödinger, dispersive.
- **No external potential and no memory in focusing regime**: 3D blow-up.

Each of these is a single-term ablation of the structure-forming regime. The full equation occupies a non-trivial point in parameter space that none of its single-term ablations reaches.

---

## 11. Multi-substrate protocols

The equation is a single PDE; "multi-substrate" refers to running multiple instances simultaneously with cross-coupling. Three protocols.

### 11.1 Pair coupling

Two instances $\Psi^{(A)}$ and $\Psi^{(B)}$, each obeying (1), coupled through $V_{\text{ext}}$:

$$V_{\text{ext}}^{(A)}(\mathbf{x}, t) = \kappa \, |\Psi^{(B)}(\mathbf{x}, t)|^2, \quad V_{\text{ext}}^{(B)}(\mathbf{x}, t) = \kappa \, |\Psi^{(A)}(\mathbf{x}, t)|^2. \tag{23}$$

$\kappa \geq 0$ is the coupling strength. Both substrates run in lockstep, with $V_{\text{ext}}$ updated at each step from the partner's instantaneous density.

### 11.2 Ring of $N$

$N$ instances arranged cyclically:

$$V_{\text{ext}}^{(i)}(\mathbf{x}, t) = \kappa \, |\Psi^{(i+1 \mod N)}(\mathbf{x}, t)|^2, \quad i = 0, \dots, N-1. \tag{24}$$

For $N = 3$, this is a minimal closed recursion (no outermost level). Different per-instance seeds and initial momenta produce distinguishable substrates; the ring topology preserves identity along structural axes (peak orientation, lattice family, $k_*$) across the coupling sweep.

### 11.3 Sequence of inputs

A single instance $\Psi$ runs through $K$ consecutive segments, each with a different external potential $V_{\text{ext}}^{(k)}(\mathbf{x})$:

| Segment | Duration | $V_{\text{ext}}$ |
|---|---|---|
| 1 | $T_{\text{seg}}$ | $V_{\text{input}}^{(1)}$ |
| 2 | $T_{\text{seg}}$ | $V_{\text{input}}^{(2)}$ |
| ... | ... | ... |
| $K$ | $T_{\text{seg}}$ | $V_{\text{input}}^{(K)}$ |

The wavefunction and the auxiliary memory fields are carried forward across segment boundaries (no reset). The substrate's signature at the end of segment $k$ depends on the input at segment $k$ **and** on the history of prior segments via $V_{\text{mem}}$.

### 11.4 Full collective (ring + sequences + FDT)

Combination of §11.2 and §11.3: $N$ instances in a ring, each receiving its own input sequence, all with FDT active. The cross-substrate coupling acts at every step; the input sequences are switched at segment boundaries. This is the maximal protocol the framework defines; it tests every structural mechanism simultaneously.

### 11.5 Implementation

The ring is implemented by running $N$ instances in lockstep with shared synchronisation at each step. With per-substrate batching, all $N$ states can be stored in a single array of shape `(N, M, N_grid^D)` (memory) and `(N, N_grid^D)` (wavefunction), and the ring coupling becomes a circular shift along axis 0. Performance scales linearly in $N$ with the per-step cost of a single substrate.

---

## 12. Reading outputs

What to look for in a run, and how to read it.

### 12.1 Norm trajectory

$\|\Psi(t)\|^2$ over time.

- **Linear / unitary**: flat line at 1.0 within precision floor.
- **Thermal**: rapid rise to a finite plateau; equilibrium value $\sim f_{\text{FDT}} / (2\Gamma)$ in natural units (set by FDT balance).
- **Full, structure-forming**: similar rise to FDT plateau, possibly with transient over/undershoot during initial settling.

Norm drift in the linear mode is a bug. Norm not reaching a plateau in thermal mode means either insufficient run time or incorrect FDT scaling.

### 12.2 Peak density vs. spatial support

Plot $\rho_{\max}(t)$ and $P(t)$ (participation ratio) on log scale.

- **Collapse trajectory** (no-memory, 3D supercritical $\Lambda < 0$): $\rho_{\max}$ rises monotonically to large values, $P$ falls to $O(1)$ (single grid point).
- **Dispersive trajectory** (no nonlinearity): $\rho_{\max}$ falls as $t^{-D/2}$, $P$ rises as $t^{D/2}$.
- **Anti-collapse trajectory** (full, $\Lambda < 0$, $\lambda_j > 0$): $\rho_{\max}$ rises briefly then falls 4–5 orders of magnitude; $P$ rises 4–5 orders of magnitude. Both stabilise.
- **Structure-forming trajectory** (full B0): $\rho_{\max}$ and $P$ both reach moderate stable values; $P$ in the range of 10–100 grid points; the field forms a persistent filament.

### 12.3 Memory amplitude

Plot $\|y_j(t)\|^2 / \|\Psi(t)\|^4$ per mode.

- **Slow modes ($\nu_j \ll 1$)** build up over time and persist; their late-time amplitude reports how much of the cumulative history the field carries.
- **Fast modes ($\nu_j \gg 1/dt$)** track the instantaneous density; their amplitude $\sim \|\Psi\|^4$.
- **Comparison across modes** shows the hierarchy: well-separated $\nu_j$ produce distinct amplitudes, with the slowest accumulating the most.

### 12.4 Spectral diagnostics

Plot the radial power $P(|\mathbf{k}|, t)$ at selected times. For crystallising regimes:

- **Early**: peak near $|\mathbf{k}| = 0$ (the initial Gaussian).
- **Modulational growth**: power transferred from DC to a finite $k_*$.
- **Crystalline phase**: sharp peak at $k_*$, dominant for $t > t_{\text{cryst}}$.
- **Bravais family** (3D): apply the lattice detector of §8.4 once $k_*$ is identified.

### 12.5 Cross-seed variance

The standard report includes seed-mean ± seed-std for each scalar observable, with the number of seeds quoted. For the structure-forming regime, the typical coefficients of variation across seeds:

| Quantity | CV |
|---|---|
| IPR (late) | 2–3% |
| Structure persistence | 2–3% |
| Memory persistence | 5–10% |
| Stabilisation score | 50–60% |
| Time-to-stabilise | 40–50% |

Asymptotic structural quantities are robust across seeds; transient timing is seed-sensitive. This is the expected behaviour for a stochastic attractor.

### 12.6 What to report

A reproducible run reports:

1. Full parameter set (every entry in §0.2 and §10.1, or whichever regime is used).
2. Solver version and validation status (§7).
3. Seed list and number of seeds.
4. Scalar observables with seed-mean ± seed-std.
5. At least one time-resolved diagnostic plot.
6. Ablation comparison: same configuration with one term disabled, for each term whose role is being tested.

A run that doesn't include the ablation comparison cannot distinguish a structural effect from a parameter-fitting effect; the ablation is mandatory.

---

## 13. Computational complexity

Summary of costs.

### 13.1 Per-step cost

| Operation | Cost (1D) | Cost (3D) |
|---|---|---|
| FFT (×2) | $\mathcal{O}(N \log N)$ | $\mathcal{O}(N^3 \log N)$ |
| Real-space multiply | $\mathcal{O}(N)$ | $\mathcal{O}(N^3)$ |
| Memory update (per mode) | $\mathcal{O}(N)$ | $\mathcal{O}(N^3)$ |
| Noise generation | $\mathcal{O}(N)$ | $\mathcal{O}(N^3)$ |

Total per step in 3D: dominated by the two FFTs, $\mathcal{O}(N^3 \log N)$.

### 13.2 Total run cost

$\mathcal{O}(n_{\text{steps}} \cdot N^D \log N)$ where $n_{\text{steps}} = T_{\text{total}} / dt$. For 3D / $N = 128$ / $T = 6$ / $dt = 0.0025$:

- CPU (single core, NumPy FFTs): ~6 hours.
- GPU (CuPy on RTX 4060): ~10 minutes.

For 1D / $N = 128$ / $T = 20$: CPU 1–2 seconds.

### 13.3 Memory footprint

| Quantity | 1D $N=128$ | 3D $N=128$ | 3D $N=256$ |
|---|---|---|---|
| $\Psi$ | 2 KB | 32 MB | 256 MB |
| $y$ ($M=3$) | 3 KB | 96 MB | 768 MB |
| Propagator | 1 KB | 32 MB | 256 MB |
| FFT scratch | – | 32–64 MB | 256–512 MB |
| **Total working set** | 10–20 KB | 200–250 MB | 1.5–2.0 GB |

3D at $N = 128$ fits in any modern GPU. At $N = 256$, requires 8 GB minimum; an RTX 4060 8 GB fits with some headroom.

### 13.4 Multi-instance scaling

Pair coupling: $\times 2$ memory and compute.

Ring of $N$: $\times N$ memory and compute. The ring synchronisation does not add cost; each instance's step is independent given the partner's density.

Sequence of inputs: same cost as single instance per segment, so $K$ segments cost $K \times$ the single-segment cost.

Full collective ($N$-ring with $K$ input segments): $N \times K \times$ single-instance cost.

---

## Appendix A — Default parameter sets

### A.1 1D B0 (structure-forming, canonical reference)

```python
hbar  = 1.0
m     = 1.0
omega = 0.05      # harmonic V_ext
Lambda = -0.5
alpha = 0.15
sigma = 1.5
Gamma = 0.05
f_FDT = 0.002     # = 2 γ₀ k_B T
nu    = (2.0, 0.5, 0.1)
lam   = (-0.3, -0.2, -0.1)
L     = 32.0
N     = 128
dt    = 0.005
T     = 20.0
mode  = "full"
seed  = 0
```

### A.2 3D structure-forming (R5 default)

```python
hbar  = 1.0
m     = 1.0
Lambda = -8.0
sigma = 1.5
alpha = 0.0       # set 0 unless fractional dispersion explicitly under test
Gamma = 0.0       # unitary R5 baseline; use 0.01 for R5-FDT
T_bath = 0.0      # use 0.001 for R5-FDT
nu    = (10.0, 0.5)
lam   = (1.125, 0.375)   # = 0.75 * 1.5, 0.25 * 1.5
L     = 20.0
N     = 128
dt    = 0.0025
T     = 15.0
init_sigma = 0.5
init_k0    = (0, 0, 0)   # vary for R5 inputs
seed  = 42
```

### A.3 3D anti-collapse (focusing supercritical NLS with memory)

```python
hbar  = 1.0
m     = 1.0
Lambda = -10.0    # strongly focusing
sigma = 2.0
alpha = 0.0
Gamma = 0.0
f_FDT = 0.0
nu    = (10.0, 0.5)
lam   = (3.0, 1.0)        # repulsive memory (anti-collapse)
V_ext = 0.0
L     = 20.0
N     = 128
dt    = 0.0025
T     = 6.0
init_sigma = 0.5
seed  = 42
```

Match this against $\lambda_j = (0, 0)$ for the no-memory collapse trajectory.

---

## Appendix B — RNG conventions

Use `numpy.random.default_rng(seed)` (NumPy ≥ 1.17). The reference solver passes the seed through `PsiParams.seed` and creates the generator once at the start. Per-step calls use `rng.standard_normal(N)`; two such calls per step (real and imaginary part of $\xi$).

For multi-seed analysis, use sequential seeds starting from a fixed base (e.g. 42, 43, 44, ...). For batched runs on GPU, use `cupy.random.default_rng` or seed a `Generator` array; each batch index gets its own seed.

For exact reproducibility, the seed and the NumPy version must be reported; bit-exact reproduction requires the same FFT implementation (NumPy 2.x is bit-stable; CuPy 13+ likewise; mixing across versions may produce small numerical drift).

---

## Appendix C — Validation reference numbers

For the 1D B0 configuration (Appendix A.1) at the default seed:

| Quantity | Expected value (mode = "linear") | Expected value (mode = "full") |
|---|---|---|
| Norm at $t = T$ | $1.0 \pm 10^{-12}$ (fp64) | $\sim 1.0$ (FDT plateau) |
| Peak density at $t = T$ | (depends on $V_{\text{ext}}$) | $\sim 0.5$–$2$ |
| Time to stabilize | (n/a) | $\sim 4$–$15$ |
| IPR (late) | (n/a) | $\sim 0.009$–$0.013$ |
| Structure persistence | (n/a) | $\sim 0.62$–$0.66$ |

For the 3D anti-collapse configuration (Appendix A.3):

| Quantity | No memory ($\lambda_j = 0$) | With memory |
|---|---|---|
| Peak density at $t = T$ | $\sim 50$–$80$ | $\sim 10^{-3}$ |
| Participation ratio at $t = T$ | $\sim 20$ | $\sim 10^6$ |
| Trajectory shape | monotonic rise to plateau | rise then fall by 4–5 decades |

These numbers serve as a sanity check; any new implementation should produce results within $\sim 10\%$ of these at the default seeds.

---

## Appendix D — Reference code skeleton

A minimal self-contained reference. This is the 1D solver in roughly 80 lines; 3D extension follows §5.

```python
"""
Triad equation 1D reference solver.
i ℏ ∂_t Ψ = [-ℏ²/(2m) ∂_x² + V_ext + Λ|Ψ|² + V_mem + α(-Δ)^(σ/2) - iΓ] Ψ + η
V_mem = Σ_j λ_j y_j,    ∂_t y_j = ν_j (|Ψ|² - y_j)
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class TriadParams:
    L:     float = 32.0
    N:     int   = 128
    dt:    float = 0.005
    T:     float = 20.0
    hbar:  float = 1.0
    m:     float = 1.0
    omega: float = 0.05         # harmonic V_ext frequency
    Lambda: float = -0.5
    alpha: float = 0.15
    sigma: float = 1.5
    Gamma: float = 0.05
    f_FDT: float = 0.002        # = 2 γ₀ k_B T
    nu:    tuple  = (2.0, 0.5, 0.1)
    lam:   tuple  = (-0.3, -0.2, -0.1)
    mode:  str    = "full"       # linear | thermal | full
    seed:  int    = 0


def integrate(p: TriadParams):
    rng = np.random.default_rng(p.seed)
    
    # grid
    x  = np.linspace(-p.L/2, p.L/2, p.N, endpoint=False)
    dx = x[1] - x[0]
    k  = 2 * np.pi * np.fft.fftfreq(p.N, d=dx)
    abs_k = np.abs(k)
    
    # V_ext (harmonic)
    V_ext = 0.5 * p.m * p.omega**2 * x**2
    
    # mode-dependent effective parameters
    Lambda_eff = p.Lambda if p.mode == "full" else 0.0
    alpha_eff  = p.alpha  if p.mode == "full" else 0.0
    Gamma_eff  = p.Gamma  if p.mode != "linear" else 0.0
    f_FDT_eff  = p.f_FDT  if p.mode != "linear" else 0.0
    lam_eff    = np.asarray(p.lam) if p.mode == "full" else np.zeros(len(p.lam))
    
    # initial state
    psi = np.exp(-x**2 / 8.0).astype(np.complex128)
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)
    
    # memory
    M = len(p.nu)
    y = np.zeros((M, p.N), dtype=np.float64)
    
    # propagator (eq. 9)
    H_lin_k = p.hbar * k**2 / (2*p.m) + alpha_eff * abs_k**p.sigma
    half_lin = np.exp(-1j * H_lin_k * p.dt / (2*p.hbar)
                      - Gamma_eff * p.dt / (2*p.hbar))
    
    # noise amplitude (eq. 4)
    noise_amp = np.sqrt(f_FDT_eff * p.dt / dx)
    
    n_steps = int(p.T / p.dt)
    rec_density = np.zeros((p.N, n_steps + 1))
    rec_t       = np.zeros(n_steps + 1)
    
    for step in range(n_steps + 1):
        rec_density[:, step] = np.abs(psi)**2
        rec_t[step] = step * p.dt
        
        if step == n_steps:
            break
        
        # Strang split-step
        psi = np.fft.ifft(np.fft.fft(psi) * half_lin)
        
        rho   = np.abs(psi)**2
        V_mem = (lam_eff[:, None] * y).sum(axis=0)
        V_tot = V_ext + Lambda_eff * rho + V_mem
        psi   = psi * np.exp(-1j * V_tot * p.dt / p.hbar)
        
        if p.mode == "full":
            for j in range(M):
                y[j] += p.dt * p.nu[j] * (rho - y[j])
        
        if noise_amp > 0:
            xi = (rng.standard_normal(p.N) + 1j * rng.standard_normal(p.N)) / np.sqrt(2.0)
            psi = psi + noise_amp * xi
        
        psi = np.fft.ifft(np.fft.fft(psi) * half_lin)
    
    return {"t": rec_t, "x": x, "density": rec_density, "params": p}


if __name__ == "__main__":
    for mode in ["linear", "thermal", "full"]:
        out = integrate(TriadParams(mode=mode, T=10.0, seed=0))
        norm_t = out["density"].sum(axis=0) * (out["x"][1] - out["x"][0])
        print(f"mode={mode:8s}  norm in [{norm_t.min():.4f}, {norm_t.max():.4f}]")
```

Running this script should print three lines, one per mode. In `linear`, `norm` should be 1.0 to within machine precision; in `thermal` and `full`, it should reach a finite plateau above 1.0 set by FDT balance.

---

## Appendix E — Notation index

| Symbol | First defined | Section |
|---|---|---|
| $\Psi$ | wavefunction | §0.1 |
| $\rho$ | density $|\Psi|^2$ | §0.1 |
| $y_j$ | memory field, mode $j$ | §0.1, §1.2 |
| $V_{\text{ext}}$ | external potential | §0.1, §2.6 |
| $V_{\text{mem}}$ | memory potential | §0.1, §1.2 |
| $\eta$ | stochastic forcing | §0.1, §1.3 |
| $\hbar, m$ | quantum scales | §0.2 |
| $\Lambda$ | cubic coupling | §0.2, §2.2 |
| $\alpha, \sigma$ | fractional Laplacian | §0.2, §2.4 |
| $\Gamma$ | dissipation | §0.2, §2.5 |
| $\nu_j, \lambda_j$ | memory parameters | §0.2, §2.3 |
| $f_{\text{FDT}} = 2\gamma_0 k_B T$ | noise amplitude | §0.2, §1.3 |
| $M$ | number of memory modes | §0.2 |
| $D$ | spatial dimension | §0.4 |
| $L, N, dx, dt$ | grid parameters | §0.5, §3.7 |
| $k_*$ | dominant wavenumber | §8.3 |
| IPR | inverse participation ratio | §8.1 |
| $C(t)$ | crystallinity | §8.3 |
| $\kappa$ | cross-substrate coupling | §11.1 |

---

*End of reference.*
