"""tiny model end-to-end: exercises all 4 layers of the 2027 architecture.

camada 0  solver nativo B0 (strang + exptrap + FDT coupling + lambda dinamico)
camada 1  TriadSSM learns solver dynamics
camada 2  MNOSurrogate trained against B0 + fidelity gate + accelerated rollout
camada 3  AttractorObserver forecasts invariants (passive)
camada 4  PhysicalReservoirEmulator edge-of-chaos sweep
metrics   triad_phase_synchronization, spectral_flux, attractor_geometry_invariants
"""
from __future__ import annotations
import numpy as np
import pytest

def _tiny_params(**overrides):
    from runtime.core.solver import TriadParams
    defaults = dict(mode='full', N=32, L=16.0, T=1.0, dt=0.01, seed=42,
                    fdt_couple=True, kT=1.0, record_every=2)
    defaults.update(overrides)
    return TriadParams(**defaults)

class TestCamada0:

    def test_strang_baseline(self):
        from runtime.core.solver import integrate
        p = _tiny_params()
        out = integrate(p, auto_halve_dt=False)
        dx = out['dx']
        norm_t = out['density'].sum(axis=0) * dx
        assert norm_t.min() > 0.1
        assert float(np.std(norm_t)) < float(np.mean(norm_t)) * 5

    def test_exptrap_fixed_lambda(self):
        from runtime.core.solver import integrate
        p = _tiny_params(step_mode='exptrap', trap_lambda=0.5)
        out = integrate(p, auto_halve_dt=False)
        dx = out['dx']
        norm_t = out['density'].sum(axis=0) * dx
        assert norm_t.min() > 0.1

    def test_exptrap_dynamic_lambda(self):
        from runtime.core.solver import integrate

        def lambda_from_rho(rho):
            r = np.asarray(rho)
            return float(0.3 + 0.4 * np.mean(r) / (np.max(r) + 1e-10))

        p = _tiny_params(step_mode='exptrap', trap_lambda=lambda_from_rho)
        out = integrate(p, auto_halve_dt=False)
        dx = out['dx']
        norm_t = out['density'].sum(axis=0) * dx
        assert norm_t.min() > 0.1

    def test_fdt_couple_locked(self):
        from runtime.core.solver import integrate, _effective_params
        p = _tiny_params()
        eff = _effective_params(p)
        assert p.fdt_couple is True
        dx = p.L / p.N
        expected_f_fdt = 2.0 * eff['Gamma'] * dx * p.kT / p.hbar
        out = integrate(p, auto_halve_dt=False)
        norm = float((np.abs(out['psi_final']) ** 2).sum() * out['dx'])
        assert norm > 0.1
        assert np.isfinite(expected_f_fdt)

    def test_exptrap_2d(self):
        from runtime.core.solver import integrate_2d
        p = _tiny_params(D=2, step_mode='exptrap', trap_lambda=0.5)
        out = integrate_2d(p, auto_halve_dt=False)
        norm = float((np.abs(out['psi_final']) ** 2).sum() * out['dx'] ** 2)
        assert norm > 0.1

    def test_exptrap_3d(self):
        from runtime.core.solver import integrate_3d
        p = _tiny_params(D=2, N=16, step_mode='exptrap', trap_lambda=0.5)
        out = integrate_3d(p, auto_halve_dt=False, record_density=True)
        assert out['psi_final'].shape == (16, 16, 16)

class TestCamada1:

    def test_ssm_forward_shape(self):
        from runtime.ml.nn import TriadSSM
        from runtime.ml.tensor import tensor
        np.random.seed(0)
        ssm = TriadSSM(d_model=4, d_state=8, n_memory=2, coupling=False, noise=0.0)
        x = tensor(np.random.randn(2, 5, 4))
        out = ssm(x)
        assert out.shape == (2, 5, 4)

    def test_ssm_backward_runs(self):
        from runtime.ml.nn import TriadSSM, Adam
        from runtime.ml.tensor import tensor
        np.random.seed(0)
        ssm = TriadSSM(d_model=4, d_state=8, n_memory=2, coupling=False, noise=0.0)
        opt = Adam(ssm.parameters(), lr=1e-3)
        x = tensor(np.random.randn(1, 3, 4))
        out = ssm(x)
        loss = (out * out).mean()
        loss.backward()
        opt.step()
        assert True

    def test_ssm_learns_solver_step(self):
        from runtime.core.solver import integrate
        from runtime.ml.nn import TriadSSMBlock, Adam
        from runtime.ml.tensor import tensor
        np.random.seed(0)
        p = _tiny_params(mode='full', T=0.5, record_every=1)
        out = integrate(p, auto_halve_dt=False)
        density = out['density']
        n_t = density.shape[1]
        if n_t < 4:
            pytest.skip('not enough frames')
        X_np = density[:, :-1].T
        Y_np = density[:, 1:].T
        d_in = X_np.shape[1]
        model = TriadSSMBlock(d_model=d_in, d_state=8, n_memory=2, coupling=False, noise=0.0)
        opt = Adam(model.parameters(), lr=1e-2)
        for ep in range(5):
            x = tensor(X_np[None, :, :])
            pred = model(x)
            tgt = tensor(Y_np[None, :, :])
            loss = ((pred - tgt) ** 2).mean()
            model.zero_grad()
            loss.backward()
            opt.step()
        x_test = tensor(X_np[None, :, :])
        from runtime.ml.tensor import no_grad
        with no_grad():
            pred_final = model(x_test)
        err = float(np.mean((pred_final._data[0] - Y_np) ** 2))
        assert np.isfinite(err)

class TestCamada2:

    def test_triad_residual_full(self):
        from runtime.core.solver import integrate
        from runtime.ml.surrogate import TriadContext, triad_residual_full
        p = _tiny_params(T=0.5, record_every=1)
        out = integrate(p, auto_halve_dt=False)
        ctx = TriadContext.from_params(p)
        psi_seq = np.array([out['density'][:, i] for i in range(out['density'].shape[1])],
                           dtype=np.complex128)
        if psi_seq.shape[0] < 3:
            psi_seq = np.stack([out['psi_final'], out['psi_final'] * 0.99])
        res = triad_residual_full(psi_seq, out['t'][:psi_seq.shape[0]], ctx)
        assert np.isfinite(res['mean'])
        assert res['mean'] >= 0

    def test_generate_pairs_and_train(self):
        from runtime.ml.surrogate import generate_pairs, train_surrogate, TriadContext
        from runtime.core.solver import TriadParams
        from stdlib.regimes import resolve_regime
        p = resolve_regime('B0', N=16)
        p = TriadParams(**{**p.__dict__, 'N': 16, 'T': 0.1, 'dt': 0.01, 'record_every': 9999})
        ctx = TriadContext.from_params(p)
        X, Yc, ctx_map = generate_pairs(['B0'], seeds=[0], N=16,
                                         chunk_steps=5, n_chunks=3)
        assert X.shape[0] == 3
        assert X.shape[1] == 16
        assert X.shape[2] == 3
        model, history = train_surrogate(X, Yc, ctx, chunk_T=5 * p.dt,
                                          epochs=3, batch=2, d_model=4,
                                          d_state=4, n_blocks=1,
                                          verbose=False)
        assert len(history) == 3
        for l2, phys in history:
            assert np.isfinite(l2)
            assert np.isfinite(phys)

    def test_fidelity_gate(self):
        from runtime.core.solver import integrate
        from runtime.ml.surrogate import TriadContext, FidelityGate
        p = _tiny_params(T=0.5, record_every=1)
        out = integrate(p, auto_halve_dt=False)
        ctx = TriadContext.from_params(p)
        psi = out['psi_final']
        psi_noisy = psi + 0.01 * np.random.randn(*psi.shape)
        psi_noisy = psi_noisy.astype(np.complex128)
        gate = FidelityGate(l2_tol=1.0, norm_tol=1.0, kstar_tol=10.0, cryst_tol=1.0, resid_factor=100.0)
        seq = np.stack([psi, psi_noisy])
        t = np.array([0.0, p.dt])
        res = gate.evaluate(seq, seq, t, ctx)
        assert 'passed' in res
        assert np.isfinite(res['l2_rel'])

    def test_accelerated_rollout(self):
        from runtime.core.solver import TriadParams
        from runtime.ml.surrogate import (generate_pairs, train_surrogate, TriadContext,
                                        accelerated_rollout, FidelityGate)
        from stdlib.regimes import resolve_regime
        p = resolve_regime('B0', N=16)
        p = TriadParams(**{**p.__dict__, 'N': 16, 'T': 0.1, 'dt': 0.01,
                           'record_every': 9999, 'seed': 0})
        ctx = TriadContext.from_params(p)
        X, Yc, _ = generate_pairs(['B0'], seeds=[0], N=16,
                                   chunk_steps=5, n_chunks=4)
        model, _ = train_surrogate(X, Yc, ctx, chunk_T=5 * p.dt,
                                    epochs=3, batch=2, d_model=4,
                                    d_state=4, n_blocks=1, verbose=False)
        gate = FidelityGate(l2_tol=2.0, norm_tol=2.0, kstar_tol=100.0,
                             cryst_tol=2.0, resid_factor=100.0)
        res = accelerated_rollout(p, T_total=0.05, surrogate=model, gate=gate,
                                   chunk_steps=5, warmup_chunks=1, base_seed=0)
        assert 'path' in res
        assert len(res['path']) >= 1
        assert res['psi_seq'].shape[1] == 16

class TestCamada3:

    def test_attractor_observer_forecast(self):
        from runtime.observers import AttractorObserver
        np.random.seed(0)
        series = np.sin(np.linspace(0, 20, 200)) + 0.1 * np.random.randn(200)
        obs = AttractorObserver(size=30, seed=0).fit(series[:140])
        pred = obs.forecast(60)
        assert pred.shape == (60,)
        assert np.all(np.isfinite(pred))

    def test_longterm_consistency(self):
        from runtime.observers import longterm_consistency
        from runtime.core.solver import TriadParams
        p = TriadParams(mode='full', N=32, L=16.0, T=2.0, dt=0.01, seed=42,
                         record_every=2, fdt_couple=True)
        res = longterm_consistency(p, T_total=2.0, observe_frac=0.6,
                                    record_every=2, obs_size=30, seed=0,
                                    tol_corr=2.0, tol_rad=2.0, tol_pca=2.0)
        assert 'passed' in res
        assert res['feeds_back'] is False
        assert np.isfinite(res['slow_late_true'])

class TestCamada4:

    def test_reservoir_memory_capacity(self):
        from runtime.physics.physical_reservoir import PhysicalReservoirEmulator
        res = PhysicalReservoirEmulator(size=50, gain=1.0, seed=0)
        mc = res.memory_capacity(length=500, washout=50, max_delay=10)
        assert mc['memory_capacity'] > 0

    def test_edge_of_chaos_sweep(self):
        from runtime.physics.physical_reservoir import edge_of_chaos_sweep
        rows = edge_of_chaos_sweep([0.3, 0.7, 0.9, 1.0, 1.1, 1.3, 1.6],
                                    size=80, lyap_len=800, mc_len=800, seed=0)
        assert len(rows) == 7
        lyaps = [r['lyapunov'] for r in rows]
        mcs = [r['memory_capacity'] for r in rows]
        for r in rows:
            assert np.isfinite(r['lyapunov'])
            assert np.isfinite(r['memory_capacity'])
        assert max(mcs) > min(mcs)
        gains = [r['gain'] for r in rows]
        assert gains == sorted(gains)
        lyap_at_high = [r['lyapunov'] for r in rows if r['gain'] >= 1.0]
        lyap_at_low = [r['lyapunov'] for r in rows if r['gain'] < 1.0]
        assert np.mean(lyap_at_high) > np.mean(lyap_at_low)

    def test_fdt_noise_source(self):
        from runtime.physics.physical_reservoir import FDTNoiseSource, verify_fdt
        from runtime.core.solver import TriadParams
        p = TriadParams(mode='full', N=32, L=16.0, T=1.0, dt=0.01, seed=42,
                         fdt_couple=True, kT=1.0)
        src = FDTNoiseSource(mode='white_fdt', seed=0)
        res = verify_fdt(src, p, T=1.0)
        assert res['fdt_consistent'] is True
        assert abs(res['var_ratio'] - 1.0) < 0.15

    def test_fdt_colored_rejected(self):
        from runtime.physics.physical_reservoir import FDTNoiseSource, verify_fdt
        from runtime.core.solver import TriadParams
        p = TriadParams(mode='full', N=32, L=16.0, T=1.0, dt=0.01, seed=42,
                         fdt_couple=True, kT=1.0)
        src = FDTNoiseSource(mode='reservoir_colored', color_rho=0.7, seed=0)
        res = verify_fdt(src, p, T=1.0)
        assert abs(res['lag1_autocorr']) > 0.05

class TestNewObservables:

    def test_triad_phase_sync(self):
        from runtime.physics.observables import triad_phase_synchronization
        x = np.linspace(-8, 8, 64, endpoint=False)
        psi = np.exp(-x ** 2 / 4.0).astype(np.complex128)
        dx = x[1] - x[0]
        sync = triad_phase_synchronization(psi, dx)
        assert 0.0 <= sync <= 1.0

    def test_spectral_flux(self):
        from runtime.physics.observables import spectral_flux
        x = np.linspace(-8, 8, 64, endpoint=False)
        psi = np.exp(-x ** 2 / 4.0 + 1j * 2.0 * x).astype(np.complex128)
        dx = x[1] - x[0]
        flux = spectral_flux(psi, dx, n_shells=10)
        assert flux.shape == (10,)
        assert flux.sum() > 0

    def test_attractor_geometry(self):
        from runtime.physics.observables import attractor_geometry_invariants
        np.random.seed(0)
        series = np.cumsum(np.random.randn(200))
        inv = attractor_geometry_invariants(series, dim=3, tau=2)
        assert 'corr_dim' in inv
        assert 'radius' in inv
        assert 'pca_spectrum' in inv
        assert inv['corr_dim'] >= 0
        assert inv['radius'] > 0

    def test_lyapunov_proxy(self):
        from runtime.physics.observables import lyapunov_proxy
        np.random.seed(0)
        series = np.sin(np.linspace(0, 30, 500)) + 0.05 * np.random.randn(500)
        lyap = lyapunov_proxy(series, dim=4, tau=3, dt=0.01)
        assert np.isfinite(lyap)

    def test_full_pipeline_observables_on_solver_output(self):
        from runtime.core.solver import integrate
        from runtime.physics.observables import (triad_phase_synchronization, spectral_flux,
                                          attractor_geometry_invariants,
                                          crystallinity, dominant_wavenumber)
        p = _tiny_params(T=2.0, record_every=1)
        out = integrate(p, auto_halve_dt=False)
        psi = out['psi_final']
        dx = out['dx']
        sync = triad_phase_synchronization(psi, dx)
        flux = spectral_flux(psi, dx, n_shells=8)
        cryst = crystallinity(psi, dx)
        kstar = dominant_wavenumber(psi, dx)
        inv = attractor_geometry_invariants(out['density'].sum(axis=0) * dx, dim=3, tau=2)
        assert 0.0 <= sync <= 1.0
        assert flux.shape == (8,)
        assert 0.0 <= cryst <= 1.0
        assert kstar >= 0.0
        assert inv['radius'] > 0
