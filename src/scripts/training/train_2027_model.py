
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from runtime.core.solver import TriadParams, integrate
from runtime.ml.serialization import load_weights, save_weights
from runtime.ml.surrogate import (
    FidelityGate,
    TriadContext,
    accelerated_rollout,
    generate_pairs,
    train_surrogate,
)
from runtime.observers import longterm_consistency
from runtime.physics.observables import (
    attractor_geometry_invariants,
    crystallinity,
    dominant_wavenumber,
    lyapunov_proxy,
    memory_persistence,
    spectral_flux,
    triad_phase_synchronization,
)
from runtime.physics.physical_reservoir import FDTNoiseSource, chaos_threshold_sweep, verify_fdt
from stdlib.regimes import resolve_regime
from triad import ntri as np

SEED = 42
N_TRAIN = 64
N_EVAL = 128
CHUNK_STEPS = 20
N_CHUNKS_TRAIN = 30
N_CHUNKS_EVAL = 15
EPOCHS = 40
BATCH = 16
D_MODEL = 32
D_STATE = 32
N_BLOCKS = 2
LR = 2e-3
W_PHYS = 0.05
REGIMES_TRAIN = ['B0', 'anti_collapse', '_pure']
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def header(msg):
    print(f'\n{"=" * 60}')
    print(f'  {msg}')
    print(f'{"=" * 60}\n')

def camada0_solver():
    header('camada 0  solver nativo (ground )')
    results = {}
    for regime in ['B0', 'anti_collapse']:
        for mode in ['strang', 'exptrap']:
            p = resolve_regime(regime, seed=SEED, N=N_EVAL)
            overrides = dict(step_mode=mode, fdt_couple=True, kT=1.0,
                              record_every=4, T=min(p.T, 6.0))
            p = TriadParams(**{**p.__dict__, **overrides})

            def dyn_lambda(rho):
                r = np.asarray(rho)
                return float(0.3 + 0.4 * np.mean(r) / (np.max(r) + 1e-10))

            if mode == 'exptrap':
                p_trap = TriadParams(**{**p.__dict__, 'trap_lambda': dyn_lambda})
            else:
                p_trap = p

            t0 = time.perf_counter()
            out = integrate(p_trap, auto_halve_dt=False)
            dt_wall = time.perf_counter() - t0

            dx = out['dx']
            psi_f = out['psi_final']
            norm_t = out['density'].sum(axis=0) * dx
            cryst = crystallinity(psi_f, dx)
            kstar = dominant_wavenumber(psi_f, dx, k_min=2 * np.pi / p.L)
            sync = triad_phase_synchronization(psi_f, dx)
            flux = spectral_flux(psi_f, dx, n_shells=12)
            peak_flux_shell = int(np.argmax(flux[1:])) + 1 if len(flux) > 1 else 0
            rho_series = out['density'].sum(axis=0) * dx
            inv = attractor_geometry_invariants(rho_series, dim=4, tau=3)
            lyap = lyapunov_proxy(rho_series, dim=4, tau=3, dt=p.record_every * p.dt)
            mp = memory_persistence(rho_series, dt=p.record_every * p.dt)

            key = f'{regime}/{mode}'
            results[key] = {
                'wall_s': round(dt_wall, 4),
                'norm_range': [round(float(norm_t.min()), 6), round(float(norm_t.max()), 6)],
                'crystallinity': round(cryst, 4),
                'k_star': round(kstar, 4),
                'triad_sync': round(sync, 4),
                'peak_flux_shell': peak_flux_shell,
                'corr_dim': round(inv['corr_dim'], 4),
                'radius': round(inv['radius'], 4),
                'lyapunov': round(lyap, 6),
                'memory_persistence': round(mp, 4),
            }
            print(f'  {key:25s}  wall={dt_wall:.3f}s  norm=[{norm_t.min():.4f},{norm_t.max():.4f}]')
            print(f'  {"":25s}  cryst={cryst:.4f}  k*={kstar:.4f}  sync={sync:.4f}  lyap={lyap:.5f}')
            print(f'  {"":25s}  corr_dim={inv["corr_dim"]:.3f}  radius={inv["radius"]:.3f}  mem_persist={mp:.3f}')

    return results

def camada1_ssm():
    header('camada 1  TriadSSM learns solver dynamics')
    p = resolve_regime('B0', seed=SEED, N=N_TRAIN)
    p = TriadParams(**{**p.__dict__, 'T': 5.0, 'record_every': 1,
                        'fdt_couple': True, 'kT': 1.0})
    out = integrate(p, auto_halve_dt=False)
    density = out['density']
    n_t = density.shape[1]
    X_np = density[:, :-1].T
    Y_np = density[:, 1:].T
    d_in = X_np.shape[1]

    from runtime.ml.language import TriadtriadBlock
    from runtime.ml.nn import Adam
    from runtime.ml.tensor import tensor
    np.random.seed(SEED)
    model = TriadtriadBlock(d_model=d_in, N=64, n_memory=3)
    opt = Adam(model.parameters(), lr=LR)
    x_t = tensor(X_np[None, :, :])
    y_t = tensor(Y_np[None, :, :])
    history = []
    t0 = time.perf_counter()
    for ep in range(20):
        pred = model(x_t)
        loss = ((pred - y_t) ** 2).mean()
        model.zero_grad()
        loss.backward()
        opt.step()
        err = float(loss._data)
        history.append(err)
        if (ep + 1) % 5 == 0:
            print(f'  epoch {ep+1:3d}/20  loss={err:.6f}')
    dt_train = time.perf_counter() - t0

    from runtime.ml.tensor import no_grad
    with no_grad():
        pred_f = model(x_t)
    final_err = float(np.mean((pred_f._data[0] - Y_np) ** 2))
    rel_err = final_err / (np.mean(Y_np ** 2) + 1e-30)
    print(f'  trained in {dt_train:.2f}s  final_mse={final_err:.6f}  rel={rel_err:.4f}')

    os.makedirs(MODELS_DIR, exist_ok=True)
    ssm_path = os.path.join(MODELS_DIR, 'ssm_b0.weights')
    save_weights(model, ssm_path)
    print(f'  saved: {ssm_path}')

    return {'train_s': round(dt_train, 3), 'final_mse': final_err,
            'rel_err': round(rel_err, 4), 'epochs': 20}

def camada2_surrogate():
    header('camada 2  MNO surrogate multi-regime + fidelity gate')
    np.random.seed(SEED)
    t0 = time.perf_counter()
    X, Yc, ctx_map = generate_pairs(REGIMES_TRAIN, seeds=[SEED, SEED + 1],
                                     N=N_TRAIN, chunk_steps=CHUNK_STEPS,
                                     n_chunks=N_CHUNKS_TRAIN)
    dt_gen = time.perf_counter() - t0
    print(f'  generated {X.shape[0]} pairs from {REGIMES_TRAIN} in {dt_gen:.2f}s')
    print(f'  X shape={X.shape}  Y shape={Yc.shape}')

    p_b0 = resolve_regime('B0', seed=SEED, N=N_TRAIN)
    p_b0 = TriadParams(**{**p_b0.__dict__, 'fdt_couple': True, 'kT': 1.0})
    ctx_b0 = TriadContext.from_params(p_b0)
    chunk_T = CHUNK_STEPS * p_b0.dt

    t0 = time.perf_counter()
    model, history = train_surrogate(X, Yc, ctx_b0, chunk_T=chunk_T,
                                      epochs=EPOCHS, batch=BATCH, lr=LR,
                                      w_phys=W_PHYS, d_model=D_MODEL,
                                      d_state=D_STATE, n_blocks=N_BLOCKS,
                                      seed=SEED, verbose=True)
    dt_train = time.perf_counter() - t0
    print(f'  trained {EPOCHS} epochs in {dt_train:.2f}s')

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'mno_surrogate.weights')
    ckpt_path = os.path.join(MODELS_DIR, 'mno_surrogate.ckpt')
    save_weights(model, model_path)
    n_params = len(model.parameters())
    total_weights = sum(p._data.size for p in model.parameters())
    print(f'  saved model: {model_path}  ({n_params} params, {total_weights} weights)')
    size_kb = os.path.getsize(model_path) / 1024
    print(f'  file size: {size_kb:.1f} KB')

    from runtime.ml.surrogate import MNOSurrogate
    model_loaded = MNOSurrogate(d_model=D_MODEL, d_state=D_STATE,
                                 n_blocks=N_BLOCKS, n_memory=len(ctx_b0.nu))
    psi_test = X[0][None, :, :]
    from runtime.ml.tensor import no_grad
    with no_grad():
        pred_before = model.predict(Yc[0])
    load_weights(model_loaded, model_path)
    with no_grad():
        pred_after = model_loaded.predict(Yc[0])
    diff = float(np.mean(np.abs(pred_before - pred_after)))
    print(f'  load+predict check: max_diff={diff:.2e} (should be ~0)')
    assert diff < 1e-10, f'model load mismatch: {diff}'
    print('  model verified: save -> load -> predict matches')
    model = model_loaded

    meta_path = os.path.join(MODELS_DIR, 'mno_surrogate.meta.json')
    meta = {
        'regimes': REGIMES_TRAIN,
        'N_train': N_TRAIN, 'epochs': EPOCHS,
        'd_model': D_MODEL, 'd_state': D_STATE,
        'n_blocks': N_BLOCKS, 'n_memory': len(ctx_b0.nu),
        'chunk_steps': CHUNK_STEPS, 'chunk_T': chunk_T,
        'lr': LR, 'w_phys': W_PHYS, 'seed': SEED,
        'loss_last_l2': round(history[-1][0], 6) if history else None,
        'loss_last_phys': round(history[-1][1], 6) if history else None,
    }
    Path(meta_path).write_text(json.dumps(meta, indent=2))
    print(f'  saved: {meta_path}')

    gate = FidelityGate(l2_tol=0.3, norm_tol=0.3, kstar_tol=5.0,
                         cryst_tol=0.15, resid_factor=5.0)

    print('\n  accelerated rollout B0:')
    t0 = time.perf_counter()
    res_roll = accelerated_rollout(p_b0, T_total=2.0, surrogate=model, gate=gate,
                                    chunk_steps=CHUNK_STEPS, warmup_chunks=2,
                                    base_seed=SEED)
    dt_roll = time.perf_counter() - t0
    path = res_roll['path']
    n_sur = path.count('surrogate')
    n_nat = path.count('native')
    total_chunks = len(path)
    frac = n_sur / max(total_chunks, 1)
    print(f'    chunks={total_chunks}  surrogate={n_sur}  native={n_nat}  frac={frac:.2f}')
    print(f'    t_surrogate={res_roll["t_surrogate"]:.4f}s  t_native={res_roll["t_native"]:.4f}s')

    t0 = time.perf_counter()
    ref_roll = accelerated_rollout(p_b0, T_total=2.0, surrogate=model, gate=gate,
                                    chunk_steps=CHUNK_STEPS, warmup_chunks=2,
                                    base_seed=SEED)
    dt_ref = time.perf_counter() - t0

    speedup = None
    if res_roll['t_surrogate'] > 0:
        speedup = dt_ref / max(res_roll['t_surrogate'], 1e-9)
    print(f'    wall={dt_roll:.3f}s  reference_all_native={dt_ref:.3f}s')
    if speedup:
        print(f'    surrogate speedup factor: {speedup:.1f}x')

    dx = ctx_b0.dx
    psi_s = res_roll['psi_seq'][-1]
    psi_r = ref_roll['psi_seq'][-1]
    cryst_s = crystallinity(psi_s, dx)
    cryst_r = crystallinity(psi_r, dx)
    kstar_s = dominant_wavenumber(psi_s, dx, k_min=2 * np.pi / ctx_b0.L)
    kstar_r = dominant_wavenumber(psi_r, dx, k_min=2 * np.pi / ctx_b0.L)
    print(f'    cryst: sur={cryst_s:.4f} ref={cryst_r:.4f}  d={abs(cryst_s-cryst_r):.4f}')
    print(f'    k*:    sur={kstar_s:.4f} ref={kstar_r:.4f}  d={abs(kstar_s-kstar_r):.4f}')

    print('\n  accelerated rollout anti_collapse:')
    p_ac = resolve_regime('anti_collapse', seed=SEED, N=N_TRAIN)
    p_ac = TriadParams(**{**p_ac.__dict__, 'T': 4.0, 'fdt_couple': True, 'kT': 1.0})
    res_ac = accelerated_rollout(p_ac, T_total=2.0, surrogate=model, gate=gate,
                                  chunk_steps=CHUNK_STEPS, warmup_chunks=2,
                                  base_seed=SEED)
    path_ac = res_ac['path']
    n_sur_ac = path_ac.count('surrogate')
    n_nat_ac = path_ac.count('native')
    frac_ac = n_sur_ac / max(len(path_ac), 1)
    print(f'    chunks={len(path_ac)}  surrogate={n_sur_ac}  native={n_nat_ac}  frac={frac_ac:.2f}')
    ctx_ac = TriadContext.from_params(p_ac)
    psi_ac_s = res_ac['psi_seq'][-1]
    cryst_ac = crystallinity(psi_ac_s, ctx_ac.dx)
    kstar_ac = dominant_wavenumber(psi_ac_s, ctx_ac.dx, k_min=2 * np.pi / p_ac.L)
    print(f'    cryst={cryst_ac:.4f}  k*={kstar_ac:.4f}')

    return {
        'pairs': X.shape[0], 'gen_s': round(dt_gen, 3),
        'train_s': round(dt_train, 3), 'epochs': EPOCHS,
        'loss_first': round(history[0][0], 6) if history else None,
        'loss_last': round(history[-1][0], 6) if history else None,
        'phys_first': round(history[0][1], 6) if history else None,
        'phys_last': round(history[-1][1], 6) if history else None,
        'chunks': total_chunks, 'surrogate_hits': n_sur,
        'frac_surrogate': round(frac, 3),
        'cryst_sur': round(cryst_s, 4), 'cryst_ref': round(cryst_r, 4),
    }

def camada3_observer():
    header('camada 3  AttractorObserver passive forecast')
    p = resolve_regime('B0', seed=SEED, N=N_EVAL)
    p = TriadParams(**{**p.__dict__, 'T': 5.0, 'dt': 0.01,
                        'record_every': 2, 'fdt_couple': True, 'kT': 1.0})
    t0 = time.perf_counter()
    res = longterm_consistency(p, T_total=5.0, observe_frac=0.6,
                                record_every=2, obs_size=60, seed=SEED,
                                tol_corr=3.0, tol_rad=3.0, tol_pca=3.0)
    dt = time.perf_counter() - t0
    print(f'  ran in {dt:.2f}s')
    print(f'  observed={res["n_observed"]}  forecast={res["n_future"]}')
    print(f'  feeds_back={res["feeds_back"]}  consistent={res["consistent"]}  passed={res["passed"]}')
    print(f'  corr_dim: true={res["inv_true"]["corr_dim"]:.3f}  pred={res["inv_pred"]["corr_dim"]:.3f}')
    print(f'  radius:   true={res["inv_true"]["radius"]:.3f}  pred={res["inv_pred"]["radius"]:.3f}')
    print(f'  slow_late: true={res["slow_late_true"]:.4f}  pred={res["slow_late_pred"]:.4f}')
    print(f'  memory_persist: true={res["mp_true"]:.4f}  pred={res["mp_pred"]:.4f}')
    dist = res['dist']
    print(f'  distance: d_corr={dist["d_corr_dim"]:.4f}  d_rad={dist["d_radius"]:.4f}  d_pca={dist["d_pca"]:.4f}')
    return {
        'wall_s': round(dt, 3), 'passed': res['passed'],
        'feeds_back': res['feeds_back'],
        'corr_dim_true': round(res['inv_true']['corr_dim'], 4),
        'corr_dim_pred': round(res['inv_pred']['corr_dim'], 4),
        'radius_true': round(res['inv_true']['radius'], 4),
        'radius_pred': round(res['inv_pred']['radius'], 4),
    }

def camada4_reservoir():
    header('camada 4  Physical reservoir threshold-of-chaos + FDT')
    gains = np.concatenate([np.linspace(0.3, 0.9, 4), [0.95, 1.0, 1.05],
                             np.linspace(1.1, 2.0, 4)])
    t0 = time.perf_counter()
    rows = chaos_threshold_sweep(gains.tolist(), size=150, leak=0.9,
                                lyap_len=1500, mc_len=1500, seed=SEED)
    dt_sweep = time.perf_counter() - t0
    print(f'  sweep {len(gains)} gains in {dt_sweep:.2f}s')
    print(f'  {"gain":>6s}  {"lyapunov":>10s}  {"memory_cap":>11s}')
    near_zero_g = None
    peak_mc_g = None
    min_abs_lyap = float('inf')
    max_mc = -1
    for r in rows:
        g = r['gain']
        l = r['lyapunov']
        mc = r['memory_capacity']
        print(f'  {g:6.2f}  {l:10.5f}  {mc:11.3f}')
        if abs(l) < min_abs_lyap:
            min_abs_lyap = abs(l)
            near_zero_g = g
        if mc > max_mc:
            max_mc = mc
            peak_mc_g = g
    print(f'\n  near-zero lyapunov at gain={near_zero_g:.2f}')
    print(f'  peak memory_capacity at gain={peak_mc_g:.2f}')
    print(f'  delta = {abs(near_zero_g - peak_mc_g):.2f}')

    print('\n  FDT noise verification:')
    p = resolve_regime('B0', seed=SEED, N=N_EVAL)
    p = TriadParams(**{**p.__dict__, 'T': 5.0, 'fdt_couple': True, 'kT': 1.0})
    src_white = FDTNoiseSource(mode='white_fdt', seed=SEED)
    res_white = verify_fdt(src_white, p, T=5.0)
    print(f'    white:   var_ratio={res_white["var_ratio"]:.4f}  lag1={res_white["lag1_autocorr"]:.4f}  fdt_ok={res_white["fdt_consistent"]}')
    src_color = FDTNoiseSource(mode='reservoir_colored', color_rho=0.7, seed=SEED)
    res_color = verify_fdt(src_color, p, T=5.0)
    print(f'    colored: var_ratio={res_color["var_ratio"]:.4f}  lag1={res_color["lag1_autocorr"]:.4f}  fdt_ok={res_color["fdt_consistent"]}')

    return {
        'sweep_s': round(dt_sweep, 3), 'n_gains': len(gains),
        'near_zero_gain': round(near_zero_g, 3),
        'peak_mc_gain': round(peak_mc_g, 3),
        'fdt_white': res_white['fdt_consistent'],
        'fdt_white_ratio': round(res_white['var_ratio'], 4),
        'fdt_colored_lag1': round(res_color['lag1_autocorr'], 4),
    }

def main():
    header('triad 2027 model  triad stack')
    print(f'  regimes: {REGIMES_TRAIN}')
    print(f'  N_train={N_TRAIN}  N_eval={N_EVAL}  epochs={EPOCHS}')
    print(f'  d_model={D_MODEL}  d_state={D_STATE}  n_blocks={N_BLOCKS}')
    print(f'  loss = L2(B0) + {W_PHYS} * triad_residual  (zero crystallinity terms)')

    t_total = time.perf_counter()
    report = {}

    report['camada0'] = camada0_solver()
    report['camada1'] = camada1_ssm()
    report['camada2'] = camada2_surrogate()
    report['camada3'] = camada3_observer()
    report['camada4'] = camada4_reservoir()

    dt_total = time.perf_counter() - t_total
    report['total_s'] = round(dt_total, 2)

    header('report')
    print(json.dumps(report, indent=2))
    print(f'\n  total wall time: {dt_total:.1f}s')

    f_path = os.path.join(MODELS_DIR, 'report.json')
    with open(f_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'  saved to {f_path}')

if __name__ == '__main__':
    main()
