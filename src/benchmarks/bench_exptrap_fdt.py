import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from runtime.core.solver import TriadParams, integrate
from stdlib.regimes import resolve_regime
from runtime.physics.observables import crystallinity, dominant_wavenumber, norm as field_norm

N = 256
T = 20.0
SEED = 0
k_min = 2 * np.pi / 32.0

def base_B0(**over):
    p = resolve_regime('B0', N=N)
    d = {**p.__dict__, 'T': T, 'seed': SEED}
    d.update(over)
    return TriadParams(**d)

def norm_drift(out):
    n_t = out['density'].sum(axis=0) * out['dx']
    return float(np.max(np.abs(n_t - n_t[0]))), float(n_t[0])

def final_obs(out):
    psi = out['psi_final']
    dx = out['dx']
    return dict(k_star=dominant_wavenumber(psi, dx, k_min=k_min),
                crystallinity=crystallinity(psi, dx),
                norm=field_norm(psi, dx))

def timed(p):
    t0 = time.perf_counter()
    out = integrate(p, auto_halve_dt=False)
    return out, time.perf_counter() - t0

print('=' * 72)
print(f'  FASE 1  exp-trapezoidal + FDT   (B0, N={N}, T={T})')
print('=' * 72)

print('\n[1] Ablacao conservativa declarada (Gamma=0, f_FDT=0, eta=0)')
print(f'    {"esquema":<14}{"max|norm-norm0|":>20}{"norm0":>14}')
for mode in ('strang', 'exptrap'):
    p = base_B0(Gamma=0.0, f_FDT=0.0, step_mode=mode)
    out, _ = timed(p)
    drift, n0 = norm_drift(out)
    flag = 'ok' if drift < 1e-6 else 'FALHA'
    print(f'    {mode:<14}{drift:>20.3e}{n0:>14.8f}   {flag}')

print('\n[2] B0 full (P3 ativo), mesmo dt e seed: observaveis emergentes')
ref = None
print(f'    {"esquema":<14}{"k_star":>10}{"crystallinity":>16}{"norm":>12}{"t (s)":>10}')
for mode in ('strang', 'exptrap'):
    p = base_B0(step_mode=mode)
    out, dt_s = timed(p)
    o = final_obs(out)
    if mode == 'strang':
        ref = o
    print(f'    {mode:<14}{o["k_star"]:>10.4f}{o["crystallinity"]:>16.4f}{o["norm"]:>12.5f}{dt_s:>10.3f}')
dk = abs(final_obs(timed(base_B0(step_mode='exptrap'))[0])['k_star'] - ref['k_star'])
print(f'    desvio k_star (exptrap vs strang): {dk:.4e}')

print('\n[3] Passo maior em modo conservativo (referencia = strang dt/2)')
dt0 = base_B0().dt
ref_out, _ = timed(base_B0(Gamma=0.0, f_FDT=0.0, step_mode='strang', dt=dt0 / 2))
ref_k = final_obs(ref_out)['k_star']
for label, mode, dt in (('strang  dt', 'strang', dt0),
                        ('exptrap dt', 'exptrap', dt0),
                        ('exptrap 2dt', 'exptrap', 2 * dt0)):
    out, _ = timed(base_B0(Gamma=0.0, f_FDT=0.0, step_mode=mode, dt=dt))
    o = final_obs(out)
    err = abs(o['k_star'] - ref_k)
    print(f'    {label:<14} dt={dt:<7.4f} k_star={o["k_star"]:.4f}  |dk vs ref|={err:.4e}')

print('\n[4] FDT por construcao: f_FDT = 2*Gamma*dx*kT/hbar (acoplado a Gamma)')
dx = 32.0 / N
for kT in (0.05, 0.16, 0.5):
    p = base_B0(fdt_couple=True, kT=kT, step_mode='exptrap')
    out, _ = timed(p)
    o = final_obs(out)
    f_implied = 2.0 * p.Gamma * dx * kT / p.hbar
    print(f'    kT={kT:<5} -> f_FDT_efetivo={f_implied:.5f}  crystallinity={o["crystallinity"]:.4f}  norm={o["norm"]:.4f}')

print('\n' + '=' * 72)
print('  Avanco da Fase 1 = [1] ambos < 1e-6  E  [2] desvio k_star dentro da tolerancia.')
print('=' * 72)
