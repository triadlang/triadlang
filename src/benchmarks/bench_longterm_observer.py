"""Fase 3: observador passivo de longo prazo (PhyxMamba/PINNMamba-like) sobre o reservoir B0.

Le a janela inicial da trajetoria, preve o resto, e compara INVARIANTES de atrator (nao a
trajetoria ponto-a-ponto). Veredito do relatorio: invariantes consistentes com B0 E nenhuma
realimentacao na equacao.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from runtime.core.solver import TriadParams, integrate
from runtime.observers import (longterm_consistency, AttractorObserver,
                               slow_memory_series, density_scalar_series,
                               OBSERVER_FEEDS_BACK)
from runtime.physics.observables import memory_persistence, slow_state_late_mean
from stdlib.regimes import resolve_regime

N = 128
T_TOTAL = 60.0
RECORD_EVERY = 2

print('=' * 72)
print(f'  FASE 3  observador passivo de longo prazo vs B0   (N={N}, T={T_TOTAL})')
print('=' * 72)

p = resolve_regime('B0', N=N)

t0 = time.perf_counter()
res = longterm_consistency(p, T_TOTAL, observe_frac=0.6, record_every=RECORD_EVERY,
                           kind='participation', embed_dim=5, tau=3, obs_size=200, seed=0)
dt_run = time.perf_counter() - t0

print(f'\n[1] janela observada: {res["n_observed"]} frames | futuro previsto: {res["n_future"]} frames')
print(f'    ({dt_run:.2f}s)')

print('\n[2] invariantes de atrator  (verdadeiro vs previsto)')
it, ip = res['inv_true'], res['inv_pred']
print(f'    corr_dim   true={it["corr_dim"]:.3f}   pred={ip["corr_dim"]:.3f}   '
      f'|d|={res["dist"]["d_corr_dim"]:.3f}')
print(f'    radius     true={it["radius"]:.3f}   pred={ip["radius"]:.3f}   '
      f'd_rel={res["dist"]["d_radius"]:.3f}')
print(f'    pca[:3]    true={[round(v,3) for v in it["pca_spectrum"][:3]]}   '
      f'pred={[round(v,3) for v in ip["pca_spectrum"][:3]]}   L1={res["dist"]["d_pca"]:.3f}')

print('\n[3] persistencia de memoria e estado lento')
print(f'    memory_persistence  true={res["mp_true"]:.3f}   pred={res["mp_pred"]:.3f}')
print(f'    slow_state_late     true={res["slow_late_true"]:.4f}   pred={res["slow_late_pred"]:.4f}')

out_y = integrate(TriadParams(**{**p.__dict__, 'T': T_TOTAL, 'record_every': RECORD_EVERY}),
                  auto_halve_dt=False, record_y=True)
slow = slow_memory_series(out_y['y_traj'])
print(f'    slow_memory_series (campo y mais lento): late_mean='
      f'{slow_state_late_mean(slow):.4f}  persistence='
      f'{memory_persistence(slow, dt=RECORD_EVERY * p.dt):.3f}')

print('\n[4] passividade (guardrail da Camada 3)')
print(f'    OBSERVER_FEEDS_BACK = {OBSERVER_FEEDS_BACK}   (deve ser False)')
assert OBSERVER_FEEDS_BACK is False, 'Camada 3 violou a passividade'

print('\n' + '=' * 72)
print(f'  consistencia de invariantes: {res["consistent"]}')
print(f'  sem realimentacao: {not res["feeds_back"]}')
print(f'  Avanco da Fase 3: {res["passed"]}')
print('  (invariantes consistentes com B0 E nenhuma realimentacao na equacao)')
print('=' * 72)
