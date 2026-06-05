"""Fase 4: co-design com reservoir fisico. Verifica a borda do caos (Lyapunov ~ 0, onde a
capacidade de memoria tem pico) e a consistencia FDT do ruido fisico hardware-in-the-loop.

Veredito do relatorio: ruido fisico verificadamente FDT-consistente E reservoir na borda do
caos (Lyapunov maximo ~ 0).
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from runtime.core.solver import TriadParams
from runtime.physics.physical_reservoir import (PhysicalReservoirEmulator, FDTNoiseSource,
                                        verify_fdt, edge_of_chaos_sweep)
from stdlib.regimes import resolve_regime

print('=' * 72)
print('  FASE 4  co-design reservoir fisico: borda do caos + FDT')
print('=' * 72)

gains = [0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]
print('\n[1] varredura: Lyapunov maximo e capacidade de memoria vs gain')
print(f'    {"gain":>6}{"lyapunov":>12}{"mem_capacity":>16}')
t0 = time.perf_counter()
rows = edge_of_chaos_sweep(gains, size=300, leak=0.9, seed=0)
for r in rows:
    mark = '  <- borda do caos' if abs(r['lyapunov']) < 0.05 else ''
    print(f'    {r["gain"]:>6.2f}{r["lyapunov"]:>12.4f}{r["memory_capacity"]:>16.2f}{mark}')
mc_peak = max(rows, key=lambda r: r['memory_capacity'])
print(f'    pico de MC em gain={mc_peak["gain"]:.2f}  (Lyapunov={mc_peak["lyapunov"]:.4f})')
print(f'    ({time.perf_counter() - t0:.1f}s)')

p = resolve_regime('B0', N=128)
print('\n[2] FDT do ruido fisico hardware-in-the-loop (solver + noise_provider)')
src_ok = FDTNoiseSource(mode='white_fdt', seed=1)
v_ok = verify_fdt(src_ok, p, T=20.0)
src_bad = FDTNoiseSource(mode='reservoir_colored', color_rho=0.7, seed=1)
v_bad = verify_fdt(src_bad, p, T=20.0)
print(f'    {"fonte":<20}{"var_ratio":>12}{"lag1":>10}{"FDT ok":>9}')
print(f'    {"white_fdt":<20}{v_ok["var_ratio"]:>12.4f}{v_ok["lag1_autocorr"]:>10.4f}{str(v_ok["fdt_consistent"]):>9}')
print(f'    {"reservoir_colored":<20}{v_bad["var_ratio"]:>12.4f}{v_bad["lag1_autocorr"]:>10.4f}{str(v_bad["fdt_consistent"]):>9}')
print('    (colored preserva variancia mas correlaciona no tempo: viola a FDT branca,')
print('     exatamente o artefato que o guardrail manda evitar)')

print('\n[3] composicao com a Fase 1 (fdt_couple): eta fisico honra f_FDT = 2*Gamma*dx*kT/hbar')
p_coupled = TriadParams(**{**p.__dict__, 'fdt_couple': True, 'kT': 0.16})
src_c = FDTNoiseSource(mode='white_fdt', seed=2)
v_c = verify_fdt(src_c, p_coupled, T=20.0)
print(f'    var_ratio={v_c["var_ratio"]:.4f}  lag1={v_c["lag1_autocorr"]:.4f}  '
      f'FDT ok={v_c["fdt_consistent"]}  norma_final={v_c["final_norm"]:.4f}')

edge_ok = abs(mc_peak['lyapunov']) < 0.1
fdt_ok = v_ok['fdt_consistent'] and v_c['fdt_consistent'] and not v_bad['fdt_consistent']
print('\n' + '=' * 72)
print(f'  borda do caos verificada (Lyapunov~0 no pico de MC): {edge_ok}')
print(f'  ruido fisico FDT-consistente (e artefato detectado): {fdt_ok}')
print(f'  Avanco da Fase 4: {edge_ok and fdt_ok}')
print('  Se o hardware nao satisfizer a FDT, o relatorio restringe a Camada 4 a readout')
print('  puro: jamais acoplar eta nao-FDT.')
print('=' * 72)
