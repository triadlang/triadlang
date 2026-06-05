"""Fase 2: surrogate MNO contra o B0, com gate de fidelidade e fallback ao solver nativo.

Pipeline:
  1. Gera pares (Psi(t_k), Psi(t_{k+1})) do solver nativo (ground truth), regime B0.
  2. Treina o surrogate: loss = L2(dados B0) + residuo da equacao Triad. Sem termo de
     crystallinity, sem residuo de phase-field. Cristalizacao continua emergente.
  3. Referencia 100% nativa para o gate.
  4. Rollout acelerado: por chunk tenta o surrogate; se o residuo Triad exceder o piso do
     solver nativo, cai de volta no nativo (Camada 0 e sempre o fallback).
  5. Gate de fidelidade + veredito da Fase 2.

Treino e em N pequeno para caber em CPU; em producao treina-se no N alvo.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from runtime.ml.surrogate import (generate_pairs, train_surrogate, FidelityGate,
                               accelerated_rollout, native_reference,
                               triad_residual_full, TriadContext)
from stdlib.regimes import resolve_regime

N = 64
CHUNK = 20            
SEEDS = [0, 1, 2]
NCHUNKS_TRAIN = 8
EPOCHS = 20

print('=' * 72)
print(f'  FASE 2  surrogate MNO vs B0   (N={N}, chunk={CHUNK} passos)')
print('=' * 72)

t0 = time.perf_counter()
X, Yc, ctx_map = generate_pairs(['B0'], SEEDS, N=N, chunk_steps=CHUNK,
                                n_chunks=NCHUNKS_TRAIN)
ctx = ctx_map['B0']
chunk_T = CHUNK * resolve_regime('B0', N=N).dt
print(f'\n[1] dados B0: {X.shape[0]} pares, N={N}, dt_chunk={chunk_T:.3f}  '
      f'({time.perf_counter() - t0:.1f}s)')

print('\n[2] treino (loss = L2 + residuo Triad local; nenhum termo de crystallinity)')
model, hist = train_surrogate(X, Yc, ctx, chunk_T, epochs=EPOCHS, batch=16,
                              lr=2e-3, w_phys=0.05, d_model=32, d_state=32,
                              n_blocks=2, verbose=True)

p = resolve_regime('B0', N=N)
T_total = chunk_T * 10
ref = native_reference(p, T_total, chunk_steps=CHUNK, base_seed=7)
print(f'\n[3] referencia nativa: {len(ref["t"])} snapshots, T={T_total:.2f}, '
      f'{ref["elapsed"]:.2f}s')

gate = FidelityGate()
acc = accelerated_rollout(p, T_total, model, gate, chunk_steps=CHUNK,
                          warmup_chunks=1, base_seed=7)
print(f'\n[4] rollout acelerado: {acc["n_surrogate"]}/{acc["n_chunks"]} chunks '
      f'pelo surrogate (frac={acc["frac_surrogate"]:.2f})')
print(f'    caminho por chunk: {acc["path"]}')

verdict = gate.evaluate(acc['psi_seq'], ref['psi_seq'], ref['t'], ctx)
print('\n[5] gate de fidelidade (surrogate-acelerado vs nativo):')
for kdx in ('l2_rel', 'norm_dev', 'dk_star', 'd_crystallinity',
            'residual_sur', 'residual_ref'):
    print(f'    {kdx:<18}{verdict[kdx]:.4e}')
print(f'    gate passou: {verdict["passed"]}')

nat_per_chunk = ref['elapsed'] / max(acc['n_chunks'], 1)
sur_per_chunk = (acc['t_surrogate'] / acc['n_surrogate']) if acc['n_surrogate'] else float('nan')
ratio = nat_per_chunk / sur_per_chunk if sur_per_chunk == sur_per_chunk and sur_per_chunk > 0 else float('nan')
print('\n[speedup] medido, nao alegado (ref. cetica: surrogate pode ser mais lento)')
print(f'    nativo  por chunk: {nat_per_chunk * 1e3:.2f} ms')
print(f'    surrogate por chunk: {sur_per_chunk * 1e3:.2f} ms')
print(f'    razao potencial (se validar por amostragem, nao a cada chunk): {ratio:.2f}x')

print('\n' + '=' * 72)
ok_obs = (verdict['dk_star'] <= gate.kstar_tol
          and verdict['d_crystallinity'] <= gate.cryst_tol)
ok_res = verdict['residual_sur'] <= gate.resid_factor * verdict['residual_ref'] + 1e-30
print('  Avanco da Fase 2 (relatorio): speedup >= 10x  E  observaveis dentro da')
print('  tolerancia de ruido  E  residuo abaixo do limiar em rollout longo.')
print(f'    observaveis ok: {ok_obs}   residuo ok: {ok_res}   speedup>=10x: {ratio >= 10 if ratio == ratio else False}')
print('  Se o speedup nao chega a 10x ou o gate falha em regime aberto: o relatorio')
print('  manda abandonar a Camada 2 e manter so o ganho de discretizacao da Fase 1.')
print('=' * 72)
