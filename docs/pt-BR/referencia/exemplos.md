# Exemplos — mapa por domínio

Cada pasta de `examples/` roda com `./triad run` (ou `check`).
Comece onde sua curiosidade estiver.

```
  basic ────────── a linguagem em si (comece aqui)
  interop ──────── pontes Python (numpy, flask, click…)
  solver ───────── SAT, benches, paridade nativa
  quantum ──────── circuitos (bell, grover, qft, vqe, shor15…)
  triad ────────── DSL física (reg/ring/OBSERVE)
  ml ───────────── tensores, redes, treino
  llm ──────────── modelos de linguagem em triad
  crystalformer ── arquiteturas crystal attention
  resonanceformer─ arquiteturas de ressonância
  crossdomain ──── sagas s1..s6 (neurônio, clima, cosmos…)
  games ────────── 3D para brincar (memory, golf, predator-prey)
  blockchain ───── demo de encadeamento
  plotting ─────── figuras
```

## Dez portas, dez arquivos

| Se você quer… | Rode isto primeiro |
|---|---|
| hello world | `basic/hello.tri` |
| o sistema de tipos | `basic/types.tri` |
| vetores | `basic/broadcast_test.tri` |
| Python dentro | `interop/boundary_numpy.tri` |
| uma resposta web | `interop/boundary_flask_testclient.tri` |
| lógica resolvida | `solver/sat_demo.tri` |
| emaranhamento | `quantum/bell_state.tri` |
| um campo observado | `triad/anti_collapse.tri` |
| uma rede treinada | `ml/nn_test.tri` |
| uma saga | `crossdomain/s1_neuro_climate.tri` |

Confira tudo rápido:

```sh
for f in examples/*/*.tri; do if ./triad check "$f" | grep -q "OK"; then echo "OK $f"; else echo "FAIL $f"; fi; done
```
