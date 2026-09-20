# Ejemplos — mapa por dominio

Cada carpeta de `examples/` corre con `./triad run` (o `check`).
Empieza donde esté tu curiosidad.

```
  basic ────────── el lenguaje en sí (empieza aquí)
  interop ──────── puentes Python (numpy, flask, click…)
  solver ───────── SAT, benches, paridad nativa
  quantum ──────── circuitos (bell, grover, qft, vqe, shor15…)
  triad ────────── DSL física (reg/ring/OBSERVE)
  ml ───────────── tensores, redes, entreno
  llm ──────────── modelos de lenguaje en triad
  crystalformer ── arquitecturas crystal attention
  resonanceformer─ arquitecturas de resonancia
  crossdomain ──── sagas s1..s6 (neurona, clima, cosmos…)
  games ────────── 3D para jugar (memory, golf, predator-prey)
  blockchain ───── demo de encadenado
  plotting ─────── figuras
```

## Diez puertas, diez archivos

| Si quieres… | Corre esto primero |
|---|---|
| hello world | `basic/hello.tri` |
| el sistema de tipos | `basic/types.tri` |
| vectores | `basic/broadcast_test.tri` |
| Python dentro | `interop/boundary_numpy.tri` |
| una respuesta web | `interop/boundary_flask_testclient.tri` |
| lógica resuelta | `solver/sat_demo.tri` |
| entrelazamiento | `quantum/bell_state.tri` |
| un campo observado | `triad/anti_collapse.tri` |
| una red entrenada | `ml/nn_test.tri` |
| una saga | `crossdomain/s1_neuro_climate.tri` |

Chequea todo rápido:

```sh
for f in examples/*/*.tri; do if ./triad check "$f" | grep -q "OK"; then echo "OK $f"; else echo "FAIL $f"; fi; done
```
