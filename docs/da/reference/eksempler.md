# Eksempler — kort per domæne

Hver mappe under `examples/` kører med `./triad run` (eller `check`).
Start hvor din nysgerrighed er.

```
  basic ────────── selve sproget (start her)
  interop ──────── Python-broer (numpy, flask, click…)
  solver ───────── SAT, benches, nativ paritet
  quantum ──────── kredsløb (bell, grover, qft, vqe, shor15…)
  triad ────────── fysisk DSL (reg/ring/OBSERVE)
  ml ───────────── tensorer, net, træning
  llm ──────────── sprogmodeller på triad
  crystalformer ── crystal-attention-arkitekturer
  resonanceformer─ resonansarkitekturer
  crossdomain ──── sagaer s1..s6 (neuron, klima, kosmos…)
  games ────────── 3D at lege med (memory, golf, predator-prey)
  blockchain ───── kædningsdemo
  plotting ─────── figurer
```

## Ti døre, ti filer

| Hvis du vil… | Kør dette først |
|---|---|
| hello world | `basic/hello.tri` |
| typesystemet | `basic/types.tri` |
| vektorer | `basic/broadcast_test.tri` |
| Python indeni | `interop/boundary_numpy.tri` |
| et websvar | `interop/boundary_flask_testclient.tri` |
| løst logik | `solver/sat_demo.tri` |
| sammenfiltring | `quantum/bell_state.tri` |
| et observeret felt | `triad/anti_collapse.tri` |
| et trænet net | `ml/nn_test.tri` |
| en saga | `crossdomain/s1_neuro_climate.tri` |

Tjek alt hurtigt:

```sh
for f in examples/*/*.tri; do if ./triad check "$f" | grep -q "OK"; then echo "OK $f"; else echo "FAIL $f"; fi; done
```
