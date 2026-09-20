# Exempel — karta per domän

Varje mapp under `examples/` körs med `./triad run` (eller `check`).
Börja där din nyfikenhet är.

```
  basic ────────── själva språket (börja här)
  interop ──────── Python-bryggor (numpy, flask, click…)
  solver ───────── SAT, benches, nativ paritet
  quantum ──────── kretsar (bell, grover, qft, vqe, shor15…)
  triad ────────── fysikalisk DSL (reg/ring/OBSERVE)
  ml ───────────── tensorer, nät, träning
  llm ──────────── språkmodeller på triad
  crystalformer ── crystal-attention-arkitekturer
  resonanceformer─ resonansarkitekturer
  crossdomain ──── sagor s1..s6 (neuron, klimat, kosmos…)
  games ────────── 3D att leka med (memory, golf, predator-prey)
  blockchain ───── kedjningsdemo
  plotting ─────── figurer
```

## Tio dörrar, tio filer

| Om du vill… | Kör detta först |
|---|---|
| hello world | `basic/hello.tri` |
| typsystemet | `basic/types.tri` |
| vektorer | `basic/broadcast_test.tri` |
| Python inuti | `interop/boundary_numpy.tri` |
| ett webbsvar | `interop/boundary_flask_testclient.tri` |
| löst logik | `solver/sat_demo.tri` |
| sammanflätning | `quantum/bell_state.tri` |
| ett observerat fält | `triad/anti_collapse.tri` |
| ett tränat nät | `ml/nn_test.tri` |
| en saga | `crossdomain/s1_neuro_climate.tri` |

Kolla allt snabbt:

```sh
for f in examples/*/*.tri; do if ./triad check "$f" | grep -q "OK"; then echo "OK $f"; else echo "FAIL $f"; fi; done
```
