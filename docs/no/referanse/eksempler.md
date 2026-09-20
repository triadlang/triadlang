# Eksempler — kart per domene

Hver mappe under `examples/` kjører med `./triad run` (eller `check`).
Start der nysgjerrigheten din er.

```
  basic ────────── selve språket (start her)
  interop ──────── Python-broer (numpy, flask, click…)
  solver ───────── SAT, bencher, nativ paritet
  quantum ──────── kretser (bell, grover, qft, vqe, shor15…)
  triad ────────── fysisk DSL (reg/ring/OBSERVE)
  ml ───────────── tensorer, nett, trening
  llm ──────────── språkmodeller på triad
  crystalformer ── crystal-attention-arkitekturer
  resonanceformer─ resonansarkitekturer
  crossdomain ──── sagaer s1..s6 (nevron, klima, kosmos…)
  games ────────── 3D å leke med (memory, golf, predator-prey)
  blockchain ───── kjedingsdemo
  plotting ─────── figurer
```

## Ti dører, ti filer

| Hvis du vil… | Kjør dette først |
|---|---|
| hello world | `basic/hello.tri` |
| typesystemet | `basic/types.tri` |
| vektorer | `basic/broadcast_test.tri` |
| Python inni | `interop/boundary_numpy.tri` |
| et nettsvar | `interop/boundary_flask_testclient.tri` |
| løst logikk | `solver/sat_demo.tri` |
| sammenfiltring | `quantum/bell_state.tri` |
| et observert felt | `triad/anti_collapse.tri` |
| et trent nett | `ml/nn_test.tri` |
| en saga | `crossdomain/s1_neuro_climate.tri` |

Sjekk alt fort:

```sh
for f in examples/*/*.tri; do if ./triad check "$f" | grep -q "OK"; then echo "OK $f"; else echo "FAIL $f"; fi; done
```
