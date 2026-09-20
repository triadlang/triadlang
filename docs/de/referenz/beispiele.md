# Beispiele — Karte nach Domäne

Jeder Ordner unter `examples/` läuft mit `./triad run` (oder `check`).
Fang an, wo deine Neugier ist.

```
  basic ────────── die Sprache selbst (hier starten)
  interop ──────── Python-Brücken (numpy, flask, click…)
  solver ───────── SAT, Benches, native Parität
  quantum ──────── Schaltkreise (bell, grover, qft, vqe, shor15…)
  triad ────────── physikalische DSL (reg/ring/OBSERVE)
  ml ───────────── Tensoren, Netze, Training
  llm ──────────── Sprachmodelle auf triad
  crystalformer ── Crystal-Attention-Architekturen
  resonanceformer─ Resonanzarchitekturen
  crossdomain ──── Sagas s1..s6 (Neuron, Klima, Kosmos…)
  games ────────── 3D zum Spielen (memory, golf, predator-prey)
  blockchain ───── Verkettungs-Demo
  plotting ─────── Abbildungen
```

## Zehn Türen, zehn Dateien

| Wenn du willst… | Starte hiermit |
|---|---|
| hello world | `basic/hello.tri` |
| das Typsystem | `basic/types.tri` |
| Vektoren | `basic/broadcast_test.tri` |
| Python innen | `interop/boundary_numpy.tri` |
| eine Web-Antwort | `interop/boundary_flask_testclient.tri` |
| gelöste Logik | `solver/sat_demo.tri` |
| Verschränkung | `quantum/bell_state.tri` |
| ein beobachtetes Feld | `triad/anti_collapse.tri` |
| ein trainiertes Netz | `ml/nn_test.tri` |
| eine Saga | `crossdomain/s1_neuro_climate.tri` |

Alles schnell prüfen:

```sh
for f in examples/*/*.tri; do if ./triad check "$f" | grep -q "OK"; then echo "OK $f"; else echo "FAIL $f"; fi; done
```
