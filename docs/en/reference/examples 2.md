# Examples — map by domain

Every folder under `examples/` runs with `./triad run` (or `check`).
Start where your curiosity is.

```
  basic ────────── the language itself (start here)
  interop ──────── Python bridges (numpy, flask, click…)
  solver ───────── SAT, benches, native parity
  quantum ──────── circuits (bell, grover, qft, vqe, shor15…)
  triad ────────── physical DSL (reg/ring/OBSERVE)
  ml ───────────── tensors, nets, training
  llm ──────────── language models on triad
  crystalformer ── crystal attention architectures
  resonanceformer─ resonance architectures
  crossdomain ──── s1..s6 sagas (neuron, climate, cosmos…)
  games ────────── 3D play (memory, golf, predator-prey)
  blockchain ───── chaining demo
  plotting ─────── figures
```

## Ten doors, ten files

| If you want… | Run this first |
|---|---|
| hello world | `basic/hello.tri` |
| the type system | `basic/types.tri` |
| vectors | `basic/broadcast_test.tri` |
| Python inside | `interop/boundary_numpy.tri` |
| a web answer | `interop/boundary_flask_testclient.tri` |
| logic solved | `solver/sat_demo.tri` |
| entanglement | `quantum/bell_state.tri` |
| a field observed | `triad/anti_collapse.tri` |
| a net trained | `ml/nn_test.tri` |
| a saga | `crossdomain/s1_neuro_climate.tri` |

Check everything fast:

```sh
for f in examples/*/*.tri; do if ./triad check "$f" | grep -q "OK"; then echo "OK $f"; else echo "FAIL $f"; fi; done
```
