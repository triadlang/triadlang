# triad.* — the 39 modules

Import with `import triad.NAME [as x];`. One line each; details live
in `examples/` and the source.

## Numbers and fields

| Module | It is |
|---|---|
| `triad.ntri` | numpy-manners arrays: `array zeros ones linspace arange sin cos exp sqrt sum max min mean`… |
| `triad.tensor` | autograd tensors: `tensor randn arange relu softmax bmm transpose layer_norm`… + device (`set_device cuda_available sync`) |
| `triad.equation_runtime` | evolving the integral equation directly |

## Learning

| Module | It is |
|---|---|
| `triad.nn` | layers: `triad Conv1d Conv2d Embedding MultiHeadAttention Transformer TriadLM Adam SGD…` + memories (`CrystalMemory AttractorMemory…`) |
| `triad.net` | ready networks (see `examples/crystalformer/`) |
| `triad.data` / `losses` / `metrics` | datasets, loss functions, scores |
| `triad.train` / `inference` / `generate` | training loops, running, sampling |
| `triad.llm` | language-model blocks (`CausalSelfAttention KVCache TriadLM…`) |
| `triad.native_qwen35` | native Qwen-3.5 bridge |
| `triad.checkpoint` / `amp` | saving models / mixed precision |
| `triad.solver_grad_bridge` / `param_bridge` | solver↔gradient bridges |

## Physics and quanta

| Module | It is |
|---|---|
| `triad.kernel` | `prepare run extract` — regimes to observables |
| `triad.sat` | `SATInstance solve_sat` — logic as interference |
| `triad.qubits` | `TriadCircuit h cnot run` — circuits and counts |
| `triad.qalgo` | quantum algorithms (grover, qft, qpe…) |
| `triad.equilibrium` | equilibrium finding |
| `triad.energy` / `atoms` | energy and atom bookkeeping |
| `triad.calibrate` / `curriculum` | calibration and curricula |
| `triad.causal` / `consistency` / `steering` | traces, checks, steering |
| `triad.consciousness` | experimental cognitive sector |

## System and world

| Module | It is |
|---|---|
| `triad` | the base module itself |
| `triad.engine3d` | 3D engine (`./triad play`) |
| `triad.memory` | crystal memory (`./triad memory …`) |
| `triad.chain` | chaining (see `examples/blockchain/`) |
| `triad.http` / `fs` / `time` / `re` | triad-flavored system access |
| `triad.viz` | visualization helpers |

```
  math on arrays ──► ntri → tensor → nn → train
  physics ────────► kernel → sat / qubits / equilibrium
  worlds ─────────► engine3d → memory → chain
```
