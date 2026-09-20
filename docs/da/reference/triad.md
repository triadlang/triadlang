# triad.* — de 39 moduler

Importer med `import triad.NAVN [as x];`. Én linje hver; detaljer i
`examples/` og kilden.

## Tal og felter

| Modul | Det er |
|---|---|
| `triad.ntri` | arrays med numpy-manerer: `array zeros ones linspace arange sin cos exp sqrt sum max min mean`… |
| `triad.tensor` | tensorer med autograd: `tensor randn arange relu softmax bmm transpose layer_norm`… + enhed (`set_device cuda_available sync`) |
| `triad.equation_runtime` | udvikl integralligningen direkte |

## Læring

| Modul | Det er |
|---|---|
| `triad.nn` | lag: `triad Conv1d Conv2d Embedding MultiHeadAttention Transformer TriadLM Adam SGD…` + hukommelser (`CrystalMemory AttractorMemory…`) |
| `triad.net` | færdige net (se `examples/crystalformer/`) |
| `triad.data` / `losses` / `metrics` | datasæt, tab, mål |
| `triad.train` / `inference` / `generate` | træn, kør, sample |
| `triad.llm` | LM-blokke (`CausalSelfAttention KVCache TriadLM…`) |
| `triad.native_qwen35` | nativ Qwen-3.5-bro |
| `triad.checkpoint` / `amp` | gem modeller / blandet præcision |
| `triad.solver_grad_bridge` / `param_bridge` | solver↔gradient-broer |

## Fysik og kvanter

| Modul | Det er |
|---|---|
| `triad.kernel` | `prepare run extract` — regimer til observabler |
| `triad.sat` | `SATInstance solve_sat` — logik som interferens |
| `triad.qubits` | `TriadCircuit h cnot run` — kredsløb og counts |
| `triad.qalgo` | kvantealgoritmer (grover, qft, qpe…) |
| `triad.equilibrium` | find ligevægte |
| `triad.energy` / `atoms` | energi- og atomregnskab |
| `triad.calibrate` / `curriculum` | kalibrering og læreplaner |
| `triad.causal` / `consistency` / `steering` | spor, kontroller, styring |
| `triad.consciousness` | eksperimentel kognitiv sektor |

## System og verden

| Modul | Det er |
|---|---|
| `triad` | selve basismodulet |
| `triad.engine3d` | 3D-motor (`./triad play`) |
| `triad.memory` | krystalhukommelse (`./triad memory …`) |
| `triad.chain` | kædning (se `examples/blockchain/`) |
| `triad.http` / `fs` / `time` / `re` | systemadgang med triad-accent |
| `triad.viz` | visualiseringshjælpere |

```
  matte på arrays ─► ntri → tensor → nn → train
  fysik ───────────► kernel → sat / qubits / equilibrium
  verdener ────────► engine3d → memory → chain
```
