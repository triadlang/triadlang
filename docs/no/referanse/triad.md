# triad.* — de 39 modulene

Importer med `import triad.NAVN [as x];`. Én linje hver; detaljer i
`examples/` og kilden.

## Tall og felt

| Modul | Det er |
|---|---|
| `triad.ntri` | arrayer med numpy-manerer: `array zeros ones linspace arange sin cos exp sqrt sum max min mean`… |
| `triad.tensor` | tensorer med autograd: `tensor randn arange relu softmax bmm transpose layer_norm`… + enhet (`set_device cuda_available sync`) |
| `triad.equation_runtime` | utvikle integrallikningen direkte |

## Læring

| Modul | Det er |
|---|---|
| `triad.nn` | lag: `triad Conv1d Conv2d Embedding MultiHeadAttention Transformer TriadLM Adam SGD…` + minner (`CrystalMemory AttractorMemory…`) |
| `triad.net` | ferdige nett (se `examples/crystalformer/`) |
| `triad.data` / `losses` / `metrics` | datasett, tap, mål |
| `triad.train` / `inference` / `generate` | trene, kjøre, sample |
| `triad.llm` | LM-blokker (`CausalSelfAttention KVCache TriadLM…`) |
| `triad.native_qwen35` | nativ Qwen-3.5-bro |
| `triad.checkpoint` / `amp` | lagre modeller / blandet presisjon |
| `triad.solver_grad_bridge` / `param_bridge` | solver↔gradient-broer |

## Fysikk og kvanter

| Modul | Det er |
|---|---|
| `triad.kernel` | `prepare run extract` — regimer til observabler |
| `triad.sat` | `SATInstance solve_sat` — logikk som interferens |
| `triad.qubits` | `TriadCircuit h cnot run` — kretser og counts |
| `triad.qalgo` | kvantealgoritmer (grover, qft, qpe…) |
| `triad.equilibrium` | finne likevekter |
| `triad.energy` / `atoms` | energi- og atomregnskap |
| `triad.calibrate` / `curriculum` | kalibrering og læreplaner |
| `triad.causal` / `consistency` / `steering` | spor, kontroller, styring |
| `triad.consciousness` | eksperimentell kognitiv sektor |

## System og verden

| Modul | Det er |
|---|---|
| `triad` | selve basismodulen |
| `triad.engine3d` | 3D-motor (`./triad play`) |
| `triad.memory` | krystallminne (`./triad memory …`) |
| `triad.chain` | kjeding (se `examples/blockchain/`) |
| `triad.http` / `fs` / `time` / `re` | systemtilgang med triad-aksent |
| `triad.viz` | visualiseringshjelpere |

```
  matte på arrayer ─► ntri → tensor → nn → train
  fysikk ───────────► kernel → sat / qubits / equilibrium
  verdener ─────────► engine3d → memory → chain
```
