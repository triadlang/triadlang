# triad.* — de 39 modulerna

Importera med `import triad.NAMN [as x];`. En rad var; detaljer i
`examples/` och källan.

## Tal och fält

| Modul | Det är |
|---|---|
| `triad.ntri` | arrayer med numpy-manér: `array zeros ones linspace arange sin cos exp sqrt sum max min mean`… |
| `triad.tensor` | tensorer med autograd: `tensor randn arange relu softmax bmm transpose layer_norm`… + enhet (`set_device cuda_available sync`) |
| `triad.equation_runtime` | utveckla integralekvationen direkt |

## Lärande

| Modul | Det är |
|---|---|
| `triad.nn` | lager: `triad Conv1d Conv2d Embedding MultiHeadAttention Transformer TriadLM Adam SGD…` + minnen (`CrystalMemory AttractorMemory…`) |
| `triad.net` | färdiga nät (se `examples/crystalformer/`) |
| `triad.data` / `losses` / `metrics` | dataset, förluster, mått |
| `triad.train` / `inference` / `generate` | träna, köra, sampla |
| `triad.llm` | LM-block (`CausalSelfAttention KVCache TriadLM…`) |
| `triad.native_qwen35` | nativ Qwen-3.5-brygga |
| `triad.checkpoint` / `amp` | spara modeller / blandad precision |
| `triad.solver_grad_bridge` / `param_bridge` | solver↔gradient-bryggor |

## Fysik och kvanta

| Modul | Det är |
|---|---|
| `triad.kernel` | `prepare run extract` — regimer till observabler |
| `triad.sat` | `SATInstance solve_sat` — logik som interferens |
| `triad.qubits` | `TriadCircuit h cnot run` — kretsar och counts |
| `triad.qalgo` | kvantalgoritmer (grover, qft, qpe…) |
| `triad.equilibrium` | hitta jämvikter |
| `triad.energy` / `atoms` | energi- och atombokföring |
| `triad.calibrate` / `curriculum` | kalibrering och läroplaner |
| `triad.causal` / `consistency` / `steering` | spår, kontroller, styrning |
| `triad.consciousness` | experimentell kognitiv sektor |

## System och värld

| Modul | Det är |
|---|---|
| `triad` | själva basmodulen |
| `triad.engine3d` | 3D-motor (`./triad play`) |
| `triad.memory` | kristallminne (`./triad memory …`) |
| `triad.chain` | kedjning (se `examples/blockchain/`) |
| `triad.http` / `fs` / `time` / `re` | systemåtkomst med triad-accent |
| `triad.viz` | visualiseringshjälpare |

```
  matte på arrayer ─► ntri → tensor → nn → train
  fysik ────────────► kernel → sat / qubits / equilibrium
  världar ──────────► engine3d → memory → chain
```
