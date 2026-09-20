# triad.* — die 39 Module

Import mit `import triad.NAME [as x];`. Eine Zeile pro Modul; Details
in `examples/` und im Quelltext.

## Zahlen und Felder

| Modul | Es ist |
|---|---|
| `triad.ntri` | Arrays mit numpy-Manieren: `array zeros ones linspace arange sin cos exp sqrt sum max min mean`… |
| `triad.tensor` | Tensoren mit Autograd: `tensor randn arange relu softmax bmm transpose layer_norm`… + Gerät (`set_device cuda_available sync`) |
| `triad.equation_runtime` | die integrale Gleichung direkt entwickeln |

## Lernen

| Modul | Es ist |
|---|---|
| `triad.nn` | Schichten: `triad Conv1d Conv2d Embedding MultiHeadAttention Transformer TriadLM Adam SGD…` + Gedächtnisse (`CrystalMemory AttractorMemory…`) |
| `triad.net` | fertige Netze (siehe `examples/crystalformer/`) |
| `triad.data` / `losses` / `metrics` | Datensätze, Verluste, Metriken |
| `triad.train` / `inference` / `generate` | Training, Ausführen, Sampeln |
| `triad.llm` | LM-Blöcke (`CausalSelfAttention KVCache TriadLM…`) |
| `triad.native_qwen35` | native Qwen-3.5-Brücke |
| `triad.checkpoint` / `amp` | Modelle speichern / gemischte Präzision |
| `triad.solver_grad_bridge` / `param_bridge` | Solver↔Gradient-Brücken |

## Physik und Quanten

| Modul | Es ist |
|---|---|
| `triad.kernel` | `prepare run extract` — Regime zu Observablen |
| `triad.sat` | `SATInstance solve_sat` — Logik als Interferenz |
| `triad.qubits` | `TriadCircuit h cnot run` — Schaltkreise und Counts |
| `triad.qalgo` | Quantenalgorithmen (grover, qft, qpe…) |
| `triad.equilibrium` | Gleichgewichte finden |
| `triad.energy` / `atoms` | Energie- und Atombuchhaltung |
| `triad.calibrate` / `curriculum` | Kalibrierung und Curricula |
| `triad.causal` / `consistency` / `steering` | Spuren, Prüfungen, Steuerung |
| `triad.consciousness` | experimenteller kognitiver Sektor |

## System und Welt

| Modul | Es ist |
|---|---|
| `triad` | das Basismodul selbst |
| `triad.engine3d` | 3D-Engine (`./triad play`) |
| `triad.memory` | Kristallgedächtnis (`./triad memory …`) |
| `triad.chain` | Verketten (siehe `examples/blockchain/`) |
| `triad.http` / `fs` / `time` / `re` | Systemzugriff mit triad-Akzent |
| `triad.viz` | Visualisierungshelfer |

```
  Mathe auf Arrays ─► ntri → tensor → nn → train
  Physik ───────────► kernel → sat / qubits / equilibrium
  Welten ───────────► engine3d → memory → chain
```
