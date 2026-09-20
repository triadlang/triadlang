# triad.* — los 39 módulos

Importa con `import triad.NOMBRE [as x];`. Una línea cada uno;
detalles en `examples/` y la fuente.

## Números y campos

| Módulo | Es |
|---|---|
| `triad.ntri` | arreglos estilo numpy: `array zeros ones linspace arange sin cos exp sqrt sum max min mean`… |
| `triad.tensor` | tensores con autograd: `tensor randn arange relu softmax bmm transpose layer_norm`… + dispositivo (`set_device cuda_available sync`) |
| `triad.equation_runtime` | evolucionar la ecuación integral directo |

## Aprendizaje

| Módulo | Es |
|---|---|
| `triad.nn` | capas: `triad Conv1d Conv2d Embedding MultiHeadAttention Transformer TriadLM Adam SGD…` + memorias (`CrystalMemory AttractorMemory…`) |
| `triad.net` | redes listas (ver `examples/crystalformer/`) |
| `triad.data` / `losses` / `metrics` | datasets, pérdidas, métricas |
| `triad.train` / `inference` / `generate` | entrenar, correr, muestrear |
| `triad.llm` | bloques de LM (`CausalSelfAttention KVCache TriadLM…`) |
| `triad.native_qwen35` | puente nativo Qwen-3.5 |
| `triad.checkpoint` / `amp` | guardar modelos / precisión mixta |
| `triad.solver_grad_bridge` / `param_bridge` | puentes solver↔gradiente |

## Física y cuántica

| Módulo | Es |
|---|---|
| `triad.kernel` | `prepare run extract` — regímenes a observables |
| `triad.sat` | `SATInstance solve_sat` — lógica como interferencia |
| `triad.qubits` | `TriadCircuit h cnot run` — circuitos y conteos |
| `triad.qalgo` | algoritmos cuánticos (grover, qft, qpe…) |
| `triad.equilibrium` | hallar equilibrios |
| `triad.energy` / `atoms` | energía y contabilidad de átomos |
| `triad.calibrate` / `curriculum` | calibración y currículos |
| `triad.causal` / `consistency` / `steering` | trazas, chequeos, dirección |
| `triad.consciousness` | sector cognitivo experimental |

## Sistema y mundo

| Módulo | Es |
|---|---|
| `triad` | el propio módulo base |
| `triad.engine3d` | engine 3D (`./triad play`) |
| `triad.memory` | memoria cristal (`./triad memory …`) |
| `triad.chain` | encadenado (ver `examples/blockchain/`) |
| `triad.http` / `fs` / `time` / `re` | acceso a sistema sabor triad |
| `triad.viz` | ayudantes de visualización |

```
  matemática en arreglos ► ntri → tensor → nn → train
  física ────────────────► kernel → sat / qubits / equilibrium
  mundos ────────────────► engine3d → memory → chain
```
