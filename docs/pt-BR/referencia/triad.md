# triad.* — os 39 módulos

Importe com `import triad.NOME [as x];`. Uma linha cada; detalhes em
`examples/` e no fonte.

## Números e campos

| Módulo | É |
|---|---|
| `triad.ntri` | arrays estilo numpy: `array zeros ones linspace arange sin cos exp sqrt sum max min mean`… |
| `triad.tensor` | tensores com autograd: `tensor randn arange relu softmax bmm transpose layer_norm`… + dispositivo (`set_device cuda_available sync`) |
| `triad.equation_runtime` | evoluir a equação integral direto |

## Aprendizado

| Módulo | É |
|---|---|
| `triad.nn` | camadas: `triad Conv1d Conv2d Embedding MultiHeadAttention Transformer TriadLM Adam SGD…` + memórias (`CrystalMemory AttractorMemory…`) |
| `triad.net` | redes prontas (ver `examples/crystalformer/`) |
| `triad.data` / `losses` / `metrics` | datasets, perdas, métricas |
| `triad.train` / `inference` / `generate` | treino, execução, amostragem |
| `triad.llm` | blocos de LM (`CausalSelfAttention KVCache TriadLM…`) |
| `triad.native_qwen35` | ponte nativa Qwen-3.5 |
| `triad.checkpoint` / `amp` | salvar modelos / precisão mista |
| `triad.solver_grad_bridge` / `param_bridge` | pontes solver↔gradiente |

## Física e quanta

| Módulo | É |
|---|---|
| `triad.kernel` | `prepare run extract` — regimes a observáveis |
| `triad.sat` | `SATInstance solve_sat` — lógica como interferência |
| `triad.qubits` | `TriadCircuit h cnot run` — circuitos e contagens |
| `triad.qalgo` | algoritmos quânticos (grover, qft, qpe…) |
| `triad.equilibrium` | achar equilíbrios |
| `triad.energy` / `atoms` | energia e contabilidade de átomos |
| `triad.calibrate` / `curriculum` | calibração e currículos |
| `triad.causal` / `consistency` / `steering` | traços, checagens, direção |
| `triad.consciousness` | setor cognitivo experimental |

## Sistema e mundo

| Módulo | É |
|---|---|
| `triad` | o próprio módulo base |
| `triad.engine3d` | engine 3D (`./triad play`) |
| `triad.memory` | memória cristal (`./triad memory …`) |
| `triad.chain` | encadeamento (ver `examples/blockchain/`) |
| `triad.http` / `fs` / `time` / `re` | acesso a sistema sabor triad |
| `triad.viz` | ajudantes de visualização |

```
  matemática em arrays ──► ntri → tensor → nn → train
  física ────────────────► kernel → sat / qubits / equilibrium
  mundos ────────────────► engine3d → memory → chain
```
