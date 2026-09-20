# triad.* — 39 个模块

用 `import triad.名字 [as x];` 导入。一个一行；
细节见 `examples/` 和源码。

## 数字和场

| 模块 | 是什么 |
|---|---|
| `triad.ntri` | numpy 习惯数组：`array zeros ones linspace arange sin cos exp sqrt sum max min mean`… |
| `triad.tensor` | 自动微分张量：`tensor randn arange relu softmax bmm transpose layer_norm`… + 设备（`set_device cuda_available sync`） |
| `triad.equation_runtime` | 直接演化积分方程 |

## 学习

| 模块 | 是什么 |
|---|---|
| `triad.nn` | 层：`triad Conv1d Conv2d Embedding MultiHeadAttention Transformer TriadLM Adam SGD…` + 记忆（`CrystalMemory AttractorMemory…`） |
| `triad.net` | 现成网络（见 `examples/crystalformer/`） |
| `triad.data` / `losses` / `metrics` | 数据集、损失、指标 |
| `triad.train` / `inference` / `generate` | 训练、运行、采样 |
| `triad.llm` | 语言模型块（`CausalSelfAttention KVCache TriadLM…`） |
| `triad.native_qwen35` | 本地 Qwen-3.5 桥 |
| `triad.checkpoint` / `amp` | 存模型 / 混合精度 |
| `triad.solver_grad_bridge` / `param_bridge` | 求解器↔梯度桥 |

## 物理和量子

| 模块 | 是什么 |
|---|---|
| `triad.kernel` | `prepare run extract` — 机制到可观测量 |
| `triad.sat` | `SATInstance solve_sat` — 逻辑即干涉 |
| `triad.qubits` | `TriadCircuit h cnot run` — 线路和计数 |
| `triad.qalgo` | 量子算法（grover、qft、qpe…） |
| `triad.equilibrium` | 找平衡 |
| `triad.energy` / `atoms` | 能量和原子记账 |
| `triad.calibrate` / `curriculum` | 校准和课程 |
| `triad.causal` / `consistency` / `steering` | 轨迹、检查、引导 |
| `triad.consciousness` | 实验认知区 |

## 系统和世界

| 模块 | 是什么 |
|---|---|
| `triad` | 基础模块自己 |
| `triad.engine3d` | 3D 引擎（`./triad play`） |
| `triad.memory` | 晶体记忆（`./triad memory …`） |
| `triad.chain` | 链接（见 `examples/blockchain/`） |
| `triad.http` / `fs` / `time` / `re` | triad 味系统访问 |
| `triad.viz` | 可视化助手 |

```
  数组数学 ──► ntri → tensor → nn → train
  物理 ──────► kernel → sat / qubits / equilibrium
  世界 ──────► engine3d → memory → chain
```
