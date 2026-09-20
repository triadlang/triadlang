# 例子 — 按领域地图

`examples/` 下每个文件夹都用 `./triad run`（或 `check`）运行。
从你的好奇心开始。

```
  basic ────────── 语言自己（从这里开始）
  interop ──────── Python 桥（numpy、flask、click…）
  solver ───────── SAT、基准、本地一致性
  quantum ──────── 线路（bell、grover、qft、vqe、shor15…）
  triad ────────── 物理 DSL（reg/ring/OBSERVE）
  ml ───────────── 张量、网络、训练
  llm ──────────── triad 上语言模型
  crystalformer ── crystal attention 架构
  resonanceformer─ 共振架构
  crossdomain ──── s1..s6 传奇（神经元、气候、宇宙…）
  games ────────── 可玩 3D（memory、golf、predator-prey）
  blockchain ───── 链接演示
  plotting ─────── 图形
```

## 十扇门，十个文件

| 想要… | 先跑这个 |
|---|---|
| hello world | `basic/hello.tri` |
| 类型系统 | `basic/types.tri` |
| 向量 | `basic/broadcast_test.tri` |
| 里面的 Python | `interop/boundary_numpy.tri` |
| 网页回答 | `interop/boundary_flask_testclient.tri` |
| 解出的逻辑 | `solver/sat_demo.tri` |
| 纠缠 | `quantum/bell_state.tri` |
| 被观测的场 | `triad/anti_collapse.tri` |
| 训好的网络 | `ml/nn_test.tri` |
| 传奇 | `crossdomain/s1_neuro_climate.tri` |

快速全查：

```sh
for f in examples/*/*.tri; do if ./triad check "$f" | grep -q "OK"; then echo "OK $f"; else echo "FAIL $f"; fi; done
```
