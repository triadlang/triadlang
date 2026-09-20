# TriadLang — 文档

```
                         ┌──────────────┐
                    ┌────│  你在这里     │────┐
                    │    │              │    │
                    │    └──────────────┘    │
                    ▼                       ▼
          ┌─────────────────┐     ┌─────────────────┐
          │  从没见过       │     │  我会编程       │
          │  TRIAD：从零    │     │  只想快速找到   │
          │  开始           │     │                 │
          └────────┬────────┘     └────────┬────────┘
                   │                       │
                   ▼                       ▼
          track/ 00 → 13          reference/
          （引导教程）              （查阅）
```

## 学习路线（从零到基底）

| 步骤 | 文件 | 学完之后 |
|---|---|---|
| 0 | [track/00.md](track/00.md) | 是什么、安装、`doctor`、第一次 `run` |
| 1 | [track/01.md](track/01.md) | hello、变量、`print`、REPL |
| 2 | [track/02.md](track/02.md) | `if`、循环、f-string |
| 3 | [track/03.md](track/03.md) | `fn`、参数、`*args/**kwargs`、`yield` |
| 4 | [track/04.md](track/04.md) | 列表、字典、`string`、`json` |
| 5 | [track/05.md](track/05.md) | `type`、`class`、继承 |
| 6 | [track/06.md](track/06.md) | `try/catch`、`throw`、`assert` |
| 7 | [track/07.md](track/07.md) | 导入、标准库、包 |
| 8 | [track/08.md](track/08.md) | `triad.ntri`、向量化 |
| 9 | [track/09.md](track/09.md) | Python 互操作、safe/unsafe |
| 10 | [track/10.md](track/10.md) | 求解器、SAT、量子比特 |
| 11 | [track/11.md](track/11.md) | `reg/ring/OBSERVE` DSL、P1/P2/P3 |
| 12 | [track/12.md](track/12.md) | 命令行：fmt、test、bench、debug、bundle… |
| 13 | [track/13.md](track/13.md) | 本地编译、C、内核 |

## 参考（直接查阅）

- [reference/syntax.md](reference/syntax.md) — 一页看懂整门语言
- [reference/stdlib.md](reference/stdlib.md) — `math`、`random`、`fs`…模块
- [reference/triad.md](reference/triad.md) — 39 个 `triad.*` 模块
- [reference/cli.md](reference/cli.md) — 每个 `./triad` 命令与示例
- [reference/examples.md](reference/examples.md) — `examples/` 按领域地图

## 本文档约定

- 这里每个 `.tri` 代码块都能运行：`./triad run 文件.tri`。
- `▸ 输出` 标记程序打印的内容。
- ```tri-frag 块是示意片段（仅语法页）；其余都可运行。
- `✎ 试试` 是邀请，不是作业。
