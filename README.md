# TriadLang

**A programming language where physics does the computing.**

```tri
import triad.sat as sat;

let inst = sat.SATInstance(3, [[1, -2, 3], [-1, 2, 3], [1, 2, -3]]);
let result = sat.solve_sat(inst, "interference", 10.0);

print(result["assignment"]);   // [0, 0, 0], zero violations
```

The language evolves a field and reads the answer back, code on top
and dynamics underneath.

---

## Run it in 90 seconds

```sh
./triad run examples/solver/sat_demo.tri
```

```text
assignment: [0, 0, 0]
violations: 0
correct: true
satisfiable: true
```

```sh
./triad run examples/quantum/bell_state.tri
./triad run examples/triad/anti_collapse.tri
```

Three runs solve logic, count entanglement, and observe a field.
The guided course lives in [`docs/`](docs/), in 8 languages.

---

## Logic as interference

Clauses go in as a field. The field settles. The assignment drops out.

```text
  clauses                     field              answer

  [ 1, -2,  3 ] ──┐
  [-1,  2,  3 ] ──┼──▶  interference  ──▶  [0, 0, 0]
  [ 1,  2, -3 ] ──┘                         violations: 0
```

```sh
./triad run examples/solver/sat_scale.tri
```

```text
  n=6 m=26
    solver violations: 0
    independent violations: 0
  n=10 m=43
    solver violations: 0
    independent violations: 0
  n=14 m=60
    solver violations: 0
    independent violations: 0
  unsat1 satisfiable: false
```

Two counters check every answer: the solver's own and an independent
one. UNSAT input comes back flagged unsatisfiable.

---

## Quantum that counts

Circuits run in `.tri` and produce shot counts.

```text
  |0> ──[H]──●──╮
                  ╞══  counts {0: 537, 3: 463}, 1000 shots
  |0> ───────X──╯
```

```sh
./triad run examples/quantum/grover.tri
./triad run examples/quantum/shor15.tri
./triad run examples/quantum/teleport.tri
```

```text
  Grover:     target |11> takes 1923 of 2000 shots (96.2%).
              16 states, 3 iterations.
  Shor:       N=15, a=7 -> order 4, factors [3, 5]
              N=21, a=2 -> order 6, factors [3, 7]
  Teleport:   fidelity 1.000000 on all 5 trials.
```

QFT, QPE, VQE on H2, BB84, and QRNG ship in the same directory.

---

## ML from zero

A neural net written in `.tri`, trained in `.tri`.

```sh
./triad run examples/ml/xor_train.tri
```

```text
  loss   0.72  █
               ▓█
               ░▓█
               ░░▓█▄▄  0.0002

  predictions:  0.00007  0.99985  0.99983  0.00022
```

Layers, autograd, Adam, Conv1d, and a Sequential MLP
(`examples/ml/nn_test.tri`), plus field-native models, transformers,
and GGUF loading in the same folder.

---

## One language, three natures

```
  ┌───────────────────────────────────────────────┐
  │  everyday .tri                                │
  │  let fn class try match yield async modules   │
  ├───────────────────────────────────────────────┤
  │  real Python inside                           │
  │  import numpy / flask / click, same objects   │
  ├───────────────────────────────────────────────┤
  │  real physics inside                          │
  │  solver, SAT, qubits, reg/ring/OBSERVE        │
  └───────────────────────────────────────────────┘
```

**Write normal code.** Variables, functions with `*args/**kwargs`,
classes with inheritance, f-strings, JSON, files, packages, a REPL,
a formatter, a debugger, a test runner.

**Use Python.** `import numpy` gives you real numpy. Same for flask,
click, os. Imports stay safe by default; pass `--unsafe` when you
mean it.

**Run physics.** `solver_solve(cfg)` evolves integral field dynamics on
CPU, Metal, or CUDA. `triad.sat` turns clauses into interference and
reads assignments back. `triad.qubits` runs gate circuits and counts
shots. Or drop into the DSL and write the experiment itself:

```tri
@T(18.0)
reg a : anti_collapse = 4;
reg b : anti_collapse = 4;
ring(a, b) kappa=-2.5 for T=18.0;
OBSERVE a k_star, crystallinity, peak, atom_count;
```

```text
a = { k_star=8.6394, crystallinity=0.9784, peak=0.3177, atom_count=0.9646 }
b = { k_star=8.6394, crystallinity=0.9784, peak=0.3180, atom_count=0.9646 }
```

---

## The engineer's tour

```sh
./triad check app.tri      # types, no execution
./triad repl               # scratchpad
./triad bench app.tri      # timings
./triad debug app.tri -b 12
./triad bundle app.tri -o app.pyz     # ship it
./triad compile app.tri --native -o app
./triad solve --N 64 --T 2.0 --dim 1  # physics, no file needed
./triad play               # 3D engine in the browser
```

Native C runtime and checker in `native/c/`. Bare-metal x86 kernel in
`native/kernel/`, the substrate as an operating system.

---

## Docs

| | |
|---|---|
| Course (zero to substrate) | [`docs/en/track/`](docs/en/track/) |
| Syntax on one page | [`docs/en/reference/syntax.md`](docs/en/reference/syntax.md) |
| Stdlib + `triad.*` modules | [`docs/en/reference/`](docs/en/reference/) |
| CLI, every command | [`docs/en/reference/cli.md`](docs/en/reference/cli.md) |
| Examples map | [`docs/en/reference/examples.md`](docs/en/reference/examples.md) |

Also: [pt-BR](docs/pt-BR/) · [de](docs/de/) · [sv](docs/sv/) · [no](docs/no/) ·
[da](docs/da/) · [es](docs/es/) · [zh](docs/zh/).

---

## Layout

```
  src/          frontend, compiler, runtime, stdlib, cli, tests
  examples/     basic, interop, solver, quantum, triad, ml, llm,
                crystalformer, crossdomain, games, blockchain, ...
  native/c      C runtime + checker + host tests
  native/kernel bare-metal x86 kernel
  docs/         course + reference, 8 languages
```

Research side (theory, studies, provenance): triad-lab,
the sibling repository.

---

*TRIAD: P1 oscillation · P2 self-reference · P3 coupling, integral, always.*
