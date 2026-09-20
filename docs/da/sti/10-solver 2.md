# 10 — Solver: regn med fysik

Motoren under sproget udvikler felter og læser svar tilbage som
observabler. Tre døre, ét hus.

## Dør 1 — `solver_solve`: udvikl et felt

```tri
let cfg = {"N": 64, "T": 2.0, "dt": 0.01, "seed": 42, "D": 1};
let r = solver_solve(cfg);
print(r["norm"]);
print(r["peak"]);
```

- `N`: gitterpunkter. `T`: simuleret tid. `dt`: skridt. `D`: 1, 2, 3.
- Tilbage kommer en dict: `norm`, `peak`, densitet, bane…

```
  cfg ──► solver_solve ──► { norm, peak, … }
   │                          │
   │ hvad udvikle             └── hvad feltet siger
```

## Dør 2 — SAT: logik som interferens

`examples/solver/sat_demo.tri`:

```tri
import triad.sat as sat;

let inst = sat.SATInstance(3, [[1, -2, 3], [-1, 2, 3], [1, 2, -3]]);
let result = sat.solve_sat(inst, "interference", 10.0);

print("assignment: " + str(result["assignment"]));
print("violations: " + str(result["violations"]));
```

`▸ output`

```text
assignment: [0, 0, 0]
violations: 0
```

Klausuler ind, opfyldende tildeling ud. Skalastige i
`examples/solver/sat_scale.tri` (n = 6, 10, 14, nul brud).

## Dør 3 — kubitter: kredsløb der kører

`examples/quantum/bell_state.tri`:

```tri
import triad.qubits as q;

let circuit = q.TriadCircuit(2);
circuit.h(0);
circuit.cnot(0, 1);

let result = circuit.run(none, 1000, 7);
print("counts: " + str(result.counts));
```

Hadamard + CNOT, 1000 shots, counts ud. Flere kredsløb i
`examples/quantum/`: grover, teleport, qft, qpe, vqe, shor15.

## Én motor, tre læsninger

```
              ┌─ solver_solve ──► felter og densitet
  TRIAD ──────┼─ sat ───────────► tildelinger
  engine      └─ qubits ────────► counts
```

✎ prøv: ændr SAT-klausulerne og se tildelingen følge med.

Næste: [11 — DSL](11-dsl.md).
