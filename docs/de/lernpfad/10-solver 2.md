# 10 — Solver: mit Physik rechnen

Die Engine unter der Sprache entwickelt Felder und liest Antworten
als Observablen zurück. Drei Türen, ein Haus.

## Tür 1 — `solver_solve`: ein Feld entwickeln

```tri
let cfg = {"N": 64, "T": 2.0, "dt": 0.01, "seed": 42, "D": 1};
let r = solver_solve(cfg);
print(r["norm"]);
print(r["peak"]);
```

- `N`: Gitterpunkte. `T`: simulierte Zeit. `dt`: Schritt. `D`: 1, 2, 3.
- Zurück kommt ein Dict: `norm`, `peak`, Dichte, Trajektorie…

```
  cfg ──► solver_solve ──► { norm, peak, … }
   │                          │
   │ was entwickeln           └── was das Feld sagt
```

## Tür 2 — SAT: Logik als Interferenz

`examples/solver/sat_demo.tri`:

```tri
import triad.sat as sat;

let inst = sat.SATInstance(3, [[1, -2, 3], [-1, 2, 3], [1, 2, -3]]);
let result = sat.solve_sat(inst, "interference", 10.0);

print("assignment: " + str(result["assignment"]));
print("violations: " + str(result["violations"]));
```

`▸ Ausgabe`

```text
assignment: [0, 0, 0]
violations: 0
```

Klauseln rein, erfüllende Belegung raus. Skalenleiter in
`examples/solver/sat_scale.tri` (n = 6, 10, 14, null Verletzungen).

## Tür 3 — Qubits: Schaltkreise, die laufen

`examples/quantum/bell_state.tri`:

```tri
import triad.qubits as q;

let circuit = q.TriadCircuit(2);
circuit.h(0);
circuit.cnot(0, 1);

let result = circuit.run(none, 1000, 7);
print("counts: " + str(result.counts));
```

Hadamard + CNOT, 1000 Shots, Counts raus. Mehr Schaltkreise in
`examples/quantum/`: grover, teleport, qft, qpe, vqe, shor15.

## Eine Engine, drei Lesarten

```
              ┌─ solver_solve ──► Felder & Dichten
  TRIAD ──────┼─ sat ───────────► Belegungen
  Engine      └─ qubits ────────► Counts
```

✎ probier es: Ändere die SAT-Klauseln und schau zu, wie die
Belegung folgt.

Weiter: [11 — DSL](11-dsl.md).
