# 10 — Lösare: räkna med fysik

Motorn under språket utvecklar fält och läser tillbaka svar som
observabler. Tre dörrar, ett hus.

## Dörr 1 — `solver_solve`: utveckla ett fält

```tri
let cfg = {"N": 64, "T": 2.0, "dt": 0.01, "seed": 42, "D": 1};
let r = solver_solve(cfg);
print(r["norm"]);
print(r["peak"]);
```

- `N`: rutnätspunkter. `T`: simulerad tid. `dt`: steg. `D`: 1, 2, 3.
- Tillbaka kommer en dict: `norm`, `peak`, täthet, bana…

```
  cfg ──► solver_solve ──► { norm, peak, … }
   │                          │
   │ vad utveckla             └── vad fältet säger
```

## Dörr 2 — SAT: logik som interferens

`examples/solver/sat_demo.tri`:

```tri
import triad.sat as sat;

let inst = sat.SATInstance(3, [[1, -2, 3], [-1, 2, 3], [1, 2, -3]]);
let result = sat.solve_sat(inst, "interference", 10.0);

print("assignment: " + str(result["assignment"]));
print("violations: " + str(result["violations"]));
```

`▸ utdata`

```text
assignment: [0, 0, 0]
violations: 0
```

Klausuler in, uppfyllande tilldelning ut. Skalstege i
`examples/solver/sat_scale.tri` (n = 6, 10, 14, noll brott).

## Dörr 3 — kubitar: kretsar som körs

`examples/quantum/bell_state.tri`:

```tri
import triad.qubits as q;

let circuit = q.TriadCircuit(2);
circuit.h(0);
circuit.cnot(0, 1);

let result = circuit.run(none, 1000, 7);
print("counts: " + str(result.counts));
```

Hadamard + CNOT, 1000 shots, counts ut. Fler kretsar i
`examples/quantum/`: grover, teleport, qft, qpe, vqe, shor15.

## En motor, tre läsningar

```
              ┌─ solver_solve ──► fält & tätheter
  TRIAD ──────┼─ sat ───────────► tilldelningar
  engine      └─ qubits ────────► counts
```

✎ prova: ändra SAT-klausulerna och se tilldelningen följa med.

Nästa: [11 — DSL](11-dsl.md).
