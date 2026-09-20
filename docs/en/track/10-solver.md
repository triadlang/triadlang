# 10 — Solver: computing with physics

The engine under the language evolves fields and reads answers back
as observables. Three doors, same house.

## Door 1 — `solver_solve`: evolve a field

```tri
let cfg = {"N": 64, "T": 2.0, "dt": 0.01, "seed": 42, "D": 1};
let r = solver_solve(cfg);
print(r["norm"]);
print(r["peak"]);
```

- `N`: grid points. `T`: simulated time. `dt`: step. `D`: 1, 2 or 3.
- Back comes a dict: `norm`, `peak`, density, trajectory…

```
  cfg ──► solver_solve ──► { norm, peak, … }
   │                          │
   │ what to evolve           └── what the field says
```

## Door 2 — SAT: logic as interference

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

Clauses go in, a satisfying assignment comes out. Scale ladder in
`examples/solver/sat_scale.tri` (n = 6, 10, 14, all zero violations).

## Door 3 — qubits: circuits that run

`examples/quantum/bell_state.tri`:

```tri
import triad.qubits as q;

let circuit = q.TriadCircuit(2);
circuit.h(0);
circuit.cnot(0, 1);

let result = circuit.run(none, 1000, 7);
print("counts: " + str(result.counts));
```

Hadamard + CNOT, 1000 shots, counts out. More circuits in
`examples/quantum/`: grover, teleport, qft, qpe, vqe, shor15.

## One engine, three readings

```
              ┌─ solver_solve ──► fields & densities
  TRIAD ──────┼─ sat ───────────► assignments
  engine      └─ qubits ────────► counts
```

✎ try it: change the SAT clauses and watch the assignment follow.

Next: [11 — DSL](11-dsl.md).
