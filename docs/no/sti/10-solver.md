# 10 — Solver: regn med fysikk

Motoren under språket utvikler felt og leser tilbake svar som
observabler. Tre dører, ett hus.

## Dør 1 — `solver_solve`: utvikle et felt

```tri
let cfg = {"N": 64, "T": 2.0, "dt": 0.01, "seed": 42, "D": 1};
let r = solver_solve(cfg);
print(r["norm"]);
print(r["peak"]);
```

- `N`: rutenettpunkter. `T`: simulert tid. `dt`: steg. `D`: 1, 2, 3.
- Tilbake kommer en dict: `norm`, `peak`, tetthet, bane…

```
  cfg ──► solver_solve ──► { norm, peak, … }
   │                          │
   │ hva utvikle              └── hva feltet sier
```

## Dør 2 — SAT: logikk som interferens

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

Klausuler inn, oppfyllende tilordning ut. Skalastige i
`examples/solver/sat_scale.tri` (n = 6, 10, 14, null brudd).

## Dør 3 — kubiter: kretser som kjører

`examples/quantum/bell_state.tri`:

```tri
import triad.qubits as q;

let circuit = q.TriadCircuit(2);
circuit.h(0);
circuit.cnot(0, 1);

let result = circuit.run(none, 1000, 7);
print("counts: " + str(result.counts));
```

Hadamard + CNOT, 1000 shots, counts ut. Flere kretser i
`examples/quantum/`: grover, teleport, qft, qpe, vqe, shor15.

## Én motor, tre lesninger

```
              ┌─ solver_solve ──► felt og tetthet
  TRIAD ──────┼─ sat ───────────► tilordninger
  engine      └─ qubits ────────► counts
```

✎ prøv: endre SAT-klausulene og se tilordningen følge etter.

Neste: [11 — DSL](11-dsl.md).
