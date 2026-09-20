# 10 — Solver: computar con física

El motor bajo el lenguaje evoluciona campos y lee respuestas como
observables. Tres puertas, misma casa.

## Puerta 1 — `solver_solve`: evolucionar un campo

```tri
let cfg = {"N": 64, "T": 2.0, "dt": 0.01, "seed": 42, "D": 1};
let r = solver_solve(cfg);
print(r["norm"]);
print(r["peak"]);
```

- `N`: puntos de grilla. `T`: tiempo simulado. `dt`: paso. `D`: 1, 2 o 3.
- Vuelve un dict: `norm`, `peak`, densidad, trayectoria…

```
  cfg ──► solver_solve ──► { norm, peak, … }
   │                          │
   │ qué evolucionar          └── qué dice el campo
```

## Puerta 2 — SAT: lógica como interferencia

`examples/solver/sat_demo.tri`:

```tri
import triad.sat as sat;

let inst = sat.SATInstance(3, [[1, -2, 3], [-1, 2, 3], [1, 2, -3]]);
let result = sat.solve_sat(inst, "interference", 10.0);

print("assignment: " + str(result["assignment"]));
print("violations: " + str(result["violations"]));
```

`▸ salida`

```text
assignment: [0, 0, 0]
violations: 0
```

Cláusulas entran, assignment válido sale. Escalera en
`examples/solver/sat_scale.tri` (n = 6, 10, 14, cero violaciones).

## Puerta 3 — cúbits: circuitos que corren

`examples/quantum/bell_state.tri`:

```tri
import triad.qubits as q;

let circuit = q.TriadCircuit(2);
circuit.h(0);
circuit.cnot(0, 1);

let result = circuit.run(none, 1000, 7);
print("counts: " + str(result.counts));
```

Hadamard + CNOT, 1000 shots, conteos. Más circuitos en
`examples/quantum/`: grover, teleport, qft, qpe, vqe, shor15.

## Un motor, tres lecturas

```
              ┌─ solver_solve ──► campos y densidades
  TRIAD ──────┼─ sat ───────────► assignments
  engine      └─ qubits ────────► conteos
```

✎ prueba: cambia las cláusulas SAT y mira el assignment seguir.

Siguiente: [11 — DSL](11-dsl.md).
