# 10 — Solver: computar com física

O motor sob a linguagem evolui campos e lê respostas como observáveis.
Três portas, mesma casa.

## Porta 1 — `solver_solve`: evoluir um campo

```tri
let cfg = {"N": 64, "T": 2.0, "dt": 0.01, "seed": 42, "D": 1};
let r = solver_solve(cfg);
print(r["norm"]);
print(r["peak"]);
```

- `N`: pontos da grade. `T`: tempo simulado. `dt`: passo. `D`: 1, 2 ou 3.
- Volta um dict: `norm`, `peak`, densidade, trajetória…

```
  cfg ──► solver_solve ──► { norm, peak, … }
   │                          │
   │ o que evoluir            └── o que o campo diz
```

## Porta 2 — SAT: lógica como interferência

`examples/solver/sat_demo.tri`:

```tri
import triad.sat as sat;

let inst = sat.SATInstance(3, [[1, -2, 3], [-1, 2, 3], [1, 2, -3]]);
let result = sat.solve_sat(inst, "interference", 10.0);

print("assignment: " + str(result["assignment"]));
print("violations: " + str(result["violations"]));
```

`▸ saída`

```text
assignment: [0, 0, 0]
violations: 0
```

Cláusulas entram, assignment válido sai. Escada de escala em
`examples/solver/sat_scale.tri` (n = 6, 10, 14, zero violações).

## Porta 3 — qubits: circuitos que rodam

`examples/quantum/bell_state.tri`:

```tri
import triad.qubits as q;

let circuit = q.TriadCircuit(2);
circuit.h(0);
circuit.cnot(0, 1);

let result = circuit.run(none, 1000, 7);
print("counts: " + str(result.counts));
```

Hadamard + CNOT, 1000 shots, contagens. Mais circuitos em
`examples/quantum/`: grover, teleport, qft, qpe, vqe, shor15.

## Um motor, três leituras

```
              ┌─ solver_solve ──► campos e densidades
  TRIAD ──────┼─ sat ───────────► assignments
  engine      └─ qubits ────────► contagens
```

✎ experimente: mude as cláusulas do SAT e veja o assignment acompanhar.

Próximo: [11 — DSL](11-dsl.md).
