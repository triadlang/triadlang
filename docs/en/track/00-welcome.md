# 00 — Welcome

## What TriadLang is, in 30 seconds

```
   .tri ──► ./triad ──► result
    │           │
    │           ├── direct execution (run)
    │           ├── type checking (check)
    │           ├── real Python inside (import numpy, flask…)
    │           └── real physics inside (solver, SAT, qubits)
    │
    └── or physical DSL: reg / ring / OBSERVE
```

TriadLang is a complete programming language (variables, functions,
classes, errors, modules) with two unusual doors: it imports the real
Python ecosystem and it ships an integral physics-dynamics engine (the
TRIAD solver) that does things like SAT and quantum circuits.

## Check everything is fine

```sh
./triad doctor
```

`▸ output` — an environment report (Python, dependencies, native).

If anything is missing, the guided setup fixes it:

```sh
./triad setup
```

## The first run (2 minutes)

File `hi.tri`:

```tri
print("Hello from TriadLang");
```

```sh
./triad run hi.tri
```

`▸ output`

```text
Hello from TriadLang
```

It worked? You already know how to run TriadLang. The rest is language.

## The two checks you will always use

```sh
./triad check hi.tri     # type-checks only, does not run
./triad fmt hi.tri       # formats the file
```

## Mental map from here on

```
  steps 1–7    everyday language (like learning any language)
       │
  steps 8–9    arrays + Python (where triadlang meets the world)
       │
  steps 10–11  solver + DSL (where triadlang becomes physics)
       │
  steps 12–13  tools + native (production and kernel)
```

✎ try it: run `./triad repl`, type `print(1 + 1);`, then `exit`.

Next: [01 — first steps](01-first-steps.md).
