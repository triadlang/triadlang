# TriadLang — Documentation

```
                         ┌──────────────┐
                    ┌────│  YOU ARE     │────┐
                    │    │    HERE      │    │
                    │    └──────────────┘    │
                    ▼                       ▼
          ┌─────────────────┐     ┌─────────────────┐
          │  NEVER SEEN     │     │  I CODE, JUST   │
          │  TRIAD: START   │     │  POINT ME       │
          │  FROM ZERO      │     │  FAST           │
          └────────┬────────┘     └────────┬────────┘
                   │                       │
                   ▼                       ▼
          track/ 00 → 13          reference/
          (guided course)         (lookup)
```

## The track (zero to substrate)

| Step | File | You walk out knowing |
|---|---|---|
| 0 | [track/00-welcome.md](track/00-welcome.md) | what it is, setup, `doctor`, first `run` |
| 1 | [track/01-first-steps.md](track/01-first-steps.md) | hello, variables, `print`, REPL |
| 2 | [track/02-control.md](track/02-control.md) | `if`, loops, f-strings |
| 3 | [track/03-functions.md](track/03-functions.md) | `fn`, args, `*args/**kwargs`, `yield` |
| 4 | [track/04-collections.md](track/04-collections.md) | lists, dicts, `string`, `json` |
| 5 | [track/05-types-classes.md](track/05-types-classes.md) | `type`, `class`, inheritance |
| 6 | [track/06-errors.md](track/06-errors.md) | `try/catch`, `throw`, `assert` |
| 7 | [track/07-modules.md](track/07-modules.md) | imports, stdlib, packages |
| 8 | [track/08-arrays.md](track/08-arrays.md) | `triad.ntri`, vectorization |
| 9 | [track/09-python.md](track/09-python.md) | Python interop, safe/unsafe |
| 10 | [track/10-solver.md](track/10-solver.md) | solver, SAT, qubits |
| 11 | [track/11-dsl.md](track/11-dsl.md) | `reg/ring/OBSERVE` DSL, P1/P2/P3 |
| 12 | [track/12-tools.md](track/12-tools.md) | CLI: fmt, test, bench, debug, bundle… |
| 13 | [track/13-native.md](track/13-native.md) | native compile, C, kernel |

## Reference (direct lookup)

- [reference/syntax.md](reference/syntax.md) — the whole language on one visual page
- [reference/stdlib.md](reference/stdlib.md) — `math`, `random`, `fs`… modules
- [reference/triad.md](reference/triad.md) — the 39 `triad.*` modules
- [reference/cli.md](reference/cli.md) — every `./triad` command with an example
- [reference/examples.md](reference/examples.md) — map of `examples/` by domain

## Conventions of this doc

- Every `.tri` block shown here runs: `./triad run file.tri`.
- `▸ output` marks what the program prints.
- ```tri-frag blocks are illustrative slices (syntax sheet); everything else runs.
- `✎ try it` is an invitation, not homework.
