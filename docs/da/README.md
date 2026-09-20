# TriadLang — Dokumentation

```
                         ┌──────────────┐
                    ┌────│  DU ER       │────┐
                    │    │   HER        │    │
                    │    └──────────────┘    │
                    ▼                       ▼
          ┌─────────────────┐     ┌─────────────────┐
          │  ALDRIG SET     │     │  JEG KAN ALLEREDE│
          │  TRIAD: START   │     │  VIS MIG BARE   │
          │  FRA NUL        │     │  HURTIGT        │
          └────────┬────────┘     └────────┬────────┘
                   │                       │
                   ▼                       ▼
          sti/ 00 → 13            reference/
          (guidet kursus)         (opslag)
```

## Stien (fra nul til substrat)

| Trin | Fil | Du kan bagefter |
|---|---|---|
| 0 | [sti/00-velkommen.md](sti/00-velkommen.md) | hvad det er, setup, `doctor`, første `run` |
| 1 | [sti/01-foerste-skridt.md](sti/01-foerste-skridt.md) | hej, variabler, `print`, REPL |
| 2 | [sti/02-styring.md](sti/02-styring.md) | `if`, løkker, f-strenge |
| 3 | [sti/03-funktioner.md](sti/03-funktioner.md) | `fn`, args, `*args/**kwargs`, `yield` |
| 4 | [sti/04-samlinger.md](sti/04-samlinger.md) | lister, dicts, `string`, `json` |
| 5 | [sti/05-typer-klasser.md](sti/05-typer-klasser.md) | `type`, `class`, arv |
| 6 | [sti/06-fejl.md](sti/06-fejl.md) | `try/catch`, `throw`, `assert` |
| 7 | [sti/07-moduler.md](sti/07-moduler.md) | importer, stdlib, pakker |
| 8 | [sti/08-arrays.md](sti/08-arrays.md) | `triad.ntri`, vektorisering |
| 9 | [sti/09-python.md](sti/09-python.md) | Python-interop, safe/unsafe |
| 10 | [sti/10-solver.md](sti/10-solver.md) | solver, SAT, kubitter |
| 11 | [sti/11-dsl.md](sti/11-dsl.md) | `reg/ring/OBSERVE`-DSL, P1/P2/P3 |
| 12 | [sti/12-vaerktoejer.md](sti/12-vaerktoejer.md) | CLI: fmt, test, bench, debug, bundle… |
| 13 | [sti/13-nativ.md](sti/13-nativ.md) | nativ kompilering, C, kerne |

## Reference (direkte opslag)

- [reference/syntaks.md](reference/syntaks.md) — hele sproget på én visuel side
- [reference/stdlib.md](reference/stdlib.md) — modulerne `math`, `random`, `fs`…
- [reference/triad.md](reference/triad.md) — de 39 `triad.*`-moduler
- [reference/cli.md](reference/cli.md) — hver `./triad`-kommando med eksempel
- [reference/eksempler.md](reference/eksempler.md) — kort over `examples/` per domæne

## Konventioner i denne dok

- Hver `.tri`-blok her kører: `./triad run fil.tri`.
- `▸ output` markerer, hvad programmet udskriver.
- ```tri-frag-blokke er illustrative udsnit (kun syntaks); resten kører.
- `✎ prøv` er en invitation, ingen lektie.
