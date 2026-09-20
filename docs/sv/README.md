# TriadLang — Dokumentation

```
                         ┌──────────────┐
                    ┌────│  DU ÄR       │────┐
                    │    │   HÄR        │    │
                    │    └──────────────┘    │
                    ▼                       ▼
          ┌─────────────────┐     ┌─────────────────┐
          │  ALDRIG SETT    │     │  JAG KAN REDAN, │
          │  TRIAD: BÖRJA   │     │  VISA MIG       │
          │  FRÅN NOLL      │     │  SNABBT         │
          └────────┬────────┘     └────────┬────────┘
                   │                       │
                   ▼                       ▼
          stig/ 00 → 13           referens/
          (guidad kurs)           (uppslag)
```

## Stigen (från noll till substrat)

| Steg | Fil | Du kan sedan |
|---|---|---|
| 0 | [stig/00-valkommen.md](stig/00-valkommen.md) | vad det är, setup, `doctor`, första `run` |
| 1 | [stig/01-forsta-steg.md](stig/01-forsta-steg.md) | hej, variabler, `print`, REPL |
| 2 | [stig/02-styrning.md](stig/02-styrning.md) | `if`, loopar, f-strängar |
| 3 | [stig/03-funktioner.md](stig/03-funktioner.md) | `fn`, args, `*args/**kwargs`, `yield` |
| 4 | [stig/04-samlingar.md](stig/04-samlingar.md) | listor, dicts, `string`, `json` |
| 5 | [stig/05-typer-klasser.md](stig/05-typer-klasser.md) | `type`, `class`, arv |
| 6 | [stig/06-fel.md](stig/06-fel.md) | `try/catch`, `throw`, `assert` |
| 7 | [stig/07-moduler.md](stig/07-moduler.md) | importer, stdlib, paket |
| 8 | [stig/08-arrayer.md](stig/08-arrayer.md) | `triad.ntri`, vektorisering |
| 9 | [stig/09-python.md](stig/09-python.md) | Python-interop, safe/unsafe |
| 10 | [stig/10-losare.md](stig/10-losare.md) | solver, SAT, kubitar |
| 11 | [stig/11-dsl.md](stig/11-dsl.md) | `reg/ring/OBSERVE`-DSL, P1/P2/P3 |
| 12 | [stig/12-verktyg.md](stig/12-verktyg.md) | CLI: fmt, test, bench, debug, bundle… |
| 13 | [stig/13-nativ.md](stig/13-nativ.md) | nativ kompilering, C, kärna |

## Referens (direkt uppslag)

- [referens/syntax.md](referens/syntax.md) — hela språket på en visuell sida
- [referens/stdlib.md](referens/stdlib.md) — modulerna `math`, `random`, `fs`…
- [referens/triad.md](referens/triad.md) — de 39 `triad.*`-modulerna
- [referens/cli.md](referens/cli.md) — varje `./triad`-kommando med exempel
- [referens/exempel.md](referens/exempel.md) — karta över `examples/` per domän

## Konventioner i denna dok

- Varje `.tri`-block här körs: `./triad run fil.tri`.
- `▸ utdata` markerar vad programmet skriver ut.
- ```tri-frag-block är illustrativa utsnitt (endast syntax); resten körs.
- `✎ prova` är en inbjudan, ingen läxa.
