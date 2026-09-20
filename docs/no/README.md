# TriadLang — Dokumentasjon

```
                         ┌──────────────┐
                    ┌────│  DU ER       │────┐
                    │    │   HER        │    │
                    │    └──────────────┘    │
                    ▼                       ▼
          ┌─────────────────┐     ┌─────────────────┐
          │  ALDRI SETT     │     │  JEG KAN ALLEREDE│
          │  TRIAD: START   │     │  BARE VIS MEG   │
          │  FRA NULL       │     │  FORT           │
          └────────┬────────┘     └────────┬────────┘
                   │                       │
                   ▼                       ▼
          sti/ 00 → 13            referanse/
          (guidet kurs)           (oppslag)
```

## Stien (fra null til substrat)

| Steg | Fil | Du kan etterpå |
|---|---|---|
| 0 | [sti/00-velkommen.md](sti/00-velkommen.md) | hva det er, setup, `doctor`, første `run` |
| 1 | [sti/01-forste-steg.md](sti/01-forste-steg.md) | hei, variabler, `print`, REPL |
| 2 | [sti/02-kontroll.md](sti/02-kontroll.md) | `if`, løkker, f-strenger |
| 3 | [sti/03-funksjoner.md](sti/03-funksjoner.md) | `fn`, args, `*args/**kwargs`, `yield` |
| 4 | [sti/04-samlinger.md](sti/04-samlinger.md) | lister, dicter, `string`, `json` |
| 5 | [sti/05-typer-klasser.md](sti/05-typer-klasser.md) | `type`, `class`, arv |
| 6 | [sti/06-feil.md](sti/06-feil.md) | `try/catch`, `throw`, `assert` |
| 7 | [sti/07-moduler.md](sti/07-moduler.md) | importer, stdlib, pakker |
| 8 | [sti/08-arrays.md](sti/08-arrays.md) | `triad.ntri`, vektorisering |
| 9 | [sti/09-python.md](sti/09-python.md) | Python-interop, safe/unsafe |
| 10 | [sti/10-solver.md](sti/10-solver.md) | solver, SAT, kubiter |
| 11 | [sti/11-dsl.md](sti/11-dsl.md) | `reg/ring/OBSERVE`-DSL, P1/P2/P3 |
| 12 | [sti/12-verktoy.md](sti/12-verktoy.md) | CLI: fmt, test, bench, debug, bundle… |
| 13 | [sti/13-nativ.md](sti/13-nativ.md) | nativ kompilering, C, kjerne |

## Referanse (direkte oppslag)

- [referanse/syntaks.md](referanse/syntaks.md) — hele språket på én visuell side
- [referanse/stdlib.md](referanse/stdlib.md) — modulene `math`, `random`, `fs`…
- [referanse/triad.md](referanse/triad.md) — de 39 `triad.*`-modulene
- [referanse/cli.md](referanse/cli.md) — hver `./triad`-kommando med eksempel
- [referanse/eksempler.md](referanse/eksempler.md) — kart over `examples/` per domene

## Konvensjoner i denne dok

- Hver `.tri`-blokk her kjører: `./triad run fil.tri`.
- `▸ utdata` markerer hva programmet skriver ut.
- ```tri-frag-blokker er illustrative utsnitt (bare syntaks); resten kjører.
- `✎ prøv` er en invitasjon, ingen lekse.
