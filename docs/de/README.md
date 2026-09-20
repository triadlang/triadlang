# TriadLang — Dokumentation

```
                         ┌──────────────┐
                    ┌────│  DU BIST     │────┐
                    │    │    HIER      │    │
                    │    └──────────────┘    │
                    ▼                       ▼
          ┌─────────────────┐     ┌─────────────────┐
          │  NOCH NIE       │     │  ICH KANN       │
          │  TRIAD: AB NULL │     │  SCHON, NUR     │
          │  ANFANGEN       │     │  SCHNELL FINDEN │
          └────────┬────────┘     └────────┬────────┘
                   │                       │
                   ▼                       ▼
          lernpfad/ 00 → 13       referenz/
          (Kurs)                  (Nachschlagen)
```

## Der Pfad (von null bis Substrat)

| Schritt | Datei | Danach kannst du |
|---|---|---|
| 0 | [lernpfad/00-willkommen.md](lernpfad/00-willkommen.md) | was es ist, Setup, `doctor`, erstes `run` |
| 1 | [lernpfad/01-erste-schritte.md](lernpfad/01-erste-schritte.md) | hello, Variablen, `print`, REPL |
| 2 | [lernpfad/02-steuerung.md](lernpfad/02-steuerung.md) | `if`, Schleifen, f-strings |
| 3 | [lernpfad/03-funktionen.md](lernpfad/03-funktionen.md) | `fn`, Args, `*args/**kwargs`, `yield` |
| 4 | [lernpfad/04-sammlungen.md](lernpfad/04-sammlungen.md) | Listen, Dicts, `string`, `json` |
| 5 | [lernpfad/05-typen-klassen.md](lernpfad/05-typen-klassen.md) | `type`, `class`, Vererbung |
| 6 | [lernpfad/06-fehler.md](lernpfad/06-fehler.md) | `try/catch`, `throw`, `assert` |
| 7 | [lernpfad/07-module.md](lernpfad/07-module.md) | Imports, Stdlib, Pakete |
| 8 | [lernpfad/08-arrays.md](lernpfad/08-arrays.md) | `triad.ntri`, Vektorisierung |
| 9 | [lernpfad/09-python.md](lernpfad/09-python.md) | Python-Interop, safe/unsafe |
| 10 | [lernpfad/10-solver.md](lernpfad/10-solver.md) | Solver, SAT, Qubits |
| 11 | [lernpfad/11-dsl.md](lernpfad/11-dsl.md) | `reg/ring/OBSERVE`-DSL, P1/P2/P3 |
| 12 | [lernpfad/12-werkzeuge.md](lernpfad/12-werkzeuge.md) | CLI: fmt, test, bench, debug, bundle… |
| 13 | [lernpfad/13-nativ.md](lernpfad/13-nativ.md) | nativ kompilieren, C, Kernel |

## Referenz (direkt nachschlagen)

- [referenz/syntax.md](referenz/syntax.md) — die ganze Sprache auf einer Seite
- [referenz/stdlib.md](referenz/stdlib.md) — Module `math`, `random`, `fs`…
- [referenz/triad.md](referenz/triad.md) — die 39 `triad.*`-Module
- [referenz/cli.md](referenz/cli.md) — jeder `./triad`-Befehl mit Beispiel
- [referenz/beispiele.md](referenz/beispiele.md) — Karte von `examples/` nach Domäne

## Konventionen dieser Doku

- Jeder `.tri`-Block hier läuft: `./triad run datei.tri`.
- `▸ Ausgabe` markiert, was das Programm druckt.
- ```tri-frag-Blöcke sind illustrative Ausschnitte (nur Syntax); der Rest läuft.
- `✎ probier es` ist eine Einladung, keine Hausaufgabe.
