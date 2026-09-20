# 00 — Välkommen

## Vad TriadLang är, på 30 sekunder

```
   .tri ──► ./triad ──► resultat
    │           │
    │           ├── direkt körning (run)
    │           ├── typkontroll (check)
    │           ├── riktig Python inuti (import numpy, flask…)
    │           └── riktig fysik inuti (solver, SAT, kubitar)
    │
    └── eller fysikalisk DSL: reg / ring / OBSERVE
```

TriadLang är ett komplett programspråk (variabler, funktioner,
klasser, fel, moduler) med två ovanliga dörrar: det importerar det
riktiga Python-ekosystemet och bär en motor för integral fältdynamik
(TRIAD-solvern) som löser saker som SAT och kvantkretsar.

## Kolla att allt stämmer

```sh
./triad doctor
```

`▸ utdata` — en miljörapport (Python, beroenden, nativt).

Om något saknas fixar den guidade installationen det:

```sh
./triad setup
```

## Den första körningen (2 minuter)

Filen `hej.tri`:

```tri
print("Hello from TriadLang");
```

```sh
./triad run hej.tri
```

`▸ utdata`

```text
Hello from TriadLang
```

Funkade det? Då kan du redan köra TriadLang. Resten är språk.

## De två kontrollerna du alltid använder

```sh
./triad check hej.tri     # kontrollerar bara typer, kör inget
./triad fmt hej.tri       # formatterar filen
```

## Mental karta framåt

```
  steg 1–7     vardagsspråk (som att lära vilket språk som helst)
       │
  steg 8–9     arrayer + Python (där triadlang möter världen)
       │
  steg 10–11   solver + DSL (där triadlang blir fysik)
       │
  steg 12–13   verktyg + nativt (produktion och kärna)
```

✎ prova: kör `./triad repl`, skriv `print(1 + 1);`, sedan `exit`.

Nästa: [01 — första steg](01-forsta-steg.md).
