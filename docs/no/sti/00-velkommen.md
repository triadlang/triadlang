# 00 — Velkommen

## Hva TriadLang er, på 30 sekunder

```
   .tri ──► ./triad ──► resultat
    │           │
    │           ├── direkte kjøring (run)
    │           ├── typekontroll (check)
    │           ├── ekte Python inni (import numpy, flask…)
    │           └── ekte fysikk inni (solver, SAT, kubiter)
    │
    └── eller fysisk DSL: reg / ring / OBSERVE
```

TriadLang er et komplett programmeringsspråk (variabler,
funksjoner, klasser, feil, moduler) med to uvanlige dører: det
importerer det ekte Python-økosystemet og bærer en motor for
integrert feltdynamikk (TRIAD-solveren) som løser ting som SAT og
kvantekretser.

## Sjekk at alt stemmer

```sh
./triad doctor
```

`▸ utdata` — en miljørapport (Python, avhengigheter, nativt).

Hvis noe mangler, fikser det guidede oppsettet det:

```sh
./triad setup
```

## Den første kjøringen (2 minutter)

Filen `hei.tri`:

```tri
print("Hello from TriadLang");
```

```sh
./triad run hei.tri
```

`▸ utdata`

```text
Hello from TriadLang
```

Fungerte det? Da kan du allerede kjøre TriadLang. Resten er språk.

## De to sjekkene du alltid bruker

```sh
./triad check hei.tri     # sjekker bare typer, kjører ingenting
./triad fmt hei.tri       # formatterer filen
```

## Mentalt kart videre

```
  steg 1–7     hverdagsspråk (som å lære et hvilket som helst språk)
       │
  steg 8–9     arrayer + Python (der triadlang møter verden)
       │
  steg 10–11   solver + DSL (der triadlang blir fysikk)
       │
  steg 12–13   verktøy + nativt (produksjon og kjerne)
```

✎ prøv: kjør `./triad repl`, skriv `print(1 + 1);`, deretter `exit`.

Neste: [01 — første steg](01-forste-steg.md).
