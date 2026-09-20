# 00 — Velkommen

## Hvad TriadLang er, på 30 sekunder

```
   .tri ──► ./triad ──► resultat
    │           │
    │           ├── direkte kørsel (run)
    │           ├── typecheck (check)
    │           ├── ægte Python indeni (import numpy, flask…)
    │           └── ægte fysik indeni (solver, SAT, kubitter)
    │
    └── eller fysisk DSL: reg / ring / OBSERVE
```

TriadLang er et komplet programmeringssprog (variabler,
funktioner, klasser, fejl, moduler) med to usædvanlige døre: det
importerer det ægte Python-økosystem og bærer en motor for
integralet feltdynamik (TRIAD-solveren), der løser ting som SAT og
kvantekredsløb.

## Tjek at alt stemmer

```sh
./triad doctor
```

`▸ output` — en miljørapport (Python, afhængigheder, nativt).

Hvis noget mangler, fikser det guidede setup det:

```sh
./triad setup
```

## Den første kørsel (2 minutter)

Filen `hej.tri`:

```tri
print("Hello from TriadLang");
```

```sh
./triad run hej.tri
```

`▸ output`

```text
Hello from TriadLang
```

Virker det? Så kan du allerede køre TriadLang. Resten er sprog.

## De to tjek du altid bruger

```sh
./triad check hej.tri     # tjekker kun typer, kører intet
./triad fmt hej.tri       # formatterer filen
```

## Mentalt kort fremad

```
  trin 1–7     hverdagssprog (som at lære et hvilket som helst sprog)
       │
  trin 8–9     arrays + Python (hvor triadlang møder verden)
       │
  trin 10–11   solver + DSL (hvor triadlang bliver fysik)
       │
  trin 12–13   værktøj + nativt (produktion og kerne)
```

✎ prøv: kør `./triad repl`, skriv `print(1 + 1);`, dernæst `exit`.

Næste: [01 — første skridt](01-foerste-skridt.md).
