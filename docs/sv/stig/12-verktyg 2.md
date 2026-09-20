# 12 — Verktyg: CLI-turen

Du kan `run`, `check`, `doctor`. Resten av bänken:

## Varje dag

```sh
./triad fmt app.tri              # formattera
./triad repl                     # kladdblock
./triad watch app.tri            # kör om vid spara
./triad test                     # projekttester
./triad bench app.tri            # tider
```

## Se inuti

```sh
./triad debug app.tri -b 12      # brytpunkt rad 12
./triad tui app.tri              # interaktiv inspektör
./triad plot app.tri             # kör + ger PNG
./triad jit-stats                # hotspots
./triad memory status            # kristallminne
```

## Leverera

```sh
./triad compile app.tri --native -o app   # nativ binär
./triad bundle app.tri -o app.pyz         # fristående .pyz
./triad serve --port 8000                 # API-server
./triad play                              # 3D-motor i webbläsaren
```

## Fysik från skalet

```sh
./triad solve --N 64 --T 2.0 --dim 1      # utveckla, utan .tri
./triad observables run.npy               # läs sparat fält
```

## Projekt och docs

```sh
./triad init minapp && ./triad install && ./triad list
./triad docgen src/ -f html               # docs från kod
./triad lsp                               # editorstöd
```

Full tabell med noter: [referens/cli.md](../referens/cli.md).

✎ prova: `./triad bench` på din fil från steg 1 och läs talen.

Nästa: [13 — nativt](13-nativ.md).
