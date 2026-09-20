# 12 — Værktøjer: CLI-turen

Du kan `run`, `check`, `doctor`. Resten af bænken:

## Hver dag

```sh
./triad fmt app.tri              # formattér
./triad repl                     # kladdeblok
./triad watch app.tri            # kør om ved lagring
./triad test                     # projekttests
./triad bench app.tri            # tider
```

## Se indeni

```sh
./triad debug app.tri -b 12      # breakpoint linje 12
./triad tui app.tri              # interaktiv inspektør
./triad plot app.tri             # kør + lav PNG
./triad jit-stats                # hotspots
./triad memory status            # krystalhukommelse
```

## Levér

```sh
./triad compile app.tri --native -o app   # nativ binær
./triad bundle app.tri -o app.pyz         # fritstående .pyz
./triad serve --port 8000                 # API-server
./triad play                              # 3D-motor i browseren
```

## Fysik fra skallen

```sh
./triad solve --N 64 --T 2.0 --dim 1      # udvikl, uden .tri
./triad observables run.npy               # læs gemt felt
```

## Projekter og docs

```sh
./triad init minapp && ./triad install && ./triad list
./triad docgen src/ -f html               # docs fra kode
./triad lsp                               # editorstøtte
```

Fuld tabel med noter: [reference/cli.md](../reference/cli.md).

✎ prøv: `./triad bench` på din fil fra trin 1 og læs tallene.

Næste: [13 — nativt](13-nativ.md).
