# 12 — Verktøy: CLI-turen

Du kan `run`, `check`, `doctor`. Resten av benken:

## Hver dag

```sh
./triad fmt app.tri              # formatter
./triad repl                     # kladdeblokk
./triad watch app.tri            # kjør om ved lagring
./triad test                     # prosjekttester
./triad bench app.tri            # tider
```

## Se inni

```sh
./triad debug app.tri -b 12      # bruddpunkt linje 12
./triad tui app.tri              # interaktiv inspektør
./triad plot app.tri             # kjør + lag PNG
./triad jit-stats                # hotspots
./triad memory status            # krystallminne
```

## Lever

```sh
./triad compile app.tri --native -o app   # nativ binær
./triad bundle app.tri -o app.pyz         # frittstående .pyz
./triad serve --port 8000                 # API-server
./triad play                              # 3D-motor i nettleseren
```

## Fysikk fra skallet

```sh
./triad solve --N 64 --T 2.0 --dim 1      # utvikle, uten .tri
./triad observables run.npy               # les lagret felt
```

## Prosjekter og docs

```sh
./triad init minapp && ./triad install && ./triad list
./triad docgen src/ -f html               # docs fra kode
./triad lsp                               # editorstøtte
```

Full tabell med noter: [referanse/cli.md](../referanse/cli.md).

✎ prøv: `./triad bench` på filen din fra steg 1 og les tallene.

Neste: [13 — nativt](13-nativ.md).
