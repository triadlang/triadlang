# CLI — varje kommando med exempel

## Köra och kontrollera

```sh
./triad run app.tri               # kör (bara safe-importer)
./triad run app.tri --unsafe      # kör (full Python)
./triad check app.tri             # typkollar, kör inget
./triad fmt app.tri               # formatterar på plats
./triad repl                      # interaktivt kladdblock
./triad watch app.tri             # kör om vid spara
```

## Förstå och mäta

```sh
./triad test                      # projekttester
./triad bench app.tri             # tider
./triad debug app.tri -b 12       # brytpunkt, rad 12
./triad tui app.tri               # interaktiv inspektör
./triad jit-stats                 # hotspots
./triad docgen src/ -f html       # docs från kod
./triad lsp                       # editorstöd (server)
```

## Leverera och visa

```sh
./triad compile app.tri --native -o app   # nativ binär
./triad bundle app.tri -o app.pyz         # fristående .pyz
./triad serve --port 8000                 # API-server
./triad play                              # 3D-motor i webbläsaren
./triad plot app.tri --out fig.png        # kör + PNG
```

## Fysik och minne

```sh
./triad solve --N 64 --T 2.0 --dim 1      # utveckla, utan .tri
./triad observables run.npy               # läs sparat fält
./triad memory status                     # kristallminne
./triad memory record "idé"               # spara
./triad memory recall "idé"               # hämta
```

## Projekt

```sh
./triad init minapp   # ställning
./triad setup          # guidad installation
./triad install        # beroenden
./triad publish        # lokala registret
./triad list           # installerade
./triad doctor         # miljörapport
```

```
  skriv ──► check ──► run ──► bench ──► bundle/compile ──► leverera
               │        │
              fmt     debug/tui/plot
```
