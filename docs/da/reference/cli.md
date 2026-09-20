# CLI — hver kommando med eksempel

## Kør og tjek

```sh
./triad run app.tri               # kør (kun safe-importer)
./triad run app.tri --unsafe      # kør (fuld Python)
./triad check app.tri             # typecheck, kører intet
./triad fmt app.tri               # formattér på plads
./triad repl                      # interaktiv kladdeblok
./triad watch app.tri             # kør om ved lagring
```

## Forstå og mål

```sh
./triad test                      # projekttests
./triad bench app.tri             # tider
./triad debug app.tri -b 12       # breakpoint, linje 12
./triad tui app.tri               # interaktiv inspektør
./triad jit-stats                 # hotspots
./triad docgen src/ -f html       # docs fra kode
./triad lsp                       # editorstøtte (server)
```

## Levér og vis

```sh
./triad compile app.tri --native -o app   # nativ binær
./triad bundle app.tri -o app.pyz         # fritstående .pyz
./triad serve --port 8000                 # API-server
./triad play                              # 3D-motor i browseren
./triad plot app.tri --out fig.png        # kør + PNG
```

## Fysik og hukommelse

```sh
./triad solve --N 64 --T 2.0 --dim 1      # udvikl, uden .tri
./triad observables run.npy               # læs gemt felt
./triad memory status                     # krystalhukommelse
./triad memory record "idé"               # gem
./triad memory recall "idé"               # hent
```

## Projekter

```sh
./triad init minapp   # skelet
./triad setup          # guidet installation
./triad install        # afhængigheder
./triad publish        # lokale register
./triad list           # installerede
./triad doctor         # miljørapport
```

```
  skriv ──► check ──► run ──► bench ──► bundle/compile ──► levér
               │        │
              fmt     debug/tui/plot
```
