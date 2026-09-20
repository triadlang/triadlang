# CLI — hver kommando med eksempel

## Kjøre og sjekke

```sh
./triad run app.tri               # kjør (bare safe-importer)
./triad run app.tri --unsafe      # kjør (full Python)
./triad check app.tri             # typesjekk, kjører ingenting
./triad fmt app.tri               # formatter på plass
./triad repl                      # interaktiv kladdeblokk
./triad watch app.tri             # kjør om ved lagring
```

## Forstå og måle

```sh
./triad test                      # prosjekttester
./triad bench app.tri             # tider
./triad debug app.tri -b 12       # bruddpunkt, linje 12
./triad tui app.tri               # interaktiv inspektør
./triad jit-stats                 # hotspots
./triad docgen src/ -f html       # docs fra kode
./triad lsp                       # editorstøtte (server)
```

## Levere og vise

```sh
./triad compile app.tri --native -o app   # nativ binær
./triad bundle app.tri -o app.pyz         # frittstående .pyz
./triad serve --port 8000                 # API-server
./triad play                              # 3D-motor i nettleseren
./triad plot app.tri --out fig.png        # kjør + PNG
```

## Fysikk og minne

```sh
./triad solve --N 64 --T 2.0 --dim 1      # utvikle, uten .tri
./triad observables run.npy               # les lagret felt
./triad memory status                     # krystallminne
./triad memory record "idé"               # lagre
./triad memory recall "idé"               # hente
```

## Prosjekter

```sh
./triad init minapp   # skjelett
./triad setup          # guidet installasjon
./triad install        # avhengigheter
./triad publish        # lokale registeret
./triad list           # installerte
./triad doctor         # miljørapport
```

```
  skriv ──► check ──► run ──► bench ──► bundle/compile ──► lever
               │        │
              fmt     debug/tui/plot
```
