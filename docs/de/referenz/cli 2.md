# CLI — jeder Befehl mit Beispiel

## Starten und prüfen

```sh
./triad run app.tri               # starten (nur safe-Imports)
./triad run app.tri --unsafe      # starten (volles Python)
./triad check app.tri             # Typen prüfen, nichts ausführen
./triad fmt app.tri               # in place formatieren
./triad repl                      # interaktiver Notizblock
./triad watch app.tri             # neu starten bei Speichern
```

## Verstehen und messen

```sh
./triad test                      # Projekttests
./triad bench app.tri             # Zeiten
./triad debug app.tri -b 12       # Breakpoint, Zeile 12
./triad tui app.tri               # interaktiver Inspektor
./triad jit-stats                 # Hotspots
./triad docgen src/ -f html       # Docs aus Code
./triad lsp                       # Editor-Support (Server)
```

## Liefern und zeigen

```sh
./triad compile app.tri --native -o app   # natives Binary
./triad bundle app.tri -o app.pyz         # standalone .pyz
./triad serve --port 8000                 # API-Server
./triad play                              # 3D-Engine im Browser
./triad plot app.tri --out fig.png        # starten + PNG
```

## Physik und Gedächtnis

```sh
./triad solve --N 64 --T 2.0 --dim 1      # entwickeln, ohne .tri
./triad observables run.npy               # gespeichertes Feld lesen
./triad memory status                     # Kristallgedächtnis
./triad memory record "idee"              # ablegen
./triad memory recall "idee"              # holen
```

## Projekte

```sh
./triad init meineapp  # Gerüst
./triad setup          # geführte Installation
./triad install        # Abhängigkeiten
./triad publish        # lokales Register
./triad list           # Installiertes
./triad doctor         # Umgebungsbericht
```

```
  schreiben ──► check ──► run ──► bench ──► bundle/compile ──► liefern
                   │        │
                  fmt     debug/tui/plot
```
