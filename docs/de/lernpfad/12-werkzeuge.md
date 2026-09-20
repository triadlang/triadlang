# 12 — Werkzeuge: die CLI-Tour

Du kennst `run`, `check`, `doctor`. Der Rest der Werkbank:

## Jeden Tag

```sh
./triad fmt app.tri              # formatieren
./triad repl                     # Notizblock
./triad watch app.tri            # neu starten bei Speichern
./triad test                     # Projekttests
./triad bench app.tri            # Zeiten
```

## Hineinsehen

```sh
./triad debug app.tri -b 12      # Breakpoint Zeile 12
./triad tui app.tri              # interaktiver Inspektor
./triad plot app.tri             # starten + PNG
./triad jit-stats                # Hotspots
./triad memory status            # Kristallgedächtnis
```

## Liefern

```sh
./triad compile app.tri --native -o app   # natives Binary
./triad bundle app.tri -o app.pyz         # standalone .pyz
./triad serve --port 8000                 # API-Server
./triad play                              # 3D-Engine im Browser
```

## Physik aus der Shell

```sh
./triad solve --N 64 --T 2.0 --dim 1      # entwickeln, ohne .tri
./triad observables run.npy               # gespeichertes Feld lesen
```

## Projekte und Docs

```sh
./triad init meineapp && ./triad install && ./triad list
./triad docgen src/ -f html               # Docs aus Code
./triad lsp                               # Editor-Support
```

Volle Tabelle mit Notizen: [referenz/cli.md](../referenz/cli.md).

✎ probier es: `./triad bench` mit deiner Datei aus Schritt 1 und
lies die Zahlen.

Weiter: [13 — Nativ](13-nativ.md).
