# CLI — cada comando con ejemplo

## Correr y chequear

```sh
./triad run app.tri               # corre (solo imports safe)
./triad run app.tri --unsafe      # corre (Python total)
./triad check app.tri             # chequea tipos, sin correr
./triad fmt app.tri               # formatea en el lugar
./triad repl                      # borrador interactivo
./triad watch app.tri             # re-corre al guardar
```

## Entender y medir

```sh
./triad test                      # pruebas del proyecto
./triad bench app.tri             # tiempos
./triad debug app.tri -b 12       # breakpoint, línea 12
./triad tui app.tri               # inspector interactivo
./triad jit-stats                 # puntos calientes
./triad docgen src/ -f html       # docs del código
./triad lsp                       # soporte editor (servidor)
```

## Entregar y mostrar

```sh
./triad compile app.tri --native -o app   # binario nativo
./triad bundle app.tri -o app.pyz         # .pyz standalone
./triad serve --port 8000                 # servidor API
./triad play                              # engine 3D en browser
./triad plot app.tri --out fig.png        # corre + PNG
```

## Física y memoria

```sh
./triad solve --N 64 --T 2.0 --dim 1      # evoluciona, sin .tri
./triad observables run.npy               # lee campo guardado
./triad memory status                     # memoria cristal
./triad memory record "idea"              # guarda
./triad memory recall "idea"              # recupera
```

## Proyectos

```sh
./triad init miapp     # esqueleto
./triad setup          # instalación guiada
./triad install        # dependencias
./triad publish        # registro local
./triad list           # instalados
./triad doctor         # reporte del entorno
```

```
  escribe ──► check ──► run ──► bench ──► bundle/compile ──► entrega
                 │        │
                fmt     debug/tui/plot
```
