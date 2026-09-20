# 12 — Herramientas: el tour del CLI

Conoces `run`, `check`, `doctor`. El resto del banco:

## Cada día

```sh
./triad fmt app.tri              # formatea
./triad repl                     # borrador
./triad watch app.tri            # re-corre al guardar
./triad test                     # pruebas del proyecto
./triad bench app.tri            # tiempos
```

## Ver por dentro

```sh
./triad debug app.tri -b 12      # breakpoint línea 12
./triad tui app.tri              # inspector interactivo
./triad plot app.tri             # corre + emite PNG
./triad jit-stats                # puntos calientes
./triad memory status            # memoria cristal
```

## Entregar

```sh
./triad compile app.tri --native -o app   # binario nativo
./triad bundle app.tri -o app.pyz         # .pyz standalone
./triad serve --port 8000                 # servidor API
./triad play                              # engine 3D en browser
```

## Física por shell

```sh
./triad solve --N 64 --T 2.0 --dim 1      # evoluciona, sin .tri
./triad observables run.npy               # lee campo guardado
```

## Proyectos y docs

```sh
./triad init miapp && ./triad install && ./triad list
./triad docgen src/ -f html               # docs del código
./triad lsp                               # soporte editor
```

Tabla completa con notas: [referencia/cli.md](../referencia/cli.md).

✎ prueba: `./triad bench` en tu archivo del paso 1 y lee los números.

Siguiente: [13 — nativo](13-nativo.md).
