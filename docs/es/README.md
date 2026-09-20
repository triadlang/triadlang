# TriadLang — Documentación

```
                         ┌──────────────┐
                    ┌────│  ESTÁS       │────┐
                    │    │   AQUÍ       │    │
                    │    └──────────────┘    │
                    ▼                       ▼
          ┌─────────────────┐     ┌─────────────────┐
          │  NUNCA VI TRIAD │     │  YA PROGRAMO,   │
          │  EMPIEZA DE     │     │  LLÉVAME        │
          │  CERO           │     │  RÁPIDO         │
          └────────┬────────┘     └────────┬────────┘
                   │                       │
                   ▼                       ▼
          ruta/ 00 → 13           referencia/
          (curso guiado)          (consulta)
```

## La ruta (de cero al sustrato)

| Paso | Archivo | Sales sabiendo |
|---|---|---|
| 0 | [ruta/00-bienvenida.md](ruta/00-bienvenida.md) | qué es, instalar, `doctor`, primer `run` |
| 1 | [ruta/01-primeros-pasos.md](ruta/01-primeros-pasos.md) | hola, variables, `print`, REPL |
| 2 | [ruta/02-control.md](ruta/02-control.md) | `if`, bucles, f-strings |
| 3 | [ruta/03-funciones.md](ruta/03-funciones.md) | `fn`, args, `*args/**kwargs`, `yield` |
| 4 | [ruta/04-colecciones.md](ruta/04-colecciones.md) | listas, dicts, `string`, `json` |
| 5 | [ruta/05-tipos-clases.md](ruta/05-tipos-clases.md) | `type`, `class`, herencia |
| 6 | [ruta/06-errores.md](ruta/06-errores.md) | `try/catch`, `throw`, `assert` |
| 7 | [ruta/07-modulos.md](ruta/07-modulos.md) | imports, stdlib, paquetes |
| 8 | [ruta/08-arreglos.md](ruta/08-arreglos.md) | `triad.ntri`, vectorización |
| 9 | [ruta/09-python.md](ruta/09-python.md) | interop Python, safe/unsafe |
| 10 | [ruta/10-solver.md](ruta/10-solver.md) | solver, SAT, cúbits |
| 11 | [ruta/11-dsl.md](ruta/11-dsl.md) | DSL `reg/ring/OBSERVE`, P1/P2/P3 |
| 12 | [ruta/12-herramientas.md](ruta/12-herramientas.md) | CLI: fmt, test, bench, debug, bundle… |
| 13 | [ruta/13-nativo.md](ruta/13-nativo.md) | compilación nativa, C, kernel |

## Referencia (consulta directa)

- [referencia/sintaxis.md](referencia/sintaxis.md) — todo el lenguaje en una página visual
- [referencia/stdlib.md](referencia/stdlib.md) — módulos `math`, `random`, `fs`…
- [referencia/triad.md](referencia/triad.md) — los 39 módulos `triad.*`
- [referencia/cli.md](referencia/cli.md) — cada comando `./triad` con ejemplo
- [referencia/ejemplos.md](referencia/ejemplos.md) — mapa de `examples/` por dominio

## Convenciones de esta doc

- Todo bloque `.tri` aquí mostrado corre: `./triad run archivo.tri`.
- `▸ salida` marca lo que imprime el programa.
- Los bloques ```tri-frag son recortes ilustrativos (solo sintaxis); el resto corre.
- `✎ prueba` es una invitación, no tarea.
