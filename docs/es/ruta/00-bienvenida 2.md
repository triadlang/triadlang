# 00 — Bienvenida

## Qué es TriadLang, en 30 segundos

```
   .tri ──► ./triad ──► resultado
    │           │
    │           ├── ejecución directa (run)
    │           ├── chequeo de tipos (check)
    │           ├── Python real dentro (import numpy, flask…)
    │           └── física real dentro (solver, SAT, cúbits)
    │
    └── o DSL físico: reg / ring / OBSERVE
```

TriadLang es un lenguaje de programación completo (variables,
funciones, clases, errores, módulos) con dos puertas fuera de lo
común: importa el ecosistema Python de verdad y trae un motor de
dinámica física integral (el solver TRIAD) que resuelve cosas como
SAT y circuitos cuánticos.

## Ver si todo está bien

```sh
./triad doctor
```

`▸ salida` — un reporte del entorno (Python, dependencias, nativo).

Si falta algo, la guía lo resuelve:

```sh
./triad setup
```

## La primera corrida (2 minutos)

Archivo `hola.tri`:

```tri
print("Hello from TriadLang");
```

```sh
./triad run hola.tri
```

`▸ salida`

```text
Hello from TriadLang
```

¿Funcionó? Ya sabes correr TriadLang. El resto es lenguaje.

## Los dos chequeos de siempre

```sh
./triad check hola.tri     # solo chequea tipos, no ejecuta
./triad fmt hola.tri       # formatea el archivo
```

## Mapa mental de aquí en más

```
  pasos 1–7    lenguaje cotidiano (como aprender cualquier lenguaje)
       │
  pasos 8–9    arreglos + Python (donde triadlang encuentra el mundo)
       │
  pasos 10–11  solver + DSL (donde triadlang se vuelve física)
       │
  pasos 12–13  herramientas + nativo (producción y kernel)
```

✎ prueba: corre `./triad repl`, escribe `print(1 + 1);`, luego `exit`.

Siguiente: [01 — primeros pasos](01-primeros-pasos.md).
