# 07 — Módulos: importar el mundo

## Las tres formas

```tri
import math;                    // módulo entero
import math as m;               // con apodo
from math import sqrt, pi;      // solo lo necesario
```

## Stdlib esencial (siempre disponible)

```
  math         sqrt sin cos log exp floor ceil pi e min max
  random       random randint choice seed shuffle uniform
  string       split join replace lower upper strip
  json         parse stringify
  fs           read_text write_text exists listdir join tempdir
  time         now sleep
  collections  len range enumerate sorted zip map filter…
  datetime     fechas y horas
  regex        expresiones regulares
  plot         gráficos (triad.plot es lo mismo)
  io           print input
```

Un ejemplo junto:

```tri
import math;
import random;

random.seed(0);
print(math.sqrt(25.0));       // 5.0
print(random.randint(1, 10)); // 7 (con seed 0)
```

## Archivos (`fs`)

```tri
import fs;

let ruta = fs.join(fs.tempdir(), "hola.txt");
fs.write_text(ruta, "Hello file!");
print(fs.read_text(ruta));   // Hello file!
print(fs.exists(ruta));      // true
```

## Tus propios módulos

Archivo `util.tri`:

```tri
fn doble(n) {
    return n * 2;
}
```

Archivo `app.tri`, en la misma carpeta:

```tri
import util;

print(util.doble(21));   // 42
```

## Paquetes (proyectos mayores)

```sh
./triad init miapp        # crea proyecto
./triad install           # instala dependencias
./triad publish           # publica en el registro local
./triad list              # lista instalados
```

Las dependencias viven en `triad_modules/`.

✎ prueba: un módulo `saluda.tri` con `fn hola(nombre)` y un
programa que lo importe y use.

Siguiente: [08 — arreglos](08-arreglos.md).
