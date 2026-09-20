# 09 — Python: el ecosistema es tuyo

TriadLang importa paquetes Python de verdad. Mismos nombres, mismos objetos.

## numpy llega como es

```tri
import numpy;

let a = numpy.array([1, 2, 3, 4, 5]);
print(numpy.sum(a));    // array(15, dtype=int64)
print(numpy.mean(a));   // array(3.0, dtype=float64)
```

From-imports también:

```tri
from numpy import sum, mean;

print(sum([10, 20, 30]));   // array(60, dtype=int64)
```

## flask sirve, click clica

```tri
import flask;

let app = flask.Flask("triad_test");
print(app.name);   // triad_test
```

```tri
import click;

print("click works");   // el binding carga
```

Una ruta que resuelve física y responde JSON (ver
`examples/interop/flask_solver.tri`): define con `app.add_url_rule`,
responde con `flask.jsonify`.

El test client lo prueba sin servidor:

```tri
// examples/interop/boundary_flask_testclient.tri
// ▸ status code = 200, json status = ok, json value = 42
```

## Safe por defecto, unsafe cuando tú dices

```
  ./triad run app.tri              safe: solo imports benditos
  ./triad run app.tri --unsafe     Python total por tu cuenta
```

El modo safe permite: `triad math random statistics json itertools
functools collections re datetime fs io string time hash csv logging
threading net os subprocess socket numpy click flask`.

Fuera de esa lista necesita `--unsafe` — o falla en el import
mostrando la lista permitida.

## El mapa interop

```
  cuenta numérica ─► import numpy / triad.ntri
  endpoints web ───► import flask (+ test client)
  CLIs ────────────► import click
  archivos/proc ───► import os (safe) / subprocess (safe)
  todo lo demás ───► --unsafe
```

Nueve ejemplos corridos en `examples/interop/` — todos verdes.

✎ prueba: un `.tri` que importe `numpy`, arme `arange(10)`
con `numpy.arange` e imprima la suma.

Siguiente: [10 — solver](10-solver.md).
