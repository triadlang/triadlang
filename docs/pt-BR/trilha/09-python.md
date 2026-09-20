# 09 — Python: o ecossistema é seu

TriadLang importa pacotes Python de verdade. Mesmos nomes, mesmos objetos.

## numpy chega como é

```tri
import numpy;

let a = numpy.array([1, 2, 3, 4, 5]);
print(numpy.sum(a));    // array(15, dtype=int64)
print(numpy.mean(a));   // array(3.0, dtype=float64)
```

From-imports também:

```tri
from numpy import sum, mean;

print(sum([10, 20, 30]));   // array(60, dtype=int64)
```

## flask serve, click clica

```tri
import flask;

let app = flask.Flask("triad_test");
print(app.name);   // triad_test
```

```tri
import click;

print("click works");   // o binding carrega
```

Uma rota que resolve física e responde JSON (ver
`examples/interop/flask_solver.tri`): define com `app.add_url_rule`,
responde com `flask.jsonify`.

O test client prova sem servidor:

```tri
// examples/interop/boundary_flask_testclient.tri
// ▸ status code = 200, json status = ok, json value = 42
```

## Safe por padrão, unsafe quando você diz

```
  ./triad run app.tri              safe: só imports abençoados
  ./triad run app.tri --unsafe     Python total por sua conta
```

O modo safe permite: `triad math random statistics json itertools
functools collections re datetime fs io string time hash csv logging
threading net os subprocess socket numpy click flask`.

Fora dessa lista precisa de `--unsafe` — ou falha no import mostrando
a lista permitida.

## O mapa interop

```
  conta numérica ──► import numpy / triad.ntri
  endpoints web ───► import flask (+ test client)
  CLIs ────────────► import click
  arquivos/proc ───► import os (safe) / subprocess (safe)
  todo o resto ────► --unsafe
```

Nove exemplos rodáveis em `examples/interop/` — todos verdes.

✎ experimente: um `.tri` que importa `numpy`, monta `arange(10)`
com `numpy.arange` e imprime a soma.

Próximo: [10 — solver](10-solver.md).
