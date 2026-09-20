# 09 — Python: økosystemet er dit

TriadLang importerer ægte Python-pakker. Samme navne, samme objekter.

## numpy kommer som det er

```tri
import numpy;

let a = numpy.array([1, 2, 3, 4, 5]);
print(numpy.sum(a));    // array(15, dtype=int64)
print(numpy.mean(a));   // array(3.0, dtype=float64)
```

From-importer også:

```tri
from numpy import sum, mean;

print(sum([10, 20, 30]));   // array(60, dtype=int64)
```

## flask serverer, click klikker

```tri
import flask;

let app = flask.Flask("triad_test");
print(app.name);   // triad_test
```

```tri
import click;

print("click works");   // bindingen loader
```

En rute, der løser fysik og svarer JSON (se
`examples/interop/flask_solver.tri`): definér med
`app.add_url_rule`, svar med `flask.jsonify`.

Testklienten beviser uden server:

```tri
// examples/interop/boundary_flask_testclient.tri
// ▸ status code = 200, json status = ok, json value = 42
```

## Sikkert som standard, usikkert på ord

```
  ./triad run app.tri              sikkert: kun velsignede importer
  ./triad run app.tri --unsafe     fuld Python på egen risiko
```

Safe tillader: `triad math random statistics json itertools
functools collections re datetime fs io string time hash csv logging
threading net os subprocess socket numpy click flask`.

Uden for listen kræves `--unsafe` — ellers falder importen og
udskriver den tilladte liste.

## Interop-kortet

```
  talknusning ─► import numpy / triad.ntri
  websvar ─────► import flask (+ testklient)
  CLI'er ──────► import click
  filer/proc ──► import os (safe) / subprocess (safe)
  alt andet ───► --unsafe
```

Ni kørbare eksempler i `examples/interop/` — alle grønne.

✎ prøv: en `.tri`, der importerer `numpy`, bygger `arange(10)`
med `numpy.arange` og udskriver summen.

Næste: [10 — solver](10-solver.md).
