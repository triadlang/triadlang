# 09 — Python: økosystemet er ditt

TriadLang importerer ekte Python-pakker. Samme navn, samme objekter.

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

print("click works");   // bindingen laster
```

En rute som løser fysikk og svarer JSON (se
`examples/interop/flask_solver.tri`): definer med
`app.add_url_rule`, svar med `flask.jsonify`.

Testklienten beviser uten server:

```tri
// examples/interop/boundary_flask_testclient.tri
// ▸ status code = 200, json status = ok, json value = 42
```

## Trygt som standard, utrygt på ord

```
  ./triad run app.tri              trygt: bare velsignede importer
  ./triad run app.tri --unsafe     full Python på egen risiko
```

Safe tillater: `triad math random statistics json itertools
functools collections re datetime fs io string time hash csv logging
threading net os subprocess socket numpy click flask`.

Utenfor listen kreves `--unsafe` — ellers faller importen og
skriver ut den tillatte listen.

## Interop-kartet

```
  tallknusing ─► import numpy / triad.ntri
  websvar ─────► import flask (+ testklient)
  CLI-er ──────► import click
  filer/pros ──► import os (safe) / subprocess (safe)
  alt annet ───► --unsafe
```

Ni kjørbare eksempler i `examples/interop/` — alle grønne.

✎ prøv: en `.tri` som importerer `numpy`, bygger `arange(10)`
med `numpy.arange` og skriver ut summen.

Neste: [10 — solver](10-solver.md).
