# 09 — Python: ekosystemet är ditt

TriadLang importerar riktiga Python-paket. Samma namn, samma objekt.

## numpy kommer som det är

```tri
import numpy;

let a = numpy.array([1, 2, 3, 4, 5]);
print(numpy.sum(a));    // array(15, dtype=int64)
print(numpy.mean(a));   // array(3.0, dtype=float64)
```

From-importer också:

```tri
from numpy import sum, mean;

print(sum([10, 20, 30]));   // array(60, dtype=int64)
```

## flask servar, click klickar

```tri
import flask;

let app = flask.Flask("triad_test");
print(app.name);   // triad_test
```

```tri
import click;

print("click works");   // bindningen laddar
```

En route som löser fysik och svarar JSON (se
`examples/interop/flask_solver.tri`): definiera med
`app.add_url_rule`, svara med `flask.jsonify`.

Testklienten bevisar utan server:

```tri
// examples/interop/boundary_flask_testclient.tri
// ▸ status code = 200, json status = ok, json value = 42
```

## Säkert som standard, osäkert på ord

```
  ./triad run app.tri              säkert: bara välsignade importer
  ./triad run app.tri --unsafe     full Python på egen risk
```

Safe tillåter: `triad math random statistics json itertools
functools collections re datetime fs io string time hash csv logging
threading net os subprocess socket numpy click flask`.

Utanför listan krävs `--unsafe` — annars faller importen och
skriver ut den tillåtna listan.

## Interop-kartan

```
  siffertugg ──► import numpy / triad.ntri
  webbsvar ────► import flask (+ testklient)
  CLI:er ──────► import click
  filer/proc ──► import os (safe) / subprocess (safe)
  allt annat ──► --unsafe
```

Nio körbara exempel i `examples/interop/` — alla gröna.

✎ prova: en `.tri` som importerar `numpy`, bygger `arange(10)`
med `numpy.arange` och skriver ut summan.

Nästa: [10 — lösare](10-losare.md).
