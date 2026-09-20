# 09 — Python: das Ökosystem gehört dir

TriadLang importiert echte Python-Pakete. Gleiche Namen, gleiche Objekte.

## numpy kommt wie es ist

```tri
import numpy;

let a = numpy.array([1, 2, 3, 4, 5]);
print(numpy.sum(a));    // array(15, dtype=int64)
print(numpy.mean(a));   // array(3.0, dtype=float64)
```

From-Imports auch:

```tri
from numpy import sum, mean;

print(sum([10, 20, 30]));   // array(60, dtype=int64)
```

## flask serviert, click klickt

```tri
import flask;

let app = flask.Flask("triad_test");
print(app.name);   // triad_test
```

```tri
import click;

print("click works");   // das Binding lädt
```

Eine Route, die Physik löst und JSON antwortet (siehe
`examples/interop/flask_solver.tri`): definieren mit
`app.add_url_rule`, antworten mit `flask.jsonify`.

Der Test-Client beweist es ohne Server:

```tri
// examples/interop/boundary_flask_testclient.tri
// ▸ status code = 200, json status = ok, json value = 42
```

## Sicher ab Werk, unsicher auf Wort

```
  ./triad run app.tri              sicher: nur gesegnete Imports
  ./triad run app.tri --unsafe     volles Python auf Risiko
```

Safe erlaubt: `triad math random statistics json itertools
functools collections re datetime fs io string time hash csv logging
threading net os subprocess socket numpy click flask`.

Außerhalb der Liste braucht es `--unsafe` — sonst scheitert der
Import und druckt die erlaubte Liste.

## Die Interop-Karte

```
  Zahlen batzen ──► import numpy / triad.ntri
  Web-Antworten ──► import flask (+ Test-Client)
  CLIs ───────────► import click
  Dateien/Proz. ──► import os (safe) / subprocess (safe)
  alles andere ───► --unsafe
```

Neun lauffähige Beispiele in `examples/interop/` — alle grün.

✎ probier es: Ein `.tri`, das `numpy` importiert, `arange(10)`
mit `numpy.arange` baut und die Summe druckt.

Weiter: [10 — Solver](10-solver.md).
