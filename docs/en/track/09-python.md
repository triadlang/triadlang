# 09 — Python: the ecosystem is yours

TriadLang imports real Python packages. Same names, same objects.

## numpy arrives as-is

```tri
import numpy;

let a = numpy.array([1, 2, 3, 4, 5]);
print(numpy.sum(a));    // array(15, dtype=int64)
print(numpy.mean(a));   // array(3.0, dtype=float64)
```

From-imports too:

```tri
from numpy import sum, mean;

print(sum([10, 20, 30]));   // array(60, dtype=int64)
```

## flask serves, click clicks

```tri
import flask;

let app = flask.Flask("triad_test");
print(app.name);   // triad_test
```

```tri
import click;

print("click works");   // the binding loads
```

A route that solves physics and answers JSON (see
`examples/interop/flask_solver.tri`): define with `app.add_url_rule`,
answer with `flask.jsonify`.

The test client proves it without a server:

```tri
// examples/interop/boundary_flask_testclient.tri
// ▸ status code = 200, json status = ok, json value = 42
```

## Safe by default, unsafe when you say so

```
  ./triad run app.tri              safe: only blessed imports
  ./triad run app.tri --unsafe     full Python at your risk
```

Safe mode allows: `triad math random statistics json itertools
functools collections re datetime fs io string time hash csv logging
threading net os subprocess socket numpy click flask`.

Anything outside that list needs `--unsafe` — or it fails at import
with the allowed list printed.

## The interop map

```
  number crunching ──► import numpy / triad.ntri
  web endpoints ─────► import flask (+ test client)
  CLIs ──────────────► import click
  files/processes ───► import os (safe) / subprocess (safe)
  everything else ───► --unsafe
```

Nine runnable examples live in `examples/interop/` — all green.

✎ try it: a `.tri` that imports `numpy`, builds `arange(10)`
with `numpy.arange`, and prints the sum.

Next: [10 — solver](10-solver.md).
