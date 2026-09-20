# Syntax — die ganze Sprache auf einer Seite

## Sätze und Werte

```tri-frag
let x = 10;          const K = 2;          // änderbar / eingefroren
10  3.14  "Text"  true  false  none       // Int Float String Bool none
"a" + "b"  2 ** 10  7 % 3                 // kleben, Potenz, Modulo
f"x={x} {x * 2}"                          // f-string rechnet innen
str(42)  type(x)  len([1])                // wandeln / verraten / zählen
```

## Entscheiden und Schleifen

```tri-frag
if a { } elif b { } else { }              // and or not
for i in range(3) { }                     // 0 1 2
for v in [1, 2] { }                       // geht durch Gehbares
while ok { }  break;  continue;           // klassisches Trio
match v { case 1 { } case 2 { } }         // nach Wert wählen
```

## Funktionen und Fehler

```tri-frag
fn f(a, b=2, *args, **kw) { return a; }   // Standard + Überlauf
yield v;                                  // pausiert-und-reicht (in fn)
try { } catch e { } finally { }           // riskieren / fallen / immer
throw "grund";  assert ok;                // werfen / verriegeln
async fn f() { }  await g();              // await nur in async
```

## Datenformen

```tri
let l = [1, 2]; let d = {"k": 1, "n": 0};
[1, 2]  l[0]  l.push(3)                   // Liste
{"k": 1}  d["k"]  d["n"] = 2  del d["k"]  // Dict
type P { name: String; alter: Int; }      // Struct-Form
class C { fn __init__(self) {} }          // Verhaltens-Form
class D inherits C { }                    // wiederverwenden + ändern
```

## Module

```tri
import math;  import math as m;           // ganz / Spitzname
from math import sqrt, pi;                // gewählt
import triad.ntri as np;                  // triad.*-Familie (39)
import numpy;  import flask;               // echtes Python (safe-Liste)
```

## Physikalische DSL (gleicher Runner)

```tri
@T(18.0)
reg a : anti_collapse = 4;
reg b : anti_collapse = 4;
ring(a, b) kappa=-2.5 for T=18.0;
OBSERVE a k_star, crystallinity, peak, atom_count;
```

## Schlüsselworte (alle)

```text
let const fn return if else elif for in while break continue
type class super import from as true false none and or not
try catch finally throw self match case yield async await with
reg entity world couple pair ring observe OBSERVE run evolve
sequence via each_for substrate composed_of assert persistent
extended structurally_open mem_memory atomic anti_collapsed
over_seeds is pass del inherits
```
