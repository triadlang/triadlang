# Syntax — hela språket på en sida

## Meningar och värden

```tri-frag
let x = 10;          const K = 2;          // ändrbar / fryst
10  3.14  "text"  true  false  none       // Int Float String Bool none
"a" + "b"  2 ** 10  7 % 3                 // klibba, potens, modulo
f"x={x} {x * 2}"                          // f-sträng räknar inuti
str(42)  type(x)  len([1])                // omvandla / avslöja / räkna
```

## Beslut och loopar

```tri-frag
if a { } elif b { } else { }              // and or not
for i in range(3) { }                     // 0 1 2
for v in [1, 2] { }                       // gå igenom gångbart
while ok { }  break;  continue;           // klassiska trion
match v { case 1 { } case 2 { } }         // välj på värde
```

## Funktioner och fel

```tri-frag
fn f(a, b=2, *args, **kw) { return a; }   // standard + spill
yield v;                                  // pausa-och-räck (i fn)
try { } catch e { } finally { }           // riskera / fall / alltid
throw "orsak";  assert ok;                // kasta / lås
async fn f() { }  await g();              // await bara i async
```

## Dataformer

```tri
let l = [1, 2]; let d = {"k": 1, "n": 0};
[1, 2]  l[0]  l.push(3)                   // lista
{"k": 1}  d["k"]  d["n"] = 2  del d["k"]  // dict
type P { namn: String; alder: Int; }      // strukturform
class C { fn __init__(self) {} }          // beteendeform
class D inherits C { }                    // återanvänd + variera
```

## Moduler

```tri
import math;  import math as m;           // hel / smeknamn
from math import sqrt, pi;                // valda
import triad.ntri as np;                  // triad.*-familjen (39)
import numpy;  import flask;               // riktig Python (safe-lista)
```

## Fysikalisk DSL (samma runner)

```tri
@T(18.0)
reg a : anti_collapse = 4;
reg b : anti_collapse = 4;
ring(a, b) kappa=-2.5 for T=18.0;
OBSERVE a k_star, crystallinity, peak, atom_count;
```

## Nyckelord (alla)

```text
let const fn return if else elif for in while break continue
type class super import from as true false none and or not
try catch finally throw self match case yield async await with
reg entity world couple pair ring observe OBSERVE run evolve
sequence via each_for substrate composed_of assert persistent
extended structurally_open mem_memory atomic anti_collapsed
over_seeds is pass del inherits
```
