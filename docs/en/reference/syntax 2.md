# Syntax — the whole language on one page

## Sentences and values

```tri-frag
let x = 10;          const K = 2;          // changeable / frozen
10  3.14  "text"  true  false  none        // Int Float String Bool none
"a" + "b"  2 ** 10  7 % 3                 // glue, power, modulo
f"x={x} {x * 2}"                          // f-string computes inside
str(42)  type(x)  len([1])                // convert / reveal / count
```

## Decisions and loops

```tri-frag
if a { } elif b { } else { }              // and or not
for i in range(3) { }                     // 0 1 2
for v in [1, 2] { }                       // walk anything walkable
while ok { }  break;  continue;           // classic trio
match v { case 1 { } case 2 { } }         // choose by value
```

## Functions and errors

```tri-frag
fn f(a, b=2, *args, **kw) { return a; }   // defaults + spillover
yield v;                                  // pause-and-hand (in fn)
try { } catch e { } finally { }           // risk / fall / always
throw "reason";  assert ok;               // raise / lock
async fn f() { }  await g();              // await only inside async
```

## Data shapes

```tri
let l = [1, 2]; let d = {"k": 1, "n": 0};
[1, 2]  l[0]  l.push(3)                   // list
{"k": 1}  d["k"]  d["n"] = 2  del d["k"]  // dict
type P { name: String; age: Int; }        // struct mold
class C { fn __init__(self) {} }          // behavior mold
class D inherits C { }                    // reuse + vary
```

## Modules

```tri
import math;  import math as m;           // whole / nicknamed
from math import sqrt, pi;                // picked
import triad.ntri as np;                  // triad.* family (39)
import numpy;  import flask;               // real Python (safe list)
```

## Physical DSL (same runner)

```tri
@T(18.0)
reg a : anti_collapse = 4;
reg b : anti_collapse = 4;
ring(a, b) kappa=-2.5 for T=18.0;
OBSERVE a k_star, crystallinity, peak, atom_count;
```

## Keywords (all)

```text
let const fn return if else elif for in while break continue
type class super import from as true false none and or not
try catch finally throw self match case yield async await with
reg entity world couple pair ring observe OBSERVE run evolve
sequence via each_for substrate composed_of assert persistent
extended structurally_open mem_memory atomic anti_collapsed
over_seeds is pass del inherits
```
