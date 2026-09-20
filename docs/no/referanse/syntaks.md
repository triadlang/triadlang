# Syntaks — hele språket på én side

## Setninger og verdier

```tri-frag
let x = 10;          const K = 2;          // endrbar / fryst
10  3.14  "tekst"  true  false  none      // Int Float String Bool none
"a" + "b"  2 ** 10  7 % 3                 // lime, potens, modulo
f"x={x} {x * 2}"                          // f-streng regner inni
str(42)  type(x)  len([1])                // omdann / avslør / tell
```

## Beslutninger og løkker

```tri-frag
if a { } elif b { } else { }              // and or not
for i in range(3) { }                     // 0 1 2
for v in [1, 2] { }                       // gå gjennom gangbart
while ok { }  break;  continue;           // klassiske trioen
match v { case 1 { } case 2 { } }         // velg på verdi
```

## Funksjoner og feil

```tri-frag
fn f(a, b=2, *args, **kw) { return a; }   // standard + overløp
yield v;                                  // pause-og-rekk (i fn)
try { } catch e { } finally { }           // risiker / fall / alltid
throw "årsak";  assert ok;                // kast / lås
async fn f() { }  await g();              // await bare i async
```

## Dataformer

```tri
let l = [1, 2]; let d = {"k": 1, "n": 0};
[1, 2]  l[0]  l.push(3)                   // liste
{"k": 1}  d["k"]  d["n"] = 2  del d["k"]  // dict
type P { navn: String; alder: Int; }      // strukturform
class C { fn __init__(self) {} }          // atferdsform
class D inherits C { }                    // gjenbruk + varier
```

## Moduler

```tri
import math;  import math as m;           // hel / kallenavn
from math import sqrt, pi;                // valgte
import triad.ntri as np;                  // triad.*-familien (39)
import numpy;  import flask;               // ekte Python (safe-liste)
```

## Fysisk DSL (samme runner)

```tri
@T(18.0)
reg a : anti_collapse = 4;
reg b : anti_collapse = 4;
ring(a, b) kappa=-2.5 for T=18.0;
OBSERVE a k_star, crystallinity, peak, atom_count;
```

## Nøkkelord (alle)

```text
let const fn return if else elif for in while break continue
type class super import from as true false none and or not
try catch finally throw self match case yield async await with
reg entity world couple pair ring observe OBSERVE run evolve
sequence via each_for substrate composed_of assert persistent
extended structurally_open mem_memory atomic anti_collapsed
over_seeds is pass del inherits
```
