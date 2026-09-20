# Sintaxis — todo el lenguaje en una página

## Frases y valores

```tri-frag
let x = 10;          const K = 2;          // mutable / congelada
10  3.14  "texto"  true  false  none      // Int Float String Bool none
"a" + "b"  2 ** 10  7 % 3                 // pega, potencia, módulo
f"x={x} {x * 2}"                          // f-string calcula dentro
str(42)  type(x)  len([1])                // convierte / revela / cuenta
```

## Decisiones y bucles

```tri-frag
if a { } elif b { } else { }              // and or not
for i in range(3) { }                     // 0 1 2
for v in [1, 2] { }                       // recorre lo recorrible
while ok { }  break;  continue;           // trío clásico
match v { case 1 { } case 2 { } }         // elige por valor
```

## Funciones y errores

```tri-frag
fn f(a, b=2, *args, **kw) { return a; }   // defecto + sobra
yield v;                                  // pausa-y-entrega (en fn)
try { } catch e { } finally { }           // arriesga / cae / siempre
throw "motivo";  assert ok;               // lanza / traba
async fn f() { }  await g();              // await solo en async
```

## Formas de datos

```tri
let l = [1, 2]; let d = {"k": 1, "n": 0};
[1, 2]  l[0]  l.push(3)                   // lista
{"k": 1}  d["k"]  d["n"] = 2  del d["k"]  // dict
type P { nombre: String; edad: Int; }     // molde struct
class C { fn __init__(self) {} }          // molde conducta
class D inherits C { }                    // reúsa + varía
```

## Módulos

```tri
import math;  import math as m;           // entero / apodo
from math import sqrt, pi;                // elegidos
import triad.ntri as np;                  // familia triad.* (39)
import numpy;  import flask;               // Python real (lista safe)
```

## DSL físico (mismo runner)

```tri
@T(18.0)
reg a : anti_collapse = 4;
reg b : anti_collapse = 4;
ring(a, b) kappa=-2.5 for T=18.0;
OBSERVE a k_star, crystallinity, peak, atom_count;
```

## Palabras clave (todas)

```text
let const fn return if else elif for in while break continue
type class super import from as true false none and or not
try catch finally throw self match case yield async await with
reg entity world couple pair ring observe OBSERVE run evolve
sequence via each_for substrate composed_of assert persistent
extended structurally_open mem_memory atomic anti_collapsed
over_seeds is pass del inherits
```
