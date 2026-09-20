# 08 — Arreglos: pensar en vectores

## `triad.ntri`: modales numpy, acento triad

```tri
import triad.ntri as np;

let a = np.array([1.0, 2.0, 3.0, 4.0]);
let b = np.array([10.0, 20.0, 30.0, 40.0]);

print(a + b);    // array([11.0, 22.0, 33.0, 44.0], dtype=float64)
print(b - a);    // array([9.0, 18.0, 27.0, 36.0], dtype=float64)
print(a * b);    // array([10.0, 40.0, 90.0, 160.0], dtype=float64)
print(a ** 2);   // array([1.0, 4.0, 9.0, 16.0], dtype=float64)
```

Un operador, todo el arreglo. Ningún bucle a la vista.

## Broadcasting: escalar encuentra vector

```tri
import triad.ntri as np;

let a = np.array([1.0, 2.0, 3.0]);
print(a * 3.0 + 1.0);   // array([4.0, 7.0, 10.0], dtype=float64)
```

El escalar se estira sobre cada posición.

## Constructores y reducciones

```tri
import triad.ntri as np;

let z = np.zeros(4);
print(z);                 // array([0.0, 0.0, 0.0, 0.0], dtype=float64)
print(np.sum(z + 1.0));   // array(4.0, dtype=float64)

let x = np.linspace(0.0, 6.28, 100);
let y = np.sin(x) * np.cos(x);
print(np.max(y));         // ≈ array(0.5, dtype=float64)
print(np.min(y));         // ≈ array(-0.5, dtype=float64)
```

```
  construye ─► array / zeros / ones / linspace / arange
  moldea ────► + - * / ** sin cos exp sqrt …
  resume ────► sum max min mean
```

## ¿Bucle o vector?

```tri
import triad.ntri as np;

// bucle: bien para cientos
let s = 0.0;
for v in np.array([1.0, 2.0, 3.0]) {
    s = s + v;
}
print(s);   // 6.0

// vector: el camino para miles en adelante
let big = np.ones(1000000);
print(np.sum(big));   // array(1000000.0, dtype=float64)
```

Regla práctica: expresa la matemática, no el caminar.

✎ prueba: `sin` de `linspace(0, 3.14, 5)` impreso con `np.max`.

Siguiente: [09 — Python](09-python.md).
