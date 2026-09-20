# 08 — Arrays: tænk i vektorer

## `triad.ntri`: numpy-manerer, triad-accent

```tri
import triad.ntri as np;

let a = np.array([1.0, 2.0, 3.0, 4.0]);
let b = np.array([10.0, 20.0, 30.0, 40.0]);

print(a + b);    // array([11.0, 22.0, 33.0, 44.0], dtype=float64)
print(b - a);    // array([9.0, 18.0, 27.0, 36.0], dtype=float64)
print(a * b);    // array([10.0, 40.0, 90.0, 160.0], dtype=float64)
print(a ** 2);   // array([1.0, 4.0, 9.0, 16.0], dtype=float64)
```

Én operator, hele arrayet. Ingen løkke i sigte.

## Broadcasting: skalar møder vektor

```tri
import triad.ntri as np;

let a = np.array([1.0, 2.0, 3.0]);
print(a * 3.0 + 1.0);   // array([4.0, 7.0, 10.0], dtype=float64)
```

Skalaren strækker sig over hver plads.

## Byggere og reduktioner

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
  byg ────► array / zeros / ones / linspace / arange
  form ───► + - * / ** sin cos exp sqrt …
  krymp ──► sum max min mean
```

## Løkke eller vektor?

```tri
import triad.ntri as np;

// løkke: god til hundredtal
let s = 0.0;
for v in np.array([1.0, 2.0, 3.0]) {
    s = s + v;
}
print(s);   // 6.0

// vektor: vejen for tusindtal og op
let big = np.ones(1000000);
print(np.sum(big));   // array(1000000.0, dtype=float64)
```

Tommelfingerregel: udtryk matten, ikke vandringen.

✎ prøv: `sin` af `linspace(0, 3.14, 5)` udskrevet med `np.max`.

Næste: [09 — Python](09-python.md).
