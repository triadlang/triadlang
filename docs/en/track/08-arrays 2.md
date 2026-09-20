# 08 — Arrays: thinking in vectors

## `triad.ntri`: numpy manners, triad accent

```tri
import triad.ntri as np;

let a = np.array([1.0, 2.0, 3.0, 4.0]);
let b = np.array([10.0, 20.0, 30.0, 40.0]);

print(a + b);    // array([11.0, 22.0, 33.0, 44.0], dtype=float64)
print(b - a);    // array([9.0, 18.0, 27.0, 36.0], dtype=float64)
print(a * b);    // array([10.0, 40.0, 90.0, 160.0], dtype=float64)
print(a ** 2);   // array([1.0, 4.0, 9.0, 16.0], dtype=float64)
```

One operator, the whole array. No loop in sight.

## Broadcasting: scalar meets vector

```tri
import triad.ntri as np;

let a = np.array([1.0, 2.0, 3.0]);
print(a * 3.0 + 1.0);   // array([4.0, 7.0, 10.0], dtype=float64)
```

The scalar stretches over every slot.

## Builders and reductions

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
  build it ──► array / zeros / ones / linspace / arange
  shape it ──► + - * / ** sin cos exp sqrt …
  shrink it ─► sum max min mean
```

## Loop or vector?

```tri
import triad.ntri as np;

// loop: fine for hundreds
let s = 0.0;
for v in np.array([1.0, 2.0, 3.0]) {
    s = s + v;
}
print(s);   // 6.0

// vector: the way for thousands and up
let big = np.ones(1000000);
print(np.sum(big));   // array(1000000.0, dtype=float64)
```

Rule of thumb: express the math, not the walking.

✎ try it: `sin` of `linspace(0, 3.14, 5)` printed with `np.max`.

Next: [09 — Python](09-python.md).
