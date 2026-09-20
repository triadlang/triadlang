# 03 — Funktionen

## Einfaches `fn`

```
  fn name(Parameter) {
      ...
      return Wert;
  }
```

```tri
fn add(a, b) {
    return a + b;
}

print(add(2, 3));   // 5
```

Ohne `return` gibt eine Funktion `none` zurück.

## Standardwerte

```tri
fn potenz(basis, exp=2) {
    return basis ** exp;
}

print(potenz(3));      // 9
print(potenz(2, 10));  // 1024
```

## `*args` und `**kwargs`: so viele wie kommen

```tri
fn alles_summe(*zahlen) {
    let s = 0;
    for n in zahlen {
        s = s + n;
    }
    return s;
}
print(alles_summe(1, 2, 3, 4));   // 10

fn zeig(**optionen) {
    for key in optionen {
        print(str(key) + " = " + str(optionen[key]));
    }
}
zeig(farbe="blau", groesse=42);
// ▸ farbe = blau / groesse = 42
```

## Rekursion geht

```tri
fn fib(n) {
    if n <= 1 {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}

print(fib(10));   // 55
```

## `yield`: Funktion, die stückweise liefert

```tri
fn zaehler() {
    yield 1;
    yield 2;
    yield 3;
}

for v in zaehler() {
    print(v);
}
// ▸ 1 2 3
```

Jedes `yield` pausiert und reicht einen Wert; das `for` macht dort
weiter, wo es aufgehört hat.

✎ probier es: Schreibe `fn gerade(n)`, das `true` liefert, falls
`n % 2 == 0`.

Weiter: [04 — Sammlungen](04-sammlungen.md).
