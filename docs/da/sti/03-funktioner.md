# 03 — Funktioner

## Simpel `fn`

```
  fn navn(parametre) {
      ...
      return værdi;
  }
```

```tri
fn adder(a, b) {
    return a + b;
}

print(adder(2, 3));   // 5
```

Uden `return` giver funktionen `none` tilbage.

## Standardværdier

```tri
fn potens(grund, eks=2) {
    return grund ** eks;
}

print(potens(3));      // 9
print(potens(2, 10));  // 1024
```

## `*args` og `**kwargs`: så mange som kommer

```tri
fn summer_alt(*tal) {
    let s = 0;
    for n in tal {
        s = s + n;
    }
    return s;
}
print(summer_alt(1, 2, 3, 4));   // 10

fn vis(**valg) {
    for noegle in valg {
        print(str(noegle) + " = " + str(valg[noegle]));
    }
}
vis(farve="blå", stoerrelse=42);
// ▸ farve = blå / stoerrelse = 42
```

## Rekursion virker

```tri
fn fib(n) {
    if n <= 1 {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}

print(fib(10));   // 55
```

## `yield`: funktion der leverer bitvist

```tri
fn taeller() {
    yield 1;
    yield 2;
    yield 3;
}

for v in taeller() {
    print(v);
}
// ▸ 1 2 3
```

Hver `yield` pauser og rækker en værdi; `for` fortsætter hvor
den slap.

✎ prøv: skriv `fn lige(n)`, der giver `true` hvis `n % 2 == 0`.

Næste: [04 — samlinger](04-samlinger.md).
