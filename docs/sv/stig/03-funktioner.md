# 03 — Funktioner

## Enkelt `fn`

```
  fn namn(parametrar) {
      ...
      return värde;
  }
```

```tri
fn addera(a, b) {
    return a + b;
}

print(addera(2, 3));   // 5
```

Utan `return` ger funktionen tillbaka `none`.

## Standardvärden

```tri
fn potens(bas, exp=2) {
    return bas ** exp;
}

print(potens(3));      // 9
print(potens(2, 10));  // 1024
```

## `*args` och `**kwargs`: hur många som kommer

```tri
fn summa_allt(*tal) {
    let s = 0;
    for n in tal {
        s = s + n;
    }
    return s;
}
print(summa_allt(1, 2, 3, 4));   // 10

fn visa(**alternativ) {
    for nyckel in alternativ {
        print(str(nyckel) + " = " + str(alternativ[nyckel]));
    }
}
visa(farg="blå", storlek=42);
// ▸ farg = blå / storlek = 42
```

## Rekursion funkar

```tri
fn fib(n) {
    if n <= 1 {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}

print(fib(10));   // 55
```

## `yield`: funktion som levererar bitvis

```tri
fn raknare() {
    yield 1;
    yield 2;
    yield 3;
}

for v in raknare() {
    print(v);
}
// ▸ 1 2 3
```

Varje `yield` pausar och räcker ett värde; `for` fortsätter där
den slutade.

✎ prova: skriv `fn jamn(n)` som ger `true` om `n % 2 == 0`.

Nästa: [04 — samlingar](04-samlingar.md).
