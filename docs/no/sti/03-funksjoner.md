# 03 — Funksjoner

## Enkel `fn`

```
  fn navn(parametre) {
      ...
      return verdi;
  }
```

```tri
fn adder(a, b) {
    return a + b;
}

print(adder(2, 3));   // 5
```

Uten `return` gir funksjonen tilbake `none`.

## Standardverdier

```tri
fn potens(grunn, eks=2) {
    return grunn ** eks;
}

print(potens(3));      // 9
print(potens(2, 10));  // 1024
```

## `*args` og `**kwargs`: så mange som kommer

```tri
fn summer_alt(*tall) {
    let s = 0;
    for n in tall {
        s = s + n;
    }
    return s;
}
print(summer_alt(1, 2, 3, 4));   // 10

fn vis(**valg) {
    for nokkel in valg {
        print(str(nokkel) + " = " + str(valg[nokkel]));
    }
}
vis(farge="blå", storrelse=42);
// ▸ farge = blå / storrelse = 42
```

## Rekursjon fungerer

```tri
fn fib(n) {
    if n <= 1 {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}

print(fib(10));   // 55
```

## `yield`: funksjon som leverer bitvis

```tri
fn teller() {
    yield 1;
    yield 2;
    yield 3;
}

for v in teller() {
    print(v);
}
// ▸ 1 2 3
```

Hver `yield` pauser og rekker en verdi; `for` fortsetter der
den slapp.

✎ prøv: skriv `fn partall(n)` som gir `true` hvis `n % 2 == 0`.

Neste: [04 — samlinger](04-samlinger.md).
