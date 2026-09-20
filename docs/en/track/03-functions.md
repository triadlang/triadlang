# 03 — Functions

## Basic `fn`

```
  fn name(parameters) {
      ...
      return value;
  }
```

```tri
fn add(a, b) {
    return a + b;
}

print(add(2, 3));   // 5
```

Without `return`, a function gives back `none`.

## Default values

```tri
fn power(base, exp=2) {
    return base ** exp;
}

print(power(3));      // 9
print(power(2, 10));  // 1024
```

## `*args` and `**kwargs`: as many as come

```tri
fn sum_all(*numbers) {
    let s = 0;
    for n in numbers {
        s = s + n;
    }
    return s;
}
print(sum_all(1, 2, 3, 4));   // 10

fn show(**options) {
    for key in options {
        print(str(key) + " = " + str(options[key]));
    }
}
show(color="blue", size=42);
// ▸ color = blue / size = 42
```

## Recursion works

```tri
fn fib(n) {
    if n <= 1 {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}

print(fib(10));   // 55
```

## `yield`: a function that delivers in pieces

```tri
fn counter() {
    yield 1;
    yield 2;
    yield 3;
}

for v in counter() {
    print(v);
}
// ▸ 1 2 3
```

Each `yield` pauses and hands out a value; the `for` resumes where
it stopped.

✎ try it: write `fn even(n)` returning `true` if `n % 2 == 0`.

Next: [04 — collections](04-collections.md).
