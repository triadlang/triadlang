# 01 — First steps

## Variables: `let` and `const`

```tri
let x = 10;
let name = "Triad";
let active = true;
const PI2 = 3.14159 * 2;

print(x + 20);      // 30
print(name);        // Triad
print(active);      // true
print(PI2);         // 6.28318
```

- `let` creates a variable that may change: `x = 99;` is fine.
- `const` creates a constant: changing it is an error.
- The semicolon ends the sentence. Always.

## The four types you use 99% of the time

```
  10            3.14           "text"          true / false / none
  └── Int      └── Float      └── String       └── Bool (none = empty)
```

```tri
print(10 + 3.14);            // number with number works
print("hi " + "dev");        // text with text glues
print("n = " + str(42));     // number becomes text with str()
```

`▸ output`

```text
13.14
hi dev
n = 42
```

## `print` shows, `str` converts, `type` reveals

```tri
print(str(3.14));     // "3.14"
print(type(10));   // <class 'int'>
print(type("hi"));   // <class 'str'>
```

## Comments

```tri
// one line

print("fine"); // at end of line too
```

## The REPL is your scratchpad

```sh
./triad repl
```

```text
>>> print(2 * 21);
42
>>> exit
```

✎ try it: create `me.tri` that prints your name and `2 ** 10`
(`**` is power). Run it with `./triad run me.tri`.

Next: [02 — control](02-control.md).
