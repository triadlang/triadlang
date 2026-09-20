# 01 — Första steg

## Variabler: `let` och `const`

```tri
let x = 10;
let namn = "Triad";
let aktiv = true;
const PI2 = 3.14159 * 2;

print(x + 20);      // 30
print(namn);        // Triad
print(aktiv);       // true
print(PI2);         // 6.28318
```

- `let` skapar en variabel som får ändras: `x = 99;` gäller.
- `const` skapar en konstant: att ändra är ett fel.
- Semikolon avslutar meningen. Alltid.

## De fyra typerna du använder 99 % av tiden

```
  10            3.14           "text"          true / false / none
  └── Int      └── Float      └── String       └── Bool (none = tom)
```

```tri
print(10 + 3.14);            // tal med tal funkar
print("hej " + "dev");       // text med text klibbar
print("n = " + str(42));     // tal blir text med str()
```

`▸ utdata`

```text
13.14
hej dev
n = 42
```

## `print` visar, `str` omvandlar, `type` avslöjar

```tri
print(str(3.14));     // "3.14"
print(type(10));   // <class 'int'>
print(type("hej"));   // <class 'str'>
```

## Kommentarer

```tri
// en rad

print("gäller"); // i radslut också
```

## REPL:en är ditt kladdblock

```sh
./triad repl
```

```text
>>> print(2 * 21);
42
>>> exit
```

✎ prova: skapa `jag.tri` som skriver ditt namn och `2 ** 10`
(`**` är potens). Kör med `./triad run jag.tri`.

Nästa: [02 — styrning](02-styrning.md).
