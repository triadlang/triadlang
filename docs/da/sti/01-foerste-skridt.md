# 01 — Første skridt

## Variabler: `let` og `const`

```tri
let x = 10;
let navn = "Triad";
let aktiv = true;
const PI2 = 3.14159 * 2;

print(x + 20);      // 30
print(navn);        // Triad
print(aktiv);       // true
print(PI2);         // 6.28318
```

- `let` laver en variabel, der må ændres: `x = 99;` gælder.
- `const` laver en konstant: at ændre er en fejl.
- Semikolon afslutter sætningen. Altid.

## De fire typer du bruger 99 % af tiden

```
  10            3.14           "tekst"         true / false / none
  └── Int      └── Float      └── String       └── Bool (none = tom)
```

```tri
print(10 + 3.14);            // tal med tal virker
print("hej " + "dev");       // tekst med tekst limer
print("n = " + str(42));     // tal bliver tekst med str()
```

`▸ output`

```text
13.14
hej dev
n = 42
```

## `print` viser, `str` omdanner, `type` afslører

```tri
print(str(3.14));     // "3.14"
print(type(10));   // <class 'int'>
print(type("hej"));   // <class 'str'>
```

## Kommentarer

```tri
// én linje

print("gælder"); // i linjeslutning også
```

## REPL'en er din kladdeblok

```sh
./triad repl
```

```text
>>> print(2 * 21);
42
>>> exit
```

✎ prøv: lav `jeg.tri`, der skriver dit navn og `2 ** 10`
(`**` er potens). Kør med `./triad run jeg.tri`.

Næste: [02 — styring](02-styring.md).
