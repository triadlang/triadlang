# 01 — Første steg

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

- `let` lager en variabel som kan endres: `x = 99;` gjelder.
- `const` lager en konstant: å endre er en feil.
- Semikolon avslutter setningen. Alltid.

## De fire typene du bruker 99 % av tiden

```
  10            3.14           "tekst"         true / false / none
  └── Int      └── Float      └── String       └── Bool (none = tom)
```

```tri
print(10 + 3.14);            // tall med tall fungerer
print("hei " + "dev");       // tekst med tekst limer
print("n = " + str(42));     // tall blir tekst med str()
```

`▸ utdata`

```text
13.14
hei dev
n = 42
```

## `print` viser, `str` omdanner, `type` avslører

```tri
print(str(3.14));     // "3.14"
print(type(10));   // <class 'int'>
print(type("hei"));   // <class 'str'>
```

## Kommentarer

```tri
// én linje

print("gjelder"); // på linjeslutt også
```

## REPL-et er kladdeblokken din

```sh
./triad repl
```

```text
>>> print(2 * 21);
42
>>> exit
```

✎ prøv: lag `jeg.tri` som skriver navnet ditt og `2 ** 10`
(`**` er potens). Kjør med `./triad run jeg.tri`.

Neste: [02 — kontroll](02-kontroll.md).
