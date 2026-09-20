# 01 — Primeros pasos

## Variables: `let` y `const`

```tri
let x = 10;
let nombre = "Triad";
let activo = true;
const PI2 = 3.14159 * 2;

print(x + 20);      // 30
print(nombre);      // Triad
print(activo);      // true
print(PI2);         // 6.28318
```

- `let` crea variable que puede cambiar: `x = 99;` vale.
- `const` crea constante: cambiarla es error.
- El punto y coma cierra la frase. Siempre.

## Los cuatro tipos del 99% del tiempo

```
  10            3.14           "texto"         true / false / none
  └── Int      └── Float      └── String       └── Bool (none = vacío)
```

```tri
print(10 + 3.14);            // número con número funciona
print("hola " + "dev");      // texto con texto pega
print("n = " + str(42));     // número se vuelve texto con str()
```

`▸ salida`

```text
13.14
hola dev
n = 42
```

## `print` muestra, `str` convierte, `type` revela

```tri
print(str(3.14));     // "3.14"
print(type(10));   // <class 'int'>
print(type("hola"));   // <class 'str'>
```

## Comentarios

```tri
// una línea

print("vale"); // al final de la línea también
```

## El REPL es tu cuaderno de borrador

```sh
./triad repl
```

```text
>>> print(2 * 21);
42
>>> exit
```

✎ prueba: crea `yo.tri` que imprima tu nombre y `2 ** 10`
(`**` es potencia). Córrelo con `./triad run yo.tri`.

Siguiente: [02 — control](02-control.md).
