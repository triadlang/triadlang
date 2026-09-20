# 03 — Funciones

## `fn` básico

```
  fn nombre(parámetros) {
      ...
      return valor;
  }
```

```tri
fn suma(a, b) {
    return a + b;
}

print(suma(2, 3));   // 5
```

Sin `return`, la función devuelve `none`.

## Valores por defecto

```tri
fn potencia(base, exponente=2) {
    return base ** exponente;
}

print(potencia(3));      // 9
print(potencia(2, 10));  // 1024
```

## `*args` y `**kwargs`: cuantos vengan

```tri
fn sumar_todo(*numeros) {
    let s = 0;
    for n in numeros {
        s = s + n;
    }
    return s;
}
print(sumar_todo(1, 2, 3, 4));   // 10

fn mostrar(**opciones) {
    for clave in opciones {
        print(str(clave) + " = " + str(opciones[clave]));
    }
}
mostrar(color="azul", tamano=42);
// ▸ color = azul / tamano = 42
```

## La recursión funciona

```tri
fn fib(n) {
    if n <= 1 {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}

print(fib(10));   // 55
```

## `yield`: función que entrega de a poco

```tri
fn conteo() {
    yield 1;
    yield 2;
    yield 3;
}

for v in conteo() {
    print(v);
}
// ▸ 1 2 3
```

Cada `yield` pausa y entrega un valor; el `for` sigue donde paró.

✎ prueba: escribe `fn par(n)` que devuelva `true` si `n % 2 == 0`.

Siguiente: [04 — colecciones](04-colecciones.md).
