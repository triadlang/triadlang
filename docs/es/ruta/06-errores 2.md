# 06 — Errores: romper con elegancia

## `try` / `catch` / `finally`

```
  try {
      arriesga aquí
  } catch error {
      si cae, cae aquí (error trae el mensaje)
  } finally {
      siempre pasa por aquí
  }
```

```tri
try {
    throw "boom";
} catch e {
    print("atrapé: " + str(e));
} finally {
    print("siempre corre");
}
```

`▸ salida`

```text
atrapé: boom
siempre corre
```

`throw` lanza cualquier valor (texto, número, dict).

## `assert`: traba tus certezas

```tri
fn dividir(a, b) {
    assert b != 0;
    return a / b;
}

print(dividir(10, 2));   // 5.0
// dividir(10, 0) aborta con error de aserción
```

`assert` es documentación ejecutable: si la condición falla, el
programa para ahí y apunta la línea.

## `match`: elige por valor

```tri
let v = 2;

match v {
    case 1 {
        print("uno");
    }
    case 2 {
        print("dos");
    }
}
// ▸ dos
```

Cada `case` prueba un valor; su bloque corre.

## El trío de emergencia

```
  ¿algo se rompió? ─► throw "motivo"
  ¿puede romperse? ─► try / catch
  ¿nunca puede? ────► assert condición
```

✎ prueba: `fn raiz(x)` que lance `throw` si `x < 0`, si no
devuelva `math.sqrt(x)` (necesita `import math;`).

Siguiente: [07 — módulos](07-modulos.md).
