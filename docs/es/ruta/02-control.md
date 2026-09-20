# 02 — Control: decisiones y repeticiones

## `if` / `elif` / `else`

```tri
let nota = 7;

if nota >= 7 {
    print("aprobado");
} elif nota >= 4 {
    print("recuperación");
} else {
    print("reprobado");
}
```

`▸ salida`: `aprobado`

Comparaciones: `==` `!=` `<` `<=` `>` `>=`.
Lógica: `and`, `or`, `not`.

```tri
let edad = 20;
let tiene_invite = true;

if edad >= 18 and tiene_invite {
    print("puede entrar");
}

if not false {
    print("verdad");
}
```

## `for`: dos formas

```tri
// 1. contando
for i in range(3) {
    print(i);
}
// ▸ 0 1 2

// 2. recorriendo
for nombre in ["ana", "bia"] {
    print("hola " + nombre);
}
// ▸ hola ana / hola bia
```

`range(5)` → 0..4. `range(2, 8)` → 2..7.

## `while`, `break`, `continue`

```tri
let x = 0;
while x < 10 {
    x = x + 1;
    if x == 3 {
        continue;   // salta el 3
    }
    if x == 6 {
        break;      // para en el 6
    }
    print(x);
}
// ▸ 1 2 4 5
```

## f-strings: texto con cuenta dentro

```tri
let total = 250;
print(f"suma 0..99 = {total}");
print(f"doble = {total * 2}");
```

`▸ salida`

```text
suma 0..99 = 250
doble = 500
```

Todo entre `{ }` se calcula en el acto.

✎ prueba: imprime la tabla del 7 con `for` + f-string.

Siguiente: [03 — funciones](03-funciones.md).
