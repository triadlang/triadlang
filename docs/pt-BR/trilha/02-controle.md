# 02 — Controle: decisões e repetições

## `if` / `elif` / `else`

```tri
let nota = 7;

if nota >= 7 {
    print("passou");
} elif nota >= 4 {
    print("recuperação");
} else {
    print("reprovou");
}
```

`▸ saída`: `passou`

Comparações: `==` `!=` `<` `<=` `>` `>=`.
Lógica: `and`, `or`, `not`.

```tri
let idade = 20;
let tem_convite = true;

if idade >= 18 and tem_convite {
    print("pode entrar");
}

if not false {
    print("verdade");
}
```

## `for`: duas formas

```tri
// 1. contando
for i in range(3) {
    print(i);
}
// ▸ 0 1 2

// 2. percorrendo
for nome in ["ana", "bia"] {
    print("oi " + nome);
}
// ▸ oi ana / oi bia
```

`range(5)` → 0..4. `range(2, 8)` → 2..7.

## `while`, `break`, `continue`

```tri
let x = 0;
while x < 10 {
    x = x + 1;
    if x == 3 {
        continue;   // pula o 3
    }
    if x == 6 {
        break;      // para no 6
    }
    print(x);
}
// ▸ 1 2 4 5
```

## f-strings: texto com conta dentro

```tri
let total = 250;
print(f"soma 0..99 = {total}");
print(f"dobro = {total * 2}");
```

`▸ saída`

```text
soma 0..99 = 250
dobro = 500
```

Tudo entre `{ }` é calculado na hora.

✎ experimente: imprima a tabuada do 7 com `for` + f-string.

Próximo: [03 — funções](03-funcoes.md).
