# 02 — Kontroll: beslutninger og gjentakelse

## `if` / `elif` / `else`

```tri
let karakter = 7;

if karakter >= 7 {
    print("bestått");
} elif karakter >= 4 {
    print("kontinuasjon");
} else {
    print("stryk");
}
```

`▸ utdata`: `bestått`

Sammenligninger: `==` `!=` `<` `<=` `>` `>=`.
Logikk: `and`, `or`, `not`.

```tri
let alder = 20;
let har_invitt = true;

if alder >= 18 and har_invitt {
    print("kom inn");
}

if not false {
    print("sant");
}
```

## `for`: to former

```tri
// 1. telle
for i in range(3) {
    print(i);
}
// ▸ 0 1 2

// 2. gå gjennom
for navn in ["ana", "bia"] {
    print("hei " + navn);
}
// ▸ hei ana / hei bia
```

`range(5)` → 0..4. `range(2, 8)` → 2..7.

## `while`, `break`, `continue`

```tri
let x = 0;
while x < 10 {
    x = x + 1;
    if x == 3 {
        continue;   // hopper over 3
    }
    if x == 6 {
        break;      // stopper ved 6
    }
    print(x);
}
// ▸ 1 2 4 5
```

## f-strenger: tekst med regning inni

```tri
let total = 250;
print(f"sum 0..99 = {total}");
print(f"dobbelt = {total * 2}");
```

`▸ utdata`

```text
sum 0..99 = 250
dobbelt = 500
```

Alt innenfor `{ }` regnes ut på stedet.

✎ prøv: skriv ut 7-gangen med `for` + f-streng.

Neste: [03 — funksjoner](03-funksjoner.md).
