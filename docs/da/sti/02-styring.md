# 02 — Styring: beslutninger og gentagelse

## `if` / `elif` / `else`

```tri
let karakter = 7;

if karakter >= 7 {
    print("bestået");
} elif karakter >= 4 {
    print("reeksamen");
} else {
    print("dumpet");
}
```

`▸ output`: `bestået`

Sammenligninger: `==` `!=` `<` `<=` `>` `>=`.
Logik: `and`, `or`, `not`.

```tri
let alder = 20;
let har_invitation = true;

if alder >= 18 and har_invitation {
    print("kom ind");
}

if not false {
    print("sandt");
}
```

## `for`: to former

```tri
// 1. tælle
for i in range(3) {
    print(i);
}
// ▸ 0 1 2

// 2. gå igennem
for navn in ["ana", "bia"] {
    print("hej " + navn);
}
// ▸ hej ana / hej bia
```

`range(5)` → 0..4. `range(2, 8)` → 2..7.

## `while`, `break`, `continue`

```tri
let x = 0;
while x < 10 {
    x = x + 1;
    if x == 3 {
        continue;   // springer 3 over
    }
    if x == 6 {
        break;      // stopper ved 6
    }
    print(x);
}
// ▸ 1 2 4 5
```

## f-strenge: tekst med regning indeni

```tri
let total = 250;
print(f"sum 0..99 = {total}");
print(f"dobbelt = {total * 2}");
```

`▸ output`

```text
sum 0..99 = 250
dobbelt = 500
```

Alt inden for `{ }` regnes ud på stedet.

✎ prøv: skriv 7-tabellen med `for` + f-streng.

Næste: [03 — funktioner](03-funktioner.md).
