# 02 — Styrning: beslut och upprepning

## `if` / `elif` / `else`

```tri
let betyg = 7;

if betyg >= 7 {
    print("godkänd");
} elif betyg >= 4 {
    print("komplettering");
} else {
    print("underkänd");
}
```

`▸ utdata`: `godkänd`

Jämförelser: `==` `!=` `<` `<=` `>` `>=`.
Logik: `and`, `or`, `not`.

```tri
let alder = 20;
let har_inbjudan = true;

if alder >= 18 and har_inbjudan {
    print("kom in");
}

if not false {
    print("sant");
}
```

## `for`: två former

```tri
// 1. räkna
for i in range(3) {
    print(i);
}
// ▸ 0 1 2

// 2. gå igenom
for namn in ["ana", "bia"] {
    print("hej " + namn);
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
        continue;   // hoppar 3
    }
    if x == 6 {
        break;      // stannar vid 6
    }
    print(x);
}
// ▸ 1 2 4 5
```

## f-strängar: text med räkning inuti

```tri
let total = 250;
print(f"summa 0..99 = {total}");
print(f"dubbelt = {total * 2}");
```

`▸ utdata`

```text
summa 0..99 = 250
dubbelt = 500
```

Allt inom `{ }` räknas ut på plats.

✎ prova: skriv ut 7:ans tabell med `for` + f-sträng.

Nästa: [03 — funktioner](03-funktioner.md).
