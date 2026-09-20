# 02 — Steuerung: Entscheiden und Wiederholen

## `if` / `elif` / `else`

```tri
let note = 7;

if note >= 7 {
    print("bestanden");
} elif note >= 4 {
    print("Nachprüfung");
} else {
    print("durchgefallen");
}
```

`▸ Ausgabe`: `bestanden`

Vergleiche: `==` `!=` `<` `<=` `>` `>=`.
Logik: `and`, `or`, `not`.

```tri
let alter = 20;
let hat_einladung = true;

if alter >= 18 and hat_einladung {
    print("herein");
}

if not false {
    print("wahr");
}
```

## `for`: zwei Formen

```tri
// 1. zählen
for i in range(3) {
    print(i);
}
// ▸ 0 1 2

// 2. durchgehen
for name in ["ana", "bia"] {
    print("hi " + name);
}
// ▸ hi ana / hi bia
```

`range(5)` → 0..4. `range(2, 8)` → 2..7.

## `while`, `break`, `continue`

```tri
let x = 0;
while x < 10 {
    x = x + 1;
    if x == 3 {
        continue;   // überspringt 3
    }
    if x == 6 {
        break;      // stoppt bei 6
    }
    print(x);
}
// ▸ 1 2 4 5
```

## f-strings: Text mit Rechnung innen

```tri
let total = 250;
print(f"Summe 0..99 = {total}");
print(f"doppelt = {total * 2}");
```

`▸ Ausgabe`

```text
Summe 0..99 = 250
doppelt = 500
```

Alles in `{ }` wird auf der Stelle berechnet.

✎ probier es: Drucke das Einmaleins von 7 mit `for` + f-string.

Weiter: [03 — Funktionen](03-funktionen.md).
