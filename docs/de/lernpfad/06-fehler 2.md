# 06 — Fehler: elegant brechen

## `try` / `catch` / `finally`

```
  try {
      hier riskieren
  } catch fehler {
      fällt es, fällt es hier (fehler hat die Nachricht)
  } finally {
      läuft immer hier durch
  }
```

```tri
try {
    throw "boom";
} catch e {
    print("gefangen: " + str(e));
} finally {
    print("läuft immer");
}
```

`▸ Ausgabe`

```text
gefangen: boom
läuft immer
```

`throw` wirft jeden Wert (Text, Zahl, Dict).

## `assert`: Gewissheiten verriegeln

```tri
fn teilen(a, b) {
    assert b != 0;
    return a / b;
}

print(teilen(10, 2));   // 5.0
// teilen(10, 0) bricht mit Assert-Fehler ab
```

`assert` ist ausführbare Dokumentation: Fällt die Bedingung, stoppt
das Programm dort und zeigt auf die Zeile.

## `match`: nach Wert wählen

```tri
let v = 2;

match v {
    case 1 {
        print("eins");
    }
    case 2 {
        print("zwei");
    }
}
// ▸ zwei
```

Jedes `case` testet einen Wert; sein Block läuft.

## Das Notfall-Trio

```
  kaputt? ─────────► throw "grund"
  könnte brechen? ─► try / catch
  darf nie? ───────► assert Bedingung
```

✎ probier es: `fn wurzel(x)`, das wirft, falls `x < 0`, sonst
`math.sqrt(x)` liefert (braucht `import math;`).

Weiter: [07 — Module](07-module.md).
