# 06 — Fel: gå sönder med elegans

## `try` / `catch` / `finally`

```
  try {
      riskera här
  } catch fel {
      faller det, faller det här (fel har meddelandet)
  } finally {
      passerar alltid här
  }
```

```tri
try {
    throw "boom";
} catch e {
    print("fångade: " + str(e));
} finally {
    print("kör alltid");
}
```

`▸ utdata`

```text
fångade: boom
kör alltid
```

`throw` kastar vilket värde som helst (text, tal, dict).

## `assert`: lås dina vissheter

```tri
fn dividera(a, b) {
    assert b != 0;
    return a / b;
}

print(dividera(10, 2));   // 5.0
// dividera(10, 0) avbryter med assert-fel
```

`assert` är körbar dokumentation: faller villkoret stannar
programmet där och pekar på raden.

## `match`: välj på värde

```tri
let v = 2;

match v {
    case 1 {
        print("ett");
    }
    case 2 {
        print("två");
    }
}
// ▸ två
```

Varje `case` testar ett värde; dess block körs.

## Nödtrion

```
  gick sönder? ──► throw "orsak"
  kan gå sönder? ► try / catch
  får aldrig? ───► assert villkor
```

✎ prova: `fn rot(x)` som kastar om `x < 0`, annars ger
`math.sqrt(x)` (behöver `import math;`).

Nästa: [07 — moduler](07-moduler.md).
