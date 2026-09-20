# 06 — Feil: gå i stykker med eleganse

## `try` / `catch` / `finally`

```
  try {
      risiker her
  } catch feil {
      faller det, faller det her (feil har meldingen)
  } finally {
      passerer alltid her
  }
```

```tri
try {
    throw "boom";
} catch e {
    print("fanget: " + str(e));
} finally {
    print("kjører alltid");
}
```

`▸ utdata`

```text
fanget: boom
kjører alltid
```

`throw` kaster enhver verdi (tekst, tall, dict).

## `assert`: lås visshetene dine

```tri
fn divider(a, b) {
    assert b != 0;
    return a / b;
}

print(divider(10, 2));   // 5.0
// divider(10, 0) avbryter med assert-feil
```

`assert` er kjørbar dokumentasjon: svikter betingelsen stopper
programmet der og peker på linjen.

## `match`: velg på verdi

```tri
let v = 2;

match v {
    case 1 {
        print("en");
    }
    case 2 {
        print("to");
    }
}
// ▸ to
```

Hver `case` tester en verdi; blokken dens kjører.

## Nødtrioen

```
  gikk i stykker? ──► throw "årsak"
  kan gå i stykker? ► try / catch
  får aldri? ───────► assert betingelse
```

✎ prøv: `fn rot(x)` som kaster hvis `x < 0`, ellers gir
`math.sqrt(x)` (trenger `import math;`).

Neste: [07 — moduler](07-moduler.md).
