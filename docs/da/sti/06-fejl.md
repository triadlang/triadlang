# 06 — Fejl: gå i stykker med elegance

## `try` / `catch` / `finally`

```
  try {
      risikér her
  } catch fejl {
      falder det, falder det her (fejl har beskeden)
  } finally {
      passerer altid her
  }
```

```tri
try {
    throw "boom";
} catch e {
    print("fanget: " + str(e));
} finally {
    print("kører altid");
}
```

`▸ output`

```text
fanget: boom
kører altid
```

`throw` kaster enhver værdi (tekst, tal, dict).

## `assert`: lås dine vissheder

```tri
fn divider(a, b) {
    assert b != 0;
    return a / b;
}

print(divider(10, 2));   // 5.0
// divider(10, 0) afbryder med assert-fejl
```

`assert` er kørbar dokumentation: svigter betingelsen, stopper
programmet der og peger på linjen.

## `match`: vælg på værdi

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

Hver `case` tester en værdi; dens blok kører.

## Nødtrioen

```
  gik i stykker? ──► throw "årsag"
  kan gå i stykker? ► try / catch
  må aldrig? ───────► assert betingelse
```

✎ prøv: `fn rod(x)`, der kaster hvis `x < 0`, ellers giver
`math.sqrt(x)` (behøver `import math;`).

Næste: [07 — moduler](07-moduler.md).
