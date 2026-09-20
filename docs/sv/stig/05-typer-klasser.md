# 05 — Typer och klasser

## `type`: en enkel dataform

```tri
type Person {
    namn: String;
    alder: Int;
}

let p = Person(namn="Ana", alder=30);
print(p.namn);    // Ana
print(p.alder);   // 30
```

`type` är ren struktur: typade fält, inget beteende.

## `class`: data + beteende

```
  class Namn {
      fn __init__(self, ...) { ... }   // föds här
      fn metod(self, ...) { ... }      // bor här
  }
```

```tri
class Konto {
    fn __init__(self, agare) {
        self.agare = agare;
        self.saldo = 0;
    }

    fn insattning(self, belopp) {
        self.saldo = self.saldo + belopp;
    }
}

let k = Konto("Ana");
k.insattning(100);
print(k.saldo);   // 100
```

- `self` är själva instansen, alltid första parametern.
- `__init__` körs vid skapandet.

## Arv: `inherits`

```tri
class Djur {
    fn __init__(self, namn) {
        self.namn = namn;
    }
    fn later(self) {
        return "...";
    }
}

class Katt inherits Djur {
    fn later(self) {
        return "mjau";
    }
}

let k = Katt("Tom");
print(k.namn + " säger " + k.later());
// ▸ Tom säger mjau
```

`Katt` ärver `namn` och byter `later` — resten kommer från pappa.

## När vad

```
  bara hålla fält ───────► type
  hålla + göra ──────────► class
  återanvända + variera ─► inherits
```

✎ prova: `class Rektangel` med `bredd/hojd` och `fn area(self)`.

Nästa: [06 — fel](06-fel.md).
