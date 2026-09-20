# 05 — Typen und Klassen

## `type`: eine einfache Datenform

```tri
type Person {
    name: String;
    alter: Int;
}

let p = Person(name="Ana", alter=30);
print(p.name);    // Ana
print(p.alter);   // 30
```

`type` ist reine Struktur: typisierte Felder, kein Verhalten.

## `class`: Daten + Verhalten

```
  class Name {
      fn __init__(self, ...) { ... }   // hier geboren
      fn methode(self, ...) { ... }    // hier lebend
  }
```

```tri
class Konto {
    fn __init__(self, inhaber) {
        self.inhaber = inhaber;
        self.stand = 0;
    }

    fn einzahlen(self, betrag) {
        self.stand = self.stand + betrag;
    }
}

let k = Konto("Ana");
k.einzahlen(100);
print(k.stand);   // 100
```

- `self` ist die Instanz selbst, immer der erste Parameter.
- `__init__` läuft bei der Erzeugung.

## Vererbung: `inherits`

```tri
class Tier {
    fn __init__(self, name) {
        self.name = name;
    }
    fn spricht(self) {
        return "...";
    }
}

class Katze inherits Tier {
    fn spricht(self) {
        return "miau";
    }
}

let k = Katze("Tom");
print(k.name + " sagt " + k.spricht());
// ▸ Tom sagt miau
```

`Katze` erbt `name` und ersetzt `spricht` — der Rest kommt vom Vater.

## Wann was

```
  nur Felder halten ──────► type
  halten + tun ───────────► class
  wiederverwenden ────────► inherits
```

✎ probier es: `class Rechteck` mit `breite/hoehe` und `fn flaeche(self)`.

Weiter: [06 — Fehler](06-fehler.md).
