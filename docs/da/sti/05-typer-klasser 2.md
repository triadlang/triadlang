# 05 — Typer og klasser

## `type`: en simpel dataform

```tri
type Person {
    navn: String;
    alder: Int;
}

let p = Person(navn="Ana", alder=30);
print(p.navn);    // Ana
print(p.alder);   // 30
```

`type` er ren struktur: typede felter, ingen adfærd.

## `class`: data + adfærd

```
  class Navn {
      fn __init__(self, ...) { ... }   // fødes her
      fn metode(self, ...) { ... }     // bor her
  }
```

```tri
class Konto {
    fn __init__(self, ejer) {
        self.ejer = ejer;
        self.saldo = 0;
    }

    fn saet_ind(self, beloeb) {
        self.saldo = self.saldo + beloeb;
    }
}

let k = Konto("Ana");
k.saet_ind(100);
print(k.saldo);   // 100
```

- `self` er selve instansen, altid første parameter.
- `__init__` kører ved oprettelse.

## Arv: `inherits`

```tri
class Dyr {
    fn __init__(self, navn) {
        self.navn = navn;
    }
    fn siger(self) {
        return "...";
    }
}

class Kat inherits Dyr {
    fn siger(self) {
        return "mjav";
    }
}

let k = Kat("Tom");
print(k.navn + " siger " + k.siger());
// ▸ Tom siger mjav
```

`Kat` arver `navn` og skifter `siger` — resten kommer fra far.

## Hvornår hvad

```
  kun holde felter ───────► type
  holde + gøre ───────────► class
  genbruge + variere ─────► inherits
```

✎ prøv: `class Rektangel` med `bredde/hoejde` og `fn areal(self)`.

Næste: [06 — fejl](06-fejl.md).
