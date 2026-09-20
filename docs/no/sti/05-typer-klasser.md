# 05 — Typer og klasser

## `type`: en enkel dataform

```tri
type Person {
    navn: String;
    alder: Int;
}

let p = Person(navn="Ana", alder=30);
print(p.navn);    // Ana
print(p.alder);   // 30
```

`type` er ren struktur: typede felt, ingen atferd.

## `class`: data + atferd

```
  class Navn {
      fn __init__(self, ...) { ... }   // fødes her
      fn metode(self, ...) { ... }     // bor her
  }
```

```tri
class Konto {
    fn __init__(self, eier) {
        self.eier = eier;
        self.saldo = 0;
    }

    fn sett_inn(self, belop) {
        self.saldo = self.saldo + belop;
    }
}

let k = Konto("Ana");
k.sett_inn(100);
print(k.saldo);   // 100
```

- `self` er selve instansen, alltid første parameter.
- `__init__` kjører ved opprettelse.

## Arv: `inherits`

```tri
class Dyr {
    fn __init__(self, navn) {
        self.navn = navn;
    }
    fn sier(self) {
        return "...";
    }
}

class Katt inherits Dyr {
    fn sier(self) {
        return "mjau";
    }
}

let k = Katt("Tom");
print(k.navn + " sier " + k.sier());
// ▸ Tom sier mjau
```

`Katt` arver `navn` og bytter `sier` — resten kommer fra pappa.

## Når hva

```
  bare holde felt ───────► type
  holde + gjøre ─────────► class
  gjenbruke + variere ───► inherits
```

✎ prøv: `class Rektangel` med `bredde/hoyde` og `fn areal(self)`.

Neste: [06 — feil](06-feil.md).
