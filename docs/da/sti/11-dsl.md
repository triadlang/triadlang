# 11 — DSL: skriv fysik direkte

Ved siden af hverdags-`.tri` findes en anden dialekt: få ord,
hvert en fysisk dåd. Registre, koblinger, observationer.

## De fem ord

```
  reg a : anti_collapse = 4;     et register holder et felt
  ring(a, b) kappa=-2.5 ...;     kobler dem i ring
  OBSERVE a k_star, peak, ...;   læser observabler
  @T(18.0)                       tidshorisont øverst
```

## Et komplet program

`examples/triad/anti_collapse.tri`:

```tri
@T(18.0)
reg a : anti_collapse = 4;
reg b : anti_collapse = 4;
ring(a, b) kappa=-2.5 for T=18.0;
OBSERVE a k_star, crystallinity, peak, atom_count;
OBSERVE b k_star, crystallinity, peak, atom_count;
```

```sh
./triad run examples/triad/anti_collapse.tri
```

`▸ output`

```text
a = { k_star=8.6394, crystallinity=0.9784, peak=0.3177, atom_count=0.9646 }
b = { k_star=8.6394, crystallinity=0.9784, peak=0.3180, atom_count=0.9646 }
```

To registre, én ring, ti tal. Et helt eksperiment.

## Ideen under: P1 / P2 / P3

```
  P1 oscillation ─────── feltet sidder aldrig stille
  P2 selvreference ───── nutid + hukommelse sam-evolverer
  P3 kobling ─────────── relationer former evolutionen
```

Alt kører sammen — oscillation, hukommelse og bad i én integreret
evolution. Ingen term slukkes nogensinde for at "forenkle".

Flere felter at vandre: `examples/triad/` (ligevægt, hukommelse,
observabler, egne potentialer).

✎ prøv: ændr `kappa` til `-1.0` og sammenlign de ti tal.

Næste: [12 — værktøjer](12-vaerktoejer.md).
