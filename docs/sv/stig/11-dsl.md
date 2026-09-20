# 11 — DSL: skriv fysik direkt

Bredvid vardaglig `.tri` finns en andra dialekt: få ord, varje ett
fysikaliskt dåd. Register, kopplingar, observationer.

## De fem orden

```
  reg a : anti_collapse = 4;     ett register håller ett fält
  ring(a, b) kappa=-2.5 ...;     kopplar dem i ring
  OBSERVE a k_star, peak, ...;   läser observabler
  @T(18.0)                       tidshorisont överst
```

## Ett komplett program

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

`▸ utdata`

```text
a = { k_star=8.6394, crystallinity=0.9784, peak=0.3177, atom_count=0.9646 }
b = { k_star=8.6394, crystallinity=0.9784, peak=0.3180, atom_count=0.9646 }
```

Två register, en ring, tio tal. Ett helt experiment.

## Idén under: P1 / P2 / P3

```
  P1 oscillation ─────── fältet sitter aldrig still
  P2 självreferens ───── nutid + minne sam-evolverar
  P3 koppling ────────── relationer formar evolutionen
```

Allt körs ihop — oscillation, minne och bad i en integral
evolution. Ingen term stängs någonsin av för att "förenkla".

Fler fält att vandra: `examples/triad/` (jämvikt, minne,
observabler, egna potentialer).

✎ prova: ändra `kappa` till `-1.0` och jämför de tio talen.

Nästa: [12 — verktyg](12-verktyg.md).
