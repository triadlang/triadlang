# 11 — DSL: skriv fysikk direkte

Ved siden av hverdagslig `.tri` finnes en andre dialekt: få ord,
hvert et fysisk dåd. Registre, koblinger, observasjoner.

## De fem ordene

```
  reg a : anti_collapse = 4;     et register holder et felt
  ring(a, b) kappa=-2.5 ...;     kobler dem i ring
  OBSERVE a k_star, peak, ...;   leser observabler
  @T(18.0)                       tidshorisont øverst
```

## Et komplett program

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

To registre, én ring, ti tall. Et helt eksperiment.

## Ideen under: P1 / P2 / P3

```
  P1 oscillasjon ─────── feltet sitter aldri stille
  P2 selvreferanse ───── nåtid + minne sam-evolverer
  P3 kobling ─────────── relasjoner former evolusjonen
```

Alt kjører sammen — oscillasjon, minne og bad i én integrert
evolusjon. Ingen term skrus noen gang av for å "forenkle".

Flere felt å vandre: `examples/triad/` (likevekt, minne,
observabler, egne potensialer).

✎ prøv: endre `kappa` til `-1.0` og sammenlign de ti tallene.

Neste: [12 — verktøy](12-verktoy.md).
