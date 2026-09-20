# 11 — DSL: writing physics directly

Next to everyday `.tri` there is a second dialect: few words, each one
a physical act. Registers, couplings, observations.

## The five words

```
  reg a : anti_collapse = 4;     a register holding a field
  ring(a, b) kappa=-2.5 ...;     couple them in a ring
  OBSERVE a k_star, peak, ...;   read observables
  @T(18.0)                       time horizon up top
```

## A complete program

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

Two registers, one ring, ten numbers. That is a whole experiment.

## The idea underneath: P1 / P2 / P3

```
  P1 oscillation ─────── the field never sits still
  P2 self-reference ──── present + memory co-evolve
  P3 coupling ────────── relations shape the evolution
```

Everything runs together — oscillation, memory and bath in one
integral evolution. No term is ever switched off to "simplify".

More fields to walk: `examples/triad/` (equilibrium, memory,
observables, custom potentials).

✎ try it: change `kappa` to `-1.0` and compare the ten numbers.

Next: [12 — tools](12-tools.md).
