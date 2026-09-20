# 11 — DSL: Physik direkt schreiben

Neben dem alltäglichen `.tri` gibt es einen zweiten Dialekt: wenige
Worte, jedes ein physikalischer Akt. Register, Kopplungen, Messungen.

## Die fünf Worte

```
  reg a : anti_collapse = 4;     ein Register hält ein Feld
  ring(a, b) kappa=-2.5 ...;     koppelt sie im Ring
  OBSERVE a k_star, peak, ...;   liest Observablen
  @T(18.0)                       Zeithorizont oben
```

## Ein vollständiges Programm

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

`▸ Ausgabe`

```text
a = { k_star=8.6394, crystallinity=0.9784, peak=0.3177, atom_count=0.9646 }
b = { k_star=8.6394, crystallinity=0.9784, peak=0.3180, atom_count=0.9646 }
```

Zwei Register, ein Ring, zehn Zahlen. Ein ganzes Experiment.

## Die Idee darunter: P1 / P2 / P3

```
  P1 Oszillation ─────── das Feld sitzt nie still
  P2 Selbstreferenz ──── Gegenwart + Gedächtnis ko-evolvieren
  P3 Kopplung ────────── Relationen formen die Evolution
```

Alles läuft zusammen — Oszillation, Gedächtnis und Bad in einer
integralen Evolution. Kein Term wird je abgeschaltet, um zu
„vereinfachen“.

Mehr Felder zum Begehen: `examples/triad/` (Gleichgewicht,
Gedächtnis, Observablen, eigene Potenziale).

✎ probier es: Ändere `kappa` auf `-1.0` und vergleiche die zehn Zahlen.

Weiter: [12 — Werkzeuge](12-werkzeuge.md).
