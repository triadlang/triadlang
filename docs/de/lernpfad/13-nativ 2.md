# 13 — Nativ: C, Kernel und weiter

## Drei Schichten, eine Sprache

```
  ┌─────────────────────────────────┐
  │  .tri-Programme (du bist hier)  │
  ├─────────────────────────────────┤
  │  native/c   C-Runtime + Checker │  ./native/c/triad check f.tri
  ├─────────────────────────────────┤
  │  native/kernel  Bare-Metal-Boot │  asm + ld → kernel.bin
  └─────────────────────────────────┘
```

## Die C-Seite

`native/c/` hält eine C-Runtime (`libtriad_rt`), eine CLI und
Host-Tests (regex, autograd shapes, AES, Solver-Parität). Heute prüft
das C-`triad` `.tri`-Dateien (lex + parse + type-check):

```sh
./native/c/triad check examples/basic/hello.tri
```

Bauen und Host-Tests über `make -C native/c`-Targets.

## Die Kernel-Seite

`native/kernel/` ist ein Bare-Metal-x86-Kernel: Boot- und
Deskriptortabellen-Assembly, Linkerskripte, `kernel.bin`-Images,
Host-Tests. Es ist die TRIAD-OS-Richtung — das Substrat als
Betriebssystem.

## Wohin von hier

```
  Pfad fertig? ─────► referenz/ zum Nachschlagen
  Beispiele? ───────► referenz/beispiele.md (alle Domänen)
  Physik? ──────────► triad-lab-Repo (65 aufgezeichnete Studien)
  Tempo? ───────────► ./triad bench
```

✎ probier es: `./triad compile` deine Datei aus Schritt 1 mit
`--native` und starte das Binary.

Zurück zur Tür: [README](../README.md).
