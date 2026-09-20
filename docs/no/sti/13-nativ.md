# 13 — Nativt: C, kjerne og videre

## Tre lag, ett språk

```
  ┌─────────────────────────────────┐
  │  .tri-programmer (du er her)    │
  ├─────────────────────────────────┤
  │  native/c   C-runtime + checker │  ./native/c/triad check f.tri
  ├─────────────────────────────────┤
  │  native/kernel  bare-metal boot │  asm + ld → kernel.bin
  └─────────────────────────────────┘
```

## C-siden

`native/c/` bærer en C-runtime (`libtriad_rt`), en CLI og
vertstester (regex, autograd shapes, AES, solver-paritet). I dag
kontrollerer C-`triad` `.tri`-filer (lex + parse + typekontroll):

```sh
./native/c/triad check examples/basic/hello.tri
```

Bygg og vertstester via `make -C native/c`-mål.

## Kjernesiden

`native/kernel/` er en bare-metal-x86-kjerne: boot- og
deskriptortabell-assembly, linkerskript, `kernel.bin`-bilder,
vertstester. Det er TRIAD-OS-retningen — substratet som
operativsystem.

## Hvor herfra

```
  sti ferdig? ─────────► referanse/ for oppslag
  vil ha eksempler? ───► referanse/eksempler.md (alle domener)
  vil ha fysikk? ──────► triad-lab-repoet (65 loggførte studier)
  vil ha fart? ────────► ./triad bench
```

✎ prøv: `./triad compile` filen din fra steg 1 med `--native`
og kjør binæren.

Tilbake til døren: [README](../README.md).
