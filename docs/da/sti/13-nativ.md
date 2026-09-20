# 13 — Nativt: C, kerne og videre

## Tre lag, ét sprog

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
værtstests (regex, autograd shapes, AES, solver-paritet). I dag
kontrollerer C-`triad` `.tri`-filer (lex + parse + typecheck):

```sh
./native/c/triad check examples/basic/hello.tri
```

Byg og værtstests via `make -C native/c`-mål.

## Kernesiden

`native/kernel/` er en bare-metal-x86-kerne: boot- og
deskriptortabel-assembly, linkerskripte, `kernel.bin`-images,
værtstests. Det er TRIAD-OS-retningen — substratet som
operativsystem.

## Hvor herfra

```
  sti færdig? ─────────► reference/ til opslag
  vil have eksempler? ─► reference/eksempler.md (alle domæner)
  vil have fysik? ─────► triad-lab-repoet (65 loggede studier)
  vil have fart? ──────► ./triad bench
```

✎ prøv: `./triad compile` din fil fra trin 1 med `--native`
og kør binæren.

Tilbage til døren: [README](../README.md).
