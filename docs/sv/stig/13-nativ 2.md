# 13 — Nativt: C, kärna och vidare

## Tre lager, ett språk

```
  ┌─────────────────────────────────┐
  │  .tri-program (du är här)       │
  ├─────────────────────────────────┤
  │  native/c   C-runtime + checker │  ./native/c/triad check f.tri
  ├─────────────────────────────────┤
  │  native/kernel  bare-metal boot │  asm + ld → kernel.bin
  └─────────────────────────────────┘
```

## C-sidan

`native/c/` bär en C-runtime (`libtriad_rt`), en CLI och
värdtester (regex, autograd shapes, AES, solver-paritet). I dag
kontrollerar C-`triad` `.tri`-filer (lex + parse + typkontroll):

```sh
./native/c/triad check examples/basic/hello.tri
```

Bygg och värdtester via `make -C native/c`-mål.

## Kärnsidan

`native/kernel/` är en bare-metal-x86-kärna: boot- och
deskriptortabell-assembly, länskript, `kernel.bin`-avbilder,
värdtester. Det är TRIAD-OS-riktningen — substratet som
operativsystem.

## Vart härifrån

```
  stig klar? ───────► referens/ för uppslag
  vill ha exempel? ► referens/exempel.md (alla domäner)
  vill ha fysik? ───► triad-lab-repot (65 loggade studier)
  vill ha fart? ────► ./triad bench
```

✎ prova: `./triad compile` din fil från steg 1 med `--native`
och kör binären.

Tillbaka till dörren: [README](../README.md).
