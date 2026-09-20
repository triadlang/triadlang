# 13 — Native: C, kernel, and beyond

## Three layers, one language

```
  ┌─────────────────────────────────┐
  │  .tri programs (you are here)   │
  ├─────────────────────────────────┤
  │  native/c   C runtime + checker │  ./native/c/triad check f.tri
  ├─────────────────────────────────┤
  │  native/kernel  bare-metal boot │  asm + ld → kernel.bin
  └─────────────────────────────────┘
```

## The C side

`native/c/` holds a C runtime (`libtriad_rt`), a CLI, and host tests
(regex, autograd shapes, AES, solver parity). Today the C `triad`
checks `.tri` files (lex + parse + type-check):

```sh
./native/c/triad check examples/basic/hello.tri
```

Build and host tests via `make -C native/c` targets.

## The kernel side

`native/kernel/` is a bare-metal x86 kernel: boot and descriptor-table
assembly, linker scripts, `kernel.bin` images, host-side tests. It is
the TRIAD OS direction — the substrate as an operating system.

## Where to go from here

```
  finished the track? ──► reference/ for lookup
  want examples? ───────► reference/examples.md (every domain)
  want physics? ────────► triad-lab repo (65 recorded studies)
  want speed? ──────────► ./triad bench
```

✎ try it: `./triad compile` your step-1 file with `--native` and
run the binary.

Back to the door: [README](../README.md).
