# 13 — Nativo: C, kernel e além

## Três camadas, uma linguagem

```
  ┌─────────────────────────────────┐
  │  programas .tri (você está aqui)│
  ├─────────────────────────────────┤
  │  native/c   runtime C + checker │  ./native/c/triad check f.tri
  ├─────────────────────────────────┤
  │  native/kernel  boot bare-metal │  asm + ld → kernel.bin
  └─────────────────────────────────┘
```

## O lado C

`native/c/` tem runtime C (`libtriad_rt`), CLI e testes host (regex,
autograd shapes, AES, paridade do solver). Hoje o `triad` em C confere
`.tri` (lex + parse + type-check):

```sh
./native/c/triad check examples/basic/hello.tri
```

Build e testes host pelos alvos `make -C native/c`.

## O lado kernel

`native/kernel/` é um kernel x86 bare-metal: assembly de boot e
tabelas de descritores, scripts de link, imagens `kernel.bin`, testes
host. É a direção TRIAD OS — o substrato como sistema operacional.

## Para onde ir daqui

```
  terminou a trilha? ─► referencia/ para consulta
  quer exemplos? ─────► referencia/exemplos.md (todos os domínios)
  quer física? ───────► repo triad-lab (65 estudos registrados)
  quer velocidade? ───► ./triad bench
```

✎ experimente: `./triad compile` seu arquivo do passo 1 com `--native`
e rode o binário.

Volta à porta: [README](../README.md).
