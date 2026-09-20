# 13 — Nativo: C, kernel y más allá

## Tres capas, un lenguaje

```
  ┌─────────────────────────────────┐
  │  programas .tri (estás aquí)    │
  ├─────────────────────────────────┤
  │  native/c   runtime C + checker │  ./native/c/triad check f.tri
  ├─────────────────────────────────┤
  │  native/kernel  boot bare-metal │  asm + ld → kernel.bin
  └─────────────────────────────────┘
```

## El lado C

`native/c/` trae runtime C (`libtriad_rt`), CLI y pruebas host
(regex, autograd shapes, AES, paridad del solver). Hoy el `triad` en
C chequea `.tri` (lex + parse + type-check):

```sh
./native/c/triad check examples/basic/hello.tri
```

Build y pruebas host vía blancos `make -C native/c`.

## El lado kernel

`native/kernel/` es un kernel x86 bare-metal: assembly de boot y
tablas de descriptores, scripts de link, imágenes `kernel.bin`,
pruebas host. Es la dirección TRIAD OS — el sustrato como sistema
operativo.

## A dónde ir de aquí

```
  ¿terminaste la ruta? ► referencia/ para consulta
  ¿quieres ejemplos? ──► referencia/ejemplos.md (todos los dominios)
  ¿quieres física? ────► repo triad-lab (65 estudios registrados)
  ¿quieres velocidad? ─► ./triad bench
```

✎ prueba: `./triad compile` tu archivo del paso 1 con `--native`
y corre el binario.

Vuelta a la puerta: [README](../README.md).
