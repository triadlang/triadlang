# 11 — DSL: escribir física directo

Junto al `.tri` cotidiano hay un segundo dialecto: pocas palabras,
cada una un acto físico. Registros, acoples, observaciones.

## Las cinco palabras

```
  reg a : anti_collapse = 4;     un registro guardando un campo
  ring(a, b) kappa=-2.5 ...;     acopla en anillo
  OBSERVE a k_star, peak, ...;   lee observables
  @T(18.0)                       horizonte de tiempo arriba
```

## Un programa completo

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

`▸ salida`

```text
a = { k_star=8.6394, crystallinity=0.9784, peak=0.3177, atom_count=0.9646 }
b = { k_star=8.6394, crystallinity=0.9784, peak=0.3180, atom_count=0.9646 }
```

Dos registros, un anillo, diez números. Un experimento entero.

## La idea debajo: P1 / P2 / P3

```
  P1 oscilación ───────── el campo nunca se queda quieto
  P2 autorreferencia ──── presente + memoria co-evolucionan
  P3 acople ───────────── las relaciones moldean la evolución
```

Todo corre junto — oscilación, memoria y baño en una evolución
integral. Ningún término se apaga para "simplificar".

Más campos por recorrer: `examples/triad/` (equilibrio, memoria,
observables, potenciales custom).

✎ prueba: cambia `kappa` a `-1.0` y compara los diez números.

Siguiente: [12 — herramientas](12-herramientas.md).
