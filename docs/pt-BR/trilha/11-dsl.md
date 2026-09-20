# 11 — DSL: escrever física direto

Ao lado do `.tri` cotidiano há um segundo dialeto: poucas palavras,
cada uma um ato físico. Registradores, acoplamentos, observações.

## As cinco palavras

```
  reg a : anti_collapse = 4;     um registrador guardando um campo
  ring(a, b) kappa=-2.5 ...;     acopla em anel
  OBSERVE a k_star, peak, ...;   lê observáveis
  @T(18.0)                       horizonte de tempo no topo
```

## Um programa completo

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

`▸ saída`

```text
a = { k_star=8.6394, crystallinity=0.9784, peak=0.3177, atom_count=0.9646 }
b = { k_star=8.6394, crystallinity=0.9784, peak=0.3180, atom_count=0.9646 }
```

Dois registradores, um anel, dez números. Um experimento inteiro.

## A ideia por baixo: P1 / P2 / P3

```
  P1 oscilação ──────── o campo nunca fica parado
  P2 autorreferência ── presente + memória co-evoluem
  P3 acoplamento ────── relações moldam a evolução
```

Tudo roda junto — oscilação, memória e banho numa evolução integral.
Nenhum termo é desligado para "simplificar".

Mais campos para percorrer: `examples/triad/` (equilíbrio, memória,
observáveis, potenciais customizados).

✎ experimente: mude `kappa` para `-1.0` e compare os dez números.

Próximo: [12 — ferramentas](12-ferramentas.md).
