# 01 — Primeiros passos

## Variáveis: `let` e `const`

```tri
let x = 10;
let nome = "Triad";
let ativo = true;
const PI2 = 3.14159 * 2;

print(x + 20);      // 30
print(nome);        // Triad
print(ativo);       // true
print(PI2);         // 6.28318
```

- `let` cria variável que pode mudar: `x = 99;` vale.
- `const` cria constante: tentar mudar é erro.
- Ponto e vírgula fecha a frase. Sempre.

## Os quatro tipos que você usa 99% do tempo

```
  10            3.14           "texto"         true / false / none
  └── Int      └── Float      └── String       └── Bool (none = vazio)
```

```tri
print(10 + 3.14);            // número com número funciona
print("oi " + "dev");        // texto com texto cola
print("n = " + str(42));     // número vira texto com str()
```

`▸ saída`

```text
13.14
oi dev
n = 42
```

## `print` mostra, `str` converte, `type` revela

```tri
print(str(3.14));     // "3.14"
print(type(10));   // <class 'int'>
print(type("oi"));   // <class 'str'>
```

## Comentários

```tri
// uma linha

print("vale"); // no fim da linha também
```

## O REPL é seu caderno de rascunho

```sh
./triad repl
```

```text
>>> print(2 * 21);
42
>>> exit
```

✎ experimente: crie `eu.tri` que imprime seu nome e `2 ** 10`
(`**` é potência). Rode com `./triad run eu.tri`.

Próximo: [02 — controle](02-controle.md).
