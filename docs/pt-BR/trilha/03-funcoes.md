# 03 — Funções

## `fn` básico

```
  fn nome(parâmetros) {
      ...
      return valor;
  }
```

```tri
fn soma(a, b) {
    return a + b;
}

print(soma(2, 3));   // 5
```

Sem `return`, a função devolve `none`.

## Valores padrão

```tri
fn potencia(base, expoente=2) {
    return base ** expoente;
}

print(potencia(3));      // 9
print(potencia(2, 10));  // 1024
```

## `*args` e `**kwargs`: quantos vierem

```tri
fn somar_tudo(*numeros) {
    let s = 0;
    for n in numeros {
        s = s + n;
    }
    return s;
}
print(somar_tudo(1, 2, 3, 4));   // 10

fn mostrar(**opcoes) {
    for chave in opcoes {
        print(str(chave) + " = " + str(opcoes[chave]));
    }
}
mostrar(cor="azul", tamanho=42);
// ▸ cor = azul / tamanho = 42
```

## Recursão funciona

```tri
fn fib(n) {
    if n <= 1 {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}

print(fib(10));   // 55
```

## `yield`: função que entrega aos poucos

```tri
fn contagem() {
    yield 1;
    yield 2;
    yield 3;
}

for v in contagem() {
    print(v);
}
// ▸ 1 2 3
```

Cada `yield` pausa e devolve um valor; o `for` continua de onde parou.

✎ experimente: escreva `fn par(n)` que devolve `true` se `n % 2 == 0`.

Próximo: [04 — coleções](04-colecoes.md).
