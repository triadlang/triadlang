# 06 — Erros: quebrar com elegância

## `try` / `catch` / `finally`

```
  try {
      arrisque aqui
  } catch erro {
      se cair, caia aqui (erro tem a mensagem)
  } finally {
      sempre passa aqui
  }
```

```tri
try {
    throw "boom";
} catch e {
    print("peguei: " + str(e));
} finally {
    print("sempre executa");
}
```

`▸ saída`

```text
peguei: boom
sempre executa
```

`throw` lança qualquer valor (texto, número, dict).

## `assert`: trave suas certezas

```tri
fn dividir(a, b) {
    assert b != 0;
    return a / b;
}

print(dividir(10, 2));   // 5.0
// dividir(10, 0) aborta com erro de asserção
```

`assert` é documentação executável: se a condição falha, o programa
para ali e aponta a linha.

## `match`: escolha por valor

```tri
let v = 2;

match v {
    case 1 {
        print("um");
    }
    case 2 {
        print("dois");
    }
}
// ▸ dois
```

Cada `case` testa um valor; o bloco correspondente executa.

## O trio de emergência

```
  algo deu errado? ──► throw "motivo"
  pode dar errado? ──► try / catch
  nunca pode? ───────► assert condição
```

✎ experimente: `fn raiz(x)` que lança `throw` se `x < 0`, senão
devolve `math.sqrt(x)` (precisa de `import math;`).

Próximo: [07 — módulos](07-modulos.md).
