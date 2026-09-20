# 05 — Tipos e classes

## `type`: um molde simples de dados

```tri
type Pessoa {
    nome: String;
    idade: Int;
}

let p = Pessoa(nome="Ana", idade=30);
print(p.nome);    // Ana
print(p.idade);   // 30
```

`type` é estrutura pura: campos com tipo, sem comportamento.

## `class`: dados + comportamento

```
  class Nome {
      fn __init__(self, ...) { ... }   // nasce aqui
      fn metodo(self, ...) { ... }     // vive aqui
  }
```

```tri
class Conta {
    fn __init__(self, titular) {
        self.titular = titular;
        self.saldo = 0;
    }

    fn depositar(self, valor) {
        self.saldo = self.saldo + valor;
    }
}

let c = Conta("Ana");
c.depositar(100);
print(c.saldo);   // 100
```

- `self` é a própria instância, sempre o primeiro parâmetro.
- `__init__` roda na criação.

## Herança: `inherits`

```tri
class Animal {
    fn __init__(self, nome) {
        self.nome = nome;
    }
    fn fala(self) {
        return "...";
    }
}

class Gato inherits Animal {
    fn fala(self) {
        return "miau";
    }
}

let g = Gato("Tom");
print(g.nome + " diz " + g.fala());
// ▸ Tom diz miau
```

`Gato` herda `nome` e troca `fala` — o resto vem do pai.

## Quando usar cada um

```
  só guardar campos ──────► type
  guardar + fazer coisas ─► class
  reaproveitar e variar ──► inherits
```

✎ experimente: `class Retangulo` com `largura/altura` e `fn area(self)`.

Próximo: [06 — erros](06-erros.md).
