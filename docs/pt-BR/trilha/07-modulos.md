# 07 — Módulos: importar o mundo

## As três formas

```tri
import math;                    // módulo inteiro
import math as m;               // com apelido
from math import sqrt, pi;      // só o que precisa
```

## Stdlib essencial (sempre disponível)

```
  math         sqrt sin cos log exp floor ceil pi e min max
  random       random randint choice seed shuffle uniform
  string       split join replace lower upper strip
  json         parse stringify
  fs           read_text write_text exists listdir join tempdir
  time         now sleep
  collections  len range enumerate sorted zip map filter…
  datetime     datas e horas
  regex        expressões regulares
  plot         gráficos (triad.plot é o mesmo)
  io           print input
```

Exemplo junto:

```tri
import math;
import random;

random.seed(0);
print(math.sqrt(25.0));       // 5.0
print(random.randint(1, 10)); // 7 (com seed 0)
```

## Arquivos (`fs`)

```tri
import fs;

let caminho = fs.join(fs.tempdir(), "oi.txt");
fs.write_text(caminho, "Hello file!");
print(fs.read_text(caminho));   // Hello file!
print(fs.exists(caminho));      // true
```

## Seus próprios módulos

Arquivo `util.tri`:

```tri
fn dobro(n) {
    return n * 2;
}
```

Arquivo `app.tri`, na mesma pasta:

```tri
import util;

print(util.dobro(21));   // 42
```

## Pacotes (projetos maiores)

```sh
./triad init meuapp        # cria projeto
./triad install            # instala dependências
./triad publish            # publica no registro local
./triad list               # lista instalados
```

Dependências moram em `triad_modules/`.

✎ experimente: um módulo `sauda.tri` com `fn ola(nome)` e um
programa que importa e usa.

Próximo: [08 — arrays](08-arrays.md).
