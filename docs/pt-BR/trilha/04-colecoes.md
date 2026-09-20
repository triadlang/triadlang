# 04 — Coleções: listas e dicionários

## Listas: ordem importa

```tri
let nums = [10, 20, 30];

print(nums[0]);      // 10 (índice começa em 0)
print(len(nums));    // 3

nums.push(40);       // adiciona no fim
print(nums);         // [10, 20, 30, 40]
```

Percorrendo:

```tri
let nums = [10, 20, 30, 40];
for n in nums {
    print(n * 2);
}
// ▸ 20 40 60 80
```

## Dicionários: nome importa

```tri
let pessoa = {"nome": "Ana", "idade": 30};

print(pessoa["nome"]);   // Ana
pessoa["cidade"] = "SP"; // cria chave nova
print(pessoa["cidade"]); // SP

del pessoa["idade"];     // remove chave
print(pessoa);           // {"nome": "Ana", "cidade": "SP"}
```

Percorrendo chaves:

```tri
let pessoa = {"nome": "Ana", "cidade": "SP"};
for chave in pessoa {
    print(str(chave) + ": " + str(pessoa[chave]));
}
```

## Texto é coleção de truques (`string`)

```tri
import string;

print(string.upper("triad"));              // TRIAD
print(string.split("oi dev triad"));       // ["oi", "dev", "triad"]
print(string.join("-", ["a", "b"]));       // a-b
```

## JSON entra e sai (`json`)

```tri
import json;

let dados = {"nome": "TriadLang", "n": 1};
let texto = json.stringify(dados);
print(texto);                    // {"nome": "TriadLang", "n": 1}

let volta = json.parse(texto);
print(volta["nome"]);            // TriadLang
```

## Mapa rápido

```
  preciso de ordem ──► lista [...] + push + índice
  preciso de nome ───► dict {...} + ["chave"]
  preciso de texto ──► string.split/join/upper/...
  preciso trocar ────► json.stringify / json.parse
```

✎ experimente: uma lista de compras (dict com nome e preço por item)
e imprima o total.

Próximo: [05 — tipos e classes](05-tipos-classes.md).
