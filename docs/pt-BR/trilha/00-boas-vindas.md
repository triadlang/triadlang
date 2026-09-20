# 00 — Boas-vindas

## O que é TriadLang, em 30 segundos

```
   .tri ──► ./triad ──► resultado
    │           │
    │           ├── execução direta (run)
    │           ├── checagem de tipos (check)
    │           ├── Python por dentro (import numpy, flask…)
    │           └── física por dentro (solver, SAT, qubits)
    │
    └── ou DSL física: reg / ring / OBSERVE
```

TriadLang é uma linguagem de programação completa (variáveis, funções,
classes, erros, módulos) com duas portas para fora do comum: ela importa
o ecossistema Python de verdade e carrega um motor de dinâmica física
(solver integral TRIAD) que resolve coisas como SAT e circuitos quânticos.

## Ver se está tudo certo

```sh
./triad doctor
```

`▸ saída` — um relatório do ambiente (Python, dependências, nativo).

Se algo faltar, o próprio guia resolve:

```sh
./triad setup
```

## O primeiro run (2 minutos)

Arquivo `oi.tri`:

```tri
print("Hello from TriadLang");
```

```sh
./triad run oi.tri
```

`▸ saída`

```text
Hello from TriadLang
```

Funcionou? Você já sabe rodar TriadLang. O resto é linguagem.

## As duas checagens que você vai usar sempre

```sh
./triad check oi.tri     # só confere tipos, não executa
./triad fmt oi.tri       # formata o arquivo
```

## Mapa mental daqui em diante

```
  passos 1–7    linguagem do dia a dia (igual aprender qualquer linguagem)
       │
  passos 8–9    arrays + Python (onde triadlang encontra o mundo)
       │
  passos 10–11  solver + DSL (onde triadlang vira física)
       │
  passos 12–13  ferramentas + nativo (produção e kernel)
```

✎ experimente: rode `./triad repl`, digite `print(1 + 1);`, depois `exit`.

Próximo: [01 — primeiros passos](01-primeiros-passos.md).
