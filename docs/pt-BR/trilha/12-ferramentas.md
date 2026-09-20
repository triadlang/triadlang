# 12 — Ferramentas: o tour do CLI

Você conhece `run`, `check`, `doctor`. O resto da bancada:

## Todo dia

```sh
./triad fmt app.tri              # formata
./triad repl                     # rascunho
./triad watch app.tri            # re-roda ao salvar
./triad test                     # testes do projeto
./triad bench app.tri            # tempos
```

## Ver por dentro

```sh
./triad debug app.tri -b 12      # breakpoint na linha 12
./triad tui app.tri              # inspetor interativo
./triad plot app.tri             # roda + emite PNG
./triad jit-stats                # pontos quentes
./triad memory status            # memória cristal
```

## Entregar

```sh
./triad compile app.tri --native -o app   # binário nativo
./triad bundle app.tri -o app.pyz         # .pyz standalone
./triad serve --port 8000                 # servidor API
./triad play                              # engine 3D no browser
```

## Física pelo shell

```sh
./triad solve --N 64 --T 2.0 --dim 1      # evolui, sem .tri
./triad observables run.npy               # lê campo salvo
```

## Projetos e docs

```sh
./triad init meuapp && ./triad install && ./triad list
./triad docgen src/ -f html               # docs do código
./triad lsp                               # suporte a editor
```

Tabela completa com notas: [referencia/cli.md](../referencia/cli.md).

✎ experimente: `./triad bench` no seu arquivo do passo 1 e leia os números.

Próximo: [13 — nativo](13-nativo.md).
