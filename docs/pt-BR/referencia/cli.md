# CLI — cada comando com exemplo

## Rodar e conferir

```sh
./triad run app.tri               # roda (só imports safe)
./triad run app.tri --unsafe      # roda (Python total)
./triad check app.tri             # checa tipos, sem executar
./triad fmt app.tri               # formata no lugar
./triad repl                      # rascunho interativo
./triad watch app.tri             # re-roda ao salvar
```

## Entender e medir

```sh
./triad test                      # testes do projeto
./triad bench app.tri             # tempos
./triad debug app.tri -b 12       # breakpoint, linha 12
./triad tui app.tri               # inspetor interativo
./triad jit-stats                 # pontos quentes
./triad docgen src/ -f html       # docs do código
./triad lsp                       # suporte a editor (servidor)
```

## Entregar e mostrar

```sh
./triad compile app.tri --native -o app   # binário nativo
./triad bundle app.tri -o app.pyz         # .pyz standalone
./triad serve --port 8000                 # servidor API
./triad play                              # engine 3D no browser
./triad plot app.tri --out fig.png        # roda + PNG
```

## Física e memória

```sh
./triad solve --N 64 --T 2.0 --dim 1      # evolui, sem .tri
./triad observables run.npy               # lê campo salvo
./triad memory status                     # memória cristal
./triad memory record "ideia"             # guarda
./triad memory recall "ideia"             # recupera
```

## Projetos

```sh
./triad init meuapp    # esqueleto
./triad setup          # instalação guiada
./triad install        # dependências
./triad publish        # registro local
./triad list           # instalados
./triad doctor         # relatório do ambiente
```

```
  escreve ──► check ──► run ──► bench ──► bundle/compile ──► entrega
                 │        │
                fmt     debug/tui/plot
```
