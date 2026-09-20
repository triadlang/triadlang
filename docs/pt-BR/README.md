# TriadLang — Documentação

```
                         ┌──────────────┐
                    ┌────│  VOCÊ ESTÁ   │────┐
                    │    │     AQUI     │    │
                    │    └──────────────┘    │
                    ▼                       ▼
          ┌─────────────────┐     ┌─────────────────┐
          │  NUNCA VI TRIAD │     │  JÁ PROGRAMA,   │
          │  COMECE DO ZERO │     │  QUERO ACHAR    │
          └────────┬────────┘     │  RÁPIDO         │
                   │              └────────┬────────┘
                   ▼                       ▼
          trilha/ 00 → 13         referencia/
          (curso guiado)          (consulta)
```

## A trilha (do zero ao substrato)

| Passo | Arquivo | Você sai sabendo |
|---|---|---|
| 0 | [trilha/00-boas-vindas.md](trilha/00-boas-vindas.md) | o que é, instalar, `doctor`, primeiro `run` |
| 1 | [trilha/01-primeiros-passos.md](trilha/01-primeiros-passos.md) | hello, variáveis, `print`, REPL |
| 2 | [trilha/02-controle.md](trilha/02-controle.md) | `if`, loops, f-strings |
| 3 | [trilha/03-funcoes.md](trilha/03-funcoes.md) | `fn`, args, `*args/**kwargs`, `yield` |
| 4 | [trilha/04-colecoes.md](trilha/04-colecoes.md) | listas, dicts, `string`, `json` |
| 5 | [trilha/05-tipos-classes.md](trilha/05-tipos-classes.md) | `type`, `class`, herança |
| 6 | [trilha/06-erros.md](trilha/06-erros.md) | `try/catch`, `throw`, `assert` |
| 7 | [trilha/07-modulos.md](trilha/07-modulos.md) | imports, stdlib, pacotes |
| 8 | [trilha/08-arrays.md](trilha/08-arrays.md) | `triad.ntri`, vetorização |
| 9 | [trilha/09-python.md](trilha/09-python.md) | interop Python, modo safe/unsafe |
| 10 | [trilha/10-solver.md](trilha/10-solver.md) | solver, SAT, qubits |
| 11 | [trilha/11-dsl.md](trilha/11-dsl.md) | DSL `reg/ring/OBSERVE`, P1/P2/P3 |
| 12 | [trilha/12-ferramentas.md](trilha/12-ferramentas.md) | CLI: fmt, test, bench, debug, bundle… |
| 13 | [trilha/13-nativo.md](trilha/13-nativo.md) | compilação nativa, C, kernel |

## Referência (consulta direta)

- [referencia/sintaxe.md](referencia/sintaxe.md) — a linguagem inteira numa página visual
- [referencia/stdlib.md](referencia/stdlib.md) — módulos `math`, `random`, `fs`…
- [referencia/triad.md](referencia/triad.md) — os 39 módulos `triad.*`
- [referencia/cli.md](referencia/cli.md) — cada comando `./triad` com exemplo
- [referencia/exemplos.md](referencia/exemplos.md) — mapa de `examples/` por domínio

## Convenções desta doc

- Todo bloco `.tri` mostrado aqui roda: `./triad run arquivo.tri`.
- `▸ saída` marca o que o programa imprime.
- Blocos ```tri-frag são recortes ilustrativos (só na sintaxe); o resto roda.
- `✎ experimente` é um convite, não lição de casa.
