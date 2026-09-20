# 12 — Tools: the CLI tour

You know `run`, `check`, `doctor`. The rest of the bench:

## Every day

```sh
./triad fmt app.tri              # format
./triad repl                     # scratchpad
./triad watch app.tri            # re-run on save
./triad test                     # project tests
./triad bench app.tri            # timings
```

## Seeing inside

```sh
./triad debug app.tri -b 12      # breakpoint at line 12
./triad tui app.tri              # interactive inspector
./triad plot app.tri             # run + emit PNG
./triad jit-stats                # hot spots
./triad memory status            # crystal memory
```

## Shipping

```sh
./triad compile app.tri --native -o app     # native binary
./triad bundle app.tri -o app.pyz           # standalone .pyz
./triad serve --port 8000                   # API server
./triad play                                # 3D engine in browser
```

## Physics from the shell

```sh
./triad solve --N 64 --T 2.0 --dim 1        # evolve, no .tri needed
./triad observables run.npy                 # read a saved field
```

## Projects and docs

```sh
./triad init myapp && ./triad install && ./triad list
./triad docgen src/ -f html                 # docs from code
./triad lsp                                 # editor support
```

Full table with notes: [reference/cli.md](../reference/cli.md).

✎ try it: `./triad bench` on your file from step 1 and read the numbers.

Next: [13 — native](13-native.md).
