# CLI — every command with an example

## Run and check

```sh
./triad run app.tri               # run (safe imports only)
./triad run app.tri --unsafe      # run (full Python)
./triad check app.tri             # type-check, no execution
./triad fmt app.tri               # format in place
./triad repl                      # interactive scratchpad
./triad watch app.tri             # re-run on save
```

## Understand and measure

```sh
./triad test                      # project tests
./triad bench app.tri             # timings
./triad debug app.tri -b 12       # breakpoint, line 12
./triad tui app.tri               # interactive inspector
./triad jit-stats                 # hot spots
./triad docgen src/ -f html       # docs from code
./triad lsp                       # editor support (server)
```

## Ship and show

```sh
./triad compile app.tri --native -o app   # native binary
./triad bundle app.tri -o app.pyz         # standalone .pyz
./triad serve --port 8000                 # API server
./triad play                              # 3D engine in browser
./triad plot app.tri --out fig.png        # run + PNG
```

## Physics and memory

```sh
./triad solve --N 64 --T 2.0 --dim 1      # evolve, no .tri needed
./triad observables run.npy               # read a saved field
./triad memory status                     # crystal memory
./triad memory record "idea"              # store
./triad memory recall "idea"              # retrieve
```

## Projects

```sh
./triad init myapp     # scaffold
./triad setup          # guided install
./triad install        # dependencies
./triad publish        # local registry
./triad list           # installed packages
./triad doctor         # environment report
```

```
  write ──► check ──► run ──► bench ──► bundle/compile ──► ship
               │        │
              fmt     debug/tui/plot
```
