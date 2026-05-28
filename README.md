# TriadLang

**TriadLang is a programming language whose runtime is a quantum field equation.**

Most languages compile to bit operations on registers. TriadLang compiles to evolution of a continuous wavefunction `Ψ` governed by the Triad equation — a nonlinear Schrödinger equation with hierarchical memory, fractional dispersion, and fluctuation-dissipation noise. Programs do not push bits; they shape a field that crystallizes into structure.

> *"Starts with chaos, then stabilizes. Crystallization is emergent, not imposed. It calibrates itself without imposing rules."*

```
iℏ ∂_t Ψ = [ −ℏ²/(2m) ∇² + V_ext + Λ|Ψ|² + V_mem + α(−Δ)^(σ/2) − iΓ ] Ψ + η
```

## Status

- **Version:** 1.0.0
- **Language:** Python (this distribution). A native C/CUDA build is sold separately under a commercial license.
- **Tests:** 153 passing (interpreter, parser, IR, solver audit against the equation reference)
- **License:** Proprietary source-available. See [`LICENSE`](LICENSE). Public for inspection, evaluation, and reproducibility. Commercial use requires a written license from qrv0.

## Install

Requires Python 3.10 or later. The only hard dependency is NumPy.

```bash
pip install numpy
```

Optional feature sets:

```bash
pip install -e ".[science]"     # SciPy — advanced observables, SAT peak finding
pip install -e ".[viz]"         # matplotlib — plotting helpers
pip install -e ".[quantum]"     # Qiskit + Cirq — real quantum hardware backends
pip install -e ".[lsp]"         # pygls — Language Server Protocol
pip install -e ".[all]"         # science + viz + lsp
```

You can also just clone the repo and run directly — no install needed:

```bash
git clone <repo>
cd triadlang
./triad doctor          # verify the install
./triad run python/examples/basic/hello.tri
```

## Hello, TriadLang

Create `hello.tri`:

```tri
print("Hello from TriadLang");
```

Run it:

```bash
./triad run hello.tri
```

## A First Triad Program

The whole point of TriadLang is that you can describe a piece of physics declaratively and let the equation do the work. This program declares two coupled substrates and asks what crystallizes:

```tri
@T(18.0)
reg a : anti_collapse = 4;
reg b : anti_collapse = 4;
ring(a, b) kappa=-2.5 for T=18.0;
OBSERVE a k_star, crystallinity, peak, atom_count;
OBSERVE b k_star, crystallinity, peak, atom_count;
```

Output:

```
a = { k_star=6.2832, crystallinity=0.9492, peak=0.3732, atom_count=2.7687 }
b = { k_star=6.2832, crystallinity=0.9492, peak=0.3732, atom_count=2.7687 }
```

Both substrates crystallized at `k* = 2π`, with a crystallinity score of `0.949`. You did not write a single loop, gradient, or update rule. The equation found the structure on its own.

## CLI

```
triad run <file.tri>              Run a .tri file
triad check <file.tri>            Type-check
triad compile <file.tri> --emit-ir  Emit IR as JSON
triad repl                        Interactive REPL
triad lsp                         Start LSP server
triad doctor                      Verify installation
triad test                        Run the 153-test audit suite
triad bench <file.tri>            Benchmark execution
triad debug <file.tri> [-b N]     Debugger with breakpoints
triad docgen <file|dir>           Generate documentation
triad fmt <file.tri>              Format a .tri source file
```

## Where to Go Next

Read these in order:

1. **[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)** — the equation, the three pillars (P1/P2/P3), the computational model. Read this first or nothing else will make sense.
2. **[docs/HOW_TO_USE.md](docs/HOW_TO_USE.md)** — the `.tri` language, the CLI, a guided tour of the examples.
3. **[docs/HOW_TO_READ.md](docs/HOW_TO_READ.md)** — map of the codebase. Where the lexer lives, where the parser lives, where the solver lives, what to read first if you want to understand the implementation.
4. **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — the compilation pipeline: `.tri` → AST → IR → transpiled Python → NumPy/FFT solver.
5. **[docs/QUANTUM.md](docs/QUANTUM.md)** — the qubit module. Amplitude encoding, gates, real hardware backends (IBM Quantum, Google Cirq).
6. **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)** — the speedup claim, the methodology, how to reproduce it. Real benchmark, real PDE, not fibonacci.
7. **[docs/EQUATION_REFERENCE.md](docs/EQUATION_REFERENCE.md)** — the mathematical reference. Definitions, invariants, the §7 audit tests that the solver must pass.

## Examples Included

| Path | What it shows |
|---|---|
| `python/examples/basic/hello.tri` | Hello world |
| `python/examples/basic/variables.tri` | Variables and primitive types |
| `python/examples/basic/functions.tri` | Functions, recursion, factorial |
| `python/examples/basic/loops.tri` | `for`, `while`, `break` |
| `python/examples/basic/types.tri` | Type declarations, structs, methods |
| `python/examples/basic/modules.tri` | Importing from the standard library |
| `python/examples/basic/json.tri` | JSON parse and stringify |
| `python/examples/basic/files.tri` | File I/O |
| `python/examples/triad/anti_collapse.tri` | Coupled substrates crystallizing under the anti-collapse regime |
| `python/examples/triad/custom_potential.tri` | Defining a custom external potential `V_ext` |
| `python/examples/triad/observables_test.tri` | Full observable battery (k*, crystallinity, IPR, FWHM, …) |
| `python/examples/verification/fdt_claims.tri` | Fluctuation-dissipation theorem verification |
| `python/examples/games/predator_prey.tri` | Lotka-Volterra population dynamics |
| `python/examples/llm/mock_runtime.tri` | LLM KV-cache mock — TriadLang as a host for token caches |
| `python/examples/ml/*` | Machine-learning examples |
| `python/examples/solver_*.tri` | Solver convergence, ablation, comparison studies |

## Reproduce the Speedup

```bash
python3 python/benchmarks/bench_comparison.py
```

You should see something close to:

```
Pure Python DFT-loop : ~9.3 s
NumPy FFT (solver)   : ~0.08 s
Speedup              : ~115x
```

Same equation. Same grid. Same time step. The only difference is one uses scalar DFT loops in pure Python; the other uses NumPy's vectorized FFT. The speedup *is* the language.

## Contact and Licensing

- **Web:** https://triadlang.com
- **Commercial licensing:** see [`LICENSE`](LICENSE) §6.

---

Copyright (c) 2026 qrv0. All rights reserved. Proprietary source-available — not open source. Public visibility does not grant permission to use, copy, modify, or commercialize. See [`LICENSE`](LICENSE) for full terms.
# triadlang
