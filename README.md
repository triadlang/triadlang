# TriadLang

TriadLang is a programming language for scientific and physics-oriented
computation.  The repository includes a parser, type checker, Python runtime,
native C backend, REPL, formatter, debugger, LSP and domain-specific numerical
modules.

## Requirements

- Python 3.12 or newer
- NumPy 1.24 or newer
- Optional native dependencies: a C11 compiler, FFTW, Boehm GC and CUDA

CUDA is optional.  TriadLang automatically falls back to its NumPy backend when
CuPy or a usable CUDA device is unavailable.  Set `TRIADLANG_BACKEND=cpu` to
force CPU execution.

## Development setup

```sh
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/pytest
```

Run a program directly from a checkout:

```sh
PYTHONPATH=src .venv/bin/python -m cli.main run examples/basic/hello.tri
```

Other useful commands are available with:

```sh
PYTHONPATH=src .venv/bin/python -m cli.main --help
```

## Minimal program

```triad
let message = "Hello from TriadLang";
print(message);
```

Programs run in safe mode by default. File access is restricted to the program
workspace and process, network, native-code and destructive operations require
explicit capabilities. Do not use `--unsafe` for untrusted source code.

## Native runtime

```sh
make -C native/c all
```

Use `make -C native/c rebuild` for a clean rebuild. Optional dependencies are
detected by the Makefile.

## License

See [LICENSE](LICENSE).
