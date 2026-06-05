#!/usr/bin/env bash
#
# TriadLang — Build standalone binary distribution.
# Usage: bash python/build_dist.sh
#
# Produces: dist/triadlang-<version>-<platform>/
#   triad          — standalone binary (no source exposed)
#   examples/      — demo .tri files
#   docs/          — user guide
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
DIST="$ROOT/dist"
PLATFORM="$(uname -s)-$(uname -m)"
VERSION="1.0.0"
TARGET="$DIST/triadlang-$VERSION-$PLATFORM"

echo "=== TriadLang Build ==="
echo "Platform: $PLATFORM"
echo "Target:   $TARGET"
echo ""

# clean previous build artifacts
rm -rf "$TARGET" "$ROOT/build" "$DIST/bin" 2>/dev/null || true
mkdir -p "$TARGET"

# --- build binary ---
echo "[1/4] Building binary with PyInstaller..."
pyinstaller --onefile \
    --name triad \
    --distpath "$DIST/bin" \
    --workpath "$ROOT/build" \
    --specpath "$ROOT/build" \
    --clean \
    --noconfirm \
    -p "$ROOT" \
    --hidden-import=cli --hidden-import=cli.main --hidden-import=cli.repl \
    --hidden-import=frontend --hidden-import=frontend.ast_nodes \
    --hidden-import=frontend.lexer_universal --hidden-import=frontend.parser_universal \
    --hidden-import=frontend.lexer --hidden-import=frontend.parser \
    --hidden-import=runtime --hidden-import=runtime.compiler_runtime \
    --hidden-import=runtime.interpreter --hidden-import=runtime.solver \
    --hidden-import=runtime.multi_runtime --hidden-import=runtime.consciousness \
    --hidden-import=runtime.observables --hidden-import=runtime.observables_atoms \
    --hidden-import=runtime.backend --hidden-import=runtime.codec \
    --hidden-import=runtime.codec_3d --hidden-import=runtime.tensor \
    --hidden-import=runtime.nn --hidden-import=runtime.causal \
    --hidden-import=runtime.vm --hidden-import=runtime.vm_runtime \
    --hidden-import=runtime.equation_runtime --hidden-import=runtime.projections \
    --hidden-import=runtime.memory_manager --hidden-import=runtime.viz \
    --hidden-import=runtime.sat_solver \
    --hidden-import=compiler --hidden-import=compiler.ir \
    --hidden-import=compiler.lower --hidden-import=compiler.emit_json \
    --hidden-import=compiler.formatter --hidden-import=compiler.typecheck_universal \
    --hidden-import=compiler.typecheck --hidden-import=compiler.triadc \
    --hidden-import=stdlib --hidden-import=stdlib.regimes \
    --hidden-import=stdlib.templates --hidden-import=stdlib.templates_3d \
    --hidden-import=stdlib.registry --hidden-import=stdlib.triad_module \
    --hidden-import=numpy \
    "$ROOT/entry.py" \
    2>&1 | tail -5

cp "$DIST/bin/triad" "$TARGET/triad"
chmod +x "$TARGET/triad"
echo "  -> binary: $(du -h "$TARGET/triad" | cut -f1)"

# --- examples ---
echo "[2/4] Copying examples..."
mkdir -p "$TARGET/examples"

# hello world
cat > "$TARGET/examples/hello.tri" << 'EOF'
// Hello World — TriadLang
let name = "TriadLang"
print("Hello from " + name + "!")

fn fibonacci(n) {
    if n <= 1 { return n }
    return fibonacci(n - 1) + fibonacci(n - 2)
}

for i in range(10) {
    print("fib(" + str(i) + ") = " + str(fibonacci(i)))
}
EOF

# classes + pattern matching
cat > "$TARGET/examples/classes.tri" << 'EOF'
// Classes and Pattern Matching

class Counter {
    fn init(self) { self.count = 0 }
    fn inc(self) { self.count = self.count + 1 }
    fn get(self) { return self.count }
}

let c = Counter()
c.inc()
c.inc()
c.inc()
print("counter: " + str(c.get()))

fn classify(n) {
    let result = ""
    match n {
        case 0 => { result = "zero" }
        case 1 => { result = "one" }
        case _ => { result = "many" }
    }
    return result
}

print(classify(0))
print(classify(1))
print(classify(42))
EOF

# solver demo
cat > "$TARGET/examples/solver_demo.tri" << 'EOF'
// TriadLang PDE Solver Demo
// Real nonlinear Schrodinger-type PDE with memory and dissipation

@T(5.0)
reg crystal : R5_crystal = 4;
OBSERVE crystal k_star, crystallinity, peak, norm;

@T(5.0)
reg robust : anti_collapse = 4;
OBSERVE robust k_star, crystallinity, peak, norm;
EOF

# closures + generators
cat > "$TARGET/examples/closures.tri" << 'EOF'
// Closures, Generators, and List Comprehensions

fn make_adder(n) {
    fn add(x) { return x + n }
    return add
}

let add5 = make_adder(5)
let add10 = make_adder(10)
print("add5(3) = " + str(add5(3)))
print("add10(3) = " + str(add10(3)))

let nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
let evens = [x for x in nums if x % 2 == 0]
let doubled = [x * 2 for x in nums]
print("evens: " + str(evens))
print("doubled: " + str(doubled))

fn count(n) {
    let i = 0
    while i < n {
        yield i * i
        i = i + 1
    }
}

let squares = [x for x in count(6)]
print("squares: " + str(squares))
EOF

# basic examples from repo
if [ -d "$ROOT/examples/basic" ]; then
    mkdir -p "$TARGET/examples/basic"
    for f in "$ROOT"/examples/basic/*.tri; do
        [ -f "$f" ] && cp "$f" "$TARGET/examples/basic/"
    done
fi

# crossdomain examples
if [ -d "$ROOT/crossdomain" ]; then
    mkdir -p "$TARGET/examples/crossdomain"
    for f in "$ROOT"/crossdomain/*.tri; do
        [ -f "$f" ] && cp "$f" "$TARGET/examples/crossdomain/"
    done
fi

EXAMPLE_COUNT=$(find "$TARGET/examples" -name "*.tri" | wc -l)
echo "  -> $EXAMPLE_COUNT example files"

# --- docs ---
echo "[3/4] Writing documentation..."
mkdir -p "$TARGET/docs"

cat > "$TARGET/docs/README.md" << 'DOCEOF'
# TriadLang — Public Demo

**The structural equation of persistent extended entities.**

TriadLang is a programming language designed for equation-native computation.
It compiles to optimized NumPy code and includes a built-in PDE solver for the
Triad equation — a nonlinear Schrödinger-type equation with memory, fractional
diffusion, and dissipation.

## Quick Start

```bash
./triad run examples/hello.tri        # Hello world
./triad run examples/classes.tri      # Classes + pattern matching
./triad run examples/closures.tri     # Closures + generators
./triad run examples/solver_demo.tri  # PDE solver
./triad repl                          # Interactive REPL
```

## Language Features

### Variables and Functions
```
let x = 42
const pi = 3.14159

fn greet(name) {
    print("Hello, " + name + "!")
}
```

### Classes and Inheritance
```
class Counter {
    fn init(self) { self.count = 0 }
    fn inc(self) { self.count = self.count + 1 }
    fn get(self) { return self.count }
}
```

### Pattern Matching
```
match value {
    case 0 => { print("zero") }
    case 1 => { print("one") }
    case _ => { print("other") }
}
```

### Generators
```
fn count(n) {
    let i = 0
    while i < n { yield i; i = i + 1 }
}
```

### List Comprehensions
```
let evens = [x for x in range(20) if x % 2 == 0]
```

### Closures
```
fn make_adder(n) {
    fn add(x) { return x + n }
    return add
}
```

### F-Strings
```
let name = "world"
print(f"Hello, {name}!")
```

### Try/Catch
```
try {
    risky()
} catch e {
    print("error: " + str(e))
} finally {
    cleanup()
}
```

## PDE Solver

The core of TriadLang is the equation solver. It solves:

```
iℏ ∂ₜ Ψ = [-ℏ²/(2m) D² + V_ext + Λ|Ψ|² + V_mem + α(-Δ)^(σ/2) - iΓ] Ψ + η
```

### Available Regimes

| Regime | Behavior |
|--------|----------|
| `R5_crystal` | Crystalline order |
| `anti_collapse` | Robust persistence |
| `thermal_pure` | Thermal equilibrium |
| `dispersive` | Wave dispersion |
| `B0` | Baseline |

### Solver Syntax
```
@T(5.0)
reg my_field : R5_crystal = 4;
OBSERVE my_field k_star, crystallinity, peak, norm;
```

### Coupled Substrates
```
@T(10.0)
reg a : R5_crystal = 4;
reg b : anti_collapse = 4;
ring(a, b) kappa=-2.5 for T=10.0;
OBSERVE a k_star, crystallinity;
OBSERVE b k_star, crystallinity;
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `triad run <file>` | Run a .tri file |
| `triad repl` | Interactive REPL |
| `triad check <file>` | Type check |
| `triad fmt <file>` | Format source |
| `triad bench <file>` | Benchmark |
| `triad init <name>` | Create project |
| `triad install` | Install dependencies |
| `triad publish` | Publish to local registry |
| `triad list` | List packages |
| `triad doctor` | Check installation |

## License

This is a demonstration build. All rights reserved.
DOCEOF

echo "  -> docs written"

# --- manifest ---
echo "[4/4] Writing manifest..."
cat > "$TARGET/MANIFEST" << 'EOF'
TriadLang Public Demo Build
============================
Standalone binary distribution. No installation or dependencies required.

Quick test:
  ./triad run examples/hello.tri
  ./triad run examples/solver_demo.tri
  ./triad repl

No Python needed. All computation runs locally.
EOF

# --- final ---
TOTAL_SIZE=$(du -sh "$TARGET" | cut -f1)
FILE_COUNT=$(find "$TARGET" -type f | wc -l)

echo ""
echo "=== Build Complete ==="
echo "  Output:     $TARGET"
echo "  Binary:     $(du -h "$TARGET/triad" | cut -f1)"
echo "  Total:      $TOTAL_SIZE ($FILE_COUNT files)"
echo ""
echo "  Test commands:"
echo "    $TARGET/triad run $TARGET/examples/hello.tri"
echo "    $TARGET/triad run $TARGET/examples/solver_demo.tri"
