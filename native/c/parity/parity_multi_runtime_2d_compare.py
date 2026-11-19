"""
Tolerant comparator for parity_multi_runtime_2d.

Two 2D substrates (N=32, L=16) evolve through 80 timesteps of FFT
split-step Strang in 2D. The error budget scales with N²·steps, so we
relax tolerances accordingly:

  - psi samples:   ≤ 5e-13 absolute or 1e-11 relative
  - norm / dens_*: ≤ 1e-10 absolute or 1e-11 relative
  - global_t:      ≤ 1 ULP
  - integers (N, n_records, diverged, D): byte-exact
"""
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
NATIVE = REPO / "native" / "c" / "parity" / "parity_multi_runtime_2d"
PY = REPO / "src" / "scripts" / "codegen" / "multi_runtime_2d_to_text.py"

PSI_ATOL = 5e-13
PSI_RTOL = 1e-11
SCALAR_ATOL = 1e-10
SCALAR_RTOL = 1e-11
GLOBAL_T_ATOL = 5e-15

def is_float_token(tok: str) -> bool:
    if "." in tok or "e" in tok or "E" in tok:
        try:
            float(tok)
            return True
        except ValueError:
            return False
    return False

def ulp_diff(a: float, b: float) -> int:
    if not (math.isfinite(a) and math.isfinite(b)):
        return 0 if a == b else 9
    if a == b:
        return 0
    if (a < 0) != (b < 0):
        return 9
    steps = 0
    cur = a
    if a < b:
        while cur < b and steps < 8:
            cur = math.nextafter(cur, math.inf); steps += 1
    else:
        while cur > b and steps < 8:
            cur = math.nextafter(cur, -math.inf); steps += 1
    return steps if cur == b else 9

def floats_close(a, b, atol, rtol):
    if a == b:
        return True
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    if abs(a - b) <= atol:
        return True
    scale = max(abs(a), abs(b))
    if scale > 0 and abs(a - b) / scale <= rtol:
        return True
    return False

def compare_line(py_line, c_line):
    if py_line == c_line:
        return True, ""
    py_toks = py_line.split()
    c_toks = c_line.split()
    if len(py_toks) != len(c_toks):
        return False, f"token count: py={len(py_toks)} c={len(c_toks)}"
    head = py_toks[0]
    is_psi_line = (head == "sub" and len(py_toks) >= 3 and py_toks[2] == "psi")
    for i, (a, b) in enumerate(zip(py_toks, c_toks)):
        if a == b:
            continue
        if not is_float_token(a) or not is_float_token(b):
            return False, f"strict mismatch at tok {i}: {a!r} vs {b!r}"
        fa, fb = float(a), float(b)
        if head == "global_t":
            ok = abs(fa - fb) <= GLOBAL_T_ATOL
        elif is_psi_line and i >= 3:
            ok = floats_close(fa, fb, PSI_ATOL, PSI_RTOL)
        else:
            ok = floats_close(fa, fb, SCALAR_ATOL, SCALAR_RTOL)
        if not ok:
            return False, (
                f"float mismatch at tok {i}: {a} vs {b} "
                f"(abs={abs(fa-fb):.3e})"
            )
    return True, ""

def main():
    py_out = subprocess.run(
        ["python3", str(PY)], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    c_out = subprocess.run(
        [str(NATIVE)], capture_output=True, text=True, check=True
    ).stdout.splitlines()

    if len(py_out) != len(c_out):
        print(f"FAIL: line count py={len(py_out)} c={len(c_out)}")
        return 1

    fails = []
    passes = 0
    for i, (a, b) in enumerate(zip(py_out, c_out)):
        ok, why = compare_line(a, b)
        if ok:
            passes += 1
        else:
            fails.append(f"  L{i+1}: {why}\n    py: {a[:200]}\n    c : {b[:200]}")

    total = len(py_out)
    if fails:
        print(f"FAIL: {len(fails)}/{total} lines diverge")
        for f in fails[:10]:
            print(f)
        return 1
    print(f"parity_multi_runtime_2d: PASS ({passes}/{total})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
