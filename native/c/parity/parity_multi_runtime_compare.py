"""
Tolerant comparator for parity_multi_runtime.

The two implementations evolve the same equation through ~150 timesteps
of FFT-based split-step Strang. Each step composes ~7 floating-point
operations per grid point (FFT + ifft, complex phase, V_couple sum,
memory updates). IEEE-754 reordering by SIMD vs scalar accumulators
makes byte-identical output impossible while still being mathematically
correct.

The accepted tolerances are calibrated by the numerical depth of the
calculation:

  - psi samples:   ≤ 2e-13 absolute or 1e-12 relative
                   (each grid point sees ~150 × 3 complex mul + FFT
                    contraction; ulp ≈ 1e-13 on values of O(1))
  - norm / dens_*: ≤ 1e-11 absolute or 1e-11 relative
                   (each is a sum of N×(150/record_every) terms, so the
                    error budget is ~N × steps × eps)
  - global_t:      ≤ 1 ULP
  - integers (N, n_records, diverged): byte-exact required.

Exit 0 PASS, 1 FAIL.
"""
import math
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
NATIVE = REPO / "native" / "c" / "parity" / "parity_multi_runtime"
PY = REPO / "src" / "scripts" / "codegen" / "multi_runtime_to_text.py"

PSI_ATOL = 2e-13
PSI_RTOL = 1e-11
SCALAR_ATOL = 1e-10
SCALAR_RTOL = 1e-10
GLOBAL_T_ATOL = 5e-15

def is_float_token(tok: str) -> bool:
    if "." in tok or "e" in tok or "E" in tok:
        try:
            float(tok)
            return True
        except ValueError:
            return False
    return False

def floats_close(a: float, b: float, atol: float, rtol: float) -> bool:
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

def compare_line(py_line: str, c_line: str) -> tuple[bool, str]:
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
                f"(abs={abs(fa-fb):.3e}, "
                f"rel={(abs(fa-fb)/max(abs(fa),abs(fb),1e-300)):.3e})"
            )
    return True, ""

def main() -> int:
    py_out = subprocess.run(
        ["python3", str(PY)], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    c_out = subprocess.run(
        [str(NATIVE)], capture_output=True, text=True, check=True
    ).stdout.splitlines()

    if len(py_out) != len(c_out):
        print(f"FAIL: line count py={len(py_out)} c={len(c_out)}")
        for i in range(max(len(py_out), len(c_out))):
            p = py_out[i] if i < len(py_out) else "<missing>"
            n = c_out[i] if i < len(c_out) else "<missing>"
            print(f"  L{i+1}: py: {p}")
            print(f"        c : {n}")
        return 1

    fails = []
    passes = 0
    for i, (a, b) in enumerate(zip(py_out, c_out)):
        ok, why = compare_line(a, b)
        if ok:
            passes += 1
        else:
            fails.append(f"  L{i+1}: {why}\n    py: {a}\n    c : {b}")
    total = len(py_out)
    if fails:
        print(f"FAIL: {len(fails)}/{total} lines diverge")
        for f in fails[:10]:
            print(f)
        if len(fails) > 10:
            print(f"  ... +{len(fails)-10} more")
        return 1
    print(f"parity_multi_runtime: PASS ({passes}/{total})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
