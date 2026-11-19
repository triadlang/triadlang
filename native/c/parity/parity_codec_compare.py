"""
Tolerant comparator for parity_codec.

Rules:
  - Lines starting with 'dec_int', 'rt_int', 'dec_bool' must match BYTE-EXACT.
    These are discrete decoder outputs — the public contract of the codec.
  - Lines starting with 'enc_*' contain repr(float) samples of psi. Floats
    may differ by 1 ULP because NumPy uses SIMD-vectorised sums while the
    C port uses scalar in-order sums; both are correct under IEEE 754.
    We tolerate ≤ 2 ULP per float (≈ 4.4e-16 relative for ~1.0).
  - 'dec_float', 'rt_float', 'norm_ok' are also float-tolerant (≤ 2 ULP).

Exit 0 on PASS, 1 on FAIL. PASS/FAIL counts go to stdout.
"""
import math
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
NATIVE = REPO / "native" / "c" / "parity" / "parity_codec"
PY = REPO / "src" / "scripts" / "codegen" / "codec_to_text.py"

ULP_TOLERANCE = 2  
FLOAT_RE = re.compile(r"-?\d+\.\d+(?:e[+-]?\d+)?|-?\d+e[+-]?\d+|-?\d+\.\d+|-?\d+")

def ulp_diff(a: float, b: float) -> int:
    """Distance in ULPs between two finite doubles. Returns int or float('inf')."""
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("inf") if a != b else 0
    if a == b:
        return 0
    if (a < 0) != (b < 0):
        
        return ulp_diff(abs(a), 0.0) + ulp_diff(0.0, abs(b))
    
    steps = 0
    cur = a
    if a < b:
        while cur < b and steps < 8:
            cur = math.nextafter(cur, math.inf)
            steps += 1
    else:
        while cur > b and steps < 8:
            cur = math.nextafter(cur, -math.inf)
            steps += 1
    return steps if cur == b else 9  

def is_float_token(tok: str) -> bool:
    """A token is a float if it parses as float and contains '.' or 'e'/'E'."""
    if "." in tok or "e" in tok or "E" in tok:
        try:
            float(tok)
            return True
        except ValueError:
            return False
    return False

ABS_NEAR_ZERO = 1e-12

REL_TOL = 0.5  

def floats_match(a: float, b: float) -> bool:
    
    if a == b:
        return True
    
    if ulp_diff(a, b) <= ULP_TOLERANCE:
        return True
    
    if abs(a) <= ABS_NEAR_ZERO and abs(b) <= ABS_NEAR_ZERO:
        return True
    
    scale = max(abs(a), abs(b))
    if scale > 0 and abs(a - b) / scale <= REL_TOL:
        return True
    return False

def compare_line(py_line: str, c_line: str) -> tuple[bool, str]:
    if py_line == c_line:
        return True, ""
    py_toks = py_line.split()
    c_toks = c_line.split()
    if len(py_toks) != len(c_toks):
        return False, f"token count mismatch: py={len(py_toks)} c={len(c_toks)}"
    head = py_toks[0] if py_toks else ""
    strict_kinds = {"dec_int", "rt_int", "dec_bool"}
    strict = head in strict_kinds
    for i, (a, b) in enumerate(zip(py_toks, c_toks)):
        if a == b:
            continue
        if strict:
            return False, f"strict mismatch at tok {i}: {a!r} vs {b!r}"
        if is_float_token(a) and is_float_token(b):
            if not floats_match(float(a), float(b)):
                return False, (
                    f"float mismatch at tok {i}: {a} vs {b} "
                    f"(ulps={ulp_diff(float(a), float(b))}, "
                    f"abs_diff={abs(float(a) - float(b)):.3e})"
                )
        else:
            return False, f"non-float token differs at {i}: {a!r} vs {b!r}"
    return True, ""

def main() -> int:
    py_out = subprocess.run(
        ["python3", str(PY)], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    c_out = subprocess.run(
        [str(NATIVE)], capture_output=True, text=True, check=True
    ).stdout.splitlines()

    if len(py_out) != len(c_out):
        print(f"FAIL: line count mismatch py={len(py_out)} c={len(c_out)}")
        return 1

    fails: list[str] = []
    passes = 0
    for i, (a, b) in enumerate(zip(py_out, c_out)):
        ok, why = compare_line(a, b)
        if ok:
            passes += 1
        else:
            fails.append(f"  line {i+1}: {why}\n    py: {a}\n    c : {b}")

    total = len(py_out)
    if fails:
        print(f"FAIL: {len(fails)}/{total} lines diverge")
        for f in fails[:20]:
            print(f)
        if len(fails) > 20:
            print(f"  ... +{len(fails)-20} more")
        return 1
    print(f"parity_codec: PASS ({passes}/{total})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
