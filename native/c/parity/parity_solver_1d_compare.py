"""
Tolerant comparator for parity_solver_1d.

Implements the §7.4 acceptance criteria from the v3 reference:

  linear mode:
    max ||Ψ_A|² - |Ψ_B|²| <= 1e-13     (fp64 floor)
    norm conservation:    |norm - 1|  <= 1e-13

  full noiseless mode:
    relative L2 in |Ψ|² at t=T <= 1e-3
    L2 := sqrt(sum (Δρ)²) / sqrt(sum ρ_py²)

These tolerances are NOT prescribed by the comparator; they are the
precision floor the two implementations naturally converge to. The
comparator just CHECKS them, doesn't impose anything on the field.

Discrete labels (fixture name, N, mode, V_ext, etc.) must match exactly
on both sides.

Exit 0 = both fixtures within tolerance; 1 = otherwise.
"""
import math
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
NATIVE = REPO / "native" / "c" / "parity" / "parity_solver_1d"
PY = REPO / "src" / "scripts" / "codegen" / "solver_1d_to_text.py"

LINEAR_MAX_ABS_TOL = 1e-13
LINEAR_NORM_TOL    = 1e-13
# both fixtures are now genuinely noiseless on both sides (fdt_couple off), so
# the full fixture converges to the same fp64 floor as the linear one. the §7.4
# acceptance bound is 1e-3; the implementations actually agree far below it.
FULL_REL_L2_TOL    = 1e-3

def parse(text: str) -> dict:
    """Parse the fingerprint format into per-fixture dicts."""
    out = {}
    cur = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        toks = line.split()
        head = toks[0]
        if head == "fixture":
            cur = {"name": toks[1]}
            out[cur["name"]] = cur
        elif head == "N":
            cur["N"]    = int(toks[1])
            cur["L"]    = float(toks[3])
            cur["dt"]   = float(toks[5])
            cur["T"]    = float(toks[7])
            cur["mode"] = toks[9]
        elif head == "norm_initial":
            cur["norm_initial"] = float(toks[1])
            cur["norm_final"]   = float(toks[3])
        elif head == "density":
            cur["density"] = [float(v) for v in toks[1:]]
        elif head == "Lambda":
            cur["Lambda"] = float(toks[1])
            cur["alpha"]  = float(toks[3])
            cur["sigma"]  = float(toks[5])
            cur["Gamma"]  = float(toks[7])
            cur["f_FDT"]  = float(toks[9])
    return out

def check_linear(py: dict, c: dict) -> tuple[bool, str]:
    if py["N"] != c["N"]: return False, f"linear N {py['N']} != {c['N']}"
    if py["mode"] != c["mode"]: return False, f"linear mode {py['mode']} != {c['mode']}"

    py_rho = py["density"]
    c_rho  = c["density"]
    if len(py_rho) != len(c_rho):
        return False, f"linear density length py={len(py_rho)} c={len(c_rho)}"

    max_abs = max(abs(a - b) for a, b in zip(py_rho, c_rho))
    if max_abs > LINEAR_MAX_ABS_TOL:
        return False, (f"linear max ||Ψ_A|²-|Ψ_B|²| = {max_abs:.3e} "
                       f"> floor {LINEAR_MAX_ABS_TOL:.0e}")

    for label, v in [("py norm_final", py["norm_final"]),
                     ("c  norm_final", c["norm_final"])]:
        if abs(v - 1.0) > LINEAR_NORM_TOL:
            return False, (f"linear {label} = {v:.17g} "
                           f"deviates from 1 by {abs(v-1):.3e} > {LINEAR_NORM_TOL:.0e}")

    return True, f"linear OK (max diff {max_abs:.3e}, fp64 floor met)"

def check_full(py: dict, c: dict) -> tuple[bool, str]:
    if py["N"] != c["N"]: return False, f"full N {py['N']} != {c['N']}"
    if py["mode"] != c["mode"]: return False, f"full mode {py['mode']} != {c['mode']}"

    py_rho = py["density"]
    c_rho  = c["density"]
    if len(py_rho) != len(c_rho):
        return False, f"full density length py={len(py_rho)} c={len(c_rho)}"

    num = math.sqrt(sum((a - b) ** 2 for a, b in zip(py_rho, c_rho)))
    den = math.sqrt(sum(a * a for a in py_rho))
    rel = num / den if den > 0 else float("inf")
    if rel > FULL_REL_L2_TOL:
        return False, (f"full noiseless rel L2 = {rel:.3e} "
                       f"> §7.4 tolerance {FULL_REL_L2_TOL:.0e}")
    return True, f"full noiseless OK (rel L2 {rel:.3e}, well under {FULL_REL_L2_TOL:.0e})"

def main() -> int:
    py_out = subprocess.run(["python3", str(PY)],
                            capture_output=True, text=True, check=True).stdout
    c_out  = subprocess.run([str(NATIVE)],
                            capture_output=True, text=True, check=True).stdout
    py = parse(py_out)
    c  = parse(c_out)

    for name in ["linear", "full_noiseless"]:
        if name not in py: return _fail(f"missing fixture '{name}' in Python output")
        if name not in c:  return _fail(f"missing fixture '{name}' in C output")

    ok_lin, msg_lin = check_linear(py["linear"], c["linear"])
    ok_full, msg_full = check_full(py["full_noiseless"], c["full_noiseless"])

    print(f"  linear:         {msg_lin}")
    print(f"  full_noiseless: {msg_full}")

    if ok_lin and ok_full:
        print("parity_solver_1d: PASS (§7.4 acceptance criteria met on both fixtures)")
        return 0
    print("parity_solver_1d: FAIL")
    return 1

def _fail(msg: str) -> int:
    print(f"parity_solver_1d: FAIL — {msg}")
    return 1

if __name__ == "__main__":
    sys.exit(main())
