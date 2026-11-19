"""Aggregate parity driver.

Iterates every relevant .tri fixture in the repo and runs each native
parity binary side-by-side with its Python reference. Compares stdout
byte-for-byte. Exit 0 if every fixture matches, 1 otherwise.

Native binaries (built by `make frontend`):
  - parity_ast            (universal AST JSON)        vs scripts/ast_to_json.py
  - parity_ast_legacy     (legacy AST JSON)           vs scripts/ast_to_json_legacy.py
  - parity_ir             (lowered IR JSON)           vs scripts/ir_to_json.py
  - parity_check_native   (typecheck error text)      vs scripts/check_to_text.py
  - parity_check_legacy   (legacy typecheck errors)   vs scripts/typecheck_legacy_to_text.py

Fixture sets:
  - universal: python/examples/**/*.tri and python/tests/**/*.tri that parse with parser_universal
  - legacy:    python/crossdomain/*.tri that parse with the v1 DSL parser

This script preserves the repo's existing parity model: native dumps
stdout, Python dumps stdout, we diff. No external golden files.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
PY_ROOT = ROOT / "src"
SCRIPTS = PY_ROOT / "scripts" / "codegen"

UNIVERSAL_GLOBS = [
    "examples/**/*.tri",
]
LEGACY_GLOBS = [
    "examples/crossdomain/*.tri",
]

EXCLUDE_PARTS = {"dist", "build", "node_modules", ".codegraph", ".claude"}

def collect(globs: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in globs:
        for p in ROOT.glob(pat):
            if any(part in EXCLUDE_PARTS for part in p.parts):
                continue
            out.append(p)
    return sorted(out)

def run(cmd: list[str]) -> tuple[int, str, str]:
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr

def diff_pair(name: str, fixture: Path, native_cmd: list[str], py_cmd: list[str]) -> bool:
    rc_n, out_n, err_n = run(native_cmd)
    rc_p, out_p, err_p = run(py_cmd)
    
    ok = (rc_n == rc_p) and (out_n == out_p)
    status = "PASS" if ok else "FAIL"
    try:
        rel = fixture.relative_to(ROOT)
    except ValueError:
        rel = fixture
    print(f"  [{name}] {status:4s} {rel}")
    if not ok:
        if rc_n != rc_p:
            print(f"      rc native={rc_n} python={rc_p}")
        
        nl = out_n.splitlines()
        pl = out_p.splitlines()
        for i in range(min(len(nl), len(pl))):
            if nl[i] != pl[i]:
                print(f"      L{i+1} native: {nl[i][:200]}")
                print(f"      L{i+1} python: {pl[i][:200]}")
                break
        if len(nl) != len(pl):
            print(f"      line counts: native={len(nl)} python={len(pl)}")
        if err_n.strip():
            print(f"      native stderr: {err_n.strip()[:300]}")
        if err_p.strip():
            print(f"      python stderr: {err_p.strip()[:300]}")
    return ok

def require(binary: Path, label: str) -> bool:
    if not binary.exists():
        print(f"  SKIP {label}: missing {binary.name} (run `make -C native/c frontend`)")
        return False
    return True

def main() -> int:
    py = sys.executable
    parity_ast = HERE / "parity_ast"
    parity_ast_legacy = HERE / "parity_ast_legacy"
    parity_ir = HERE / "parity_ir"
    parity_check = HERE / "parity_check_native"
    parity_check_legacy = HERE / "parity_check_legacy"
    parity_format = HERE / "parity_format"
    parity_docgen = HERE / "parity_docgen"
    parity_lsp = HERE / "parity_lsp"

    sys.path.insert(0, str(PY_ROOT))
    from stdlib.regimes import list_regimes  
    regimes_file = tempfile.NamedTemporaryFile(
        "w", suffix=".regimes", delete=False)
    for r in list_regimes():
        regimes_file.write(r + "\n")
    regimes_file.close()
    regimes_path = regimes_file.name

    universal = collect(UNIVERSAL_GLOBS)
    legacy = collect(LEGACY_GLOBS)

    print(f"Fixtures: universal={len(universal)} legacy={len(legacy)}")
    total = 0
    fails = 0

    if require(parity_ast, "AST(universal)"):
        print("== AST universal ==")
        for f in universal:
            total += 1
            ok = diff_pair(
                "ast",
                f,
                [str(parity_ast), str(f)],
                [py, str(SCRIPTS / "ast_to_json.py"), str(f)],
            )
            fails += 0 if ok else 1

    if require(parity_ir, "IR(universal)"):
        print("== IR universal ==")
        for f in universal:
            total += 1
            ok = diff_pair(
                "ir",
                f,
                [str(parity_ir), str(f)],
                [py, str(SCRIPTS / "ir_to_json.py"), str(f)],
            )
            fails += 0 if ok else 1

    if require(parity_check, "check(universal)"):
        print("== typecheck universal ==")
        for f in universal:
            total += 1
            ok = diff_pair(
                "check",
                f,
                [str(parity_check), str(f)],
                [py, str(SCRIPTS / "check_to_text.py"), str(f)],
            )
            fails += 0 if ok else 1

    if require(parity_ast_legacy, "AST(legacy)"):
        print("== AST legacy ==")
        for f in legacy:
            total += 1
            ok = diff_pair(
                "ast_legacy",
                f,
                [str(parity_ast_legacy), str(f)],
                [py, str(SCRIPTS / "ast_to_json_legacy.py"), str(f)],
            )
            fails += 0 if ok else 1

    if require(parity_docgen, "docgen"):
        print("== docgen markdown ==")
        for f in universal:
            total += 1
            ok = diff_pair(
                "docgen_md",
                f,
                [str(parity_docgen), str(f)],
                [py, str(SCRIPTS / "docgen_to_text.py"), str(f)],
            )
            fails += 0 if ok else 1
        print("== docgen html ==")
        for f in universal:
            total += 1
            ok = diff_pair(
                "docgen_html",
                f,
                [str(parity_docgen), str(f), "--html"],
                [py, str(SCRIPTS / "docgen_to_text.py"), str(f), "--html"],
            )
            fails += 0 if ok else 1

    if require(parity_format, "format(universal)"):
        print("== format universal ==")
        for f in universal:
            total += 1
            ok = diff_pair(
                "format",
                f,
                [str(parity_format), str(f)],
                [py, str(SCRIPTS / "format_to_text.py"), str(f)],
            )
            fails += 0 if ok else 1

    if require(parity_check_legacy, "check(legacy)"):
        print("== typecheck legacy ==")
        for f in legacy:
            total += 1
            ok = diff_pair(
                "check_legacy",
                f,
                [str(parity_check_legacy), str(f), "--regimes", regimes_path],
                [py, str(SCRIPTS / "typecheck_legacy_to_text.py"), str(f)],
            )
            fails += 0 if ok else 1

    try:
        os.unlink(regimes_path)
    except OSError:
        pass

    parity_codec_bin = HERE / "parity_codec"
    parity_codec_cmp = HERE / "parity_codec_compare.py"
    if require(parity_codec_bin, "codec"):
        print("== codec ==")
        total += 1
        rc = subprocess.call([py, str(parity_codec_cmp)])
        if rc == 0:
            print("  [codec] PASS (tolerant: ≤2 ULP on samples, exact on decoders)")
        else:
            print("  [codec] FAIL")
            fails += 1

    parity_atoms_bin = HERE / "parity_atoms"
    parity_atoms_py  = SCRIPTS / "atoms_to_text.py"
    if require(parity_atoms_bin, "atoms"):
        print("== atoms ==")
        total += 1
        py_out = subprocess.run([py, str(parity_atoms_py)],
                                capture_output=True, text=True).stdout
        c_out  = subprocess.run([str(parity_atoms_bin)],
                                capture_output=True, text=True).stdout
        if py_out == c_out:
            print("  [atoms] PASS (byte-exact — atom_count_nd, atoms_per_region, "
                  "atom_separation_1d, atom_centroids_nd; 7 fixtures × 1D/2D/3D)")
        else:
            print("  [atoms] FAIL (byte diff)")
            fails += 1

    parity_solver_bin = HERE / "parity_solver_1d"
    parity_solver_cmp = HERE / "parity_solver_1d_compare.py"
    if require(parity_solver_bin, "solver_1d"):
        print("== solver_1d ==")
        total += 1
        rc = subprocess.call([py, str(parity_solver_cmp)])
        if rc == 0:
            print("  [solver_1d] PASS (§7.4 acceptance — P1+P2+P3 all live, "
                  "linear floor 1e-13 met, full noiseless rel L2 ≪ 1e-3)")
        else:
            print("  [solver_1d] FAIL")
            fails += 1

    parity_compiler_bin = HERE / "parity_compiler"
    parity_compiler_py  = SCRIPTS / "compiler_to_text.py"
    if require(parity_compiler_bin, "compiler"):
        print("== compiler ==")
        total += 1
        py_out = subprocess.run([py, str(parity_compiler_py)],
                                capture_output=True, text=True).stdout
        c_out  = subprocess.run([str(parity_compiler_bin)],
                                capture_output=True, text=True).stdout
        if py_out == c_out:
            print("  [compiler] PASS (byte-exact — classify/generate/remember/"
                  "couple/sat × 13 fixtures; P1+P2+P3 in every default)")
        else:
            print("  [compiler] FAIL (byte diff)")
            fails += 1

    for label, bin_name, cmp_name in [
        ("multi_runtime",          "parity_multi_runtime",          "parity_multi_runtime_compare.py"),
        ("multi_runtime_2d",       "parity_multi_runtime_2d",       "parity_multi_runtime_2d_compare.py"),
        ("multi_runtime_3d",       "parity_multi_runtime_3d",       "parity_multi_runtime_3d_compare.py"),
        ("multi_runtime_2d_modes", "parity_multi_runtime_2d_modes", "parity_multi_runtime_2d_modes_compare.py"),
        ("multi_runtime_3d_modes", "parity_multi_runtime_3d_modes", "parity_multi_runtime_3d_modes_compare.py"),
    ]:
        parity_mr_bin = HERE / bin_name
        parity_mr_cmp = HERE / cmp_name
        if require(parity_mr_bin, label):
            print(f"== {label} ==")
            total += 1
            rc = subprocess.call([py, str(parity_mr_cmp)])
            if rc == 0:
                print(f"  [{label}] PASS (P1+P2+P3 + coupling)")
            else:
                print(f"  [{label}] FAIL")
                fails += 1

    if require(parity_lsp, "lsp"):
        
        print("== lsp ==")
        
        total += 1
        ok = diff_pair("lsp", Path("<initialize>"),
                       [str(parity_lsp), "initialize"],
                       [py, str(SCRIPTS / "lsp_to_text.py"), "initialize"])
        fails += 0 if ok else 1

        for f in universal:
            total += 1
            uri = "file://" + str(f.resolve())
            ok = diff_pair("lsp", f,
                [str(parity_lsp), "documentSymbol", str(f), uri],
                [py, str(SCRIPTS / "lsp_to_text.py"), "documentSymbol", str(f), uri])
            fails += 0 if ok else 1

        lsp_fixture = PY_ROOT / "examples/basic/functions.tri"
        if lsp_fixture.exists():
            uri = "file://" + str(lsp_fixture.resolve())
            scenarios = [
                
                ("completion", [str(lsp_fixture), "0", "0"]),
                ("completion", [str(lsp_fixture), "0", "3"]),
                ("definition", [str(lsp_fixture), "0", "3", uri]),
                ("hover",      [str(lsp_fixture), "0", "3"]),
                ("hover",      [str(lsp_fixture), "4", "3"]),
            ]
            for cmd, args in scenarios:
                total += 1
                ok = diff_pair(f"lsp/{cmd}", lsp_fixture,
                    [str(parity_lsp), cmd] + args,
                    [py, str(SCRIPTS / "lsp_to_text.py"), cmd] + args)
                fails += 0 if ok else 1

    print()
    if fails:
        print(f"parity: FAIL ({fails}/{total} mismatches)")
        return 1
    print(f"parity: PASS ({total}/{total})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
