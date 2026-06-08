"""parity between the two execution paths.

triadlang has two runtimes for the modern language:
  - the tree-walking Interpreter (runtime/interpreter.py), used by the repl
  - the task-oriented TriadCompiler (runtime/compiler_runtime.py), used by
    `triad run`, which transpiles to python and execs it.

these are meant to be observationally equivalent for ordinary programs: the
same source must print the same thing. this test runs a battery of programs
through both and compares stdout byte for byte. see docs/ARCHITECTURE.md for
which path is used when.
"""
import sys
import os
import io
import contextlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.parser_universal import parse
from runtime.interpreter import Interpreter
from runtime.compiler_runtime import TriadCompiler

PROGRAMS = {
    'arithmetic': 'let a = 7; let b = 3; print(a + b); print(a - b); print(a * b); print(a / b); print(a % b);',
    'string_concat': 'let n = "world"; print("hello " + n); print("n=" + str(5 + 1));',
    'fstring': 'let x = 42; let f = 3.5; print(f"x={x} f={f} sum={x + f}");',
    'for_loop': 'let s = 0; for v in [1, 2, 3, 4, 5] { s = s + v; } print(s);',
    'while_loop': 'let i = 0; let t = 0; while i < 5 { t = t + i; i = i + 1; } print(t);',
    'if_else': 'let x = 10; if x > 5 { print("big"); } else { print("small"); }',
    'function': 'fn add(a, b) { return a + b; } print(add(2, 3)); print(add(10, -4));',
    'recursion': 'fn fact(n) { if n <= 1 { return 1; } return n * fact(n - 1); } print(fact(5));',
    'nested_calls': 'fn sq(x) { return x * x; } fn dbl(x) { return x + x; } print(sq(dbl(3)));',
    'list_index': 'let xs = [10, 20, 30]; print(xs[0]); print(xs[2]); print(len(xs));',
    'bool_logic': 'let a = true; let b = false; if a and not b { print("yes"); }',
    'comparison_chain': 'print(1 < 2); print(2 == 2); print(3 != 4); print(5 >= 5);',
}

def _run_interpreter(src: str) -> str:
    mod = parse(src, '<parity>')
    mod.file = '<parity>'
    interp = Interpreter()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        interp.run(mod)
    return buf.getvalue()

def _run_compiler(src: str) -> str:
    mod = parse(src, '<parity>')
    mod.file = '<parity>'
    compiler = TriadCompiler()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        compiler.compile_and_run(mod)
    return buf.getvalue()

def _check(name: str):
    src = PROGRAMS[name]
    out_interp = _run_interpreter(src)
    out_comp = _run_compiler(src)
    assert out_interp == out_comp, (
        f"parity mismatch for {name!r}:\n"
        f"  interpreter: {out_interp!r}\n"
        f"  compiler:    {out_comp!r}"
    )

def test_arithmetic():
    _check('arithmetic')

def test_string_concat():
    _check('string_concat')

def test_fstring():
    _check('fstring')

def test_for_loop():
    _check('for_loop')

def test_while_loop():
    _check('while_loop')

def test_if_else():
    _check('if_else')

def test_function():
    _check('function')

def test_recursion():
    _check('recursion')

def test_nested_calls():
    _check('nested_calls')

def test_list_index():
    _check('list_index')

def test_bool_logic():
    _check('bool_logic')

def test_comparison_chain():
    _check('comparison_chain')

if __name__ == '__main__':
    for nm in PROGRAMS:
        try:
            _check(nm)
            print(f'  PASS {nm}')
        except AssertionError as e:
            print(f'  FAIL {nm}: {e}')
