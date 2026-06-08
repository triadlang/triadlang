import sys
import os
import io
import contextlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.parser_universal import parse
from runtime.compiler_runtime import TriadCompiler, set_safe_mode

def _run(src: str):
    mod = parse(src, '<safe>')
    mod.file = '<safe>'
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        TriadCompiler().compile_and_run(mod)
    return buf.getvalue()

def teardown_function(_):
    set_safe_mode(False)

def test_py_eval_allowed_without_safe():
    set_safe_mode(False)
    assert _run('print(py_eval("1 + 1"));').strip() == '2'

def test_py_eval_blocked_in_safe_mode():
    set_safe_mode(True)
    try:
        _run('print(py_eval("1 + 1"));')
        assert False, 'py_eval should be blocked in safe mode'
    except PermissionError as e:
        assert 'safe mode' in str(e)

def test_py_exec_blocked_in_safe_mode():
    set_safe_mode(True)
    try:
        _run('py_exec("x = 1");')
        assert False, 'py_exec should be blocked in safe mode'
    except PermissionError:
        pass

def test_safe_import_allows_math():
    set_safe_mode(True)
    assert _run('import math; print(math.sqrt(9.0));').strip() == '3.0'

def test_safe_import_blocks_os():
    set_safe_mode(True)
    try:
        _run('import os; print(os);')
        assert False, 'os import should be blocked in safe mode'
    except ImportError as e:
        assert 'safe mode' in str(e)

if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                teardown_function(None)
                print(f'  PASS {name}')
            except AssertionError as e:
                print(f'  FAIL {name}: {e}')
