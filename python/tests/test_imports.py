import sys, os, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend.parser_universal import parse
from runtime.interpreter import Interpreter

def run(code):
    old = sys.stdout
    sys.stdout = buf = io.StringIO()
    try:
        interp = Interpreter()
        mod = parse(code, '<test>')
        interp.run(mod)
        return buf.getvalue().strip()
    finally:
        sys.stdout = old

def test_import_math():
    assert run('import math; print(math.sqrt(16.0));') == '4.0'

def test_import_random():
    out = run('import random; print(random.randint(1, 1));')
    assert out == '1'

def test_import_json():
    out = run('import json; print(json.stringify({"a": 1}));')
    assert '"a"' in out

def test_import_string():
    out = run('import string; print(string.upper("hello"));')
    assert out == 'HELLO'

def test_import_fs():
    code = '\nimport fs;\nfs.write_text("/tmp/_triad_test_import.txt", "ok");\nprint(fs.read_text("/tmp/_triad_test_import.txt"));\n'
    assert run(code) == 'ok'

def test_import_collections():
    out = run('import collections; print(collections.len([1,2,3]));')
    assert out == '3'

def test_import_time():
    out = run('import time; let t = time.now(); print(t > 0);')
    assert out == 'true'
if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print(f'  PASS {name}')
            except Exception as e:
                print(f'  FAIL {name}: {e}')