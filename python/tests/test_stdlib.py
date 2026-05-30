import sys, os, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend.parser_universal import parse
from runtime.interpreter import Interpreter

def run(code):
    old = sys.stdout
    sys.stdout = buf = io.StringIO()
    try:
        interp = Interpreter()
        interp.run(parse(code, '<test>'))
        return buf.getvalue().strip()
    finally:
        sys.stdout = old

def test_math_sqrt():
    assert run('import math; print(math.sqrt(9.0));') == '3.0'

def test_math_pi():
    out = run('import math; print(math.pi);')
    assert out.startswith('3.14')

def test_math_clamp():
    assert run('import math; print(math.clamp(15, 0, 10));') == '10'

def test_random_randint():
    out = run('import random; random.seed(42); print(random.randint(0, 100));')
    assert out.isdigit()

def test_string_split():
    out = run('import string; print(string.split("a b c"));')
    assert 'a' in out

def test_string_upper():
    assert run('import string; print(string.upper("hi"));') == 'HI'

def test_json_roundtrip():
    code = '\nimport json;\nlet d = {"x": 1};\nlet s = json.stringify(d);\nlet p = json.parse(s);\nprint(p["x"]);\n'
    assert run(code) == '1'

def test_collections_sorted():
    out = run('import collections; print(collections.sorted([3,1,2]));')
    assert out == '[1, 2, 3]'

def test_builtins_len():
    assert run('print(len([1,2,3]));') == '3'

def test_builtins_range():
    out = run('print(range(3));')
    assert out == '[0, 1, 2]'

def test_builtins_str():
    assert run('print(str(42));') == '42'

def test_builtins_int():
    assert run('print(int(3.7));') == '3'

def test_builtins_float():
    assert run('print(float(3));') == '3.0'

def test_builtins_abs():
    assert run('print(abs(-5));') == '5'

def test_builtins_min_max():
    assert run('print(min(3, 1, 2));') == '1'
    assert run('print(max(3, 1, 2));') == '3'
if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print(f'  PASS {name}')
            except Exception as e:
                print(f'  FAIL {name}: {e}')