import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend.parser_universal import parse
from compiler.lower import lower_module
from compiler.ir import *
from compiler.emit_json import emit_json
import json

def test_lower_let():
    mod = parse('let x = 10;')
    ir = lower_module(mod)
    assert isinstance(ir.body[0], IRLet)
    assert ir.body[0].name == 'x'

def test_lower_fn():
    mod = parse('fn add(a, b) { return a + b; }')
    ir = lower_module(mod)
    assert isinstance(ir.body[0], IRFunction)

def test_lower_if():
    mod = parse('if x > 0 { print(x); }')
    ir = lower_module(mod)
    assert isinstance(ir.body[0], IRIf)

def test_lower_for():
    mod = parse('for i in range(10) { print(i); }')
    ir = lower_module(mod)
    assert isinstance(ir.body[0], IRFor)

def test_emit_json():
    mod = parse('let x = 42; print(x);')
    ir = lower_module(mod)
    j = emit_json(ir)
    data = json.loads(j)
    assert data['_type'] == 'IRModule'
    assert len(data['body']) == 2

def test_emit_json_roundtrip():
    code = '\nfn greet(name) { return "hello " + name; }\nlet msg = greet("world");\nprint(msg);\n'
    mod = parse(code)
    ir = lower_module(mod)
    j = emit_json(ir)
    data = json.loads(j)
    assert data['body'][0]['_type'] == 'IRFunction'

def test_lower_type():
    mod = parse('type P { name: String; }')
    ir = lower_module(mod)
    assert isinstance(ir.body[0], IRTypeDecl)

def test_lower_import():
    mod = parse('import math;')
    ir = lower_module(mod)
    assert isinstance(ir.body[0], IRImport)
if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print(f'  PASS {name}')
            except Exception as e:
                print(f'  FAIL {name}: {e}')