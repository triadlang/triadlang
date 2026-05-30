import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend.parser_universal import parse
from compiler.typecheck_universal import typecheck, TypeCheckError

def test_valid_program():
    mod = parse('let x = 10; print(x);')
    typecheck(mod)

def test_undefined_variable():
    mod = parse('print(undefined_var);')
    try:
        typecheck(mod)
        assert False, 'should raise'
    except TypeCheckError as e:
        assert 'undefined_var' in str(e)

def test_fn_scope():
    mod = parse('fn add(a, b) { return a + b; } print(add(1, 2));')
    typecheck(mod)

def test_for_scope():
    mod = parse('for i in range(10) { print(i); }')
    typecheck(mod)

def test_import_scope():
    mod = parse('import math; print(math);')
    typecheck(mod)

def test_nested_scope():
    code = '\nlet x = 10;\nif x > 5 {\n    let y = 20;\n    print(y);\n}\n'
    mod = parse(code)
    typecheck(mod)
if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print(f'  PASS {name}')
            except Exception as e:
                print(f'  FAIL {name}: {e}')