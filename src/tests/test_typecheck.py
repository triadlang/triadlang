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

def test_type_error_int_plus_str():
    mod = parse('let x = 5; let y = x + "hello";')
    try:
        typecheck(mod)
        assert False, 'should raise on int + str'
    except TypeCheckError as e:
        assert 'E2003' in str(e)

def test_type_error_str_times_int():
    mod = parse('let s = "a"; let t = s * 3;')
    try:
        typecheck(mod)
        assert False, 'should raise on str * int'
    except TypeCheckError as e:
        assert 'E2003' in str(e)

def test_type_ok_str_concat():
    typecheck(parse('let a = "x"; let b = a + "y"; print(b);'))

def test_type_ok_num_arith():
    typecheck(parse('let a = 5; let b = a + 3; let c = a * b; print(c);'))

def test_type_ok_str_with_str_cast():
    typecheck(parse('let n = 5; print("v=" + str(n));'))

def test_type_ok_reassign_changes_type():
    
    typecheck(parse('let x = 5; x = "now a string"; let y = x + "!";'))

if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print(f'  PASS {name}')
            except Exception as e:
                print(f'  FAIL {name}: {e}')
