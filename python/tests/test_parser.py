import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend.parser_universal import parse, ParseError
from frontend.ast_nodes import *

def test_let():
    m = parse('let x = 10;')
    assert isinstance(m.body[0], LetStmt)
    assert m.body[0].name == 'x'

def test_const():
    m = parse('const PI = 3.14;')
    assert isinstance(m.body[0], ConstStmt)

def test_fn():
    m = parse('fn add(a, b) { return a + b; }')
    assert isinstance(m.body[0], FnDecl)
    assert m.body[0].name == 'add'
    assert len(m.body[0].params) == 2

def test_if():
    m = parse('if x > 10 { print("big"); } else { print("small"); }')
    assert isinstance(m.body[0], IfStmt)
    assert m.body[0].else_body is not None

def test_for():
    m = parse('for i in range(10) { print(i); }')
    assert isinstance(m.body[0], ForStmt)
    assert m.body[0].var == 'i'

def test_while():
    m = parse('while x < 10 { x = x + 1; }')
    assert isinstance(m.body[0], WhileStmt)

def test_type():
    m = parse('type Person { name: String; age: Int; }')
    assert isinstance(m.body[0], TypeDecl)
    assert len(m.body[0].fields) == 2

def test_import():
    m = parse('import math;')
    assert isinstance(m.body[0], ImportStmt)
    assert m.body[0].path == ['math']

def test_dotted_import():
    m = parse('import games.world;')
    assert isinstance(m.body[0], ImportStmt)
    assert m.body[0].path == ['games', 'world']

def test_list():
    m = parse('let xs = [1, 2, 3];')
    s = m.body[0]
    assert isinstance(s, LetStmt)
    assert isinstance(s.value, ListExpr)
    assert len(s.value.elements) == 3

def test_map():
    m = parse('let m = {"a": 1, "b": 2};')
    s = m.body[0]
    assert isinstance(s.value, MapExpr)

def test_method_call():
    m = parse('xs.push(4);')
    s = m.body[0]
    assert isinstance(s, ExprStmt)
    assert isinstance(s.expr, MethodCallExpr)

def test_field_access():
    m = parse('let n = p.name;')
    s = m.body[0]
    assert isinstance(s.value, FieldExpr)

def test_nested_expr():
    m = parse('let r = (a + b) * c;')
    s = m.body[0]
    assert isinstance(s.value, BinOp)

def test_entity():
    m = parse('entity player : Actor { health = 100; }')
    assert isinstance(m.body[0], EntityDecl)
    assert m.body[0].base == 'Actor'

def test_reg():
    m = parse('reg neuron : HodgkinHuxley = 4;')
    assert isinstance(m.body[0], RegStmt)
    assert m.body[0].regime == 'HodgkinHuxley'

def test_parse_error():
    try:
        parse('let = ;')
        assert False, 'should raise'
    except ParseError:
        pass

def test_lambda():
    m = parse('let f = fn(x) { return x * 2; };')
    s = m.body[0]
    assert isinstance(s.value, LambdaExpr)

def test_elif():
    m = parse('if x > 10 { print(1); } elif x > 5 { print(2); } else { print(3); }')
    s = m.body[0]
    assert isinstance(s, IfStmt)
    assert len(s.elif_clauses) == 1

def test_kwargs_call():
    m = parse('Person(name="Ana", age=30);')
    s = m.body[0]
    assert isinstance(s.expr, CallExpr)
    assert 'name' in s.expr.kwargs
if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print(f'  PASS {name}')
            except Exception as e:
                print(f'  FAIL {name}: {e}')