import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend.ast_nodes import *

def test_pos():
    p = Pos(1, 5, 'test.tri')
    assert p.line == 1
    assert p.col == 5

def test_intlit():
    n = IntLit(42)
    assert n.value == 42

def test_binop():
    b = BinOp('+', IntLit(1), IntLit(2))
    assert b.op == '+'

def test_fn_decl():
    f = FnDecl('add', [Param('a'), Param('b')], body=[ReturnStmt(BinOp('+', Ident('a'), Ident('b')))])
    assert f.name == 'add'
    assert len(f.params) == 2

def test_module():
    m = Module('test', [LetStmt('x', value=IntLit(10))])
    assert m.name == 'test'
    assert len(m.body) == 1

def test_type_decl():
    t = TypeDecl('Person', [TypeField('name', 'String'), TypeField('age', 'Int')])
    assert len(t.fields) == 2

def test_entity_decl():
    e = EntityDecl('player', base='Actor', fields={'health': IntLit(100)})
    assert e.base == 'Actor'

def test_import_stmt():
    i = ImportStmt(['games', 'world'])
    assert i.path == ['games', 'world']

def test_for_stmt():
    f = ForStmt('i', CallExpr(Ident('range'), [IntLit(10)]), [ExprStmt(CallExpr(Ident('print'), [Ident('i')]))])
    assert f.var == 'i'

def test_all_expr_types():
    exprs = [IntLit(1), FloatLit(3.14), BoolLit(True), StringLit('hi'), NoneLit(), Ident('x'), BinOp('+', IntLit(1), IntLit(2)), UnaryOp('-', IntLit(1)), ListExpr([IntLit(1)]), MapExpr([(StringLit('a'), IntLit(1))])]
    assert len(exprs) == 10
if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print(f'  PASS {name}')
            except Exception as e:
                print(f'  FAIL {name}: {e}')