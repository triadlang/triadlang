import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.parser_universal import parse
from compiler.lower import lower_module
from compiler.ir import IRInt, IRFloat, IRString, IRBool, IRReturn, IRFunction, IRIf, IRWhile
from compiler.ir_passes import constant_fold, dead_code_elimination, optimize

def _lower_unoptimized(src):
    return lower_module(parse(src, '<t>'), optimize=False)

def test_fold_int_arithmetic():
    ir = optimize(_lower_unoptimized('let x = 2 + 3 * 4;'))
    v = ir.body[0].value
    assert isinstance(v, IRInt) and v.value == 14

def test_fold_division_is_float():
    ir = optimize(_lower_unoptimized('let x = 7 / 2;'))
    v = ir.body[0].value
    assert isinstance(v, IRFloat) and v.value == 3.5

def test_fold_string_concat():
    ir = optimize(_lower_unoptimized('let s = "a" + "b" + "c";'))
    v = ir.body[0].value
    assert isinstance(v, IRString) and v.value == 'abc'

def test_no_fold_division_by_zero():
    
    ir = optimize(_lower_unoptimized('let x = 1 / 0;'))
    assert not isinstance(ir.body[0].value, (IRInt, IRFloat))

def test_fold_unary_negate():
    ir = optimize(_lower_unoptimized('let x = -5 + 2;'))
    v = ir.body[0].value
    assert isinstance(v, IRInt) and v.value == -3

def test_dce_after_return():
    ir = optimize(_lower_unoptimized('fn f() { return 1; let dead = 5; print(dead); }'))
    fn = ir.body[0]
    assert isinstance(fn, IRFunction)
    assert len(fn.body) == 1
    assert isinstance(fn.body[0], IRReturn)

def test_dce_if_false():
    ir = optimize(_lower_unoptimized('if false { print(1); } else { print(2); }'))
    
    assert len(ir.body) == 1
    node = ir.body[0]
    assert isinstance(node, IRIf)
    assert node.else_body is None

def test_dce_while_false_removed():
    ir = optimize(_lower_unoptimized('let x = 1; while false { x = 2; } print(x);'))
    assert not any(isinstance(s, IRWhile) for s in ir.body)

def test_lower_optimizes_by_default():
    ir = lower_module(parse('let x = 10 + 20;', '<t>'))
    v = ir.body[0].value
    assert isinstance(v, IRInt) and v.value == 30

if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print(f'  PASS {name}')
            except AssertionError as e:
                print(f'  FAIL {name}: {e}')
