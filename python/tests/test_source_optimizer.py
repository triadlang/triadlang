import pytest
from compiler.source_optimizer import optimize_source, _fold_constants, _dead_code_elimination

class TestConstantFolding:

    def test_int_add(self):
        src = 'x = 3 + 4'
        assert optimize_source(src).strip() == 'x = 7'

    def test_int_sub(self):
        src = 'x = 10 - 3'
        assert optimize_source(src).strip() == 'x = 7'

    def test_int_mul(self):
        src = 'y = 5 * 6'
        assert optimize_source(src).strip() == 'y = 30'

    def test_int_div(self):
        src = 'z = 15 / 3'
        assert optimize_source(src).strip() == 'z = 5.0'

    def test_int_div_non_integral(self):
        src = 'z = 7 / 2'
        assert optimize_source(src).strip() == 'z = 3.5'

    def test_int_mod(self):
        src = 'r = 10 % 3'
        assert optimize_source(src).strip() == 'r = 1'

    def test_int_pow(self):
        src = 'p = 2 ** 10'
        assert optimize_source(src).strip() == 'p = 1024'

    def test_float_add(self):
        src = 'f = 1.5 + 2.5'
        assert optimize_source(src).strip() == 'f = 4.0'

    def test_float_mul(self):
        src = 'f = 2.0 * 3.5'
        assert optimize_source(src).strip() == 'f = 7.0'

    def test_negative_operand(self):
        src = 'x = -3 + 7'
        assert optimize_source(src).strip() == 'x = 4'

    def test_no_fold_variables(self):
        src = 'x = a + b'
        assert optimize_source(src).strip() == 'x = a + b'

    def test_no_fold_mixed(self):
        src = 'x = 3 + y'
        assert optimize_source(src).strip() == 'x = 3 + y'

    def test_no_fold_function_call(self):
        src = 'x = foo(3) + 4'
        assert optimize_source(src).strip() == 'x = foo(3) + 4'

    def test_preserves_indent(self):
        lines = ['    x = 3 + 4']
        result = _fold_constants(lines)
        assert result == ['    x = 7']

    def test_preserves_defs(self):
        src = 'def foo():\n    x = 3 + 4'
        lines = src.split('\n')
        result = _fold_constants(lines)
        assert result[0] == 'def foo():'
        assert result[1] == '    x = 7'

    def test_div_by_zero_not_folded(self):
        src = 'x = 5 / 0'
        assert optimize_source(src).strip() == 'x = 5 / 0'

    def test_mod_by_zero_not_folded(self):
        src = 'x = 5 % 0'
        assert optimize_source(src).strip() == 'x = 5 % 0'

    def test_huge_power_not_folded(self):
        src = 'x = 2 ** 200'
        assert optimize_source(src).strip() == 'x = 2 ** 200'

    def test_multiple_lines(self):
        src = 'a = 1 + 2\nb = 3 * 4\nc = a + b'
        result = optimize_source(src)
        assert 'a = 3' in result
        assert 'b = 12' in result
        assert 'c = a + b' in result

class TestDeadCodeElimination:

    def test_remove_after_return(self):
        src = 'def foo():\n    return 1\n    x = 2\n    y = 3'
        result = optimize_source(src)
        assert 'x = 2' not in result
        assert 'y = 3' not in result

    def test_remove_after_break(self):
        src = 'while True:\n    break\n    x = 1'
        result = optimize_source(src)
        assert 'x = 1' not in result

    def test_remove_after_continue(self):
        src = 'for i in range(10):\n    continue\n    x = 1'
        result = optimize_source(src)
        assert 'x = 1' not in result

    def test_remove_after_raise(self):
        src = "try:\n    raise Exception('x')\n    z = 1"
        result = optimize_source(src)
        assert 'z = 1' not in result

    def test_remove_if_false_block(self):
        src = 'if False:\n    x = 1\n    y = 2\nz = 3'
        result = optimize_source(src)
        assert 'x = 1' not in result
        assert 'y = 2' not in result
        assert 'z = 3' in result

    def test_unwrap_if_true(self):
        src = 'if True:\n    x = 1\ny = 2'
        result = optimize_source(src)
        assert 'x = 1' in result
        assert 'if True' not in result

    def test_if_true_skip_elif(self):
        src = 'if True:\n    x = 1\nelif False:\n    x = 2\ny = 3'
        result = optimize_source(src)
        assert 'x = 1' in result
        assert 'elif False' not in result
        assert 'x = 2' not in result
        assert 'y = 3' in result

    def test_if_true_skip_else(self):
        src = 'if True:\n    x = 1\nelse:\n    x = 99\ny = 2'
        result = optimize_source(src)
        assert 'x = 1' in result
        assert 'else:' not in result
        assert '99' not in result

    def test_preserves_normal_code(self):
        src = 'x = 1\ny = x + 2\nprint(y)'
        result = optimize_source(src)
        assert result.strip() == src.strip()

    def test_preserves_nested_blocks(self):
        src = 'def foo():\n    if x > 0:\n        return x\n    return 0'
        result = optimize_source(src)
        assert 'def foo():' in result
        assert 'if x > 0:' in result

class TestIntegration:

    def test_fold_then_dce(self):
        src = 'if True:\n    x = 3 + 4\n    y = x * 2\nz = 10'
        result = optimize_source(src)
        assert 'if True' not in result
        assert 'x = 7' in result
        assert 'z = 10' in result

    def test_level_1_only_folding(self):
        src = 'if False:\n    x = 3 + 4\ny = 1 + 2'
        result = optimize_source(src, level=1)
        assert 'if False:' in result
        assert 'x = 7' in result
        assert 'y = 3' in result

    def test_level_2_folding_and_dce(self):
        src = 'if False:\n    x = 3 + 4\ny = 1 + 2'
        result = optimize_source(src, level=2)
        assert 'if False:' not in result
        assert 'x = 7' not in result
        assert 'y = 3' in result