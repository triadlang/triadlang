import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend.lexer_universal import tokenize, LexError

def test_basic_tokens():
    toks = tokenize('let x = 10;')
    kinds = [t.kind for t in toks if t.kind != 'EOF']
    assert kinds == ['KEYWORD', 'IDENT', 'SYMBOL', 'NUMBER', 'SYMBOL']

def test_string():
    toks = tokenize('"hello world"')
    assert toks[0].kind == 'STRING'
    assert toks[0].value == 'hello world'

def test_operators():
    toks = tokenize('a + b * c == d')
    vals = [t.value for t in toks if t.kind != 'EOF']
    assert vals == ['a', '+', 'b', '*', 'c', '==', 'd']

def test_comments():
    toks = tokenize('x // comment\ny')
    idents = [t.value for t in toks if t.kind == 'IDENT']
    assert idents == ['x', 'y']

def test_block_comment():
    toks = tokenize('a /* block */ b')
    idents = [t.value for t in toks if t.kind == 'IDENT']
    assert idents == ['a', 'b']

def test_keywords():
    toks = tokenize('let const fn if else for while return')
    kws = [t.value for t in toks if t.kind == 'KEYWORD']
    assert 'let' in kws
    assert 'fn' in kws
    assert 'return' in kws

def test_float():
    toks = tokenize('3.14')
    assert toks[0].kind == 'NUMBER'
    assert '.' in toks[0].value

def test_hex():
    toks = tokenize('0xFF')
    assert toks[0].kind == 'NUMBER'

def test_escape_string():
    toks = tokenize('"hello\\nworld"')
    assert toks[0].value == 'hello\nworld'

def test_multichar_symbols():
    toks = tokenize('== != <= >= -> **')
    vals = [t.value for t in toks if t.kind == 'SYMBOL']
    assert vals == ['==', '!=', '<=', '>=', '->', '**']

def test_error_unterminated_string():
    try:
        tokenize('"unterminated')
        assert False, 'should raise'
    except LexError:
        pass
if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print(f'  PASS {name}')
            except Exception as e:
                print(f'  FAIL {name}: {e}')