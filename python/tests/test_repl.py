import sys, os, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend.parser_universal import parse
from runtime.interpreter import Interpreter, Environment

def repl_eval(code):
    old = sys.stdout
    sys.stdout = buf = io.StringIO()
    try:
        interp = Interpreter()
        env = Environment(interp.globals)
        mod = parse(code, '<repl>')
        result = interp.run_repl_line(mod, env)
        output = buf.getvalue().strip()
        return (result, output)
    finally:
        sys.stdout = old

def test_expr():
    r, _ = repl_eval('2 + 3')
    assert r == 5

def test_let():
    interp = Interpreter()
    env = Environment(interp.globals)
    mod = parse('let x = 10;', '<repl>')
    interp.run_repl_line(mod, env)
    mod2 = parse('x + 5', '<repl>')
    r = interp.run_repl_line(mod2, env)
    assert r == 15

def test_fn():
    interp = Interpreter()
    env = Environment(interp.globals)
    mod = parse('fn add(a, b) { return a + b; }', '<repl>')
    interp.run_repl_line(mod, env)
    mod2 = parse('add(2, 3)', '<repl>')
    r = interp.run_repl_line(mod2, env)
    assert r == 5

def test_print():
    _, out = repl_eval('print("hello");')
    assert out == 'hello'
if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print(f'  PASS {name}')
            except Exception as e:
                print(f'  FAIL {name}: {e}')