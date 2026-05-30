import sys, os, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend.parser_universal import parse
from runtime.interpreter import Interpreter, TriadError

def run(code):
    old = sys.stdout
    sys.stdout = buf = io.StringIO()
    try:
        interp = Interpreter()
        mod = parse(code, '<test>')
        interp.run(mod)
        return buf.getvalue().strip()
    finally:
        sys.stdout = old

def test_hello():
    assert run('print("hello");') == 'hello'

def test_arithmetic():
    assert run('print(2 + 3);') == '5'
    assert run('print(10 - 3);') == '7'
    assert run('print(4 * 5);') == '20'
    assert run('print(10 / 2);') == '5.0'
    assert run('print(2 ** 3);') == '8'
    assert run('print(10 % 3);') == '1'

def test_variables():
    assert run('let x = 10; let y = 20; print(x + y);') == '30'

def test_string_concat():
    assert run('print("a" + "b");') == 'ab'

def test_function():
    code = 'fn add(a, b) { return a + b; } print(add(2, 3));'
    assert run(code) == '5'

def test_recursion():
    code = '\nfn fact(n) {\n    if n <= 1 { return 1; }\n    return n * fact(n - 1);\n}\nprint(fact(5));\n'
    assert run(code) == '120'

def test_for_loop():
    code = '\nlet s = 0;\nfor i in range(5) { s = s + i; }\nprint(s);\n'
    assert run(code) == '10'

def test_while_loop():
    code = '\nlet x = 0;\nwhile x < 5 { x = x + 1; }\nprint(x);\n'
    assert run(code) == '5'

def test_list():
    code = '\nlet xs = [1, 2, 3];\nxs.push(4);\nprint(len(xs));\nprint(xs[3]);\n'
    out = run(code)
    assert '4' in out

def test_map():
    code = '\nlet m = {"a": 1, "b": 2};\nprint(m["a"]);\n'
    assert run(code) == '1'

def test_type_decl():
    code = '\ntype P { name: String; age: Int; }\nlet p = P(name="Ana", age=30);\nprint(p.name);\n'
    assert run(code) == 'Ana'

def test_import_math():
    code = '\nimport math;\nprint(math.sqrt(25.0));\n'
    assert run(code) == '5.0'

def test_if_else():
    code = '\nlet x = 5;\nif x > 10 { print("big"); } else { print("small"); }\n'
    assert run(code) == 'small'

def test_boolean_ops():
    assert run('print(true and false);') == 'false'
    assert run('print(true or false);') == 'true'
    assert run('print(not true);') == 'false'

def test_comparison():
    assert run('print(3 > 2);') == 'true'
    assert run('print(3 < 2);') == 'false'
    assert run('print(3 == 3);') == 'true'
    assert run('print(3 != 4);') == 'true'

def test_string_methods():
    assert run('print("hello".upper());') == 'HELLO'
    assert run('print("HELLO".lower());') == 'hello'

def test_nested_function():
    code = '\nfn outer(x) {\n    fn inner(y) { return x + y; }\n    return inner(10);\n}\nprint(outer(5));\n'
    assert run(code) == '15'

def test_break():
    code = '\nlet s = 0;\nfor i in range(100) {\n    if i >= 5 { break; }\n    s = s + 1;\n}\nprint(s);\n'
    assert run(code) == '5'

def test_continue():
    code = '\nlet s = 0;\nfor i in range(10) {\n    if i % 2 == 0 { continue; }\n    s = s + 1;\n}\nprint(s);\n'
    assert run(code) == '5'

def test_undefined_variable():
    try:
        run('print(undefined_var);')
        assert False, 'should raise'
    except TriadError:
        pass

def test_class_basic():
    code = '\nclass Point {\n    x;\n    y;\n    fn describe(self) { return self.x + self.y; }\n}\nlet p = Point(1, 2);\nprint(p.describe());\n'
    assert run(code) == '3'

def test_class_field_access():
    code = '\nclass Point {\n    x;\n    y;\n}\nlet p = Point(10, 20);\nprint(p.x);\n'
    assert run(code) == '10'

def test_class_inheritance():
    code = '\nclass Animal {\n    name;\n    fn speak(self) { return self.name + " speaks"; }\n}\nclass Dog : Animal {\n    breed;\n    fn speak(self) { return self.name + " barks"; }\n}\nlet d = Dog("Rex", "Labrador");\nprint(d.speak());\n'
    assert run(code) == 'Rex barks'

def test_match_basic():
    code = '\nlet x = 2;\nmatch x {\n    case 1 => { print("one"); }\n    case 2 => { print("two"); }\n    else => { print("other"); }\n}\n'
    assert run(code) == 'two'

def test_match_else():
    code = '\nlet x = 99;\nmatch x {\n    case 1 => { print("one"); }\n    else => { print("other"); }\n}\n'
    assert run(code) == 'other'

def test_class_method_field():
    code = '\nclass Counter {\n    count = 0;\n    fn inc(self) { self.count = self.count + 1; }\n    fn get(self) { return self.count; }\n}\nlet c = Counter();\nc.inc();\nc.inc();\nprint(c.get());\n'
    assert run(code) == '2'

def test_match_wildcard():
    code = '\nlet x = 42;\nmatch x {\n    case _ => { print("caught"); }\n}\n'
    assert run(code) == 'caught'

def test_match_binding():
    code = '\nlet x = 7;\nmatch x {\n    case n => { print(n); }\n}\n'
    assert run(code) == '7'

def test_match_list_destructuring():
    code = '\nlet xs = [10, 20, 30];\nmatch xs {\n    case [a, b, c] => { print(a + b + c); }\n}\n'
    assert run(code) == '60'

def test_match_list_destructuring_partial():
    code = '\nlet xs = [1, 2];\nmatch xs {\n    case [a, b] => { print(a * b); }\n    case _ => { print("no match"); }\n}\n'
    assert run(code) == '2'

def test_match_nested_destructuring():
    code = '\nlet xs = [[1, 2], [3, 4]];\nmatch xs {\n    case [[a, b], [c, d]] => { print(a + b + c + d); }\n}\n'
    assert run(code) == '10'

def test_match_mixed():
    code = '\nlet xs = [1, 2];\nmatch xs {\n    case [0, 0] => { print("zeros"); }\n    case [a, b] => { print(a - b); }\n}\n'
    assert run(code) == '-1'

def test_generator_basic():
    code = '\nfn gen(n) {\n    for i in range(n) {\n        yield i;\n    }\n}\nlet g = gen(3);\nfor x in g {\n    print(x);\n}\n'
    assert run(code) == '0\n1\n2'

def test_generator_manual():
    code = '\nfn three() {\n    yield 10;\n    yield 20;\n    yield 30;\n}\nfor x in three() {\n    print(x);\n}\n'
    assert run(code) == '10\n20\n30'

def test_try_catch():
    code = '\ntry {\n    throw "oops";\n} catch e {\n    print(e);\n}\n'
    assert run(code) == 'oops'

def test_try_catch_finally():
    code = '\nlet x = 0;\ntry {\n    x = 1;\n} catch e {\n    x = 2;\n} finally {\n    x = x + 10;\n}\nprint(x);\n'
    assert run(code) == '11'

def test_async_fn():
    code = '\nasync fn greet(name) {\n    return "hello " + name;\n}\nlet result = greet("world");\nprint(result);\n'
    assert run(code) == 'hello world'

def test_map_destructuring():
    code = '\nlet m = {"x": 10, "y": 20, "z": 30};\nlet {x, z} = m;\nprint(x + z);\n'
    assert run(code) == '40'

def test_tuple_basic():
    code = '\nlet t = (1, 2, 3);\nprint(t[0] + t[1] + t[2]);\n'
    assert run(code) == '6'

def test_tuple_return():
    code = '\nfn swap(a, b) {\n    return (b, a);\n}\nlet result = swap(10, 20);\nprint(result[0]);\n'
    assert run(code) == '20'

def test_tuple_destructuring():
    code = '\nlet t = (100, 200);\nlet (a, b) = t;\nprint(a - b);\n'
    assert run(code) == '-100'

def test_yield_expr():
    code = '\nfn gen() {\n    yield 10;\n    yield 20;\n    yield 30;\n}\nlet g = gen();\nfor v in g {\n    print(v);\n}\n'
    assert run(code) == '10\n20\n30'
if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print(f'  PASS {name}')
            except Exception as e:
                print(f'  FAIL {name}: {e}')