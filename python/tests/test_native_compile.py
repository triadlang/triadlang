import os
import subprocess
import sys
import tempfile
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIAD = [sys.executable, '-m', 'cli.main']
RT_DIR = os.path.join(ROOT, 'native', 'c')

def _compile_native(src: str, extra_flags: list=None) -> str:
    with tempfile.NamedTemporaryFile(suffix='.tri', mode='w', delete=False) as f:
        f.write(src)
        tri_path = f.name
    bin_path = tri_path.replace('.tri', '')
    c_path = bin_path + '.c'
    try:
        r = subprocess.run(TRIAD + ['compile', tri_path, '--native', '-o', bin_path], capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, f'compile failed: {r.stderr}'
        assert os.path.exists(bin_path), 'binary not created'
        return bin_path
    finally:
        for p in [tri_path, c_path]:
            if os.path.exists(p):
                os.unlink(p)

def _run_native(bin_path: str, timeout=10) -> str:
    r = subprocess.run([bin_path], capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()

def _run_python(src: str) -> str:
    with tempfile.NamedTemporaryFile(suffix='.tri', mode='w', delete=False) as f:
        f.write(src)
        tri_path = f.name
    try:
        r = subprocess.run(TRIAD + ['run', tri_path], capture_output=True, text=True, cwd=ROOT)
        return r.stdout.strip()
    finally:
        os.unlink(tri_path)

class TestNativeBasic:

    def test_hello_world(self, tmp_path):
        bin_path = _compile_native('print("hello native")')
        try:
            out = _run_native(bin_path)
            assert out == 'hello native'
        finally:
            os.unlink(bin_path)

    def test_arithmetic(self, tmp_path):
        src = 'let x = 10 + 20 * 3\nprint(x)'
        bin_path = _compile_native(src)
        try:
            out = _run_native(bin_path)
            assert out == '70'
        finally:
            os.unlink(bin_path)

    def test_function_recursive(self, tmp_path):
        src = 'fn fib(n) {\n    if n <= 1 { return n }\n    return fib(n - 1) + fib(n - 2)\n}\nprint(fib(10))'
        bin_path = _compile_native(src)
        try:
            out = _run_native(bin_path)
            assert out == '55'
        finally:
            os.unlink(bin_path)

    def test_for_loop(self, tmp_path):
        src = 'let total = 0\nfor i in range(5) {\n    total = total + i\n}\nprint(total)'
        bin_path = _compile_native(src)
        try:
            out = _run_native(bin_path)
            assert out == '10'
        finally:
            os.unlink(bin_path)

    def test_list_dict(self, tmp_path):
        src = 'let nums = [10, 20, 30]\nprint(nums[1])\nlet m = {"key": 42}\nprint(m["key"])'
        bin_path = _compile_native(src)
        try:
            out = _run_native(bin_path)
            assert '20' in out
            assert '42' in out
        finally:
            os.unlink(bin_path)

    def test_fstring(self, tmp_path):
        src = 'let name = "TriadLang"\nlet v = 2\nprint(f"{name} v{v}")'
        bin_path = _compile_native(src)
        try:
            out = _run_native(bin_path)
            assert out == 'TriadLang v2'
        finally:
            os.unlink(bin_path)

    def test_while(self, tmp_path):
        src = 'let x = 1\nlet i = 0\nwhile i < 5 {\n    x = x * 2\n    i = i + 1\n}\nprint(x)'
        bin_path = _compile_native(src)
        try:
            out = _run_native(bin_path)
            assert out == '32'
        finally:
            os.unlink(bin_path)

    def test_if_elif_else(self, tmp_path):
        src = 'fn classify(x) {\n    if x < 0 { return "neg" }\n    elif x == 0 { return "zero" }\n    else { return "pos" }\n}\nprint(classify(-1))\nprint(classify(0))\nprint(classify(5))'
        bin_path = _compile_native(src)
        try:
            out = _run_native(bin_path)
            lines = out.split('\n')
            assert lines == ['neg', 'zero', 'pos']
        finally:
            os.unlink(bin_path)

    def test_try_catch(self, tmp_path):
        src = 'fn safe(a, b) {\n    try {\n        if b == 0 { throw "zero" }\n        return a / b\n    } catch e {\n        return -1\n    }\n}\nprint(safe(10, 2))\nprint(safe(10, 0))'
        bin_path = _compile_native(src)
        try:
            out = _run_native(bin_path)
            lines = out.split('\n')
            assert lines[0] == '5'
            assert lines[1] == '-1'
        finally:
            os.unlink(bin_path)

    def test_string_methods(self, tmp_path):
        src = 'let s = "hello"\nprint(s.upper())\nprint(s.upper().lower())'
        bin_path = _compile_native(src)
        try:
            out = _run_native(bin_path)
            lines = out.split('\n')
            assert lines[0] == 'HELLO'
            assert lines[1] == 'hello'
        finally:
            os.unlink(bin_path)

    def test_python_native_parity(self, tmp_path):
        src = 'fn fib(n) {\n    if n <= 1 { return n }\n    return fib(n - 1) + fib(n - 2)\n}\nlet results = []\nfor i in range(10) {\n    results.push(fib(i))\n}\nprint(results)\nlet total = 0\nfor r in results {\n    total = total + r\n}\nprint(total)'
        bin_path = _compile_native(src)
        try:
            native_out = _run_native(bin_path)
            python_out = _run_python(src)
            assert native_out == python_out
        finally:
            os.unlink(bin_path)

class TestNativeClosures:

    def test_closure_capture_param(self, tmp_path):
        src = 'fn make_adder(n) {\n    fn add(a) {\n        return a + n\n    }\n    return add\n}\nlet f = make_adder(5)\nprint(f(3))'
        bin_path = _compile_native(src)
        try:
            out = _run_native(bin_path)
            assert out == '8'
        finally:
            os.unlink(bin_path)

    def test_closure_no_capture(self, tmp_path):
        src = 'fn make_id() {\n    fn id(x) {\n        return x\n    }\n    return id\n}\nlet f = make_id()\nprint(f(42))'
        bin_path = _compile_native(src)
        try:
            out = _run_native(bin_path)
            assert out == '42'
        finally:
            os.unlink(bin_path)

    def test_closure_top_level_var(self, tmp_path):
        src = 'let factor = 3\nfn make_scaler() {\n    fn scale(x) {\n        return x * factor\n    }\n    return scale\n}\nlet s = make_scaler()\nprint(s(7))'
        bin_path = _compile_native(src)
        try:
            out = _run_native(bin_path)
            assert out == '21'
        finally:
            os.unlink(bin_path)

    def test_closure_multiple_captures(self, tmp_path):
        src = 'fn make_combo(a, b) {\n    fn combo(x) {\n        return x * a + b\n    }\n    return combo\n}\nlet c = make_combo(3, 1)\nprint(c(4))'
        bin_path = _compile_native(src)
        try:
            out = _run_native(bin_path)
            assert out == '13'
        finally:
            os.unlink(bin_path)

    def test_closure_multiple_calls(self, tmp_path):
        src = 'fn make_adder(n) {\n    fn add(a) {\n        return a + n\n    }\n    return add\n}\nlet add5 = make_adder(5)\nlet add10 = make_adder(10)\nprint(add5(1))\nprint(add10(1))\nprint(add5(100))'
        bin_path = _compile_native(src)
        try:
            out = _run_native(bin_path)
            lines = out.split('\n')
            assert lines == ['6', '11', '105']
        finally:
            os.unlink(bin_path)

class TestNativeMatch:

    def test_match_int(self, tmp_path):
        src = 'let x = 2\nmatch x {\n    case 1 { print("one") }\n    case 2 { print("two") }\n    case _ { print("other") }\n}'
        bin_path = _compile_native(src)
        try:
            out = _run_native(bin_path)
            assert out == 'two'
        finally:
            os.unlink(bin_path)

    def test_match_in_function(self, tmp_path):
        src = 'fn classify(n) {\n    match n {\n        case 0 { return "zero" }\n        case 1 { return "one" }\n        case _ { return "many" }\n    }\n}\nprint(classify(0))\nprint(classify(1))\nprint(classify(99))'
        bin_path = _compile_native(src)
        try:
            out = _run_native(bin_path)
            assert out.split('\n') == ['zero', 'one', 'many']
        finally:
            os.unlink(bin_path)

    def test_match_wildcard_fallback(self, tmp_path):
        src = 'let v = 7\nmatch v {\n    case 1 { print("a") }\n    case 2 { print("b") }\n    case _ { print("c") }\n}'
        bin_path = _compile_native(src)
        try:
            out = _run_native(bin_path)
            assert out == 'c'
        finally:
            os.unlink(bin_path)

class TestNativeClass:

    def test_class_constructor_and_method(self, tmp_path):
        src = 'class Point {\n    x\n    y\n    fn new(x, y) {\n        self.x = x\n        self.y = y\n    }\n    fn dist() {\n        return sqrt(self.x * self.x + self.y * self.y)\n    }\n}\nlet p = Point(3, 4)\nprint(p.dist())'
        bin_path = _compile_native(src)
        try:
            out = _run_native(bin_path)
            assert out == '5'
        finally:
            os.unlink(bin_path)

    def test_class_mutable_state(self, tmp_path):
        src = 'class Counter {\n    count\n    fn new() {\n        self.count = 0\n    }\n    fn inc() {\n        self.count = self.count + 1\n    }\n    fn get() {\n        return self.count\n    }\n}\nlet c = Counter()\nprint(c.get())\nc.inc()\nc.inc()\nc.inc()\nprint(c.get())'
        bin_path = _compile_native(src)
        try:
            out = _run_native(bin_path)
            assert out.split('\n') == ['0', '3']
        finally:
            os.unlink(bin_path)

    def test_class_field_access(self, tmp_path):
        src = 'class Pair {\n    a\n    b\n    fn new(a, b) {\n        self.a = a\n        self.b = b\n    }\n}\nlet p = Pair(10, 20)\nprint(p.a)\nprint(p.b)'
        bin_path = _compile_native(src)
        try:
            out = _run_native(bin_path)
            assert out.split('\n') == ['10', '20']
        finally:
            os.unlink(bin_path)

class TestNativeGenerators:

    def test_generator_basic(self, tmp_path):
        src = 'fn count(n) {\n    let i = 0\n    while i < n {\n        yield i\n        i = i + 1\n    }\n}\nlet results = count(4)\nprint(results)'
        bin_path = _compile_native(src)
        try:
            out = _run_native(bin_path)
            assert out == '[0, 1, 2, 3]'
        finally:
            os.unlink(bin_path)

    def test_generator_for_in(self, tmp_path):
        src = 'fn squares(n) {\n    let i = 0\n    while i < n {\n        yield i * i\n        i = i + 1\n    }\n}\nfor x in squares(4) {\n    print(x)\n}'
        bin_path = _compile_native(src)
        try:
            out = _run_native(bin_path)
            assert out.split('\n') == ['0', '1', '4', '9']
        finally:
            os.unlink(bin_path)

    def test_generator_fibonacci(self, tmp_path):
        src = 'fn fib_gen(n) {\n    let a = 0\n    let b = 1\n    let count = 0\n    while count < n {\n        yield a\n        let tmp = b\n        b = a + b\n        a = tmp\n        count = count + 1\n    }\n}\nlet results = fib_gen(6)\nprint(results)'
        bin_path = _compile_native(src)
        try:
            out = _run_native(bin_path)
            assert out == '[0, 1, 1, 2, 3, 5]'
        finally:
            os.unlink(bin_path)

class TestNativeSolver:

    def test_solver_basic(self, tmp_path):
        src = 'let r = solver_solve({"N": 64, "T": 1.0, "seed": 42})\nprint(f"norm={r["norm"]}")\nprint(f"peak={r["peak"]}")'
        bin_path = _compile_native(src)
        try:
            out = _run_native(bin_path, timeout=30)
            assert 'norm=' in out
            assert 'peak=' in out
            norm_val = float(out.split('norm=')[1].split('\n')[0])
            assert 0.5 < norm_val < 3.0
        finally:
            os.unlink(bin_path)

    def test_solver_returns_density(self, tmp_path):
        src = 'let r = solver_solve({"N": 64, "T": 0.5, "seed": 99})\nlet d = r["density"]\nprint(f"density_len={len(d)}")\nprint(f"x_len={len(r["x"])}")'
        bin_path = _compile_native(src)
        try:
            out = _run_native(bin_path, timeout=30)
            assert 'density_len=64' in out
            assert 'x_len=64' in out
        finally:
            os.unlink(bin_path)

    def test_solver_full_config(self, tmp_path):
        src = 'let r = solver_solve({\n    "N": 64, "L": 16.0, "dt": 0.01, "T": 1.0,\n    "Lambda": -0.3, "alpha": 0.1, "sigma": 1.0,\n    "Gamma": 0.03, "f_FDT": 0.001, "mode": 2, "seed": 7\n})\nprint(f"N={r["N"]}")\nprint(f"dx={r["dx"]}")'
        bin_path = _compile_native(src)
        try:
            out = _run_native(bin_path, timeout=30)
            assert 'N=64' in out
            dx_val = float(out.split('dx=')[1].strip())
            assert abs(dx_val - 16.0 / 64) < 0.01
        finally:
            os.unlink(bin_path)

class TestNativeML:

    def test_ml_tensor_create(self):
        src = '\nlet t = ml_tensor([1.0, 2.0, 3.0], false)\nlet v = ml_item(ml_tensor_sum(t))\nprint(v)\n'
        bp = _compile_native(src)
        try:
            out = _run_native(bp)
            assert abs(float(out) - 6.0) < 0.001
        finally:
            os.unlink(bp)

    def test_ml_tensor_2d(self):
        src = '\nlet t = ml_tensor([[1.0, 2.0], [3.0, 4.0]], false)\nlet v = ml_item(ml_tensor_sum(t))\nprint(v)\n'
        bp = _compile_native(src)
        try:
            out = _run_native(bp)
            assert abs(float(out) - 10.0) < 0.001
        finally:
            os.unlink(bp)

    def test_ml_linear_forward(self):
        src = '\nlet lin = ml_linear(3, 2)\nlet x = ml_tensor([1.0, 1.0, 1.0], false)\nlet y = ml_forward(lin, x)\nml_print(y)\n'
        bp = _compile_native(src)
        try:
            out = _run_native(bp)
            assert '[' in out or '.' in out
        finally:
            os.unlink(bp)

    def test_ml_relu_sigmoid(self):
        src = '\nlet x = ml_tensor([-1.0, 0.0, 1.0], false)\nlet r = ml_relu(x)\nlet s = ml_sigmoid(x)\nml_print(r)\nml_print(s)\n'
        bp = _compile_native(src)
        try:
            out = _run_native(bp)
            lines = out.strip().split('\n')
            assert '0.0000' in lines[0]
            assert '0.5000' in lines[1]
        finally:
            os.unlink(bp)

    def test_ml_mse_loss(self):
        src = '\nlet pred = ml_tensor([1.0, 2.0, 3.0], true)\nlet tgt = ml_tensor([1.0, 2.0, 3.0], false)\nlet loss = ml_mse_loss(pred, tgt)\nprint(ml_item(loss))\n'
        bp = _compile_native(src)
        try:
            out = _run_native(bp)
            assert abs(float(out)) < 0.001
        finally:
            os.unlink(bp)

    def test_ml_backward(self):
        src = '\nlet a = ml_tensor([1.0, 2.0, 3.0], true)\nlet b = ml_tensor([4.0, 5.0, 6.0], true)\nlet c = ml_tensor_add(a, b)\nlet loss = ml_tensor_sum(c)\nml_backward(loss)\nprint("ok")\n'
        bp = _compile_native(src)
        try:
            out = _run_native(bp)
            assert 'ok' in out
        finally:
            os.unlink(bp)

    def test_ml_xor_training(self):
        src = '\nlet model = ml_seq_new(4)\nlet l1 = ml_linear(2, 8)\nlet l2 = ml_linear(8, 1)\nml_seq_set(model, 0, 0, l1)\nml_seq_set(model, 1, 1, l1)\nml_seq_set(model, 2, 0, l2)\nml_seq_set(model, 3, 2, l2)\nlet opt = ml_adam(model, 0.01)\nlet x0 = ml_tensor([0.0, 0.0], false)\nlet x1 = ml_tensor([0.0, 1.0], false)\nlet x2 = ml_tensor([1.0, 0.0], false)\nlet x3 = ml_tensor([1.0, 1.0], false)\nlet y0 = ml_tensor([0.0], false)\nlet y1 = ml_tensor([1.0], false)\nlet y2 = ml_tensor([1.0], false)\nlet y3 = ml_tensor([0.0], false)\nlet inputs = [x0, x1, x2, x3]\nlet targets = [y0, y1, y2, y3]\nfor epoch in range(1500) {\n    for s in range(4) {\n        ml_adam_zero(opt)\n        let pred = ml_seq_forward(model, inputs[s])\n        let loss = ml_mse_loss(pred, targets[s])\n        ml_backward(loss)\n        ml_adam_step(opt)\n    }\n}\nlet correct = 0\nfor s in range(4) {\n    let pred = ml_seq_forward(model, inputs[s])\n    let p = ml_item(pred)\n    let label = 0\n    if p > 0.5 { label = 1 }\n    let tgts = [0, 1, 1, 0]\n    if label == tgts[s] { correct = correct + 1 }\n}\nprint(correct)\n'
        bp = _compile_native(src)
        try:
            out = _run_native(bp, timeout=30)
            assert int(out.strip()) == 4, f'XOR accuracy {out}/4'
        finally:
            os.unlink(bp)