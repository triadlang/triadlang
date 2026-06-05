"""
interop test suite for triadlang <-> python frameworks.
covers: embedding, import bridge, register/get, engine delegation, adapters, discovery.
P1+P2+P3 always together in solver calls.
"""
import sys, os, math, unittest, tempfile

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SRC)
os.chdir(ROOT)

EXAMPLES = os.path.join(ROOT, 'examples', 'interop')

class TestTriadEngine(unittest.TestCase):

    def test_run_source_hello(self):
        from embed.engine import TriadEngine
        engine = TriadEngine()
        ns = engine.run_source('let x = 42;')
        self.assertEqual(ns['x'], 42)

    def test_run_file(self):
        from embed.engine import TriadEngine
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tri', delete=False) as f:
            f.write('let val = 99;\n')
            tmp = f.name
        try:
            engine = TriadEngine()
            ns = engine.run_file(tmp)
            self.assertEqual(ns['val'], 99)
        finally:
            os.unlink(tmp)

    def test_solver_p1p2p3(self):
        from embed.engine import TriadEngine
        engine = TriadEngine()
        ns = engine.run_source('''
import triad
let p = triad.regime("R5_crystal", seed=42, N=64, T=2.0, dt=0.01);
let result = triad.solve(p);
let cryst = result.crystallinity;
let pk = result.peak;
let nm = result.norm;
''')
        cryst = ns['cryst']
        pk = ns['pk']
        nm = ns['nm']
        self.assertTrue(math.isfinite(cryst))
        self.assertTrue(math.isfinite(pk))
        self.assertTrue(math.isfinite(nm))
        self.assertGreaterEqual(cryst, 0.0)

class TestRegisterGet(unittest.TestCase):

    def test_register_python_object(self):
        from embed.engine import TriadEngine
        engine = TriadEngine()
        engine.register('factor', 6)
        ns = engine.run_source('let out = factor * 7;')
        self.assertEqual(ns['out'], 42)

    def test_register_python_function(self):
        from embed.engine import TriadEngine
        engine = TriadEngine()
        engine.register('double', lambda x: x * 2)
        ns = engine.run_source('let out = double(21);')
        self.assertEqual(ns['out'], 42)

    def test_get_callback(self):
        from embed.engine import TriadEngine
        engine = TriadEngine()
        engine.run_source('fn add(a, b) { return a + b; }')
        add = engine.get('add')
        self.assertEqual(add(3, 4), 7)

class TestImportBridge(unittest.TestCase):

    def test_import_numpy(self):
        from embed.engine import TriadEngine
        engine = TriadEngine()
        ns = engine.run_source('''
import numpy
let val = numpy.sum(numpy.array([1, 2, 3]));
''')
        self.assertEqual(ns['val'], 6)

    def test_from_import_basic(self):
        from embed.engine import TriadEngine
        engine = TriadEngine()
        ns = engine.run_source('''
from numpy import array, sum
let val = sum(array([4, 5, 6]));
''')
        self.assertEqual(ns['val'], 15)

    def test_from_import_as(self):
        from embed.engine import TriadEngine
        engine = TriadEngine()
        ns = engine.run_source('''
from numpy import sum as npsum
let val = npsum([10, 20, 30]);
''')
        self.assertEqual(ns['val'], 60)

    def test_dotted_import(self):
        from embed.engine import TriadEngine
        engine = TriadEngine()
        ns = engine.run_source('''
import os.path
let val = os.path.exists(".");
''')
        self.assertTrue(ns['val'])

    def test_import_flask(self):
        from embed.engine import TriadEngine
        engine = TriadEngine()
        ns = engine.run_source('''
import flask
let app = flask.Flask("test");
let name = app.name;
''')
        self.assertEqual(ns['name'], 'test')

    def test_import_error_detail(self):
        from embed.engine import TriadEngine
        engine = TriadEngine()
        with self.assertRaises(ImportError) as ctx:
            engine.run_source('import nonexistent_mod_xyz;')
        msg = str(ctx.exception)
        self.assertIn('tried:', msg)

    def test_prefix_policy_blocks(self):
        from embed.engine import TriadEngine
        from runtime.compiler_runtime import set_allowed_import_prefixes
        set_allowed_import_prefixes(['numpy'])
        try:
            engine = TriadEngine()
            with self.assertRaises(ImportError) as ctx:
                engine.run_source('import flask;')
            self.assertIn('blocked by allowed-prefix', str(ctx.exception))
        finally:
            set_allowed_import_prefixes(None)

    def test_prefix_policy_allows(self):
        from embed.engine import TriadEngine
        from runtime.compiler_runtime import set_allowed_import_prefixes
        set_allowed_import_prefixes(['numpy'])
        try:
            engine = TriadEngine()
            ns = engine.run_source('import numpy; let v = 1;')
            self.assertEqual(ns['v'], 1)
        finally:
            set_allowed_import_prefixes(None)

class TestKernel(unittest.TestCase):

    def test_kernel_run_extract(self):
        from embed.engine import TriadEngine
        engine = TriadEngine()
        ns = engine.run_source('''
import triad.kernel as kernel
let params = {"regime": "R5_crystal", "seed": 42, "N": 64, "T": 2.0, "dt": 0.01};
let result = kernel.run(params);
let obs = kernel.extract(result);
let cryst = obs["crystallinity"];
let pk = obs["peak"];
let nm = obs["norm"];
let ks = obs["k_star"];
''')
        cryst = ns['cryst']
        self.assertTrue(math.isfinite(cryst))
        self.assertTrue(math.isfinite(ns['pk']))
        self.assertTrue(math.isfinite(ns['nm']))
        self.assertTrue(math.isfinite(ns['ks']))

    def test_kernel_custom_observables(self):
        from embed.engine import TriadEngine
        engine = TriadEngine()
        ns = engine.run_source('''
import triad.kernel as kernel
let params = {"regime": "R5_crystal", "seed": 42, "N": 64, "T": 2.0};
let result = kernel.run(params);
let obs = kernel.extract(result, ["crystallinity", "ipr", "fwhm"]);
''')
        obs = ns['obs']
        self.assertIn('crystallinity', obs)
        self.assertIn('ipr', obs)
        self.assertIn('fwhm', obs)

class TestHeadlessAdapter(unittest.TestCase):

    def test_headless_adapter(self):
        from adapters.headless import HeadlessAdapter
        adapter = HeadlessAdapter(tri_file=os.path.join(EXAMPLES, 'headless_solver.tri'))
        adapter.start()
        cryst = adapter.get('crystallinity')
        self.assertTrue(math.isfinite(cryst))
        self.assertGreaterEqual(cryst, 0.0)

class TestFlaskAdapter(unittest.TestCase):

    def test_flask_health(self):
        from adapters.flask import FlaskAdapter
        adapter = FlaskAdapter(tri_file=os.path.join(EXAMPLES, 'flask_solver.tri'))
        client = adapter.test_client()
        resp = client.get('/health')
        data = resp.get_json()
        self.assertEqual(data['status'], 'ok')

    def test_flask_simulate(self):
        from adapters.flask import FlaskAdapter
        adapter = FlaskAdapter(tri_file=os.path.join(EXAMPLES, 'flask_solver.tri'))
        client = adapter.test_client()
        resp = client.get('/simulate')
        data = resp.get_json()
        self.assertTrue(math.isfinite(data['crystallinity']))
        self.assertTrue(math.isfinite(data['peak']))
        self.assertTrue(math.isfinite(data['norm']))
        self.assertTrue(math.isfinite(data['k_star']))
        self.assertGreaterEqual(data['crystallinity'], 0.0)

class TestDiscovery(unittest.TestCase):

    def test_detect_flask(self):
        from adapters.discovery import detect_framework
        frameworks = detect_framework(os.path.join(EXAMPLES, 'flask_solver.tri'))
        self.assertIn('flask', frameworks)

    def test_detect_headless(self):
        from adapters.discovery import detect_framework
        frameworks = detect_framework(os.path.join(EXAMPLES, 'headless_solver.tri'))
        self.assertNotIn('flask', frameworks)

    def test_auto_load_flask(self):
        from adapters.discovery import auto_load
        from adapters.flask import FlaskAdapter
        adapter = auto_load(os.path.join(EXAMPLES, 'flask_solver.tri'))
        self.assertIsInstance(adapter, FlaskAdapter)

    def test_auto_load_headless(self):
        from adapters.discovery import auto_load
        from adapters.headless import HeadlessAdapter
        adapter = auto_load(os.path.join(EXAMPLES, 'headless_solver.tri'))
        self.assertIsInstance(adapter, HeadlessAdapter)

    def test_detect_unknown_framework(self):
        from adapters.discovery import detect_framework
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tri', delete=False) as f:
            f.write('import unknownframework;\n')
            tmp = f.name
        try:
            frameworks = detect_framework(tmp)
            self.assertEqual(frameworks, [])
        finally:
            os.unlink(tmp)

class TestDecorators(unittest.TestCase):

    def test_flask_route_decorator(self):
        from embed.engine import TriadEngine
        engine = TriadEngine()
        ns = engine.run_source('''
import flask;
let app = flask.Flask("dec_test");

@app.route("/hello")
fn hello() { return "hello decorated"; }

let client = app.test_client();
let resp = client.get("/hello");
let body = resp.data.decode("utf-8");
''')
        self.assertEqual(ns['body'], 'hello decorated')
        self.assertEqual(ns['resp'].status_code, 200)

    def test_flask_multiple_decorator_routes(self):
        from embed.engine import TriadEngine
        engine = TriadEngine()
        ns = engine.run_source('''
import flask;
let app = flask.Flask("multi_dec");

@app.route("/a")
fn route_a() { return "a"; }

@app.route("/b")
fn route_b() { return "b"; }

let client = app.test_client();
let ra = client.get("/a");
let rb = client.get("/b");
''')
        self.assertEqual(ns['ra'].status_code, 200)
        self.assertEqual(ns['rb'].status_code, 200)

    def test_simple_name_decorator(self):
        from embed.engine import TriadEngine
        engine = TriadEngine()
        ns = engine.run_source('''
fn wrap(original) {
    fn callme() {
        let inner = original();
        return inner + " wrapped";
    }
    return callme;
}

@wrap
fn base() { return "base"; }

let result = base();
''')
        self.assertEqual(ns['result'], 'base wrapped')

    def test_triad_annotations_unaffected(self):
        from embed.engine import TriadEngine
        engine = TriadEngine()
        ns = engine.run_source('''
@par
fn double(x) { return x * 2; }
let val = double(21);
''')
        self.assertEqual(ns['val'], 42)

if __name__ == '__main__':
    unittest.main(verbosity=2)
