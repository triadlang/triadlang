import sys, os, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIAD = os.path.join(ROOT, 'triad')

def _run(tri_file):
    path = os.path.join(ROOT, tri_file)
    r = subprocess.run([sys.executable, TRIAD, 'run', path], capture_output=True, text=True, timeout=30)
    return (r.returncode, r.stdout, r.stderr)

def test_hello():
    rc, out, err = _run('examples/basic/hello.tri')
    assert rc == 0, err
    assert 'Hello from TriadLang' in out

def test_variables():
    rc, out, err = _run('examples/basic/variables.tri')
    assert rc == 0, err
    assert '30' in out

def test_functions():
    rc, out, err = _run('examples/basic/functions.tri')
    assert rc == 0, err
    assert '5' in out
    assert '120' in out

def test_loops():
    rc, out, err = _run('examples/basic/loops.tri')
    assert rc == 0, err

def test_types():
    rc, out, err = _run('examples/basic/types.tri')
    assert rc == 0, err
    assert 'Ana' in out
    assert '5.0' in out

def test_modules():
    rc, out, err = _run('examples/basic/modules.tri')
    assert rc == 0, err
    assert '5.0' in out

def test_files():
    rc, out, err = _run('examples/basic/files.tri')
    assert rc == 0, err
    assert 'Hello from TriadLang file I/O' in out

def test_json():
    rc, out, err = _run('examples/basic/json.tri')
    assert rc == 0, err
    assert 'TriadLang' in out

def test_check():
    path = os.path.join(ROOT, 'examples/basic/hello.tri')
    r = subprocess.run([sys.executable, TRIAD, 'check', path], capture_output=True, text=True, timeout=10)
    assert r.returncode == 0
    assert 'OK' in r.stdout

def test_compile_emit_ir():
    path = os.path.join(ROOT, 'examples/basic/hello.tri')
    r = subprocess.run([sys.executable, TRIAD, 'compile', path, '--emit-ir'], capture_output=True, text=True, timeout=10)
    assert r.returncode == 0
    assert 'IRModule' in r.stdout
if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print(f'  PASS {name}')
            except Exception as e:
                print(f'  FAIL {name}: {e}')