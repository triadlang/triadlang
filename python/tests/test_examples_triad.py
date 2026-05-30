import sys, os, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIAD = os.path.join(ROOT, 'triad')

def _run(tri_file):
    path = os.path.join(ROOT, tri_file)
    r = subprocess.run([sys.executable, TRIAD, 'run', path], capture_output=True, text=True, timeout=60)
    return (r.returncode, r.stdout, r.stderr)

def test_anti_collapse():
    rc, out, err = _run('examples/triad/anti_collapse.tri')
    assert rc == 0, err
    assert 'crystallinity' in out

def test_predator_prey():
    rc, out, err = _run('examples/games/predator_prey.tri')
    assert rc == 0, err
    assert 'Final' in out

def test_mock_runtime():
    rc, out, err = _run('examples/llm/mock_runtime.tri')
    assert rc == 0, err
    assert 'Final cache size' in out

def test_fdt_claims():
    rc, out, err = _run('examples/verification/fdt_claims.tri')
    assert rc == 0, err
    assert 'Evidence field' in out
if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print(f'  PASS {name}')
            except Exception as e:
                print(f'  FAIL {name}: {e}')