import sys, os, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIAD = os.path.join(ROOT, 'triad')

def _cli(*args):
    r = subprocess.run([sys.executable, TRIAD] + list(args), capture_output=True, text=True, timeout=30)
    return (r.returncode, r.stdout, r.stderr)

def test_no_args():
    rc, out, err = _cli()
    assert rc == 0
    assert 'TriadLang' in out

def test_run():
    rc, out, err = _cli('run', os.path.join(ROOT, 'examples/basic/hello.tri'))
    assert rc == 0
    assert 'Hello' in out

def test_check():
    rc, out, err = _cli('check', os.path.join(ROOT, 'examples/basic/hello.tri'))
    assert rc == 0
    assert 'OK' in out

def test_compile():
    rc, out, err = _cli('compile', os.path.join(ROOT, 'examples/basic/hello.tri'), '--emit-ir')
    assert rc == 0

def test_doctor():
    rc, out, err = _cli('doctor')
    assert 'TriadLang Doctor' in out

def test_shortcut():
    rc, out, err = _cli(os.path.join(ROOT, 'examples/basic/hello.tri'))
    assert rc == 0
    assert 'Hello' in out

def test_missing_file():
    rc, out, err = _cli('run', '/tmp/nonexistent_triad_file.tri')
    assert rc == 1
if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print(f'  PASS {name}')
            except Exception as e:
                print(f'  FAIL {name}: {e}')