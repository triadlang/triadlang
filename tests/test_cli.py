import os
import subprocess
import sys


def _run(*args):
    env = {**os.environ, 'PYTHONPATH': 'src', 'TRIADLANG_BACKEND': 'cpu'}
    return subprocess.run(
        [sys.executable, '-m', 'cli.main', *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_help_flag_is_supported():
    result = _run('--help')
    assert result.returncode == 0
    assert 'Usage:' in result.stdout


def test_run_and_check_smoke():
    run = _run('run', 'examples/basic/hello.tri')
    check = _run('check', 'examples/basic/hello.tri')
    assert run.returncode == 0, run.stderr
    assert 'Hello from TriadLang' in run.stdout
    assert check.returncode == 0, check.stderr
    assert 'OK' in check.stdout
