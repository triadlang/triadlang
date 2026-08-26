from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys

PY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PY_ROOT) if os.path.basename(PY_ROOT) in ('python', 'src') else PY_ROOT

EXTRAS = {
    'science':  'scipy (analysis helpers)',
    'viz':      'matplotlib (plots)',
    'api':      'FastAPI + uvicorn (REST server: triad serve)',
    'web':      'Flask (web interop examples)',
    'quantum':  'qiskit + cirq (quantum bridge)',
    'llm':      'LLM inference: transformers, safetensors, gguf, torch',
    'tui':      'textual + rich (triad tui)',
    'dev':      'pytest, ruff, black',
}
DEFAULT_EXTRAS = ['science', 'viz', 'api', 'tui', 'dev']

def _ok(msg):   print(f'  [ok]      {msg}')
def _miss(msg): print(f'  [missing] {msg}')
def _info(msg): print(f'  [info]    {msg}')

def _has_header(*paths) -> bool:
    return any(os.path.isfile(p) for p in paths)

def _detect() -> dict:
    d = {}
    d['os'] = platform.system()
    d['python'] = sys.version.split()[0]
    d['python_ok'] = sys.version_info >= (3, 12)
    d['pip'] = shutil.which('pip') or shutil.which('pip3')
    d['gcc'] = shutil.which('gcc') or shutil.which('cc')
    d['make'] = shutil.which('make')
    d['nvcc'] = shutil.which('nvcc')
    d['libgc'] = _has_header('/usr/include/gc/gc.h', '/usr/local/include/gc/gc.h',
                             '/opt/homebrew/include/gc/gc.h')
    d['fftw'] = _has_header('/usr/include/fftw3.h', '/usr/local/include/fftw3.h',
                            '/opt/homebrew/include/fftw3.h')
    d['in_repo'] = os.path.isfile(os.path.join(REPO_ROOT, 'pyproject.toml'))
    d['tty'] = sys.stdin.isatty()
    try:
        from triad import ntri
        d['numpy'] = True
    except ImportError:
        d['numpy'] = False
    return d

def _ask_yn(question: str, default: bool, assume_yes: bool) -> bool:
    if assume_yes or not sys.stdin.isatty():
        return default
    suffix = ' [S/n] ' if default else ' [s/N] '
    while True:
        ans = input(question + suffix).strip().lower()
        if ans == '':
            return default
        if ans in ('s', 'sim', 'y', 'yes'):
            return True
        if ans in ('n', 'nao', 'não', 'no'):
            return False
        print('  answer y or n')

def _ask_extras(detected: dict, assume_yes: bool) -> list[str]:
    if assume_yes or not sys.stdin.isatty():
        extras = list(DEFAULT_EXTRAS)
        if detected['nvcc']:
            extras.append('gpu')
        return extras
    print('\nAvailable extras:')
    keys = list(EXTRAS)
    for i, k in enumerate(keys, 1):
        mark = '*' if k in DEFAULT_EXTRAS else ' '
        print(f'  {i}. [{mark}] {k:8s} — {EXTRAS[k]}')
    if detected['nvcc']:
        print(f'  {len(keys)+1}. [ ] gpu      — cupy for CUDA 12 (nvcc detected)')
        keys.append('gpu')
    print('  (* = recommended for your machine)')
    raw = input('Which ones? (comma-separated numbers, Enter = recommended, 0 = none) ').strip()
    if raw == '':
        return [k for k in DEFAULT_EXTRAS if k in keys] + (['gpu'] if 'gpu' in keys and False else [])
    if raw == '0':
        return []
    chosen = []
    for part in raw.split(','):
        part = part.strip()
        if part.isdigit() and 1 <= int(part) <= len(keys):
            chosen.append(keys[int(part) - 1])
    return chosen

def _pip_install(target: str) -> int:
    cmd = [sys.executable, '-m', 'pip', 'install', '-e', target]
    print(f'\n$ {" ".join(cmd)}')
    rc = subprocess.call(cmd, cwd=REPO_ROOT)
    if rc != 0:
        print('\npip failed; retrying with --user ...')
        rc = subprocess.call(cmd + ['--user'], cwd=REPO_ROOT)
    if rc != 0:
        print('\npip failed; retrying with --break-system-packages ...')
        rc = subprocess.call(cmd + ['--break-system-packages'], cwd=REPO_ROOT)
    return rc

def _build_native() -> int:
    cdir = os.path.join(REPO_ROOT, 'native', 'c')
    print(f'\n$ make -j (in {cdir})')
    return subprocess.call(['make', '-j4'], cwd=cdir)

def cmd_setup(args) -> int:
    assume_yes = '--yes' in args or '-y' in args
    want_all = '--all' in args
    skip_install = '--skip-install' in args
    force_native = '--native' in args
    skip_native = '--no-native' in args
    extras_flag = None
    for a in args:
        if a.startswith('--extras='):
            extras_flag = [e.strip() for e in a.split('=', 1)[1].split(',') if e.strip()]

    print('TriadLang Setup')
    print('=' * 40)
    d = _detect()
    print('\nYour machine:')
    _ok(f"{d['os']}, Python {d['python']}")
    if not d['python_ok']:
        _miss('Python >= 3.12 is required (PEP 701 f-strings). Install a newer Python and run again.')
        return 1
    _ok('pip available') if d['pip'] else _miss('pip not found')
    _ok('gcc + make (native C runtime possible)') if (d['gcc'] and d['make']) else _info('no gcc/make — the C runtime stays out (optional)')
    if d['gcc'] and d['make']:
        _ok('Boehm GC (libgc)') if d['libgc'] else _info('no libgc — the build falls back to malloc/free on its own')
        _ok('FFTW') if d['fftw'] else _info('no FFTW — runtime uses its own FFT')
    _ok('CUDA (nvcc)') if d['nvcc'] else _info('no CUDA — everything runs on CPU')
    if not d['in_repo']:
        _miss(f'pyproject.toml not found at {REPO_ROOT} — run setup from inside the repository')
        return 1

    if want_all:
        extras = [k for k in EXTRAS] + (['gpu'] if d['nvcc'] else [])
    elif extras_flag is not None:
        extras = extras_flag
    else:
        extras = _ask_extras(d, assume_yes)

    if skip_install:
        print('\n(--skip-install: skipping the Python install)')
        rc = 0
    else:
        target = '.[{}]'.format(','.join(extras)) if extras else '.'
        rc = _pip_install(target)
        if rc != 0:
            print('\nPython install failed — see pip output above.')
            return rc

    if d['gcc'] and d['make'] and not skip_native:
        build = force_native or _ask_yn('\nBuild the native C runtime now?', True, assume_yes)
        if build:
            rc_n = _build_native()
            if rc_n == 0:
                _ok('native runtime built (native/c)')
            else:
                _miss('C runtime build failed — everything else works without it')

    print('\nFinal diagnosis:')
    from cli.main import cmd_doctor
    cmd_doctor([])
    print('\nDone. Start with: triad run examples/basic/hello.tri')
    print('Doc for people who never programmed: docs/beginners.md')
    return 0
