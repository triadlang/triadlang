from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipapp

PY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PY_ROOT) if os.path.basename(PY_ROOT) in ('python', 'src') else PY_ROOT

def _tri_files(root: str) -> dict[str, float]:
    out = {}
    if os.path.isfile(root):
        out[root] = os.stat(root).st_mtime
        root = os.path.dirname(root) or '.'
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ('__pycache__', '__triadcache__', '.git')]
        for fn in filenames:
            if fn.endswith('.tri'):
                p = os.path.join(dirpath, fn)
                try:
                    out[p] = os.stat(p).st_mtime
                except OSError:
                    pass
    return out

def _run_child(subcmd: str, path: str) -> int:
    code = (
        f"import sys; sys.path.insert(0, {PY_ROOT!r}); "
        "from cli.main import main; "
        "sys.argv = ['triad'] + sys.argv[1:]; "
        "sys.exit(main())"
    )
    return subprocess.call([sys.executable, '-c', code, subcmd, path])

def cmd_watch(args) -> int:
    if not args:
        print('usage: triad watch <file.tri|dir> [--cmd run|check] [--interval S] [--no-clear]',
              file=sys.stderr)
        return 1
    target = args[0]
    subcmd = 'run'
    interval = 0.5
    clear = True
    i = 1
    while i < len(args):
        a = args[i]
        if a == '--cmd' and i + 1 < len(args):
            subcmd = args[i + 1]; i += 2
        elif a == '--interval' and i + 1 < len(args):
            try:
                interval = float(args[i + 1])
            except ValueError:
                print(f'watch: invalid --interval {args[i + 1]!r}', file=sys.stderr)
                return 1
            if not interval >= 0.05:
                print('watch: --interval must be >= 0.05', file=sys.stderr)
                return 1
            i += 2
        elif a == '--no-clear':
            clear = False; i += 1
        else:
            i += 1
    if subcmd not in ('run', 'check'):
        print('watch: --cmd accepts run or check', file=sys.stderr)
        return 1
    if not os.path.exists(target):
        print(f'watch: does not exist: {target}', file=sys.stderr)
        return 1
    entry = target if os.path.isfile(target) else None
    if entry is None:

        cand = os.path.join(target, 'src', 'main.tri')
        if not os.path.isfile(cand):
            cand = os.path.join(target, 'main.tri')
        if not os.path.isfile(cand):
            tris = sorted(p for p in _tri_files(target))
            if not tris:
                print(f'watch: no .tri under {target}', file=sys.stderr)
                return 1
            cand = tris[0]
        entry = cand

    print(f'watch: {entry}  (Ctrl+C to quit)')
    seen = _tri_files(target)
    last_rc = None
    try:
        while True:
            if clear:
                os.system('cls' if os.name == 'nt' else 'clear')
            stamp = time.strftime('%H:%M:%S')
            print(f'── triad watch · {stamp} · {subcmd} {entry} ──')
            last_rc = _run_child(subcmd, entry)
            status = 'ok' if last_rc == 0 else f'exit {last_rc}'
            print(f'── {status} · saved? I re-run ──')
            while True:
                time.sleep(interval)
                now = _tri_files(target)
                if now != seen:
                    seen = now
                    break
    except KeyboardInterrupt:
        print('\nwatch: stopped')
        return 0 if (last_rc in (0, None)) else last_rc

_BUNDLE_PACKAGES = ('frontend', 'runtime', 'compiler', 'stdlib', 'cli', 'adapters', 'embed', 'triad')

_MAIN_PY = '''\
import os
import sys
import tempfile
import zipfile

def _extract_app(tmp):
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.isdir(here):
        src = os.path.join(here, "app")
        for fn in os.listdir(src):
            with open(os.path.join(src, fn), "rb") as f:
                data = f.read()
            with open(os.path.join(tmp, fn), "wb") as f:
                f.write(data)
    else:
        with zipfile.ZipFile(here) as z:
            for info in z.infolist():
                if info.filename.startswith("app/") and not info.is_dir():
                    out = os.path.join(tmp, os.path.basename(info.filename))
                    with open(out, "wb") as f:
                        f.write(z.read(info))

def main():
    with tempfile.TemporaryDirectory(prefix="triad_app_") as tmp:
        _extract_app(tmp)
        entry = os.path.join(tmp, ENTRY)
        from cli.main import main as triad_main
        sys.argv = ["triad", "run", entry] + sys.argv[1:]
        raise SystemExit(triad_main())

ENTRY = {entry!r}
main()
'''

def cmd_bundle(args) -> int:
    if not args:
        print('usage: triad bundle <app.tri> [-o out.pyz]', file=sys.stderr)
        return 1
    entry = args[0]
    out = None
    for i, a in enumerate(args):
        if a == '-o' and i + 1 < len(args):
            out = args[i + 1]
    if not os.path.isfile(entry):
        print(f'bundle: does not exist: {entry}', file=sys.stderr)
        return 1
    if out is None:
        out = os.path.splitext(os.path.basename(entry))[0] + '.pyz'

    app_dir = os.path.dirname(os.path.abspath(entry)) or '.'
    with tempfile.TemporaryDirectory(prefix='triad_bundle_') as staging:

        for pkg in _BUNDLE_PACKAGES:
            src_pkg = os.path.join(PY_ROOT, pkg)
            if os.path.isdir(src_pkg):
                shutil.copytree(
                    src_pkg, os.path.join(staging, pkg),
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'tui'))

        app_dst = os.path.join(staging, 'app')
        os.makedirs(app_dst, exist_ok=True)
        for fn in os.listdir(app_dir):
            if fn.endswith('.tri'):
                shutil.copy2(os.path.join(app_dir, fn), os.path.join(app_dst, fn))
        with open(os.path.join(staging, '__main__.py'), 'w') as f:
            f.write(_MAIN_PY.replace('{entry!r}', repr(os.path.basename(entry))))
        zipapp.create_archive(staging, out,
                              interpreter='/usr/bin/env python3',
                              compressed=True)
    size = os.path.getsize(out) / 1e6
    print(f'bundle: {out} ({size:.1f} MB)')
    print('requirements on the target machine: python >= 3.12 with numpy installed')
    print(f'run it: python {out}')
    return 0

