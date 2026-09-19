from __future__ import annotations

import os
import sys

PY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PY_ROOT) if os.path.basename(PY_ROOT) in ('python', 'src') else PY_ROOT
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

def cmd_run(args):
    sys.setrecursionlimit(50000)
    backend = 'solver'
    safe = True
    remaining = []
    i = 0
    while i < len(args):
        if args[i] == '--backend' and i + 1 < len(args):
            backend = args[i + 1]
            i += 2
        elif args[i] == '--safe':
            safe = True
            i += 1
        elif args[i] == '--unsafe':
            safe = False
            i += 1
        else:
            remaining.append(args[i])
            i += 1
    args = remaining
    if not args:
        print('usage: triad run <file.tri> [--backend solver|vm] [--safe|--unsafe]', file=sys.stderr)
        return 1
    path = args[0]
    if not os.path.exists(path):
        print(f'error: file not found: {path}', file=sys.stderr)
        return 1
    from frontend.errors import LexError, ParseError
    from frontend.parser_universal import parse
    from runtime.compiler_runtime import CompileError, TriadCompiler, set_safe_mode
    from runtime.security import default_policy, set_policy
    project_dir = os.path.dirname(os.path.abspath(path))
    set_policy(default_policy(project_dir))
    try:
        with open(path) as f:
            src = f.read()
        mod = parse(src, path)
        set_safe_mode(safe)
        compiler = TriadCompiler()
        if backend == 'vm':
            compiler.set_backend('vm')
        compiler.compile_and_run(mod, safe=safe)
        return 0
    except (LexError, ParseError) as e:
        print(str(e), file=sys.stderr)
        return 1
    except CompileError as e:
        print(str(e), file=sys.stderr)
        return 1
    except Exception as e:
        sourcemap = compiler._sourcemap if compiler._sourcemap else None
        if sourcemap and sys.exc_info()[2] is not None:
            tb_text = sourcemap.format_traceback(sys.exc_info()[2], src)
            print(f'{tb_text}\n{type(e).__name__}: {e}', file=sys.stderr)
        else:
            print(f'runtime error: {e}', file=sys.stderr)
        return 1

def cmd_check(args):
    if not args:
        print('usage: triad check <file.tri>', file=sys.stderr)
        return 1
    path = args[0]
    if not os.path.exists(path):
        print(f'error: file not found: {path}', file=sys.stderr)
        return 1
    from compiler.typecheck_universal import TypeCheckError, typecheck
    from frontend.errors import LexError, ParseError
    from frontend.parser_universal import parse
    try:
        with open(path) as f:
            src = f.read()
        mod = parse(src, path)
        typecheck(mod)
        print(f'check: {path} OK')
        return 0
    except (LexError, ParseError) as e:
        print(str(e), file=sys.stderr)
        return 1
    except TypeCheckError as e:
        print(str(e), file=sys.stderr)
        return 1

def cmd_compile(args):
    path = None
    emit_ir = False
    emit_json_path = None
    native = False
    no_boehm = False
    output_path = None
    i = 0
    while i < len(args):
        if args[i] == '--emit-ir':
            emit_ir = True
        elif args[i] == '--emit-ir-json':
            i += 1
            emit_json_path = args[i] if i < len(args) else 'out.json'
        elif args[i] == '--native':
            native = True
        elif args[i] == '--no-boehm':
            no_boehm = True
        elif args[i] == '-o':
            i += 1
            output_path = args[i] if i < len(args) else 'a.out'
        else:
            path = args[i]
        i += 1
    if not path:
        print('usage: triad compile <file.tri> [--emit-ir] [--emit-ir-json out.json] [--native] [-o output]', file=sys.stderr)
        return 1
    if not os.path.exists(path):
        print(f'error: file not found: {path}', file=sys.stderr)
        return 1
    from compiler.emit_json import emit_json, emit_json_file
    from compiler.lower import lower_module
    from frontend.errors import LexError, ParseError
    from frontend.parser_universal import parse
    try:
        with open(path) as f:
            src = f.read()
        if native:
            import subprocess

            from compiler.c_codegen import compile_to_c
            c_code = compile_to_c(src, path)
            base = os.path.splitext(os.path.basename(path))[0]
            c_path = output_path + '.c' if output_path else base + '.c'
            bin_path = output_path or base
            with open(c_path, 'w') as f:
                f.write(c_code)
            rt_dir = os.path.join(REPO_ROOT, 'native', 'c')
            cc = os.environ.get('CC', 'gcc')
            cflags = ['-std=c11', '-O2', '-I', os.path.join(rt_dir, 'include')]
            py_libs = []
            if 'triad_python.h' in c_code:
                import sysconfig
                src_dir = os.path.join(REPO_ROOT, 'src')
                script_dir = os.path.dirname(os.path.abspath(path))
                cflags.extend(['-I', sysconfig.get_paths()['include']])
                cflags.append(f'-DTRIAD_SRC_DIR="{src_dir}"')
                cflags.append(f'-DTRIAD_SCRIPT_DIR="{script_dir}"')
                cflags.append(f'-DTRIAD_SITE_DIR="{sysconfig.get_paths()["purelib"]}"')
                libdir = sysconfig.get_config_var('LIBDIR')
                pyver = sysconfig.get_config_var('LDVERSION') or sysconfig.get_config_var('VERSION')
                if libdir:
                    if os.name == 'nt':
                        py_libs.extend(['-L', libdir])
                    else:
                        py_libs.extend(['-L', libdir, f'-Wl,-rpath,{libdir}'])
                py_libs.append(f'-lpython{pyver}')
                for extra in (sysconfig.get_config_var('LIBS') or '').split():
                    py_libs.append(extra)

            static_lib = os.path.join(rt_dir, 'libtriad_rt.a')
            if os.path.exists(static_lib):
                libs = [static_lib, '-lm']
            else:
                libs = ['-L', rt_dir, '-ltriad_rt', '-lm']
            if os.name != 'nt':
                libs.append('-lpthread')
            _fftw_inc = os.path.join(os.path.expanduser('~'), '.local', 'include', 'fftw3.h')
            if os.path.exists('/usr/include/fftw3.h'):
                cflags.append('-DUSE_FFTW')
                libs.append('-lfftw3')
            elif os.path.exists(_fftw_inc):
                cflags.append('-DUSE_FFTW')
                cflags.extend(['-I', os.path.dirname(_fftw_inc)])
                _fftw_lib = os.path.join(os.path.expanduser('~'), '.local', 'lib')
                libs.extend(['-L', _fftw_lib, '-lfftw3'])
            gc_header = os.path.exists('/usr/include/gc/gc.h')
            if no_boehm or not gc_header:
                cflags.append('-DTRIAD_NO_BOEHM')
            else:
                libs.append('-lgc')
            cmd = [cc] + cflags + [c_path] + libs + py_libs + ['-o', bin_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f'cc error:\n{result.stderr}', file=sys.stderr)
                return 1
            if result.stderr:
                for line in result.stderr.strip().split('\n'):
                    if 'warning' in line:
                        print(f'  {line}')
            print(f'compiled {path} -> {bin_path} (native)')
            return 0
        mod = parse(src, path)
        ir = lower_module(mod)
        if emit_json_path:
            emit_json_file(ir, emit_json_path)
            print(f'IR written to {emit_json_path}')
        elif emit_ir:
            print(emit_json(ir))
        else:
            print(emit_json(ir))
        return 0
    except (LexError, ParseError) as e:
        print(str(e), file=sys.stderr)
        return 1

def cmd_repl(args):
    from cli.repl import run_repl
    run_repl()
    return 0

def cmd_doctor(args):
    from triad import __version__
    print(f'TriadLang Doctor v{__version__}')
    print('=' * 40)
    checks = []
    v = sys.version_info
    checks.append(('Python >= 3.12', v >= (3, 12)))
    for mod_name in ['frontend.lexer_universal', 'frontend.parser_universal', 'frontend.ast_nodes', 'compiler.ir', 'compiler.lower', 'compiler.emit_json', 'runtime.interpreter', 'cli.repl']:
        try:
            __import__(mod_name)
            checks.append((f'import {mod_name}', True))
        except Exception as e:
            checks.append((f'import {mod_name}: {e}', False))
    _saved_path = list(sys.path)
    try:
        sys.path = [p for p in sys.path if os.path.abspath(p or '.') != os.path.abspath(PY_ROOT)]
        sys.modules.pop('numpy', None)
        import numpy as _np
        _origin = os.path.abspath(getattr(_np, '__file__', '') or '')
        _ver = getattr(_np, '__version__', '?')
        if _origin.startswith(os.path.abspath(PY_ROOT)):
            checks.append(('numpy upstream (sombreado pelo shim triad-native)', False))
        else:
            checks.append((f'numpy upstream ({_ver})', True))
    except ImportError:
        checks.append(('numpy (optional, needed for triad-native)', False))
    finally:
        sys.path = _saved_path
    try:
        from triad import ntri as _ntri

        assert _ntri is not None
        checks.append(('ntri fallback', True))
    except ImportError:
        checks.append(('ntri fallback', False))
    ex_dir = os.path.join(REPO_ROOT, 'examples', 'basic')
    checks.append(('examples/basic/ exists', os.path.isdir(ex_dir)))
    for label, ok in checks:
        status = 'OK' if ok else 'MISSING'
        print(f'  [{status:7s}] {label}')
    all_ok = all((ok for _, ok in checks))
    print()
    if all_ok:
        print('All checks passed.')
    else:
        print('Some checks failed — see above.')
    return 0 if all_ok else 1

def cmd_jit_stats(args):

    from runtime.jit_tiered import get_jit
    stats = get_jit().stats()
    print('triadlang hot-spot tracker')
    print('=' * 40)
    for key in ('enabled', 'threshold', 'compile_count', 'cache_size'):
        if key in stats:
            print(f'  {key:14s}: {stats[key]}')
    counts = stats.get('counts') or {}
    if counts:
        print('  loop hit counts:')
        for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            hot = ' (hot)' if count >= stats.get('threshold', 128) else ''
            print(f'    {name}: {count}{hot}')
    else:
        print('  no loops tracked in this process yet')
    return 0

def cmd_test(args):
    import subprocess

    tests_dir = os.path.join(PY_ROOT, 'tests')
    e2e = os.path.join(REPO_ROOT, 'test_e2e_triad.py')
    env = {**os.environ, 'PYTHONPATH': PY_ROOT}
    overall = 0

    print('running pytest src/tests ...')
    rc_pytest = subprocess.call([sys.executable, '-m', 'pytest', tests_dir, '-q'], env=env)
    if rc_pytest != 0:
        overall = 1

    if os.path.exists(e2e):
        print('\nrunning test_e2e_triad.py ...')
        rc_e2e = subprocess.call([sys.executable, e2e], cwd=REPO_ROOT, env=env)
        if rc_e2e != 0:
            overall = 1
    else:
        print('\ntest_e2e_triad.py not found, skipping e2e', file=sys.stderr)

    print('\nall test suites passed.' if overall == 0 else '\nsome test suites failed.')
    return overall

def cmd_setup(args):
    from cli.setup_cmd import cmd_setup as _setup
    return _setup(args)

def cmd_watch(args):
    from cli.devtools import cmd_watch as _watch
    return _watch(args)

def cmd_bundle(args):
    from cli.devtools import cmd_bundle as _bundle
    return _bundle(args)

def cmd_fmt(args):
    if not args:
        print('usage: triad fmt <file.tri>', file=sys.stderr)
        return 1
    path = args[0]
    if not os.path.exists(path):
        print(f'error: file not found: {path}', file=sys.stderr)
        return 1
    from compiler.formatter import format_universal
    from frontend.errors import LexError, ParseError
    from frontend.parser_universal import parse
    try:
        with open(path) as f:
            src = f.read()
        mod = parse(src, path)
        print(format_universal(mod), end='')
        return 0
    except (LexError, ParseError) as e:
        print(str(e), file=sys.stderr)
        return 1

def cmd_bench(args):
    safe = False
    if not args:
        print('usage: triad bench <file.tri>', file=sys.stderr)
        return 1
    import time
    path = args[0]
    from frontend.parser_universal import parse
    from runtime.compiler_runtime import TriadCompiler
    with open(path) as f:
        src = f.read()
    mod = parse(src, path)
    start = time.perf_counter()
    compiler = TriadCompiler()
    compiler.compile_and_run(mod, safe=safe)
    elapsed = time.perf_counter() - start
    print(f'\nbench: {elapsed:.4f}s')
    return 0

def cmd_init(args):
    name = args[0] if args else None
    if not name:
        print('usage: triad init <project-name>', file=sys.stderr)
        return 1
    from stdlib.registry import init_project

    project_dir = os.path.join('.', name)
    if os.path.exists(os.path.join(project_dir, 'triad.json')):
        print(f'triad project already exists at {project_dir}/', file=sys.stderr)
        return 1
    os.makedirs(project_dir, exist_ok=True)
    init_project(name, project_dir=project_dir)

    import json
    src_dir = os.path.join(project_dir, 'src')
    tests_dir = os.path.join(project_dir, 'tests')
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(tests_dir, exist_ok=True)
    main_tri = os.path.join(src_dir, 'main.tri')
    if not os.path.exists(main_tri):
        with open(main_tri, 'w') as f:
            f.write(
                f'// {name} — entry point\n'
                'print("hello from ' + name + '");\n\n'
                '// a field to get started:\n'
                '// substrate field : B0;\n'
                '// evolve field for T=1.0;\n'
                '// observe field norm, crystallinity;\n')
    smoke = os.path.join(tests_dir, 'smoke.tri')
    if not os.path.exists(smoke):
        with open(smoke, 'w') as f:
            f.write('// run with: triad run tests/smoke.tri\n'
                    'let ok = 1 + 1 == 2;\n'
                    'print(f"smoke: {ok}");\n')
    readme = os.path.join(project_dir, 'README.md')
    if not os.path.exists(readme):
        with open(readme, 'w') as f:
            f.write(f'# {name}\n\n'
                    'Run: `triad run src/main.tri`\n\n'
                    'Develop with reload: `triad watch .`\n\n'
                    'Package: `triad bundle src/main.tri -o ' + name + '.pyz`\n')
    manifest_path = os.path.join(project_dir, 'triad.json')
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest.setdefault('entry', 'src/main.tri')
        manifest.setdefault('scripts', {'start': 'triad run src/main.tri',
                                        'dev': 'triad watch .'})
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    except (OSError, json.JSONDecodeError):
        pass
    print(f'initialized: ./{name}/')
    print('  src/main.tri    — entry point')
    print('  tests/smoke.tri — smoke test')
    print('  triad.json      — manifest (entry, scripts)')
    print(f'next steps: cd {name} && triad watch .')
    return 0

def cmd_install(args):
    from stdlib.registry import install_dependencies, install_package, load_manifest, save_manifest
    if not args:
        installed = install_dependencies()
        if not installed:
            print('no dependencies to install')
        else:
            for n in installed:
                print(f'  installed: {n}')
        return 0
    source = args[0]
    name_override = args[1] if len(args) > 1 else None
    pkg_name = install_package(source, name=name_override)
    m = load_manifest()
    if m:
        m.dependencies[pkg_name] = source
        save_manifest(m)
        print(f'  installed: {pkg_name} (added to {m.name})')
    else:
        print(f'  installed: {pkg_name}')
    return 0

def cmd_publish(args):
    from stdlib.registry import publish_package
    reg_dir = args[0] if args else None
    try:
        path = publish_package(registry_dir=reg_dir)
        m = None
        from stdlib.registry import load_manifest
        m = load_manifest()
        print(f'published: {m.name}@{m.version} -> {path}')
        return 0
    except Exception as e:
        print(f'error: {e}', file=sys.stderr)
        return 1

def cmd_list(args):
    from stdlib.registry import list_installed
    pkgs = list_installed()
    if not pkgs:
        print('no packages installed')
        return 0
    print(f"{'name':<20} {'version':<10} {'source':<10} {'description'}")
    print('-' * 60)
    for p in pkgs:
        src = 'symlink' if p['symlink'] else 'copy'
        desc = p['description'][:30] if p['description'] else ''
        print(f"{p['name']:<20} {p['version']:<10} {src:<10} {desc}")
    return 0

def cmd_docgen(args):
    from compiler.docgen import cmd_docgen as _cmd_docgen
    return _cmd_docgen(args)

def cmd_debug(args):
    from runtime.debugger import cmd_debug as _cmd_debug
    return _cmd_debug(args)

def cmd_lsp(args):
    from cli.lsp import main as lsp_main
    lsp_main()
    return 0

def cmd_play(args):
    host = '127.0.0.1'
    port = 8000
    for i, a in enumerate(args):
        if a == '--port' and i + 1 < len(args):
            try:
                port = int(args[i + 1])
            except ValueError:
                print(f'error: invalid --port {args[i + 1]!r}', file=sys.stderr)
                return 1
            if not 1 <= port <= 65535:
                print(f'error: --port must be 1..65535, got {port}', file=sys.stderr)
                return 1
        elif a == '--host' and i + 1 < len(args):
            host = args[i + 1]
    url = f'http://{host}:{port}/engine'
    print(f'TriadLang 3D Engine: {url}')
    print('(Ctrl+C stops it)')
    try:
        import threading
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    except (ImportError, AttributeError, RuntimeError):
        pass
    return cmd_serve(['--host', host, '--port', str(port)])

def cmd_serve(args):
    host = '127.0.0.1'
    port = 8000
    reload = False
    i = 0
    while i < len(args):
        if args[i] == '--host' and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        elif args[i] == '--port' and i + 1 < len(args):
            try:
                port = int(args[i + 1])
            except ValueError:
                print(f'error: invalid --port {args[i + 1]!r}', file=sys.stderr)
                return 1
            if not 1 <= port <= 65535:
                print(f'error: --port must be 1..65535, got {port}', file=sys.stderr)
                return 1
            i += 2
        elif args[i] == '--reload':
            reload = True
            i += 1
        else:
            i += 1
    try:
        import uvicorn
    except ImportError:
        print('error: uvicorn not installed. run: pip install "triadlang[api]"', file=sys.stderr)
        return 1
    uvicorn.run('api.app:app', host=host, port=port, reload=reload)
    return 0

def cmd_solve(args):
    import argparse
    import json
    p = argparse.ArgumentParser(prog='triad solve', add_help=True)
    p.add_argument('--regime', default=None)
    p.add_argument('--N', type=int, default=128)
    p.add_argument('--T', type=float, default=20.0)
    p.add_argument('--L', type=float, default=32.0)
    p.add_argument('--dt', type=float, default=0.005)
    p.add_argument('--dim', type=int, default=None)
    p.add_argument('--output', choices=['json', 'npy'], default='json')
    p.add_argument('--out-file', default=None)
    ns = p.parse_args(args)

    from runtime.core.solver import TriadParams, integrate_nd
    from runtime.physics.observables import (
        crystallinity,
        dominant_wavenumber,
        fwhm,
        ipr,
        norm,
        peak_density,
    )
    from triad import ntri as np

    if ns.dim is None:
        ns.dim = TriadParams().D
    if ns.dim < 1:
        p.error('--dim must be >= 1')
    if ns.regime:
        from stdlib.regimes import resolve_regime
        params = resolve_regime(ns.regime, N=ns.N, L=ns.L, dt=ns.dt)
        params.T = ns.T
        params.D = ns.dim
    else:
        params = TriadParams(N=ns.N, T=ns.T, L=ns.L, dt=ns.dt, D=ns.dim)

    result = integrate_nd(params)

    psi = result['psi_final']
    dx = float(result.get('dx', ns.L / ns.N))
    k_min = 2.0 * np.pi / ns.L

    if psi.ndim == 1:
        obs = {
            'crystallinity': float(crystallinity(psi, dx)),
            'k_star': float(dominant_wavenumber(psi, dx, k_min=k_min)),
            'peak_density': float(peak_density(psi)),
            'norm': float(norm(psi, dx)),
            'ipr': float(ipr(psi, dx)),
            'fwhm': float(fwhm(psi, dx)),
        }
    else:
        rho = np.abs(psi) ** 2
        obs = {
            'peak_density': float(rho.max()),
            'norm': float(rho.sum() * dx ** psi.ndim),
        }

    if ns.output == 'json':
        t_arr = result.get('t', result.get('t_traj', np.array([ns.T])))
        out = {
            't_final': float(t_arr[-1]) if len(t_arr) > 0 else ns.T,
            'N': ns.N,
            'dim': ns.dim,
            'regime': ns.regime,
            **obs,
        }
        text = json.dumps(out, indent=2)
        if ns.out_file:
            with open(ns.out_file, 'w') as f:
                f.write(text)
            print(f'saved {ns.out_file}')
        else:
            print(text)
    else:
        fname = ns.out_file or 'psi_final.npy'
        np.save(fname, psi)
        print(f'saved {fname}')
        for k, v in obs.items():
            print(f'  {k}: {v:.6f}')
    return 0

def cmd_observables(args):
    import argparse
    import json
    p = argparse.ArgumentParser(prog='triad observables', add_help=True)
    p.add_argument('file')
    p.add_argument('--dx', type=float, default=None)
    p.add_argument('--L', type=float, default=32.0)
    p.add_argument('--N', type=int, default=None)
    ns = p.parse_args(args)

    from runtime.physics.observables import (
        crystallinity,
        dominant_wavenumber,
        fwhm,
        ipr,
        norm,
        participation_ratio,
        peak_density,
    )
    from triad import ntri as np

    psi = np.load(ns.file, allow_pickle=False)
    psi_1d = psi.ravel()
    N = ns.N or psi_1d.shape[0]
    dx = ns.dx if ns.dx is not None else (ns.L / N)
    k_min = 2.0 * np.pi / ns.L

    result = {
        'file': ns.file,
        'shape': list(psi.shape),
        'dtype': str(psi.dtype),
        'N_flat': int(N),
        'dx': float(dx),
        'crystallinity': float(crystallinity(psi_1d, dx)),
        'k_star': float(dominant_wavenumber(psi_1d, dx, k_min=k_min)),
        'peak_density': float(peak_density(psi_1d)),
        'norm': float(norm(psi_1d, dx)),
        'ipr': float(ipr(psi_1d, dx)),
        'fwhm': float(fwhm(psi_1d, dx)),
        'participation_ratio': float(participation_ratio(psi_1d, dx)),
    }
    print(json.dumps(result, indent=2))
    return 0

def cmd_plot(args):
    safe = False
    if not args:
        print('usage: triad plot <file.tri> [--out path.png]', file=sys.stderr)
        return 1
    path = args[0]
    out_path = None
    i = 1
    while i < len(args):
        if args[i] == '--out' and i + 1 < len(args):
            out_path = args[i + 1]
            i += 2
        else:
            i += 1
    if not os.path.exists(path):
        print(f'error: file not found: {path}', file=sys.stderr)
        return 1
    from frontend.errors import LexError, ParseError
    from frontend.parser_universal import parse
    from runtime.compiler_runtime import CompileError, TriadCompiler
    try:
        with open(path) as f:
            src = f.read()
        mod = parse(src, path)
        compiler = TriadCompiler()
        compiler.compile_and_run(mod, safe=safe)
        if out_path is None:
            base = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(os.path.dirname(os.path.abspath(path)),
                                    f'{base}.png')
            print(f'plot: {path} -> {out_path}')
            return 0
        if not os.path.exists(out_path):
            print(f'plot error: expected artifact missing: {out_path}', file=sys.stderr)
            return 1
        print(f'plot: {path} -> {out_path}')
        return 0
    except (LexError, ParseError) as e:
        print(str(e), file=sys.stderr)
        return 1
    except CompileError as e:
        print(str(e), file=sys.stderr)
        return 1
    except Exception as e:
        print(f'plot error: {e}', file=sys.stderr)
        return 1

def cmd_tui(args):
    try:
        from cli.tui import run_tui
    except ImportError:
        print('error: TUI dependencies not installed. run: pip install "triadlang[tui]"', file=sys.stderr)
        return 1
    return run_tui(args)

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] in ('-h', '--help', 'help'):
        print('TriadLang')
        print()
        print('Usage:')
        print('  triad run <file.tri> [--unsafe]     Run a .tri file (safe by default)')
        print('  triad check <file.tri>            Type check')
        print('  triad compile <file.tri> --native [--no-boehm] [-o out]  Compile to native binary')
        print('  triad repl                        Interactive REPL')
        print('  triad lsp                         Start LSP server')
        print('  triad init <name>                 Create new project')
        print('  triad install [source]            Install package/deps')
        print('  triad publish                     Publish to local registry')
        print('  triad list                        List installed packages')
        print('  triad doctor                      Check installation')
        print('  triad test                        Run tests')
        print('  triad bench <file.tri>            Benchmark')
        print('  triad debug <file.tri> [-b N] [--solver]  Debug with breakpoints (source + solver-native)')
        print('  triad docgen <file|dir> [-f markdown|md|html]  Generate docs')
        print('  triad fmt <file.tri>              Format')
        print('  triad serve [--host HOST] [--port PORT] [--reload]  Start API server')
        print('  triad solve [--regime NAME] [--N N] [--T T] [--dim 1|2|3] [--output json|npy]')
        print('  triad observables <file.npy> [--dx DX] [--L L]')
        print('  triad tui [file.tri] [--solver|--repl]   Interactive TUI')
        print('  triad setup [--yes] [--all] [--extras=a,b] [--native|--no-native]  Guided install')
        print('  triad watch <file|dir> [--cmd run|check]  Re-run on save (hot reload)')
        print('  triad bundle <app.tri> [-o out.pyz]   Package app as standalone .pyz')
        print('  triad play [--port P]                 3D engine in the browser')
        print('  triad plot <file.tri> [--out path.png]  Run a plotting .tri and emit a PNG')
        print('  triad memory <start|record|recall|resonate|sleep|status>  Crystal memory')
        print('  triad jit-stats                        JIT hot-spot tracker')
        return 0
    cmd = argv[0]
    rest = argv[1:]
    if cmd.endswith('.tri'):
        return cmd_run([cmd])
    commands = {'run': cmd_run, 'check': cmd_check, 'compile': cmd_compile, 'repl': cmd_repl, 'lsp': cmd_lsp, 'init': cmd_init, 'install': cmd_install, 'publish': cmd_publish, 'list': cmd_list, 'doctor': cmd_doctor, 'test': cmd_test, 'fmt': cmd_fmt, 'bench': cmd_bench, 'debug': cmd_debug, 'docgen': cmd_docgen, 'serve': cmd_serve, 'solve': cmd_solve, 'observables': cmd_observables, 'jit-stats': cmd_jit_stats, 'tui': cmd_tui, 'setup': cmd_setup, 'plot': cmd_plot, 'watch': cmd_watch, 'bundle': cmd_bundle, 'play': cmd_play, 'memory': cmd_memory}
    fn = commands.get(cmd)
    if fn is None:
        print(f'unknown command: {cmd}', file=sys.stderr)
        return 1
    return fn(rest)
def cmd_memory(args):
    import sys

    from cli.triad_memory import main as _memory_main
    saved = sys.argv[:]
    sys.argv = ['triad memory'] + list(args)
    try:
        _memory_main()
    finally:
        sys.argv = saved

if __name__ == '__main__':
    sys.exit(main())
