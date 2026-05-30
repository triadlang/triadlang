from __future__ import annotations
import sys
import os
PY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PY_ROOT) if os.path.basename(PY_ROOT) == 'python' else PY_ROOT
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

def cmd_run(args):
    sys.setrecursionlimit(50000)
    backend = 'solver'
    remaining = []
    i = 0
    while i < len(args):
        if args[i] == '--backend' and i + 1 < len(args):
            backend = args[i + 1]
            i += 2
        else:
            remaining.append(args[i])
            i += 1
    args = remaining
    if not args:
        print('usage: triad run <file.tri> [--backend solver|vm]', file=sys.stderr)
        return 1
    path = args[0]
    if not os.path.exists(path):
        print(f'error: file not found: {path}', file=sys.stderr)
        return 1
    from frontend.parser_universal import parse, ParseError
    from frontend.lexer_universal import LexError
    from runtime.compiler_runtime import TriadCompiler, CompileError
    try:
        with open(path) as f:
            src = f.read()
        mod = parse(src, path)
        compiler = TriadCompiler()
        if backend == 'vm':
            compiler._backend = 'vm'
        compiler.compile_and_run(mod)
        return 0
    except (LexError, ParseError) as e:
        print(str(e), file=sys.stderr)
        return 1
    except CompileError as e:
        print(str(e), file=sys.stderr)
        return 1
    except Exception as e:
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
    from frontend.parser_universal import parse, ParseError
    from frontend.lexer_universal import LexError
    from compiler.typecheck_universal import typecheck, TypeCheckError
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
    from frontend.parser_universal import parse, ParseError
    from frontend.lexer_universal import LexError
    from compiler.lower import lower_module
    from compiler.emit_json import emit_json, emit_json_file
    try:
        with open(path) as f:
            src = f.read()
        if native:
            from compiler.c_codegen import compile_to_c
            import subprocess
            import tempfile
            c_code = compile_to_c(src, path)
            base = os.path.splitext(os.path.basename(path))[0]
            c_path = output_path + '.c' if output_path else base + '.c'
            bin_path = output_path or base
            with open(c_path, 'w') as f:
                f.write(c_code)
            rt_dir = os.path.join(REPO_ROOT, 'native', 'c')
            cc = os.environ.get('CC', 'gcc')
            cflags = ['-std=c11', '-O2', '-I', os.path.join(rt_dir, 'include')]
            libs = ['-L', rt_dir, '-ltriad_rt', '-lm']
            if os.path.exists('/usr/include/fftw3.h'):
                cflags.append('-DUSE_FFTW')
                libs.append('-lfftw3')
            cmd = [cc] + cflags + [c_path] + libs + ['-o', bin_path]
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
    print('TriadLang Doctor v1.0')
    print('=' * 40)
    checks = []
    v = sys.version_info
    checks.append(('Python >= 3.10', v >= (3, 10)))
    for mod_name in ['frontend.lexer_universal', 'frontend.parser_universal', 'frontend.ast_nodes', 'compiler.ir', 'compiler.lower', 'compiler.emit_json', 'runtime.interpreter', 'cli.repl']:
        try:
            __import__(mod_name)
            checks.append((f'import {mod_name}', True))
        except Exception as e:
            checks.append((f'import {mod_name}: {e}', False))
    try:
        import numpy
        checks.append(('numpy available', True))
    except ImportError:
        checks.append(('numpy (optional, needed for triad-native)', False))
    ex_dir = os.path.join(PY_ROOT, 'examples', 'basic')
    checks.append((f'examples/basic/ exists', os.path.isdir(ex_dir)))
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

def cmd_test(args):
    import subprocess
    test_dir = os.path.join(PY_ROOT, 'tests')
    smoke = os.path.join(test_dir, 'smoke_all.py')
    if os.path.exists(smoke):
        return subprocess.call([sys.executable, smoke])
    print('No tests/smoke_all.py found', file=sys.stderr)
    return 1

def cmd_fmt(args):
    if not args:
        print('usage: triad fmt <file.tri>', file=sys.stderr)
        return 1
    path = args[0]
    if not os.path.exists(path):
        print(f'error: file not found: {path}', file=sys.stderr)
        return 1
    from frontend.parser_universal import parse, ParseError
    from frontend.lexer_universal import LexError
    from compiler.formatter import format_universal
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
    compiler.compile_and_run(mod)
    elapsed = time.perf_counter() - start
    print(f'\nbench: {elapsed:.4f}s')
    return 0

def cmd_init(args):
    name = args[0] if args else None
    if not name:
        print('usage: triad init <project-name>', file=sys.stderr)
        return 1
    from stdlib.registry import init_project
    m = init_project(name)
    print(f'initialized triad project: {name}')
    print(f'  {m.to_dict()}')
    return 0

def cmd_install(args):
    from stdlib.registry import install_package, install_dependencies, load_manifest, save_manifest
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

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print('TriadLang v1.0 — Universal Language')
        print()
        print('Usage:')
        print('  triad run <file.tri>              Run a .tri file')
        print('  triad check <file.tri>            Type check')
        print('  triad compile <file.tri> --native [-o out]  Compile to native binary')
        print('  triad repl                        Interactive REPL')
        print('  triad lsp                         Start LSP server')
        print('  triad init <name>                 Create new project')
        print('  triad install [source]            Install package/deps')
        print('  triad publish                     Publish to local registry')
        print('  triad list                        List installed packages')
        print('  triad doctor                      Check installation')
        print('  triad test                        Run tests')
        print('  triad bench <file.tri>            Benchmark')
        print('  triad debug <file.tri> [-b N]     Debug with breakpoints')
        print('  triad docgen <file|dir> [-f md|html]  Generate docs')
        print('  triad fmt <file.tri>              Format')
        return 0
    cmd = argv[0]
    rest = argv[1:]
    if cmd.endswith('.tri'):
        return cmd_run([cmd])
    commands = {'run': cmd_run, 'check': cmd_check, 'compile': cmd_compile, 'repl': cmd_repl, 'lsp': cmd_lsp, 'init': cmd_init, 'install': cmd_install, 'publish': cmd_publish, 'list': cmd_list, 'doctor': cmd_doctor, 'test': cmd_test, 'fmt': cmd_fmt, 'bench': cmd_bench, 'debug': cmd_debug, 'docgen': cmd_docgen}
    fn = commands.get(cmd)
    if fn is None:
        print(f'unknown command: {cmd}', file=sys.stderr)
        return 1
    return fn(rest)
if __name__ == '__main__':
    sys.exit(main())