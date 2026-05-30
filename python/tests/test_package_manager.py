import json
import os
import shutil
import tempfile
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stdlib.registry import PackageManifest, init_project, load_manifest, save_manifest, install_package, install_dependencies, publish_package, list_installed, resolve_modules_dir, MANIFEST_FILE, MODULES_DIR

class TempProject:

    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix='triad_pkg_test_')

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def path(self, *parts):
        return os.path.join(self.dir, *parts)

    def write_file(self, name, content):
        p = self.path(name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w') as f:
            f.write(content)

def test_manifest_roundtrip():
    m = PackageManifest('mypkg', '1.0.0', 'a test package', {'dep1': 'git+https://example.com/dep1'}, author='test', license='Apache-2.0')
    d = m.to_dict()
    assert d['name'] == 'mypkg'
    assert d['version'] == '1.0.0'
    assert d['dependencies']['dep1'] == 'git+https://example.com/dep1'
    m2 = PackageManifest.from_dict(d)
    assert m2.name == 'mypkg'
    assert m2.version == '1.0.0'
    assert m2.author == 'test'

def test_init_project():
    proj = TempProject()
    try:
        m = init_project('testproj', proj.dir)
        assert os.path.exists(proj.path(MANIFEST_FILE))
        assert os.path.isdir(proj.path(MODULES_DIR))
        m2 = load_manifest(proj.dir)
        assert m2 is not None
        assert m2.name == 'testproj'
    finally:
        proj.cleanup()

def test_install_path_source():
    proj = TempProject()
    try:
        src_pkg = TempProject()
        src_pkg.write_file('math_utils.tri', 'fn add(a, b) { return a + b }\nfn mul(a, b) { return a * b }\n')
        src_pkg.write_file(MANIFEST_FILE, json.dumps({'name': 'math_utils', 'version': '0.2.0', 'description': 'math helpers'}))
        init_project('consumer', proj.dir)
        name = install_package(f'path:{src_pkg.dir}', proj.dir)
        assert name == 'math_utils'
        assert os.path.isdir(proj.path(MODULES_DIR, 'math_utils'))
        assert os.path.islink(proj.path(MODULES_DIR, 'math_utils'))
        pkgs = list_installed(proj.dir)
        assert len(pkgs) == 1
        assert pkgs[0]['name'] == 'math_utils'
        assert pkgs[0]['version'] == '0.2.0'
        assert pkgs[0]['symlink'] is True
        src_pkg.cleanup()
    finally:
        proj.cleanup()

def test_install_with_name_override():
    proj = TempProject()
    try:
        src_pkg = TempProject()
        src_pkg.write_file('util.tri', 'fn id(x) { return x }\n')
        init_project('myapp', proj.dir)
        name = install_package(f'path:{src_pkg.dir}', proj.dir, name='myutil')
        assert name == 'myutil'
        assert os.path.isdir(proj.path(MODULES_DIR, 'myutil'))
        src_pkg.cleanup()
    finally:
        proj.cleanup()

def test_publish_and_install_registry():
    src = TempProject()
    reg = tempfile.mkdtemp(prefix='triad_reg_test_')
    consumer = TempProject()
    try:
        src.write_file('stats.tri', 'fn mean(arr) {\n  let s = 0\n  for x in arr { s = s + x }\n  return s / len(arr)\n}\n')
        init_project('stats_lib', src.dir)
        m = load_manifest(src.dir)
        m.version = '1.0.0'
        m.description = 'statistics library'
        save_manifest(m, src.dir)
        publish_path = publish_package(src.dir, registry_dir=reg)
        assert os.path.isdir(os.path.join(reg, 'stats_lib', '1.0.0'))
        assert os.path.exists(os.path.join(reg, 'stats_lib', '1.0.0', 'stats.tri'))
        init_project('data_app', consumer.dir)
        install_package('triad://stats_lib@1.0.0', consumer.dir, registry_dir=reg)
        assert os.path.isdir(consumer.path(MODULES_DIR, 'stats_lib'))
        assert os.path.exists(consumer.path(MODULES_DIR, 'stats_lib', 'stats.tri'))
        pkgs = list_installed(consumer.dir)
        assert any((p['name'] == 'stats_lib' for p in pkgs))
    finally:
        src.cleanup()
        consumer.cleanup()
        shutil.rmtree(reg, ignore_errors=True)

def test_install_dependencies():
    proj = TempProject()
    try:
        src1 = TempProject()
        src1.write_file('mod.tri', 'let X = 42\n')
        init_project('dep1', src1.dir)
        src2 = TempProject()
        src2.write_file('mod.tri', 'let Y = 99\n')
        init_project('dep2', src2.dir)
        init_project('app', proj.dir)
        m = load_manifest(proj.dir)
        m.dependencies = {'dep1': f'path:{src1.dir}', 'dep2': f'path:{src2.dir}'}
        save_manifest(m, proj.dir)
        installed = install_dependencies(proj.dir)
        assert set(installed) == {'dep1', 'dep2'}
        assert os.path.exists(proj.path(MODULES_DIR, 'dep1', 'mod.tri'))
        assert os.path.exists(proj.path(MODULES_DIR, 'dep2', 'mod.tri'))
        src1.cleanup()
        src2.cleanup()
    finally:
        proj.cleanup()

def test_list_empty():
    proj = TempProject()
    try:
        pkgs = list_installed(proj.dir)
        assert pkgs == []
    finally:
        proj.cleanup()

def test_resolve_modules_dir():
    d = resolve_modules_dir('/tmp/myproject')
    assert d == '/tmp/myproject/triad_modules'

def test_import_from_installed_package():
    proj = TempProject()
    try:
        src_pkg = TempProject()
        src_pkg.write_file('__init__.tri', 'fn greet(name) { return "hello " + name }\nlet VERSION = "1.0.0"\n')
        init_project('myapp', proj.dir)
        install_package(f'path:{src_pkg.dir}', proj.dir, name='greeter')
        from frontend.parser_universal import parse
        from runtime.compiler_runtime import TriadCompiler
        old_cwd = os.getcwd()
        os.chdir(proj.dir)
        try:
            code = 'import greeter\nlet v = greeter.VERSION\nprint(v)\n'
            mod = parse(code, 'test.tri')
            compiler = TriadCompiler()
            import io
            from contextlib import redirect_stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                compiler.compile_and_run(mod)
            output = buf.getvalue().strip()
            assert output == '1.0.0'
        finally:
            os.chdir(old_cwd)
        src_pkg.cleanup()
    finally:
        proj.cleanup()

def test_import_submodule_from_package():
    proj = TempProject()
    try:
        src_pkg = TempProject()
        src_pkg.write_file('math.tri', 'fn double(x) { return x * 2 }\n')
        init_project('mypkg', src_pkg.dir)
        init_project('app', proj.dir)
        install_package(f'path:{src_pkg.dir}', proj.dir, name='mypkg')
        from frontend.parser_universal import parse
        from runtime.compiler_runtime import TriadCompiler
        old_cwd = os.getcwd()
        os.chdir(proj.dir)
        try:
            code = 'from mypkg.math import double\nprint(double(21))\n'
            mod = parse(code, 'test.tri')
            compiler = TriadCompiler()
            import io
            from contextlib import redirect_stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                compiler.compile_and_run(mod)
            output = buf.getvalue().strip()
            assert output == '42'
        finally:
            os.chdir(old_cwd)
        src_pkg.cleanup()
    finally:
        proj.cleanup()

def test_cli_init():
    proj = TempProject()
    try:
        from cli.main import cmd_init
        old_cwd = os.getcwd()
        os.chdir(proj.dir)
        try:
            ret = cmd_init(['myproject'])
            assert ret == 0
            assert os.path.exists(os.path.join(proj.dir, MANIFEST_FILE))
        finally:
            os.chdir(old_cwd)
    finally:
        proj.cleanup()

def test_cli_list():
    proj = TempProject()
    try:
        src_pkg = TempProject()
        src_pkg.write_file('mod.tri', 'let X = 1\n')
        init_project('pkga', src_pkg.dir)
        init_project('app', proj.dir)
        install_package(f'path:{src_pkg.dir}', proj.dir, name='pkga')
        from cli.main import cmd_list
        old_cwd = os.getcwd()
        os.chdir(proj.dir)
        try:
            ret = cmd_list([])
            assert ret == 0
        finally:
            os.chdir(old_cwd)
        src_pkg.cleanup()
    finally:
        proj.cleanup()
if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f'  PASS  {t.__name__}')
            passed += 1
        except Exception as e:
            print(f'  FAIL  {t.__name__}: {e}')
            failed += 1
    print(f'\n{passed} passed, {failed} failed, {passed + failed} total')
    sys.exit(1 if failed else 0)