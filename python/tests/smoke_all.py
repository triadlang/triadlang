import sys, os, importlib, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_MODULES = ['tests.test_lexer', 'tests.test_parser', 'tests.test_typecheck', 'tests.test_ir', 'tests.test_interpreter', 'tests.test_imports', 'tests.test_stdlib', 'tests.test_repl', 'tests.test_cli', 'tests.test_examples_basic', 'tests.test_examples_triad', 'tests.test_ast', 'tests.test_solver_audit']

def run_all():
    total = 0
    passed = 0
    failed = 0
    errors = []
    for mod_name in TEST_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            print(f'\n  ERROR importing {mod_name}: {e}')
            failed += 1
            total += 1
            errors.append((mod_name, 'import', str(e)))
            continue
        tests = [(n, getattr(mod, n)) for n in sorted(dir(mod)) if n.startswith('test_')]
        if not tests:
            continue
        print(f'\n--- {mod_name} ({len(tests)} tests) ---')
        for name, fn in tests:
            total += 1
            try:
                fn()
                passed += 1
                print(f'  PASS {name}')
            except Exception as e:
                failed += 1
                print(f'  FAIL {name}: {e}')
                errors.append((mod_name, name, str(e)))
    print(f"\n{'=' * 50}")
    print(f'Total: {total}  Passed: {passed}  Failed: {failed}')
    if errors:
        print(f'\nFailures:')
        for mod, test, err in errors:
            print(f'  {mod}::{test} — {err}')
    print(f"{'=' * 50}")
    return 0 if failed == 0 else 1
if __name__ == '__main__':
    sys.exit(run_all())