import pytest
import os
import tempfile
from runtime.debugger import TriadDebugger

@pytest.fixture
def dbg():
    return TriadDebugger()

class TestBreakpoints:

    def test_set_breakpoints(self, dbg):
        dbg.set_breakpoints([5, 10, 15])
        assert dbg._breakpoints == {5, 10, 15}

    def test_add_breakpoint(self, dbg):
        dbg.add_breakpoint(7)
        assert 7 in dbg._breakpoints

    def test_remove_breakpoint(self, dbg):
        dbg.set_breakpoints([5, 10])
        dbg.remove_breakpoint(5)
        assert dbg._breakpoints == {10}

    def test_clear_breakpoints(self, dbg):
        dbg.set_breakpoints([1, 2, 3])
        dbg.clear_breakpoints()
        assert dbg._breakpoints == set()

class TestHookInjection:

    def test_injects_at_breakpoint_lines(self, dbg):
        dbg.set_breakpoints([2])
        source = 'line 1\nline 2\nline 3\n'
        result = dbg.inject_hooks(source, 'test.tri')
        assert '_triad_dbg._bp_hook(2,' in result

    def test_no_inject_without_breakpoints(self, dbg):
        source = 'line 1\nline 2\nline 3\n'
        result = dbg.inject_hooks(source, 'test.tri')
        assert '_triad_dbg._bp_hook' not in result

    def test_inject_preserves_indent(self, dbg):
        dbg.set_breakpoints([2])
        source = 'def foo():\n    x = 1\n    return x\n'
        result = dbg.inject_hooks(source, 'test.tri')
        lines = result.split('\n')
        assert any(('_triad_dbg._bp_hook(2' in l and l.startswith('    ') for l in lines))

    def test_multiple_breakpoints(self, dbg):
        dbg.set_breakpoints([1, 3])
        source = 'a\nb\nc\n'
        result = dbg.inject_hooks(source, 'test.tri')
        assert '_triad_dbg._bp_hook(1,' in result
        assert '_triad_dbg._bp_hook(3,' in result

class TestHookExecution:

    def test_hook_records_hit(self, dbg):
        dbg._source_lines = ['x = 1', 'y = 2', 'z = 3']
        dbg._bp_hook(1, {'x': 1})
        assert dbg._hit_count[1] == 1

    def test_hook_does_not_break_without_bp(self, dbg):
        dbg._source_lines = ['x = 1']
        dbg._breakpoints = {5}
        dbg._step_mode = False
        dbg._bp_hook(1, {'x': 1})
        assert dbg._hit_count.get(1, 0) == 1
        assert dbg._current_frame is None

class TestRunSource:

    def test_run_simple_program(self, dbg):
        source = 'let x = 10\nlet y = 20\n'
        dbg.run_source(source, '<test>')
        assert dbg._hit_count == {}

    def test_run_with_breakpoint_hits(self, dbg):
        source = 'let x = 10\nlet y = 20\nlet z = x + y\n'
        dbg.set_breakpoints([2])
        dbg._step_mode = False
        dbg.run_source(source, '<test>')
        assert 2 in dbg._hit_count

    def test_run_file(self, dbg):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tri', delete=False) as f:
            f.write('let a = 42\nlet b = a * 2\n')
            f.flush()
            try:
                dbg.run_file(f.name)
            finally:
                os.unlink(f.name)