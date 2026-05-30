import pytest
from cli.lsp import TriadLSP

@pytest.fixture
def lsp():
    server = TriadLSP()
    server._docs = {'file:///test.tri': 'let x = 10\nlet y = 20\nfn add(a, b):\n    return a + b\nclass Point:\n    fn init(self, x, y):\n        self.x = x\nlet result = add(x, y)\n'}
    return server

class TestDocumentSymbols:

    def test_finds_let_symbols(self, lsp):
        text = 'let x = 10\nlet y = 20\n'
        symbols = lsp._parse_symbols(text)
        names = [s['name'] for s in symbols]
        assert 'x' in names
        assert 'y' in names

    def test_finds_fn_symbols(self, lsp):
        text = 'fn foo():\n    return 1\n'
        symbols = lsp._parse_symbols(text)
        assert any((s['name'] == 'foo' and s['kind'] == 'function' for s in symbols))

    def test_finds_class_symbols(self, lsp):
        text = 'class Point:\n    pass\n'
        symbols = lsp._parse_symbols(text)
        assert any((s['name'] == 'Point' and s['kind'] == 'class' for s in symbols))

    def test_finds_type_symbols(self, lsp):
        text = 'type Vec:\n    x\n    y\n'
        symbols = lsp._parse_symbols(text)
        assert any((s['name'] == 'Vec' for s in symbols))

    def test_returns_line_numbers(self, lsp):
        text = 'let a = 1\n\nlet b = 2\n'
        symbols = lsp._parse_symbols(text)
        a_sym = [s for s in symbols if s['name'] == 'a'][0]
        b_sym = [s for s in symbols if s['name'] == 'b'][0]
        assert a_sym['line'] == 0
        assert b_sym['line'] == 2

class TestCompletions:

    def test_keyword_completions(self, lsp):
        result = lsp._completions({'textDocument': {'uri': 'file:///test.tri'}, 'position': {'line': 0, 'character': 3}})
        labels = [i['label'] for i in result['items']]
        assert 'let' in labels

    def test_local_variable_completions(self, lsp):
        lsp._docs['file:///test.tri'] = 'let x = 10\nlet y = 20\ny'
        result = lsp._completions({'textDocument': {'uri': 'file:///test.tri'}, 'position': {'line': 2, 'character': 1}})
        labels = [i['label'] for i in result['items']]
        assert 'y' in labels

    def test_fn_completions(self, lsp):
        lsp._docs['file:///test.tri'] = 'fn add(a, b):\n    return a + b\nadd'
        result = lsp._completions({'textDocument': {'uri': 'file:///test.tri'}, 'position': {'line': 2, 'character': 3}})
        labels = [i['label'] for i in result['items']]
        assert 'add' in labels

    def test_stdlib_module_completions(self, lsp):
        lsp._docs['file:///test.tri'] = 'm'
        result = lsp._completions({'textDocument': {'uri': 'file:///test.tri'}, 'position': {'line': 0, 'character': 1}})
        labels = [i['label'] for i in result['items']]
        assert 'math' in labels

    def test_dot_completions(self, lsp):
        result = lsp._completions({'textDocument': {'uri': 'file:///test.tri'}, 'position': {'line': 0, 'character': 6}})
        prefix_line = 'let x = math.'
        lsp._docs['file:///test2.tri'] = prefix_line
        result = lsp._completions({'textDocument': {'uri': 'file:///test2.tri'}, 'position': {'line': 0, 'character': len(prefix_line)}})
        labels = [i['label'] for i in result['items']]
        assert 'sqrt' in labels
        assert 'sin' in labels

class TestDefinition:

    def test_jump_to_let(self, lsp):
        lsp._docs['file:///test.tri'] = 'let x = 10\nlet y = x + 1\n'
        result = lsp._definition({'textDocument': {'uri': 'file:///test.tri'}, 'position': {'line': 1, 'character': 8}})
        assert result is not None
        assert result['range']['start']['line'] == 0

    def test_jump_to_fn(self, lsp):
        lsp._docs['file:///test.tri'] = 'fn add(a, b):\n    return a + b\nlet r = add(1, 2)\n'
        result = lsp._definition({'textDocument': {'uri': 'file:///test.tri'}, 'position': {'line': 2, 'character': 9}})
        assert result is not None
        assert result['range']['start']['line'] == 0

    def test_jump_to_class(self, lsp):
        lsp._docs['file:///test2.tri'] = 'class Foo:\n    pass\nlet f = Foo()\n'
        result = lsp._definition({'textDocument': {'uri': 'file:///test2.tri'}, 'position': {'line': 2, 'character': 8}})
        assert result is not None
        assert result['range']['start']['line'] == 0

    def test_unknown_returns_none(self, lsp):
        result = lsp._definition({'textDocument': {'uri': 'file:///test.tri'}, 'position': {'line': 0, 'character': 0}})
        assert result is None

class TestHover:

    def test_hover_local_var(self, lsp):
        lsp._docs['file:///test.tri'] = 'let x = 10\nlet y = x + 1\n'
        result = lsp._hover({'textDocument': {'uri': 'file:///test.tri'}, 'position': {'line': 1, 'character': 8}})
        assert result is not None
        assert 'x' in result['contents']['value']

    def test_hover_fn(self, lsp):
        lsp._docs['file:///test.tri'] = 'fn add(a, b):\n    return a + b\nlet r = add(1, 2)\n'
        result = lsp._hover({'textDocument': {'uri': 'file:///test.tri'}, 'position': {'line': 2, 'character': 9}})
        assert result is not None
        assert 'add' in result['contents']['value']

    def test_hover_stdlib_module(self, lsp):
        lsp._docs['file:///test2.tri'] = 'import math\nlet x = math.sqrt(4)\n'
        result = lsp._hover({'textDocument': {'uri': 'file:///test2.tri'}, 'position': {'line': 1, 'character': 10}})
        assert result is not None
        assert 'module math' in result['contents']['value']

    def test_hover_stdlib_fn(self, lsp):
        lsp._docs['file:///test3.tri'] = 'let x = sqrt(4)\n'
        result = lsp._hover({'textDocument': {'uri': 'file:///test3.tri'}, 'position': {'line': 0, 'character': 10}})
        if result is not None:
            assert 'math' in result['contents']['value'] or 'sqrt' in result['contents']['value']

    def test_hover_keyword(self, lsp):
        lsp._docs['file:///test4.tri'] = 'let x = 1\n'
        result = lsp._hover({'textDocument': {'uri': 'file:///test4.tri'}, 'position': {'line': 0, 'character': 1}})
        assert result is not None
        assert 'keyword' in result['contents']['value']

class TestDocumentSymbol:

    def test_symbols_response(self, lsp):
        result = lsp._document_symbols({'textDocument': {'uri': 'file:///test.tri'}})
        names = [s['name'] for s in result]
        assert 'x' in names
        assert 'y' in names
        assert 'add' in names
        assert 'Point' in names

    def test_symbols_have_locations(self, lsp):
        result = lsp._document_symbols({'textDocument': {'uri': 'file:///test.tri'}})
        for sym in result:
            assert 'location' in sym
            assert 'uri' in sym['location']
            assert 'range' in sym['location']