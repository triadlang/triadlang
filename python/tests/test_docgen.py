import pytest
import os
import tempfile
from compiler.docgen import parse_tri_docs, to_markdown, to_html, cmd_docgen

@pytest.fixture
def sample_source():
    return '// Module documentation\nimport math\n\nconst MAX = 100\n\nfn add(a, b):\n    // Add two numbers\n    return a + b\n\nclass Point:\n    x\n    y\n    fn init(self, x, y):\n        self.x = x\n        self.y = y\n    fn distance(self):\n        return (self.x ** 2 + self.y ** 2) ** 0.5\n\ntype Vec:\n    x\n    y\n    z\n'

class TestParsing:

    def test_extracts_module_doc(self, sample_source):
        doc = parse_tri_docs(sample_source)
        assert doc['module_doc'] == 'Module documentation'

    def test_extracts_imports(self, sample_source):
        doc = parse_tri_docs(sample_source)
        assert any(('import math' in i for i in doc['imports']))

    def test_extracts_constants(self, sample_source):
        doc = parse_tri_docs(sample_source)
        assert any((c['name'] == 'MAX' and c['value'] == '100' for c in doc['constants']))

    def test_extracts_functions(self, sample_source):
        doc = parse_tri_docs(sample_source)
        fns = {f['name']: f for f in doc['functions']}
        assert 'add' in fns
        assert len(fns['add']['params']) == 2
        assert fns['add']['doc'] == 'Add two numbers'

    def test_extracts_classes(self, sample_source):
        doc = parse_tri_docs(sample_source)
        classes = {c['name']: c for c in doc['classes']}
        assert 'Point' in classes
        assert len(classes['Point']['fields']) == 2
        method_names = [m['name'] for m in classes['Point']['methods']]
        assert 'init' in method_names
        assert 'distance' in method_names

    def test_extracts_types(self, sample_source):
        doc = parse_tri_docs(sample_source)
        types = {t['name']: t for t in doc['types']}
        assert 'Vec' in types
        assert len(types['Vec']['fields']) == 3

    def test_function_line_numbers(self, sample_source):
        doc = parse_tri_docs(sample_source)
        fn = [f for f in doc['functions'] if f['name'] == 'add'][0]
        assert fn['line'] == 6

    def test_empty_source(self):
        doc = parse_tri_docs('')
        assert doc['module_doc'] == ''
        assert doc['functions'] == []

    def test_function_with_defaults(self):
        src = 'fn foo(a, b=10, c="hi"):\n    return a\n'
        doc = parse_tri_docs(src)
        fn = doc['functions'][0]
        assert len(fn['params']) == 3
        assert fn['params'][1]['default'] == '10'

    def test_class_with_parent(self):
        src = 'class Vec3 < Vec:\n    z\n'
        doc = parse_tri_docs(src)
        assert doc['classes'][0]['parent'] == 'Vec'

class TestMarkdownOutput:

    def test_generates_markdown(self, sample_source):
        doc = parse_tri_docs(sample_source)
        md = to_markdown(doc, 'test.tri')
        assert '# test.tri' in md
        assert '## Functions' in md
        assert 'fn add' in md
        assert '## Classes' in md
        assert 'class Point' in md
        assert '## Types' in md
        assert 'Vec' in md

    def test_includes_doc_comments(self, sample_source):
        doc = parse_tri_docs(sample_source)
        md = to_markdown(doc)
        assert 'Add two numbers' in md

    def test_includes_constants(self, sample_source):
        doc = parse_tri_docs(sample_source)
        md = to_markdown(doc)
        assert 'MAX' in md
        assert '100' in md

class TestHtmlOutput:

    def test_generates_html(self, sample_source):
        doc = parse_tri_docs(sample_source)
        html = to_html(doc, 'test.tri')
        assert '<!DOCTYPE html>' in html
        assert 'fn add' in html
        assert 'class Point' in html

    def test_html_escapes(self):
        src = '// Doc with <special> chars\nfn foo():\n    return 1\n'
        doc = parse_tri_docs(src)
        html = to_html(doc)
        assert '&lt;' in html

class TestCLI:

    def test_docgen_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tri', delete=False) as f:
            f.write('fn hello():\n    return 1\n')
            f.flush()
            try:
                ret = cmd_docgen([f.name])
                assert ret == 0
            finally:
                os.unlink(f.name)

    def test_docgen_with_output(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tri', delete=False) as f:
            f.write('fn hello():\n    return 1\n')
            f.flush()
            outf = f.name + '.md'
            try:
                ret = cmd_docgen([f.name, '-o', outf])
                assert ret == 0
                assert os.path.exists(outf)
                with open(outf) as of:
                    content = of.read()
                assert 'hello' in content
            finally:
                os.unlink(f.name)
                if os.path.exists(outf):
                    os.unlink(outf)

    def test_docgen_html_format(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tri', delete=False) as f:
            f.write('fn hello():\n    return 1\n')
            f.flush()
            try:
                ret = cmd_docgen([f.name, '-f', 'html'])
                assert ret == 0
            finally:
                os.unlink(f.name)