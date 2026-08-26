from pathlib import Path

from compiler.typecheck_universal import typecheck
from frontend.parser_universal import parse


def test_all_examples_parse_and_typechecker_import_is_frontend_only():
    examples = sorted(Path('examples').rglob('*.tri'))
    assert examples
    for path in examples:
        parse(path.read_text(encoding='utf-8'), str(path))


def test_basic_program_typechecks():
    module = parse('let answer: int = 42;')
    typecheck(module)
