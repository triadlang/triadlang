from dataclasses import dataclass

import pytest

from frontend.ast_nodes import Pos
from runtime.compiler_runtime import CompileError, TriadCompiler


@dataclass
class UnknownStatement:
    pos: Pos


def test_unknown_statement_is_never_compiled_as_silent_noop():
    compiler = TriadCompiler()
    statement = UnknownStatement(pos=Pos(3, 7, 'bad.tri'))

    with pytest.raises(CompileError, match='unsupported statement UnknownStatement'):
        compiler._compile_stmt(statement)
