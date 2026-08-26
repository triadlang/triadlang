from cli.tui.widgets.editor_tabs import EditorTabs
from cli.tui.widgets.error_panel import format_error, write_error
from cli.tui.widgets.file_tree import TriadFileTree
from cli.tui.widgets.symbol_outline import SymbolOutline
from cli.tui.widgets.triad_editor import TriadEditor
from cli.tui.widgets.triad_repl import TriadREPL
from cli.tui.widgets.viz_canvas import (
    bar,
    field_2d_ascii,
    field_ascii,
    format_observable,
    model_arch_ascii,
    sparkline,
)

__all__ = [
    "TriadFileTree",
    "SymbolOutline",
    "TriadEditor",
    "TriadREPL",
    "format_error",
    "write_error",
    "sparkline",
    "bar",
    "field_ascii",
    "field_2d_ascii",
    "model_arch_ascii",
    "format_observable",
    "EditorTabs",
]

from cli.tui.syntax.autocomplete import get_completions
