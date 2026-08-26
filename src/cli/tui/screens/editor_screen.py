from __future__ import annotations

from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Label, RichLog

from cli.tui.widgets.triad_editor import TriadEditor


class EditorScreen(Screen):
    BINDINGS = [
        Binding("f5", "run_file", "Run"),
        Binding("f9", "run_selection", "Run Sel"),
        Binding("ctrl+s", "save_file", "Save"),
        Binding("ctrl+t", "type_check", "TypeCheck"),
        Binding("shift+f", "format_code", "Format"),
        Binding("ctrl+slash", "toggle_comment", "Comment"),
    ]

    def compose(self):
        with Vertical():
            yield TriadEditor(id="triad-editor")
            with Horizontal(id="editor-status-bar"):
                yield Label("", id="editor-pos")
                yield Label("", id="editor-file-info")

    def on_mount(self):
        editor = self.query_one("#triad-editor", TriadEditor)
        editor.focus()
        self._update_status()

    def _update_status(self):
        try:
            editor = self.query_one("#triad-editor", TriadEditor)
            row, col = editor.get_line_col()
            self.query_one("#editor-pos", Label).update(f"Ln {row+1}, Col {col+1}")
        except (LookupError, AttributeError):
            pass

    def get_source(self) -> str:
        return self.query_one("#triad-editor", TriadEditor).text

    def load_source(self, source: str):
        self.query_one("#triad-editor", TriadEditor).load_text(source)

    def action_run_file(self):
        app = self.app
        source = self.get_source()
        if not source.strip():
            return
        self._auto_save()
        output = app.query_one("#output-log", RichLog)
        error_log = app.query_one("#error-log", RichLog)
        error_log.clear()
        output.write(Text(f"--- Running {app.file_path or 'untitled'} ---", style="bold cyan"))
        app._run_source(source, output, error_log)

    def action_run_selection(self):
        editor = self.query_one("#triad-editor", TriadEditor)
        try:
            sel = editor.selected_text
        except (LookupError, AttributeError):
            sel = editor.text
        if not sel.strip():
            return
        output = self.app.query_one("#output-log", RichLog)
        error_log = self.app.query_one("#error-log", RichLog)
        error_log.clear()
        output.write(Text("--- Running selection ---", style="bold cyan"))
        self.app._run_source(sel, output, error_log)

    def action_save_file(self):
        self._auto_save()

    def _auto_save(self):
        path = self.app.file_path
        if not path:
            return
        try:
            with open(path, 'w') as f:
                f.write(self.get_source())
        except OSError:
            pass

    def action_type_check(self):
        source = self.get_source()
        if not source.strip():
            return
        output = self.app.query_one("#output-log")
        try:
            from frontend.parser_universal import parse
            mod = parse(source, self.app.file_path or '<tui>')
            from compiler.typecheck_universal import typecheck
            typecheck(mod)
            from rich.text import Text
            output.write(Text("Type check: OK", style="bold green"))
        except Exception as e:
            self.app._show_error(f"{type(e).__name__}: {e}")

    def action_format_code(self):
        source = self.get_source()
        if not source.strip():
            return
        try:
            from compiler.formatter import format_universal
            from frontend.parser_universal import parse
            mod = parse(source, self.app.file_path or '<tui>')
            formatted = format_universal(mod)
            self.query_one("#triad-editor", TriadEditor).load_text(formatted)
        except Exception as e:
            self.app._show_error(f"Format error: {e}")

    def action_toggle_comment(self):
        self.query_one("#triad-editor", TriadEditor).action_toggle_comment()
