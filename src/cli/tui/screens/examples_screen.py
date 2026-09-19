from __future__ import annotations

import os
import sys

from rich.syntax import Syntax
from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Input, Label, RichLog

PY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

class ExamplesScreen(Screen):
    BINDINGS = [
        Binding("enter", "run_selected", "Run"),
        Binding("e", "edit_selected", "Edit"),
        Binding("slash", "focus_search", "Search"),
    ]

    def compose(self):
        with Vertical():
            yield Label(" [bold cyan]Examples Browser[/bold cyan]  —  browse and run examples")
            with Horizontal():
                yield Input(placeholder="Search examples...", id="examples-search")
                yield Button("Clear", variant="default", id="ex-clear-btn")
            yield DataTable(id="examples-table", cursor_type="row")
            with Horizontal():
                with Vertical(classes="dsl-col"):
                    yield Label(" [bold]Preview:[/bold]")
                    yield RichLog(id="example-preview", highlight=True, markup=True)
                with Horizontal():
                    yield Button("Run", variant="success", id="ex-run-btn")
                    yield Button("Edit", variant="primary", id="ex-edit-btn")

    def on_mount(self):
        self._all_examples = []
        self._filtered = []
        table = self.query_one("#examples-table", DataTable)
        table.add_columns("Category", "File", "Size")
        self._load_examples()

    def _load_examples(self):
        examples_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))), "examples")
        if not os.path.isdir(examples_dir):
            return
        for cat in sorted(os.listdir(examples_dir)):
            cat_dir = os.path.join(examples_dir, cat)
            if not os.path.isdir(cat_dir) or cat.startswith('_'):
                continue
            for fname in sorted(os.listdir(cat_dir)):
                if not fname.lower().endswith('.tri') or fname.startswith(('.', '__')):
                    continue
                fpath = os.path.join(cat_dir, fname)
                try:
                    size = os.path.getsize(fpath)
                except OSError:
                    size = 0
                self._all_examples.append({"cat": cat, "file": fname, "path": fpath, "size": size})
        self._filtered = list(self._all_examples)
        self._populate_table()

    def _populate_table(self):
        table = self.query_one("#examples-table", DataTable)
        table.clear()
        for ex in self._filtered:
            table.add_row(ex["cat"], ex["file"], f"{ex['size']}B")

    def on_input_changed(self, event: Input.Changed):
        if event.input.id != "examples-search":
            return
        query = event.value.strip().lower()
        if not query:
            self._filtered = list(self._all_examples)
        else:
            self._filtered = [
                ex for ex in self._all_examples
                if query in ex["cat"].lower() or query in ex["file"].lower()
            ]
        self._populate_table()

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "ex-clear-btn":
            self.query_one("#examples-search", Input).value = ""
            return
        table = self.query_one("#examples-table", DataTable)
        if table.cursor_row is not None and table.cursor_row < len(self._filtered):
            if event.button.id == "ex-run-btn":
                self._run_example(table.cursor_row)
            elif event.button.id == "ex-edit-btn":
                self._edit_example(table.cursor_row)

    def on_data_table_row_selected(self, event: DataTable.RowSelected):
        if event.cursor_row < len(self._filtered):
            self._show_preview(event.cursor_row)

    def action_run_selected(self):
        table = self.query_one("#examples-table", DataTable)
        if table.cursor_row is not None and table.cursor_row < len(self._filtered):
            self._run_example(table.cursor_row)

    def action_edit_selected(self):
        table = self.query_one("#examples-table", DataTable)
        if table.cursor_row is not None and table.cursor_row < len(self._filtered):
            self._edit_example(table.cursor_row)

    def action_focus_search(self):
        self.query_one("#examples-search", Input).focus()

    def _show_preview(self, idx: int):
        if idx >= len(self._filtered):
            return
        ex = self._filtered[idx]
        preview = self.query_one("#example-preview", RichLog)
        preview.clear()
        try:
            with open(ex["path"]) as f:
                code = f.read()
            syntax = Syntax(code[:2000], "tri", theme="monokai", line_numbers=True)
            preview.write(syntax)
        except (OSError, ValueError):
            preview.write(Text(f"Cannot preview {ex['path']}", style="red"))

    def _run_example(self, idx: int):
        if idx >= len(self._filtered):
            return
        ex = self._filtered[idx]
        output = self.app.query_one("#output-log", RichLog)
        error_log = self.app.query_one("#error-log", RichLog)
        error_log.clear()
        output.write(Text(f"--- Running {ex['cat']}/{ex['file']} ---", style="bold cyan"))

        import io

        from cli.tui.widgets.error_panel import write_error
        from frontend.parser_universal import parse
        from runtime.compiler_runtime import TriadCompiler
        try:
            with open(ex["path"]) as f:
                src = f.read()
            mod = parse(src, ex["path"])
            compiler = TriadCompiler()
            old = sys.stdout
            sys.stdout = buf = io.StringIO()
            try:
                compiler.compile_and_run(mod)
            finally:
                captured = buf.getvalue()
                sys.stdout = old
            if captured.strip():
                for line in captured.strip().split('\n'):
                    output.write(Text(line))
            output.write(Text("OK", style="bold green"))
        except Exception as e:
            write_error(error_log, f"{type(e).__name__}: {e}", src, ex["path"])

    def _edit_example(self, idx: int):
        if idx >= len(self._filtered):
            return
        ex = self._filtered[idx]
        try:
            self.app._load_file(ex["path"])
            self.app.action_show_tab("editor")
        except (OSError, LookupError, AttributeError):
            pass

