from __future__ import annotations

import os
import sys

from rich.syntax import Syntax
from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DirectoryTree, Label, RichLog, Static

PY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

_EXAMPLES_DIR = os.path.join(PY_ROOT, "examples", "basic")

class ExamplesWidget(Static):

    DEFAULT_CSS = """
    ExamplesWidget {
        height: 1fr;
        overflow-y: auto;
    }
    #examples-main {
        layout: horizontal;
        height: 1fr;
    }
    #examples-tree-panel {
        width: 30;
        padding: 0 1;
        border-right: solid $panel;
    }
    #examples-preview {
        width: 1fr;
        padding: 0 1;
    }
    #examples-buttons {
        height: auto;
        padding: 1;
    }
    """

    BINDINGS = [
        Binding("f5", "run_example", "Run Example"),
        Binding("enter", "open_in_editor", "Open in Editor"),
    ]

    def __init__(self, **kw):
        super().__init__(**kw)
        self._current_path: str | None = None

    def compose(self):
        with Vertical():
            yield Label("[bold cyan]📚 Examples[/bold cyan]")
            with Horizontal(id="examples-main"):
                with Vertical(id="examples-tree-panel"):
                    yield Label("[bold]Files[/bold]")
                    if os.path.isdir(_EXAMPLES_DIR):
                        yield DirectoryTree(_EXAMPLES_DIR, id="examples-tree")
                    else:
                        yield Label(f"[dim]Examples dir not found: {_EXAMPLES_DIR}[/dim]")

                with Vertical(id="examples-preview"):
                    yield Label("[bold]Source[/bold]")
                    yield RichLog(id="example-source", highlight=True, markup=False, wrap=False)

            with Horizontal(id="examples-buttons"):
                yield Button("▶ Run", variant="success", id="ex-run-btn")
                yield Button("📋 Copy to Editor", variant="primary", id="ex-copy-btn")
                yield Button("📂 Open in Editor", variant="default", id="ex-open-btn")
                yield Button("💬 Run in REPL", variant="warning", id="ex-repl-btn")

    def on_directory_tree_file_selected(self, event):
        path = str(event.path)
        if path.endswith(".tri"):
            try:
                with open(path) as f:
                    source = f.read()
            except Exception as e:
                source = f"// Error loading: {e}"
            self._current_path = path
            log = self.query_one("#example-source", RichLog)
            log.clear()
            log.write(Syntax(source, "triad", theme="monokai", line_numbers=True))

    def on_button_pressed(self, event):
        bid = event.button.id
        if bid == "ex-run-btn":
            self._run_example()
        elif bid == "ex-copy-btn":
            self._copy_to_editor()
        elif bid == "ex-open-btn":
            self._open_in_editor()
        elif bid == "ex-repl-btn":
            self._run_in_repl()

    def action_run_example(self):
        self._run_example()

    def action_open_in_editor(self):
        self._open_in_editor()

    def _get_source(self) -> str:

        if self._current_path:
            try:
                with open(self._current_path) as f:
                    return f.read()
            except OSError:
                pass
        try:
            log = self.query_one("#example-source", RichLog)
            return "\n".join(str(l) for l in log.lines)
        except (LookupError, AttributeError):
            return ""

    def _run_example(self):
        source = self._get_source()
        if not source.strip():
            return
        log = self.query_one("#example-source", RichLog)
        log.clear()
        log.write(Text("▶ Running...", style="bold cyan"))

        import io as _io
        try:
            from frontend.parser_universal import parse
            from runtime.compiler_runtime import TriadCompiler
            mod = parse(source, self._current_path or "<example>")
            old = sys.stdout
            sys.stdout = buf = _io.StringIO()
            compiler = TriadCompiler()
            compiler.compile_and_run(mod)
            sys.stdout = old
            text = buf.getvalue()
            for line in text.strip().split("\n"):
                log.write(Text(line))
            log.write(Text("✓ Done", style="bold green"))
        except Exception as e:
            log.write(Text(f"✗ Error: {e}", style="red"))

    def _copy_to_editor(self):
        source = self._get_source()
        if not source.strip():
            return
        app = self.app
        if hasattr(app, "_editor_tabs"):
            tabs = app._editor_tabs()
            if tabs:
                tabs.set_source(source)
                app.action_show_tab("editor")
                app._output("✓ Example copied to editor", "green")

    def _open_in_editor(self):
        path = self._current_path
        if path and os.path.exists(path):
            app = self.app
            if hasattr(app, "_open_editor_file"):
                app._open_editor_file(path)
                app.action_show_tab("editor")

    def _run_in_repl(self):

        source = self._get_source()
        if not source.strip():
            return
        app = self.app
        if hasattr(app, "action_show_tab"):
            app.action_show_tab("repl")
        if hasattr(app, "_output"):
            app._output("▶ Running example in REPL...", "cyan")

        try:
            import io as _io

            from frontend.parser_universal import parse
            from runtime.compiler_runtime import TriadCompiler
            mod = parse(source, self._current_path or "<example>")
            old = sys.stdout
            sys.stdout = buf = _io.StringIO()
            compiler = TriadCompiler()
            compiler.compile_and_run(mod)
            sys.stdout = old
            text = buf.getvalue()
            if hasattr(app, "_output"):
                for line in text.strip().split("\n"):
                    app._output(line)
                app._output("✓ Done", "green")
        except Exception as e:
            if hasattr(app, "_error"):
                app._error(f"REPL run error: {e}")

