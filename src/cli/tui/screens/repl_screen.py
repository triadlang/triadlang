from __future__ import annotations

import io
import os
import sys

from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Input, RichLog

PY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

class REPLScreen(Screen):
    BINDINGS = [
        Binding("up", "history_up", "History Up"),
        Binding("down", "history_down", "History Down"),
        Binding("tab", "autocomplete", "Complete"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._history = []
        self._history_idx = -1
        self._session_globals = None
        self._multiline_buffer = []
        self._multiline_depth = 0

    def compose(self):
        with Vertical():
            yield RichLog(id="repl-output", highlight=True, markup=True)
            yield Input(placeholder="triad> ", id="repl-input")

    def on_mount(self):
        self._setup_session()
        output = self.query_one("#repl-output", RichLog)
        output.write(Panel(
            Text.from_markup(
                "[bold cyan]TriadLang REPL[/bold cyan]\n"
                "Type expressions or statements. [dim]:help[/dim] for commands.\n"
                "Use Tab for autocomplete, Up/Down for history."
            ),
            border_style="cyan"
        ))
        self.query_one("#repl-input", Input).focus()

    def _setup_session(self):
        import math
        import random
        self._session_globals = {
            '__name__': '__main__',
            'math': math,
            'random': random,
        }

    def on_input_submitted(self, event: Input.Submitted):
        if event.input.id != "repl-input":
            return
        code = event.value.strip()
        if not code:
            return
        self._history.append(code)
        self._history_idx = len(self._history)
        event.input.value = ""
        output = self.query_one("#repl-output", RichLog)
        output.write(Text(f"triad> {code}", style="bold cyan"))

        if code.startswith(':'):
            self._multiline_buffer = []
            self._multiline_depth = 0
            self.query_one("#repl-input", Input).placeholder = "triad> "
            self._handle_command(code, output)
            return

        open_braces = self._brace_delta(code)
        self._multiline_buffer.append(code)
        self._multiline_depth = max(0, self._multiline_depth + open_braces)

        if self._multiline_depth > 0:
            self.query_one("#repl-input", Input).placeholder = "... "
            return

        triad_code = '\n'.join(self._multiline_buffer)
        self._multiline_buffer = []
        self._multiline_depth = 0
        self.query_one("#repl-input", Input).placeholder = "triad> "
        self._eval(triad_code, output)

    @staticmethod
    def _brace_delta(code: str) -> int:
        depth = 0
        i, n = 0, len(code)
        in_str = None
        while i < n:
            c = code[i]
            if in_str:
                if c == '\\':
                    i += 2
                    continue
                if c == in_str:
                    in_str = None
                i += 1
                continue
            if c in ('"', "'"):
                in_str = c
                i += 1
                continue
            if c == '/' and i + 1 < n and code[i + 1] == '/':
                break
            if c == '#':
                break
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
            i += 1
        return depth

    def _handle_command(self, code: str, output: RichLog):
        cmd = code[1:].strip()
        if cmd == 'help':
            output.write(Markdown(
                "**REPL Commands:**\n"
                "- `:help` — Show this help\n"
                "- `:load file.tri` — Load and run a file\n"
                "- `:clear` — Clear output\n"
                "- `:quit` — Exit TUI\n\n"
                "Tab for autocomplete, Up/Down for history"
            ))
        elif cmd == 'clear':
            output.clear()
        elif cmd == 'quit':
            self.app.exit()
        elif cmd.startswith('load '):
            path = cmd[5:].strip()
            if os.path.exists(path):
                with open(path) as f:
                    src = f.read()
                self._eval(src, output)
                output.write(Text(f"Loaded: {path}", style="green"))
            else:
                output.write(Text(f"File not found: {path}", style="red"))

    def _eval(self, code: str, output: RichLog):
        captured = []
        def _capture(*args, **kwargs):
            parts = []
            for a in args:
                if isinstance(a, bool):
                    parts.append('true' if a else 'false')
                elif a is None:
                    parts.append('none')
                else:
                    parts.append(str(a))
            captured.append(' '.join(parts))

        self._session_globals['print'] = _capture
        self._session_globals['_tri_str'] = lambda x: 'none' if x is None else ('true' if x is True else ('false' if x is False else str(x)))

        try:
            from frontend.parser_universal import parse
            from runtime.compiler_runtime import TriadCompiler
            src = code
            mod = parse(src, '<repl>')
            compiler = TriadCompiler()
            old_stdout = sys.stdout
            sys.stdout = buf = io.StringIO()
            try:
                compiler.compile_and_run(mod)
            finally:
                stdout_text = buf.getvalue()
                sys.stdout = old_stdout
            if stdout_text.strip():
                for line in stdout_text.strip().split('\n'):
                    output.write(Text(line))
            for c in captured:
                output.write(Text(c))
        except Exception as e:
            from cli.tui.widgets.error_panel import format_error
            output.write(format_error(f"{type(e).__name__}: {e}", code))

    def action_history_up(self):
        if self._history and self._history_idx > 0:
            self._history_idx -= 1
            self.query_one("#repl-input", Input).value = self._history[self._history_idx]

    def action_history_down(self):
        if self._history_idx < len(self._history) - 1:
            self._history_idx += 1
            self.query_one("#repl-input", Input).value = self._history[self._history_idx]
        else:
            self._history_idx = len(self._history)
            self.query_one("#repl-input", Input).value = ""

    def action_autocomplete(self):
        inp = self.query_one("#repl-input", Input)
        text = inp.value
        if not text:
            return
        from cli.tui.syntax.autocomplete import get_completions
        completions = get_completions(text)
        if completions:
            last_word = text.split('.')[-1].split()[-1]
            base = text[:-len(last_word)] if last_word else text
            inp.value = base + completions[0]

