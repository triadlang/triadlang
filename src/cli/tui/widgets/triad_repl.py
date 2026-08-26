from __future__ import annotations

import io as _io
import json
import os
import sys

from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Input, RichLog

_HISTORY_PATH = os.path.expanduser("~/.triad/repl_history")
_HISTORY_MAX = 500

class REPLInput(Input):

    DEFAULT_CSS = """
    REPLInput {
        dock: bottom;
        height: 3;
        border-top: solid $primary;
    }
    """

    def __init__(self, **kw):
        kw.setdefault("placeholder", "triad> ")
        super().__init__(**kw)

class TriadREPL(Vertical):

    DEFAULT_CSS = """
    TriadREPL {
        height: 1fr;
    }
    TriadREPL > RichLog {
        height: 1fr;
        scrollbar-size: 1 1;
    }
    """

    BINDINGS = [
        Binding("up", "history_up", "History Up"),
        Binding("down", "history_down", "History Down"),
        Binding("tab", "autocomplete", "Complete"),
        Binding("escape", "clear_input", "Clear"),
    ]

    def __init__(self, **kw):
        super().__init__(**kw)
        self._history: list[str] = []
        self._history_idx: int = -1
        self._multiline_buffer: list[str] = []
        self._multiline_depth: int = 0
        self._session_globals: dict | None = None
        self._load_history()

    def compose(self):
        yield RichLog(id="repl-log", highlight=True, markup=True, wrap=False)
        yield REPLInput(id="repl-input")

    def on_mount(self):
        self._setup_session()
        log = self.query_one("#repl-log", RichLog)
        log.write(Panel(
            Text.from_markup(
                "[bold cyan]⚡ TriadLang REPL[/bold cyan]\n\n"
                "Type TriadLang expressions or statements.\n"
                "[dim]:help[/dim] for commands  |  [dim]Tab[/dim] autocomplete  |  [dim]↑↓[/dim] history\n"
                "Multiline: open braces `{` auto-continue until closed."
            ),
            border_style="cyan",
            title="REPL",
        ))
        self.query_one("#repl-input", REPLInput).focus()

    def _setup_session(self):
        import math
        import random
        self._session_globals = {
            '__name__': '__main__',
            'math': math,
            'random': random,
        }

    def _load_history(self):
        try:
            if os.path.exists(_HISTORY_PATH):
                with open(_HISTORY_PATH) as f:
                    data = json.load(f)
                self._history = data.get('history', [])[_HISTORY_MAX * -1:]
        except (OSError, ValueError, KeyError):
            self._history = []

    def _save_history(self):
        try:
            os.makedirs(os.path.dirname(_HISTORY_PATH), exist_ok=True)
            with open(_HISTORY_PATH, 'w') as f:
                json.dump({'history': self._history[_HISTORY_MAX * -1:]}, f)
        except OSError:
            pass

    def on_input_submitted(self, event: Input.Submitted):
        if event.input.id != "repl-input":
            return
        code = event.value.strip()
        if not code:
            return
        self._history.append(code)
        self._history_idx = len(self._history)
        self._save_history()
        event.input.value = ""
        log = self.query_one("#repl-log", RichLog)

        log.write(Syntax(code, "triad", theme="monokai", line_numbers=False, word_wrap=False))

        if code.startswith(':'):
            self._handle_command(code, log)
            return

        open_braces = code.count('{') - code.count('}')
        self._multiline_buffer.append(code)
        self._multiline_depth += open_braces

        if self._multiline_depth > 0:
            self.query_one("#repl-input", REPLInput).placeholder = f"... ({self._multiline_depth} open) "
            return

        triad_code = '\n'.join(self._multiline_buffer)
        self._multiline_buffer = []
        self._multiline_depth = 0
        self.query_one("#repl-input", REPLInput).placeholder = "triad> "
        self._eval(triad_code, log)

    def _handle_command(self, code: str, log: RichLog):
        cmd = code[1:].strip()
        if cmd == 'help' or cmd == 'h':
            log.write(Markdown(
                "**REPL Commands:**\n\n"
                "- `:help`, `:h` — Show this help\n"
                "- `:load file.tri` — Load and run a file\n"
                "- `:type expr` — Show inferred type\n"
                "- `:ast expr` — Show generated AST\n"
                "- `:ir expr` — Show lowered IR\n"
                "- `:py code` — Run raw Python (requires --unsafe)\n"
                "- `:clear` — Clear output\n"
                "- `:vars` — Show session variables\n"
                "- `:save file.tri` — Save session to file\n"
                "- `:quit`, `:q` — Exit TUI\n\n"
                "Tab for autocomplete, ↑↓ for history"
            ))
        elif cmd in ('clear', 'cls'):
            log.clear()
        elif cmd in ('quit', 'q'):
            self.app.exit()
        elif cmd.startswith('load '):
            path = cmd[5:].strip()
            abs_path = os.path.abspath(path)
            try:
                base = os.path.dirname(abs_path)
                from runtime.security import get_policy
                policy = get_policy()
                if policy.safe:
                    policy.sandbox.resolve(abs_path, mode='read', must_exist=True)
            except Exception as e:
                log.write(Text(f"✗ Cannot load outside workspace: {path} ({e})", style="red"))
                return
            if os.path.exists(abs_path):
                with open(abs_path) as f:
                    src = f.read()
                self._eval(src, log)
                log.write(Text(f"✓ Loaded: {abs_path}", style="green"))
            else:
                log.write(Text(f"✗ File not found: {path}", style="red"))
        elif cmd.startswith('type '):
            expr = cmd[5:].strip()
            self._show_type(expr, log)
        elif cmd.startswith('ast '):
            expr = cmd[4:].strip()
            self._show_ast(expr, log)
        elif cmd.startswith('ir '):
            expr = cmd[3:].strip()
            self._show_ir(expr, log)
        elif cmd.startswith('py '):
            py_code = cmd[3:].strip()
            allow = os.environ.get('TRIAD_REPL_ALLOW_PYTHON', '').lower() in ('1', 'true', 'yes')
            if not allow:
                log.write(Text("Python execution disabled in safe REPL. set TRIAD_REPL_ALLOW_PYTHON=1 to enable.", style="red"))
                return
            self._eval_python(py_code, log)
        elif cmd == 'vars':
            if self._session_globals:
                user_vars = {k: v for k, v in self._session_globals.items()
                             if not k.startswith('__') and k not in ('math', 'random')}
                if user_vars:
                    for k, v in sorted(user_vars.items()):
                        log.write(Text(f"  {k} = {v!r}", style="cyan"))
                else:
                    log.write(Text("  (no user variables)", style="dim"))
        elif cmd.startswith('save '):
            path = cmd[5:].strip()
            log.write(Text(f"Save to file not yet implemented: {path}", style="yellow"))
        else:
            log.write(Text(f"Unknown command: :{cmd} — try :help", style="yellow"))

    def _eval(self, code: str, log: RichLog):
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

        try:
            from frontend.parser_universal import parse
            from runtime.compiler_runtime import TriadCompiler
            mod = parse(code, '<repl>')
            compiler = TriadCompiler()
            old_stdout = sys.stdout
            sys.stdout = buf = _io.StringIO()
            try:
                compiler.compile_and_run(mod)
            finally:
                stdout_text = buf.getvalue()
                sys.stdout = old_stdout
            if stdout_text.strip():
                for line in stdout_text.strip().split('\n'):
                    log.write(Text(line))
            for c in captured:
                log.write(Text(c))
        except Exception as e:
            from cli.tui.widgets.error_panel import format_error
            log.write(format_error(f"{type(e).__name__}: {e}", code))

    def _show_type(self, expr: str, log: RichLog):
        try:
            from compiler.typecheck_universal import typecheck
            from frontend.parser_universal import parse
            src = f"let __type_tmp = {expr};"
            mod = parse(src, '<repl-type>')
            typecheck(mod)
            log.write(Text("✓ Type check passed", style="green"))
        except Exception as e:
            log.write(Text(f"Type error: {e}", style="red"))

    def _show_ast(self, expr: str, log: RichLog):
        try:
            from frontend.parser_universal import parse
            src = f"let __ast_tmp = {expr};" if '=' not in expr else expr
            mod = parse(src, '<repl-ast>')
            buf = _io.StringIO()
            for stmt in mod.body:
                print(stmt, file=buf)
            log.write(Text(buf.getvalue(), style="cyan"))
        except Exception as e:
            log.write(Text(f"{type(e).__name__}: {e}", style="red"))

    def _show_ir(self, expr: str, log: RichLog):
        try:
            from compiler.emit_json import emit_json
            from compiler.lower import lower_module
            from frontend.parser_universal import parse
            src = f"let __ir_tmp = {expr};" if '=' not in expr else expr
            mod = parse(src, '<repl-ir>')
            ir = lower_module(mod)
            log.write(Text(emit_json(ir).replace("\t", "  "), style="cyan"))
        except Exception as e:
            log.write(Text(f"{type(e).__name__}: {e}", style="red"))

    def _eval_python(self, py_code: str, log: RichLog):
        old_stdout = sys.stdout
        sys.stdout = buf = _io.StringIO()
        try:
            exec(py_code, self._session_globals)
        except Exception as e:
            log.write(Text(f"Python error: {e}", style="red"))
            return
        finally:
            text = buf.getvalue()
            sys.stdout = old_stdout
        if text.strip():
            for line in text.strip().split('\n'):
                log.write(Text(line))
        else:
            log.write(Text("(no output)", style="dim"))

    def action_history_up(self):
        if self._history and self._history_idx > 0:
            self._history_idx -= 1
            self.query_one("#repl-input", REPLInput).value = self._history[self._history_idx]

    def action_history_down(self):
        if self._history_idx < len(self._history) - 1:
            self._history_idx += 1
            self.query_one("#repl-input", REPLInput).value = self._history[self._history_idx]
        else:
            self._history_idx = len(self._history)
            self.query_one("#repl-input", REPLInput).value = ""

    def action_autocomplete(self):
        inp = self.query_one("#repl-input", REPLInput)
        text = inp.value
        if not text:
            return
        from cli.tui.syntax.autocomplete import get_completions
        completions = get_completions(text)
        if completions:
            last_word = text.split('.')[-1].split()[-1]
            base = text[:-len(last_word)] if last_word else text
            inp.value = base + completions[0]

    def action_clear_input(self):
        inp = self.query_one("#repl-input", REPLInput)
        inp.value = ""

