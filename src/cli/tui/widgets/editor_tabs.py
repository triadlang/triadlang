from __future__ import annotations

import os

from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Static, TextArea

try:
    from pygments.lexers import get_lexer_by_name
    from pygments.util import ClassNotFound
    get_lexer_by_name("triad")
except (ImportError, AttributeError, ClassNotFound):
    from pygments.lexers import _mapping

    from cli.tui.syntax.triad_lexer import TriadLexer
    _mapping.TRIAD_LEXER = TriadLexer

class TabSwitched(Message):

    def __init__(self, path: str | None, source: str):
        super().__init__()
        self.path = path
        self.source = source

class EditorTabs(Vertical):

    DEFAULT_CSS = """
    EditorTabs {
        height: 1fr;
    }
    #editor-tab-bar {
        height: auto;
        dock: top;
        padding: 0 1;
        border-bottom: solid $primary;
    }
    #editor-tab-bar Button {
        margin: 0 1 0 0;
        min-width: 12;
    }
    #editor-body {
        height: 1fr;
    }
    #editor-textarea {
        height: 1fr;
    }
    #editor-status-line {
        height: 1;
        dock: bottom;
        padding: 0 1;
        background: $primary-darken-2;
        color: $text;
    }
    """

    BINDINGS = [
        Binding("ctrl+slash", "toggle_comment", "Toggle Comment"),
        Binding("ctrl+pageup", "prev_tab", "Prev Tab"),
        Binding("ctrl+pagedown", "next_tab", "Next Tab"),
    ]

    def __init__(self, **kw):
        super().__init__(**kw)
        self._files: dict[str, str] = {}
        self._order: list[str] = []
        self._active: str | None = None
        self._counter: int = 0
        self._modified: set[str] = set()

    def compose(self):
        with Horizontal(id="editor-tab-bar"):
            yield Button("untitled", id="tab-0", variant="primary", classes="editor-tab")
        with Vertical(id="editor-body"):
            yield TextArea.code_editor(
                id="editor-textarea",
                language="triad",
                theme="monokai",
                show_line_numbers=True,
                tab_behavior="indent",
            )
        yield Static("Ln 1, Col 1 | UTF-8 | TriadLang", id="editor-status-line")

    def on_mount(self) -> None:

        self.new_untitled()

        ta = self.query_one("#editor-textarea", TextArea)
        ta.focus()

    def _get_textarea(self) -> TextArea:
        return self.query_one("#editor-textarea", TextArea)

    def _get_tab_bar(self) -> Horizontal:
        return self.query_one("#editor-tab-bar", Horizontal)

    def _get_status(self) -> Static:
        return self.query_one("#editor-status-line", Static)

    def new_untitled(self) -> None:

        self._counter += 1
        key = f"untitled-{self._counter}"
        self._files[key] = ""
        if key not in self._order:
            self._order.append(key)
        self._switch_to(key)

    def open_file(self, path: str, source: str) -> None:

        self._files[path] = source
        if path not in self._order:
            self._order.append(path)
        self._switch_to(path)

    def close_tab(self, key: str) -> None:

        if key in self._order:
            self._order.remove(key)
        self._files.pop(key, None)
        self._modified.discard(key)
        if self._active == key:
            if self._order:
                self._switch_to(self._order[-1])
            else:
                self.new_untitled()
        else:
            self._rebuild_tabs()

    def _switch_to(self, key: str) -> None:

        self._active = key
        source = self._files.get(key, "")
        ta = self._get_textarea()
        ta.load_text(source)
        self._rebuild_tabs()
        self._update_status()
        self.post_message(TabSwitched(
            key if not key.startswith("untitled-") else None,
            source,
        ))

    def _rebuild_tabs(self) -> None:

        bar = self._get_tab_bar()
        bar.remove_children()
        for key in self._order:
            name = os.path.basename(key) if key.startswith("/") or "\\" in key else key
            is_active = key == self._active
            is_mod = key in self._modified
            label = f"{'● ' if is_mod else ''}{name}"
            btn = Button(
                label,
                id=f"tab-{key}",
                variant="primary" if is_active else "default",
                classes="editor-tab",
            )
            bar.mount(btn)

    def _update_status(self) -> None:

        ta = self._get_textarea()
        row, col = ta.cursor_location
        fname = os.path.basename(self._active) if self._active and not self._active.startswith("untitled-") else self._active or "untitled"
        mod = " ●" if self._active in self._modified else ""
        self._get_status().update(
            f"Ln {row + 1}, Col {col + 1} | {fname}{mod} | TriadLang"
        )

    def get_source(self) -> str:

        return self._get_textarea().text

    def set_source(self, text: str) -> None:

        ta = self._get_textarea()
        ta.load_text(text)
        if self._active:
            self._files[self._active] = text

    def get_selection(self) -> str:

        ta = self._get_textarea()
        return ta.selected_text

    def find_next(self, text: str) -> bool:

        if not text:
            return False
        ta = self._get_textarea()
        src = ta.text
        row, col = ta.cursor_location

        pos = sum(len(line) + 1 for line in src.split("\n")[:row]) + col + 1
        idx = src.find(text, pos)
        if idx == -1:

            idx = src.find(text, 0, pos)
        if idx == -1:
            return False

        before = src[:idx]
        new_row = before.count("\n")
        last_nl = before.rfind("\n")
        new_col = idx - (last_nl + 1) if last_nl >= 0 else idx
        ta.cursor_location = (new_row, new_col)
        ta.selection = ((new_row, new_col), (new_row, new_col + len(text)))
        return True

    def replace_next(self, find: str, replace: str) -> bool:

        if not find:
            return False
        ta = self._get_textarea()
        sel = ta.selection
        if sel:
            start, end = sel
            sel_text = ta.text_between(start, end)
            if sel_text == find:
                ta.replace(replace, start, end)
                return True

        if self.find_next(find):
            return self.replace_next(find, replace)
        return False

    def replace_all(self, find: str, replace: str) -> int:

        if not find:
            return 0
        src = self.get_source()
        count = src.count(find)
        if count > 0:
            self.set_source(src.replace(find, replace))
        return count

    def action_toggle_comment(self) -> None:

        ta = self._get_textarea()
        row, col = ta.cursor_location
        lines = ta.text.split("\n")
        if row >= len(lines):
            return
        line = lines[row]
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]
        if stripped.startswith("//"):
            new_line = indent + stripped[2:].lstrip()
        elif stripped:
            new_line = indent + "// " + stripped
        else:
            return
        lines[row] = new_line
        ta.load_text("\n".join(lines))
        ta.cursor_location = (row, min(col, len(new_line)))

    def action_next_tab(self) -> None:
        if not self._order:
            return
        idx = self._order.index(self._active) if self._active in self._order else -1
        next_idx = (idx + 1) % len(self._order)
        self._switch_to(self._order[next_idx])

    def action_prev_tab(self) -> None:
        if not self._order:
            return
        idx = self._order.index(self._active) if self._active in self._order else -1
        prev_idx = (idx - 1) % len(self._order)
        self._switch_to(self._order[prev_idx])

    def on_button_pressed(self, ev: Button.Pressed) -> None:

        btn = ev.button
        if not hasattr(btn, "id") or not btn.id:
            return

        if btn.id.startswith("tab-"):
            key = btn.id[4:]
            if key in self._files:
                self._switch_to(key)

    def on_text_area_changed(self, ev: TextArea.Changed) -> None:

        if self._active:
            self._modified.add(self._active)
            self._files[self._active] = ev.text_area.text
            self._update_status()

    def on_text_area_selection_changed(self, ev: TextArea.SelectionChanged) -> None:

        self._update_status()

SimpleEditor = EditorTabs

