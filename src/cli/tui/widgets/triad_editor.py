from __future__ import annotations

import re

from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea

_BRACKETS_OPEN = {'{': '}', '(': ')', '[': ']'}
_BRACKETS_CLOSE = {'}': '{', ')': '(', ']': '['}
_INDENT_RE = re.compile(r'^(\s*)')

class CursorMoved(Message):
    def __init__(self, line: int, col: int) -> None:
        super().__init__()
        self.line = line
        self.col = col

class TriadEditor(TextArea):

    DEFAULT_CSS = """
    TriadEditor {
        height: 1fr;
        border: round $primary;
    }
    """

    BINDINGS = [
        Binding("ctrl+slash", "toggle_comment", "Toggle Comment"),
    ]

    def __init__(self, **kw):
        kw.setdefault("language", "python")
        super().__init__(**kw)
        self._bracket_pos: tuple[int, int] | None = None

    def on_key(self, event) -> None:
        if event.character == '\r' or event.key == "enter":
            self._auto_indent()
        super().on_key(event)

    def _auto_indent(self):
        try:
            row, col = self.cursor_location
        except (LookupError, AttributeError):
            return
        lines = self.text.split('\n')
        if row <= 0 or row > len(lines):
            return
        current_line = lines[row]
        before = current_line[:col] if col <= len(current_line) else current_line
        indent_m = _INDENT_RE.match(before)
        base_indent = indent_m.group(1) if indent_m else ""
        stripped = before.rstrip()
        extra = ""
        if stripped.endswith('{'):
            extra = "    "
        new_indent = base_indent + extra
        self.insert(f"\n{new_indent}", location=(row, col))

    def on_text_area_selection_changed(self, event):
        try:
            row, col = self.cursor_location
            self._highlight_matching_bracket(row, col)
        except (LookupError, AttributeError):
            pass

    def _highlight_matching_bracket(self, row, col):
        lines = self.text.split('\n')
        if row >= len(lines):
            return
        line = lines[row]
        if col <= 0 or col > len(line):
            return
        char = line[col - 1]
        if char in _BRACKETS_OPEN:
            self._find_matching(row, col - 1, char, 1)
        elif char in _BRACKETS_CLOSE:
            self._find_matching(row, col - 1, char, -1)

    def _find_matching(self, row, col, char, direction):
        target = _BRACKETS_OPEN[char] if direction == 1 else _BRACKETS_CLOSE[char]
        depth = 0
        lines = self.text.split('\n')
        r, c = row, col
        while 0 <= r < len(lines):
            line = lines[r]
            start_c = c + direction if (r == row) else (0 if direction == 1 else len(line) - 1)
            arange = (
                range(start_c, len(line), direction)
                if direction == 1
                else range(start_c, -1, direction)
            )
            for cc in arange:
                ch = line[cc]
                if ch == char:
                    depth += 1
                elif ch == target:
                    depth -= 1
                    if depth == 0:
                        return
            r += direction
            c = 0 if direction == 1 else (len(lines[r]) - 1 if r < len(lines) else 0)

    def action_toggle_comment(self):
        try:
            row, col = self.cursor_location
        except (LookupError, AttributeError):
            return
        lines = self.text.split('\n')
        if row >= len(lines):
            return
        line = lines[row]
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]
        if stripped.startswith('//'):
            new_line = indent + stripped[2:]
        elif stripped:
            new_line = indent + '//' + stripped
        else:
            return
        lines[row] = new_line
        self.load_text('\n'.join(lines))
        self.cursor_location = (row, col)

    def get_line_col(self) -> tuple[int, int]:
        try:
            return self.cursor_location
        except (LookupError, AttributeError):
            return (0, 0)

