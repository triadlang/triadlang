from __future__ import annotations

from dataclasses import dataclass

from rich.text import Text
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, Static


@dataclass
class Command:

    id: str
    title: str
    description: str = ""
    category: str = "General"
    keybinding: str = ""
    callback_id: str = ""

    def __repr__(self):
        return f"Command({self.id!r}, {self.title!r})"

class CommandPalette(ModalScreen[str | None]):

    DEFAULT_CSS = """
    CommandPalette {
        align: center top;
    }
    #palette-container {
        width: 70;
        max-width: 90%;
        height: auto;
        max-height: 60%;
        margin-top: 2;
        border: thick $primary;
        background: $surface;
    }
    #palette-input {
        height: 3;
        border: none;
        background: $surface;
    }
    #palette-list {
        height: auto;
        max-height: 16;
        background: $panel;
    }
    #palette-list > ListItem {
        padding: 0 1;
    }
    #palette-list > ListItem.--highlight {
        background: $primary;
        color: $background;
    }
    #palette-hint {
        height: 1;
        padding: 0 1;
        background: $panel;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss(None)", "Close"),
        Binding("ctrl+c", "dismiss(None)", "Close"),
        Binding("up", "cursor_up", "Up"),
        Binding("down", "cursor_down", "Down"),
    ]

    def __init__(self, commands: list[Command], **kw):
        super().__init__(**kw)
        self._all_commands = commands
        self._filtered: list[Command] = list(commands)
        self._query: str = ""

    def compose(self):
        with Vertical(id="palette-container"):
            yield Input(placeholder="Type a command…  (Ctrl+K to close)", id="palette-input")
            yield ListView(id="palette-list")
            yield Static("↑↓ navigate · Enter run · Esc close", id="palette-hint")

    def on_mount(self) -> None:
        self._refresh_list()
        self.query_one("#palette-input", Input).focus()

    def _refresh_list(self) -> None:

        from rich.markup import escape as _escape
        lv = self.query_one("#palette-list", ListView)
        lv.clear()
        for cmd in self._filtered[:50]:
            label = Text(_escape(cmd.title))
            if cmd.keybinding:
                label.append_text(Text(f"  {cmd.keybinding}", style="dim"))
            if cmd.description:
                label.append_text(Text(f"  {cmd.description}", style="dim"))
            item = ListItem(Label(label), id=f"cmd-{cmd.id}")
            lv.append(item)
        if self._filtered:
            lv.index = 0

    def on_input_changed(self, event: Input.Changed) -> None:

        if event.input.id != "palette-input":
            return
        self._query = event.value.lower().strip()
        if not self._query:
            self._filtered = list(self._all_commands)
        else:

            self._filtered = [
                cmd for cmd in self._all_commands
                if (self._query in cmd.title.lower() or
                    self._query in cmd.description.lower() or
                    self._query in cmd.category.lower() or
                    self._query in cmd.id.lower())
            ]
        self._refresh_list()

    def on_input_submitted(self, event: Input.Submitted) -> None:

        if event.input.id != "palette-input":
            return
        lv = self.query_one("#palette-list", ListView)
        if lv.index is not None and 0 <= lv.index < len(self._filtered):
            cmd = self._filtered[lv.index]
            self.dismiss(cmd.callback_id or cmd.id)
        elif self._filtered:
            self.dismiss(self._filtered[0].callback_id or self._filtered[0].id)

    def on_list_view_selected(self, event: ListView.Selected) -> None:

        if not event.item or not event.item.id:
            return
        cmd_id = event.item.id.replace("cmd-", "", 1)
        cmd = next((c for c in self._all_commands if c.id == cmd_id), None)
        if cmd:
            self.dismiss(cmd.callback_id or cmd.id)

class CommandBar(ModalScreen[str | None]):

    DEFAULT_CSS = """
    CommandBar {
        align: center bottom;
    }
    #cmdbar-container {
        width: 90%;
        max-width: 120;
        height: 3;
        margin-bottom: 1;
        border: thick $accent;
        background: $surface;
    }
    #cmdbar-input {
        height: 3;
        border: none;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss(None)", "Close"),
    ]

    def __init__(self, initial: str = ":", **kw):
        super().__init__(**kw)
        self._initial = initial

    def compose(self):
        yield Input(value=self._initial, id="cmdbar-input")

    def on_mount(self) -> None:
        inp = self.query_one("#cmdbar-input", Input)
        inp.focus()

        if self._initial == ":":
            inp.cursor_position = len(self._initial)

    def on_input_submitted(self, event: Input.Submitted) -> None:

        if event.input.id != "cmdbar-input":
            return
        cmd = event.value.strip()
        if cmd and cmd != ":":
            self.dismiss(cmd)
        else:
            self.dismiss(None)

