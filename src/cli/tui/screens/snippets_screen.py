from __future__ import annotations

from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen

from cli.tui.widgets.snippets import SnippetSelected, SnippetsWidget


class SnippetsScreen(ModalScreen):

    DEFAULT_CSS = """
    SnippetsScreen {
        align: center middle;
    }
    #snippets-screen-container {
        width: 90%;
        max-width: 120;
        height: 90%;
        border: thick $primary;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss_screen", "Close"),
    ]

    def compose(self):
        with Vertical(id="snippets-screen-container"):
            yield SnippetsWidget(id="snippets-inner")

    def action_dismiss_screen(self) -> None:
        self.dismiss(None)

    def on_snippet_selected(self, event: SnippetSelected) -> None:

        self.dismiss(event.snippet)

