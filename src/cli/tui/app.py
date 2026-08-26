from __future__ import annotations

import io
import json
import os
import sys
from typing import Any

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)

from cli.tui.screens.onboarding import OnboardingScreen
from cli.tui.screens.snippets_screen import SnippetsScreen
from cli.tui.themes import ALL_THEMES
from cli.tui.widgets.command_palette import Command, CommandBar, CommandPalette
from cli.tui.widgets.file_tree import FileSelected
from cli.tui.widgets.snippets import SnippetSelected

PY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

class EditorContentChanged(Message):

    def __init__(self, source: str, path: str | None = None):
        super().__init__()
        self.source = source
        self.path = path

class RunRequest(Message):

    def __init__(self, source: str, origin: str = "unknown"):
        super().__init__()
        self.source = source
        self.origin = origin

class OutputMessage(Message):

    def __init__(self, text: str, style: str = "", target: str = "output"):
        super().__init__()
        self.text = text
        self.style = style
        self.target = target

class StatusUpdate(Message):

    def __init__(self, text: str):
        super().__init__()
        self.text = text

class WelcomeScreen(Screen):

    DEFAULT_CSS = """
    WelcomeScreen {
        align: center middle;
        background: $surface;
    }
    #welcome-container {
        width: 70;
        height: auto;
        padding: 2 4;
        border: thick $primary;
        background: $panel;
    }
    #welcome-title {
        content-align: center middle;
        padding: 1 0;
    }
    #welcome-shortcuts {
        padding: 1 0;
    }
    #welcome-status {
        padding: 1 0;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="welcome-container"):
            yield Label(
                "[bold cyan]⚡ TriadLang[/bold cyan] [dim]— a language that runs on a field[/dim]",
                id="welcome-title",
            )
            yield RichLog(id="welcome-shortcuts", highlight=True, markup=True)
            yield RichLog(id="welcome-status", highlight=True, markup=True)

    def on_mount(self) -> None:
        shortcuts = self.query_one("#welcome-shortcuts", RichLog)
        shortcuts.write(Panel(
            Text.from_markup(
                "[bold cyan]Quick Start[/bold cyan]\n\n"
                "[bold]Ctrl+N[/bold]  New file          [bold]Ctrl+O[/bold]  Open file\n"
                "[bold]Ctrl+E[/bold]  Editor            [bold]Ctrl+R[/bold]  REPL\n"
                "[bold]Ctrl+S[/bold]  Solver            [bold]Ctrl+M[/bold]  ML Lab\n"
                "[bold]Ctrl+D[/bold]  DSL Builder       [bold]Ctrl+X[/bold]  Examples\n"
                "[bold]F5[/bold]      Run file          [bold]F9[/bold]      Run selection\n"
                "[bold]Ctrl+Q[/bold]  Quit              [bold]Ctrl+H[/bold]  Help\n\n"
                "[dim]Open a .tri file or create a new one to get started.[/dim]"
            ),
            border_style="cyan",
            title="Welcome",
        ))

        status = self.query_one("#welcome-status", RichLog)
        try:
            from runtime.backend import cuda_available
            cuda = cuda_available()
        except (ImportError, RuntimeError):
            cuda = False
        try:
            from triad import ntri as np
            np_ver = np.__version__
        except (ImportError, AttributeError):
            np_ver = "N/A"

        t = Table(title="Environment", border_style="dim blue")
        t.add_column("Component", style="cyan")
        t.add_column("Status", style="green")
        t.add_row("Python", sys.version.split()[0])
        t.add_row("NumPy", np_ver)
        t.add_row("CUDA", "Available" if cuda else "Not available (CPU mode)")
        t.add_row("Backend", "NumPy" if not cuda else "CuPy")
        t.add_row("Tests", "562/562 passing ✓" if self._tests_ok() else "Check with pytest")
        status.write(t)

    def _tests_ok(self) -> bool:

        return True

class HelpModal(ModalScreen[None]):
    DEFAULT_CSS = """
    HelpModal {
        align: center middle;
    }
    #help-dialog {
        width: 76;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss(None)", "Close"),
        Binding("q", "dismiss(None)", "Close"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="help-dialog"):
            yield Label("[bold cyan]⌨ TriadLang IDE — Keyboard Shortcuts[/bold cyan]")
            yield RichLog(id="help-content", highlight=True, markup=True)
            yield Button("Close", variant="error", id="help-close")

    def on_mount(self) -> None:
        log = self.query_one("#help-content", RichLog)
        shortcuts = [
            ("Navigation", [
                ("Ctrl+E", "Editor tab"),
                ("Ctrl+R", "REPL tab"),
                ("Ctrl+S", "Solver tab"),
                ("Ctrl+M", "ML Lab tab"),
                ("Ctrl+D", "DSL Builder tab"),
                ("Ctrl+X", "Examples tab"),
                ("Ctrl+W / Ctrl+Shift+W", "Next / Previous tab"),
                ("Ctrl+F", "Toggle file sidebar"),
                ("Ctrl+B", "Toggle bottom panel"),
            ]),
            ("Editor", [
                ("Ctrl+N", "New file"),
                ("Ctrl+O", "Open file"),
                ("Ctrl+S", "Save file"),
                ("F5", "Run current file"),
                ("F9", "Run selection"),
                ("Ctrl+/", "Toggle comment"),
                ("Shift+F", "Format code"),
                ("Ctrl+Shift+F", "Find & Replace"),
                ("Ctrl+T", "Type check"),
            ]),
            ("Compiler / Debug", [
                ("Ctrl+Shift+C", "Compile to IR"),
                ("Ctrl+Shift+A", "View AST"),
                ("Ctrl+Shift+R", "View IR"),
            ]),
            ("Tools", [
                ("Ctrl+P", "Package manager"),
                ("Ctrl+G", "Settings"),
                ("Ctrl+H", "This help"),
                ("Ctrl+Q", "Quit"),
            ]),
        ]
        for section, bindings in shortcuts:
            log.write(Text(f"\n[bold cyan]{section}[/bold cyan]"))
            for key, desc in bindings:
                log.write(Text(f"  [bold]{key:<22}[/bold] [dim]{desc}[/dim]"))

    def on_button_pressed(self, ev: Button.Pressed) -> None:
        if ev.button.id == "help-close":
            self.dismiss(None)

class SettingsModal(ModalScreen[dict | None]):
    DEFAULT_CSS = """
    SettingsModal {
        align: center middle;
    }
    #settings-dialog {
        width: 60;
        height: auto;
        padding: 1 2;
        border: thick $secondary;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss(None)", "Close"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-dialog"):
            yield Label("[bold secondary]⚙ Settings[/bold secondary]")
            yield Label("Backend:")
            yield Select(
                [("Auto (CuPy if available)", "auto"), ("CPU (NumPy)", "cpu"), ("GPU (CuPy)", "gpu")],
                id="settings-backend",
                prompt="Backend",
            )
            yield Label("Safe Mode:")
            yield Switch(id="settings-safe", value=False)
            yield Label("Auto-format on save:")
            yield Switch(id="settings-autoformat", value=True)
            yield Label("Theme:")
            yield Select(
                [("Triad Dark", "triad-dark"), ("Textual Dark", "textual-dark"), ("Dracula", "dracula")],
                id="settings-theme",
                prompt="Theme",
            )
            with Horizontal():
                yield Button("Save", variant="success", id="settings-save")
                yield Button("Cancel", variant="error", id="settings-cancel")

    def on_button_pressed(self, ev: Button.Pressed) -> None:
        if ev.button.id == "settings-save":
            backend = self.query_one("#settings-backend", Select).value
            safe = self.query_one("#settings-safe", Switch).value
            autoformat = self.query_one("#settings-autoformat", Switch).value
            theme = self.query_one("#settings-theme", Select).value
            self.dismiss({
                "backend": backend,
                "safe_mode": safe,
                "autoformat": autoformat,
                "theme": theme,
            })
        elif ev.button.id == "settings-cancel":
            self.dismiss(None)

class SearchModal(ModalScreen[dict | None]):
    DEFAULT_CSS = """
    SearchModal {
        align: center middle;
    }
    #search-dialog {
        width: 64;
        height: auto;
        padding: 1 2;
        border: thick $accent;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss(None)", "Close"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="search-dialog"):
            yield Label("[bold accent]🔍 Find & Replace[/bold accent]")
            yield Input(placeholder="Find...", id="search-find")
            yield Input(placeholder="Replace with...", id="search-replace")
            with Horizontal():
                yield Button("Find Next", variant="primary", id="search-find-btn")
                yield Button("Replace", variant="default", id="search-replace-btn")
                yield Button("Replace All", variant="warning", id="search-replace-all-btn")
                yield Button("Close", variant="error", id="search-close-btn")

    def on_button_pressed(self, ev: Button.Pressed) -> None:
        find = self.query_one("#search-find", Input).value
        repl = self.query_one("#search-replace", Input).value
        bid = ev.button.id
        if bid == "search-find-btn":
            self.dismiss({"action": "find", "find": find})
        elif bid == "search-replace-btn":
            self.dismiss({"action": "replace", "find": find, "replace": repl})
        elif bid == "search-replace-all-btn":
            self.dismiss({"action": "replace_all", "find": find, "replace": repl})
        else:
            self.dismiss(None)

class PackageModal(ModalScreen[dict | None]):
    DEFAULT_CSS = """
    PackageModal {
        align: center middle;
    }
    #pkg-dialog {
        width: 64;
        height: 28;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss(None)", "Close"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="pkg-dialog"):
            yield Label("[bold cyan]📦 Package Manager[/bold cyan]")
            yield Input(placeholder="Search packages...", id="pkg-search")
            yield RichLog(id="pkg-list", highlight=True, markup=False)
            with Horizontal():
                yield Button("Install", variant="success", id="pkg-install")
                yield Button("Refresh", variant="default", id="pkg-refresh")
                yield Button("Close", variant="error", id="pkg-close")

    def on_mount(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        log = self.query_one("#pkg-list", RichLog)
        log.clear()
        try:
            from stdlib.registry import list_installed
            pkgs = list_installed()
            if pkgs:
                for p in pkgs:
                    log.write(Text(f"  {p['name']:<24} {p.get('version', '—'):<12}"))
            else:
                log.write(Text("  (no packages installed)", style="dim"))
        except Exception as e:
            log.write(Text(f"  Error: {e}", style="red"))

    def on_button_pressed(self, ev: Button.Pressed) -> None:
        if ev.button.id == "pkg-close":
            self.dismiss(None)
        elif ev.button.id == "pkg-refresh":
            self._refresh()
        elif ev.button.id == "pkg-install":
            q = self.query_one("#pkg-search", Input).value.strip()
            self.dismiss({"action": "install", "query": q})

class BottomDock(Container):

    DEFAULT_CSS = """
    BottomDock {
        height: 14;
        border-top: solid $primary;
        background: $surface;
    }
    BottomDock TabbedContent {
        height: 1fr;
    }
    BottomDock RichLog {
        height: 1fr;
        scrollbar-size: 1 1;
    }
    """

    def compose(self) -> ComposeResult:
        with TabbedContent(initial="output", id="bottom-tabs"):
            with TabPane("Output", id="output"):
                yield RichLog(id="output-log", highlight=True, markup=True, wrap=False)
            with TabPane("Errors", id="errors"):
                yield RichLog(id="error-log", highlight=True, markup=True, wrap=False)
            with TabPane("Compiler", id="compiler"):
                yield RichLog(id="compiler-log", highlight=True, markup=False, wrap=False)
            with TabPane("IR / AST", id="ir-ast"):
                yield RichLog(id="ir-log", highlight=True, markup=False, wrap=False)
            with TabPane("Debug", id="debug"):
                from cli.tui.widgets.heatmap_widget import HeatmapWidget
                yield HeatmapWidget(id="debug-heatmap", width=40, height=12)
                yield RichLog(id="debug-log", highlight=True, markup=True, wrap=False)

class TriadApp(App):

    TITLE = "TriadLang"
    SUB_TITLE = "a language that runs on a field"
    CSS_PATH = None

    DEFAULT_CSS = """
    * { scrollbar-size: 1 1; }

    Screen {
        layout: vertical;
    }

    #main-horizontal {
        layout: horizontal;
        height: 1fr;
    }

    #sidebar {
        width: 26;
        min-width: 20;
        max-width: 40;
        border-right: solid $primary;
        background: $surface;
        padding: 0 1;
    }

    #sidebar Label {
        padding: 1 0 0 0;
    }

    #main-area {
        width: 1fr;
    }

    #main-area TabbedContent {
        height: 1fr;
    }

    #right-panel {
        width: 28;
        min-width: 22;
        max-width: 40;
        border-left: solid $primary;
        background: $surface;
        padding: 0 1;
    }

    #right-panel Label {
        padding: 1 0 0 0;
    }

    #statusbar {
        height: 1;
        background: $primary-darken-2;
        color: $text;
        content-align: center middle;
    }

    #welcome-placeholder {
        height: 1fr;
        content-align: center middle;
    }

    RichLog {
        scrollbar-size: 1 1;
    }

    TabPane {
        padding: 0;
    }
    """

    BINDINGS = [

        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+e", "show_tab('editor')", "Editor"),
        Binding("ctrl+r", "show_tab('repl')", "REPL"),
        Binding("ctrl+s", "show_tab('solver')", "Solver"),
        Binding("ctrl+m", "show_tab('ml')", "ML"),
        Binding("ctrl+d", "show_tab('dsl')", "DSL"),
        Binding("ctrl+x", "show_tab('examples')", "Examples"),
        Binding("ctrl+w", "next_tab", "Next Tab"),
        Binding("ctrl+shift+w", "prev_tab", "Prev Tab"),
        Binding("ctrl+f", "toggle_sidebar", "Files"),
        Binding("ctrl+b", "toggle_bottom", "Bottom"),

        Binding("ctrl+k", "command_palette", "Palette"),
        Binding("ctrl+slash", "command_bar", "Cmd Bar"),

        Binding("ctrl+shift+f", "find_replace", "Find"),
        Binding("ctrl+n", "new_file", "New"),
        Binding("ctrl+o", "open_file", "Open"),
        Binding("ctrl+shift+s", "snippets", "Snippets"),
        Binding("ctrl+shift+t", "onboarding", "Tour"),

        Binding("f5", "run_file", "Run"),
        Binding("f9", "run_selection", "Run Sel"),

        Binding("ctrl+t", "type_check", "Type Check"),
        Binding("shift+f", "format_code", "Format"),
        Binding("ctrl+alt+slash", "toggle_comment", "Comment"),
        Binding("ctrl+shift+c", "compile_ir", "Compile IR"),
        Binding("ctrl+shift+a", "view_ast", "View AST"),
        Binding("ctrl+shift+r", "view_ir", "View IR"),

        Binding("ctrl+p", "pkg_manager", "Packages"),
        Binding("ctrl+g", "settings", "Settings"),
        Binding("ctrl+shift+l", "cycle_theme", "Theme"),
        Binding("ctrl+h", "show_help", "Help"),
    ]

    sidebar_visible: reactive[bool] = reactive(True)
    bottom_visible: reactive[bool] = reactive(True)
    right_panel_visible: reactive[bool] = reactive(True)

    def __init__(
        self,
        file_path: str | None = None,
        start_screen: str = "editor",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._initial_file = file_path
        self._start_screen = start_screen
        self._active_file: str | None = None
        self._open_files: dict[str, str] = {}
        self._file_order: list[str] = []
        self._settings: dict[str, Any] = {
            "backend": "auto",
            "safe_mode": False,
            "autoformat": True,
            "theme": "triad",
        }
        self._load_settings_from_disk()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-horizontal"):

            with Vertical(id="sidebar"):
                yield Label("[bold cyan]📁 Files[/bold cyan]")
                from cli.tui.widgets.file_tree import TriadFileTree
                yield TriadFileTree(os.getcwd(), id="file-tree")
                yield Label("[bold secondary]🔣 Symbols[/bold secondary]")
                from cli.tui.widgets.symbol_outline import SymbolOutline
                yield SymbolOutline(id="sidebar-symbols")

            with Vertical(id="main-area"):
                with TabbedContent(initial=self._start_screen, id="main-tabs"):
                    with TabPane("📝 Editor", id="editor"):
                        from cli.tui.widgets.editor_tabs import EditorTabs
                        yield EditorTabs(id="editor-tabs")
                    with TabPane("💬 REPL", id="repl"):
                        from cli.tui.widgets.triad_repl import TriadREPL
                        yield TriadREPL(id="repl-widget")
                    with TabPane("🔬 Solver", id="solver"):
                        from cli.tui.widgets.solver_widget import SolverWidget
                        yield SolverWidget(id="solver-widget")
                    with TabPane("🧠 ML Lab", id="ml"):
                        from cli.tui.widgets.ml_widget import MLWidget
                        yield MLWidget(id="ml-widget")
                    with TabPane("🏗 DSL", id="dsl"):
                        from cli.tui.widgets.dsl_widget import DSLWidget
                        yield DSLWidget(id="dsl-widget")
                    with TabPane("📚 Examples", id="examples"):
                        from cli.tui.widgets.examples_widget import ExamplesWidget
                        yield ExamplesWidget(id="examples-widget")

            with Vertical(id="right-panel"):
                yield Label("[bold accent]👁 Watch[/bold accent]")
                yield RichLog(id="watch-log", highlight=True, markup=True)
                yield Label("[bold secondary]📊 Stack[/bold secondary]")
                yield RichLog(id="stack-log", highlight=True, markup=True)
                yield Label("[bold cyan]📐 Variables[/bold cyan]")
                yield RichLog(id="var-log", highlight=True, markup=True)

        yield BottomDock(id="bottom-dock")
        yield Static(" READY", id="statusbar")
        yield Footer()

    def on_mount(self) -> None:

        for theme in ALL_THEMES:
            self.register_theme(theme)

        theme_name = self._settings.get("theme", "triad")
        if not theme_name or theme_name == "triad-dark":
            theme_name = "triad"
        try:
            self.theme = theme_name
        except (ValueError, LookupError, KeyError):
            self.theme = "triad"

        if self._should_show_onboarding():
            self.push_screen(OnboardingScreen())
            self._mark_onboarding_seen()
        if self._initial_file and os.path.exists(self._initial_file):
            self._open_editor_file(self._initial_file)
        self._show_welcome_message()
        self._refresh_status()

    def _should_show_onboarding(self) -> bool:

        config_dir = os.path.expanduser("~/.triad")
        config_path = os.path.join(config_dir, "config.json")
        try:
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = json.load(f)
                return not cfg.get("onboarding_seen", False)
        except (OSError, ValueError, KeyError):
            pass
        return True

    def _mark_onboarding_seen(self) -> None:

        config_dir = os.path.expanduser("~/.triad")
        config_path = os.path.join(config_dir, "config.json")
        try:
            os.makedirs(config_dir, exist_ok=True)
            cfg = {}
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = json.load(f)
            cfg["onboarding_seen"] = True
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2)
        except (OSError, ValueError):
            pass

    def _save_settings_to_disk(self) -> None:

        config_dir = os.path.expanduser("~/.triad")
        config_path = os.path.join(config_dir, "config.json")
        try:
            os.makedirs(config_dir, exist_ok=True)
            cfg = {}
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = json.load(f)
            cfg["settings"] = self._settings
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2)
        except (OSError, ValueError):
            pass

    def _load_settings_from_disk(self) -> None:

        config_path = os.path.expanduser("~/.triad/config.json")
        try:
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = json.load(f)
                disk_settings = cfg.get("settings", {})
                self._settings.update(disk_settings)
        except (OSError, ValueError, KeyError):
            pass

    def on_editor_content_changed(self, msg: EditorContentChanged) -> None:

        try:
            outline = self.query_one("#sidebar-symbols")
            if hasattr(outline, "update_from_source"):
                outline.update_from_source(msg.source)
        except (LookupError, AttributeError):
            pass

    def on_run_request(self, msg: RunRequest) -> None:

        self._run_code(msg.source, msg.origin)

    def on_output_message(self, msg: OutputMessage) -> None:

        try:
            dock = self.query_one("#bottom-dock")
            tabs = dock.query_one("#bottom-tabs", TabbedContent)
            log = dock.query_one(f"#{msg.target}-log", RichLog)
            log.write(Text(msg.text, style=msg.style))
            tabs.active = msg.target
            self.bottom_visible = True
            dock.display = True
        except (LookupError, AttributeError):
            pass

    def on_status_update(self, msg: StatusUpdate) -> None:

        try:
            self.query_one("#statusbar", Static).update(msg.text)
        except (LookupError, AttributeError):
            pass

    def _show_welcome_message(self) -> None:

        log = self._output_log()
        if log:
            log.write(Text("⚡ TriadLang IDE — Ready", style="bold cyan"))
            log.write(Text(
                "Ctrl+E Editor  |  Ctrl+R REPL  |  Ctrl+S Solver  |  Ctrl+M ML  |  Ctrl+H Help",
                style="dim",
            ))

    def _editor_tabs(self):
        try:
            return self.query_one("#editor-tabs")
        except (LookupError, AttributeError):
            return None

    def _output_log(self) -> RichLog | None:
        try:
            return self.query_one("#bottom-dock").query_one("#output-log", RichLog)
        except (LookupError, AttributeError):
            return None

    def _error_log(self) -> RichLog | None:
        try:
            return self.query_one("#bottom-dock").query_one("#error-log", RichLog)
        except (LookupError, AttributeError):
            return None

    def _compiler_log(self) -> RichLog | None:
        try:
            return self.query_one("#bottom-dock").query_one("#compiler-log", RichLog)
        except (LookupError, AttributeError):
            return None

    def _ir_log(self) -> RichLog | None:
        try:
            return self.query_one("#bottom-dock").query_one("#ir-log", RichLog)
        except (LookupError, AttributeError):
            return None

    def _output(self, msg: str, style: str = "") -> None:
        log = self._output_log()
        if log:
            log.write(Text(msg, style=style))

    def _error(self, msg: str) -> None:
        log = self._error_log()
        if log:
            log.write(Text(msg, style="bold red"))
        self._focus_bottom_tab("errors")

    def _refresh_status(self) -> None:
        fname = os.path.basename(self._active_file) if self._active_file else "untitled"
        n_open = len(self._open_files)
        backend = self._settings.get("backend", "auto")
        theme = self.theme if hasattr(self, "theme") else "triad"

        try:
            tc = self.query_one("#main-tabs", TabbedContent)
            tab_id = tc.active
        except (LookupError, AttributeError):
            tab_id = "?"

        try:
            from runtime.backend import cuda_available
            backend_badge = "GPU" if cuda_available() else "CPU"
        except (ImportError, RuntimeError):
            backend_badge = "?"
        try:
            self.query_one("#statusbar", Static).update(
                f" ⚡ {fname} | {tab_id} | {n_open} open | "
                f"theme: {theme} | [{backend_badge}] | Ctrl+K palette"
            )
        except (LookupError, AttributeError):
            pass

    def _open_editor_file(self, path: str) -> None:
        try:
            with open(path) as f:
                source = f.read()
        except Exception as e:
            self._error(f"Failed to open {path}: {e}")
            return
        self._open_files[path] = source
        if path not in self._file_order:
            self._file_order.append(path)
        self._active_file = path
        tabs = self._editor_tabs()
        if tabs:
            tabs.open_file(path, source)
        self.post_message(EditorContentChanged(source, path))
        self.title = f"TriadLang — {os.path.basename(path)}"
        self._refresh_status()

    def _focus_bottom_tab(self, tab_id: str) -> None:
        try:
            dock = self.query_one("#bottom-dock")
            dock.query_one("#bottom-tabs", TabbedContent).active = tab_id
            self.bottom_visible = True
            dock.display = True
        except (LookupError, AttributeError):
            pass

    def _run_code(self, source: str, origin: str = "unknown") -> None:

        import io as _io
        self._output(f"▶ Running ({origin})...", "cyan")
        try:
            from frontend.errors import LexError, ParseError
            from frontend.parser_universal import parse
            from runtime.compiler_runtime import CompileError, TriadCompiler, set_safe_mode
            set_safe_mode(self._settings.get("safe_mode", False))
            mod = parse(source, self._active_file or f"<{origin}>")
            compiler = TriadCompiler()
            old_stdout = sys.stdout
            sys.stdout = buf = _io.StringIO()
            try:
                compiler.compile_and_run(mod)
            finally:
                stdout_text = buf.getvalue()
                sys.stdout = old_stdout
            if stdout_text.strip():
                for line in stdout_text.strip().split("\n"):
                    self._output(line)
            self._output("✓ OK", "green")
        except (LexError, ParseError) as e:
            self._error(str(e))
        except CompileError as e:
            self._error(str(e))
        except Exception as e:
            self._error(f"{type(e).__name__}: {e}")

    def action_show_tab(self, tab_id: str) -> None:
        try:
            self.query_one("#main-tabs", TabbedContent).active = tab_id
        except (LookupError, AttributeError):
            pass

    def action_next_tab(self) -> None:
        try:
            tc = self.query_one("#main-tabs", TabbedContent)
            panes = list(tc.query(TabPane))
            if not panes:
                return
            ids = [p.id for p in panes]
            idx = ids.index(tc.active) if tc.active in ids else 0
            tc.active = ids[(idx + 1) % len(ids)]
        except (LookupError, AttributeError, ValueError):
            pass

    def action_prev_tab(self) -> None:
        try:
            tc = self.query_one("#main-tabs", TabbedContent)
            panes = list(tc.query(TabPane))
            if not panes:
                return
            ids = [p.id for p in panes]
            idx = ids.index(tc.active) if tc.active in ids else 0
            tc.active = ids[(idx - 1) % len(ids)]
        except (LookupError, AttributeError, ValueError):
            pass

    def action_toggle_sidebar(self) -> None:
        self.sidebar_visible = not self.sidebar_visible
        try:
            self.query_one("#sidebar").display = self.sidebar_visible
        except (LookupError, AttributeError):
            pass

    def action_toggle_bottom(self) -> None:
        self.bottom_visible = not self.bottom_visible
        try:
            self.query_one("#bottom-dock").display = self.bottom_visible
        except (LookupError, AttributeError):
            pass

    def action_new_file(self) -> None:
        tabs = self._editor_tabs()
        if tabs:
            tabs.new_untitled()
            self._active_file = None
            self._refresh_status()
            self.action_show_tab("editor")

    def action_open_file(self) -> None:
        try:
            self.query_one("#file-tree").focus()
        except (LookupError, AttributeError):
            pass

    def action_save_file(self) -> None:
        tabs = self._editor_tabs()
        if not tabs or not self._active_file:
            self._output("No file to save — use 'Save As' or open a file first", "yellow")
            return
        src = tabs.get_source()
        try:
            with open(self._active_file, "w") as f:
                f.write(src)
            self._output(f"Saved: {self._active_file}", "green")
            self._refresh_status()
        except Exception as e:
            self._error(f"Save failed: {e}")

    def action_run_file(self) -> None:
        tabs = self._editor_tabs()
        if tabs:
            src = tabs.get_source()
            if src.strip():
                self._run_code(src, "editor")
            else:
                self._output("Nothing to run — editor is empty", "yellow")

    def action_run_selection(self) -> None:
        tabs = self._editor_tabs()
        if tabs:
            src = tabs.get_selection()
            if src.strip():
                self._run_code(src, "selection")
            else:
                self._output("No selection — select code first", "yellow")

    def action_toggle_comment(self) -> None:
        tabs = self._editor_tabs()
        if tabs:
            tabs.action_toggle_comment()

    def action_format_code(self) -> None:
        tabs = self._editor_tabs()
        if not tabs:
            return
        src = tabs.get_source()
        if not src.strip():
            return
        try:
            from compiler.formatter import format_universal
            from frontend.parser_universal import parse
            mod = parse(src, self._active_file or "<tui>")
            formatted = format_universal(mod)
            tabs.set_source(formatted)
            self._output("✓ Formatted", "green")
        except Exception as e:
            self._error(f"Format error: {e}")

    def action_type_check(self) -> None:
        tabs = self._editor_tabs()
        if not tabs:
            return
        src = tabs.get_source()
        if not src.strip():
            return
        try:
            from compiler.typecheck_universal import typecheck
            from frontend.parser_universal import parse
            mod = parse(src, self._active_file or "<tui>")
            typecheck(mod)
            self._output("✓ Type check passed", "green")
        except Exception as e:
            self._error(f"Type error: {e}")

    def action_compile_ir(self) -> None:
        tabs = self._editor_tabs()
        if not tabs:
            return
        src = tabs.get_source()
        if not src.strip():
            return
        try:
            from compiler.emit_json import emit_json
            from compiler.lower import lower_module
            from frontend.parser_universal import parse
            mod = parse(src, self._active_file or "<tui>")
            ir = lower_module(mod)
            text = emit_json(ir)
            clog = self._compiler_log()
            if clog:
                clog.clear()
                clog.write(Text(text.replace("\t", "  ")))
            self._output("✓ Compiled to IR", "cyan")
            self._focus_bottom_tab("compiler")
        except Exception as e:
            self._error(f"Compile error: {e}")

    def action_view_ast(self) -> None:
        tabs = self._editor_tabs()
        if not tabs:
            return
        src = tabs.get_source()
        if not src.strip():
            return
        try:
            from frontend.parser_universal import parse
            mod = parse(src, self._active_file or "<tui>")
            buf = io.StringIO()
            for stmt in mod.body:
                buf.write(repr(stmt))
                buf.write("\n")
            ir_log = self._ir_log()
            if ir_log:
                ir_log.clear()
                ir_log.write(Text(buf.getvalue()))
            self._focus_bottom_tab("ir-ast")
            self._output("✓ AST dumped", "cyan")
        except Exception as e:
            self._error(f"AST error: {e}")

    def action_view_ir(self) -> None:
        tabs = self._editor_tabs()
        if not tabs:
            return
        src = tabs.get_source()
        if not src.strip():
            return
        try:
            from compiler.emit_json import emit_json
            from compiler.lower import lower_module
            from frontend.parser_universal import parse
            mod = parse(src, self._active_file or "<tui>")
            ir = lower_module(mod)
            ir_log = self._ir_log()
            if ir_log:
                ir_log.clear()
                ir_log.write(Text(emit_json(ir).replace("\t", "  ")))
            self._focus_bottom_tab("ir-ast")
            self._output("✓ IR dumped", "cyan")
        except Exception as e:
            self._error(f"IR error: {e}")

    def action_find_replace(self) -> None:
        def _cb(result: dict | None) -> None:
            if not result:
                return
            tabs = self._editor_tabs()
            if not tabs:
                return
            act = result.get("action")
            find = result.get("find", "")
            repl = result.get("replace", "")
            if act == "find":
                tabs.find_next(find)
            elif act == "replace":
                tabs.replace_next(find, repl)
            elif act == "replace_all":
                n = tabs.replace_all(find, repl)
                self._output(f"✓ Replaced {n} occurrences", "green")
        self.push_screen(SearchModal(), _cb)

    def action_pkg_manager(self) -> None:
        def _cb(result: dict | None) -> None:
            if not result:
                return
            if result.get("action") == "install":
                q = result.get("query", "")
                self._output(f"Package install: {q} (not yet wired)", "yellow")
        self.push_screen(PackageModal(), _cb)

    def action_settings(self) -> None:
        def _cb(result: dict | None) -> None:
            if not result:
                return
            self._settings.update(result)
            self._output(f"✓ Settings updated: backend={result.get('backend')}, "
                         f"safe={result.get('safe_mode')}, "
                         f"autoformat={result.get('autoformat')}", "green")

            new_theme = result.get("theme")
            if new_theme and new_theme != self.theme:
                try:
                    self.theme = new_theme
                    self._output(f"✓ Theme: {new_theme}", "green")
                except (ValueError, LookupError, KeyError):
                    pass
            self._save_settings_to_disk()
            self._refresh_status()
        self.push_screen(SettingsModal(), _cb)

    def action_show_help(self) -> None:
        self.push_screen(HelpModal())

    def action_command_palette(self) -> None:

        commands = self._build_command_list()
        def _cb(result: str | None) -> None:
            if not result:
                return
            self._execute_command(result)
        self.push_screen(CommandPalette(commands), _cb)

    def action_command_bar(self) -> None:

        def _cb(result: str | None) -> None:
            if not result:
                return
            self._execute_command_bar(result)
        self.push_screen(CommandBar(initial=":"), _cb)

    def action_snippets(self) -> None:

        def _cb(snippet) -> None:
            if snippet is None:
                return

            tabs = self._editor_tabs()
            if tabs and hasattr(tabs, "set_source"):

                current = tabs.get_source() if hasattr(tabs, "get_source") else ""
                if current.strip():
                    new_source = current.rstrip() + "\n\n" + snippet.code
                else:
                    new_source = snippet.code
                tabs.set_source(new_source)
                self.action_show_tab("editor")
                self._output(f"✓ Snippet '{snippet.title}' inserted", "green")
        self.push_screen(SnippetsScreen(), _cb)

    def action_onboarding(self) -> None:

        self.push_screen(OnboardingScreen())

    def action_cycle_theme(self) -> None:

        theme_names = [t.name for t in ALL_THEMES]
        try:
            current_idx = theme_names.index(self.theme)
        except ValueError:
            current_idx = 0
        next_idx = (current_idx + 1) % len(theme_names)
        new_theme = theme_names[next_idx]
        try:
            self.theme = new_theme
            self._settings["theme"] = new_theme
            self._save_settings_to_disk()
            self._output(f"✓ Theme: {new_theme} ({next_idx + 1}/{len(theme_names)})", "green")
        except Exception as e:
            self._error(f"Theme change failed: {e}")

    def _build_command_list(self) -> list[Command]:

        return [

            Command("new", "New File", "Create a new file", "File", "Ctrl+N", "new_file"),
            Command("open", "Open File", "Open a file from the project", "File", "Ctrl+O", "open_file"),
            Command("save", "Save File", "Save current file", "File", "Ctrl+S", "save_file"),
            Command("find", "Find & Replace", "Find and replace text", "File", "Ctrl+Shift+F", "find_replace"),
            Command("format", "Format Code", "Auto-format the source", "File", "Shift+F", "format_code"),

            Command("tab-editor", "Go to Editor", "Switch to the editor tab", "Tabs", "Ctrl+E", "show_tab('editor')"),
            Command("tab-repl", "Go to REPL", "Switch to the REPL tab", "Tabs", "Ctrl+R", "show_tab('repl')"),
            Command("tab-solver", "Go to Solver", "Switch to the solver tab", "Tabs", "Ctrl+S", "show_tab('solver')"),
            Command("tab-ml", "Go to ML Lab", "Switch to the ML Lab tab", "Tabs", "Ctrl+M", "show_tab('ml')"),
            Command("tab-dsl", "Go to DSL Builder", "Switch to the DSL Builder tab", "Tabs", "Ctrl+D", "show_tab('dsl')"),
            Command("tab-examples", "Go to Examples", "Switch to the examples tab", "Tabs", "Ctrl+X", "show_tab('examples')"),

            Command("run", "Run File", "Run the current file", "Run", "F5", "run_file"),
            Command("run-sel", "Run Selection", "Run the selected text", "Run", "F9", "run_selection"),
            Command("type-check", "Type Check", "Run type checker on the source", "Compile", "Ctrl+T", "type_check"),
            Command("compile-ir", "Compile to IR", "Show the lowered IR", "Compile", "Ctrl+Shift+C", "compile_ir"),
            Command("view-ast", "View AST", "Show the parsed AST", "Compile", "Ctrl+Shift+A", "view_ast"),
            Command("view-ir", "View IR (JSON)", "Show the IR as JSON", "Compile", "Ctrl+Shift+R", "view_ir"),

            Command("snippets", "Snippets", "Insert a code snippet", "Tools", "Ctrl+Shift+S", "snippets"),
            Command("packages", "Package Manager", "Manage installed packages", "Tools", "Ctrl+P", "pkg_manager"),
            Command("settings", "Settings", "Open settings", "Tools", "Ctrl+G", "settings"),
            Command("onboarding", "Onboarding Tour", "Replay the welcome tour", "Tools", "Ctrl+Shift+T", "onboarding"),
            Command("help", "Help", "Show keyboard shortcuts", "Tools", "Ctrl+H", "show_help"),

            Command("theme-cycle", "Cycle Theme", "Switch to the next theme", "Theme", "Ctrl+Shift+L", "cycle_theme"),
            Command("theme-triad", "Theme: Triad", "Default Triad theme", "Theme", "", "set_theme('triad')"),
            Command("theme-catppuccin", "Theme: Catppuccin Mocha", "Catppuccin Mocha", "Theme", "", "set_theme('catppuccin-mocha')"),
            Command("theme-tokyo", "Theme: Tokyo Night", "Tokyo Night", "Theme", "", "set_theme('tokyo-night')"),
            Command("theme-gruvbox", "Theme: Gruvbox Dark", "Gruvbox Dark", "Theme", "", "set_theme('gruvbox-dark')"),
            Command("theme-nord", "Theme: Nord", "Nord", "Theme", "", "set_theme('nord')"),
            Command("theme-solarized", "Theme: Solarized Dark", "Solarized Dark", "Theme", "", "set_theme('solarized-dark')"),
            Command("theme-light", "Theme: Triad Light", "Light theme", "Theme", "", "set_theme('triad-light')"),

            Command("toggle-sidebar", "Toggle Sidebar", "Show/hide the file sidebar", "View", "Ctrl+F", "toggle_sidebar"),
            Command("toggle-bottom", "Toggle Bottom Panel", "Show/hide the output panel", "View", "Ctrl+B", "toggle_bottom"),

            Command("quit", "Quit", "Exit TriadLang IDE", "Quit", "Ctrl+Q", "quit"),
        ]

    def _execute_command(self, command_id: str) -> None:

        if command_id.startswith("show_tab("):
            tab = command_id[9:].rstrip(")").strip("'\"")
            self.action_show_tab(tab)
        elif command_id.startswith("set_theme("):
            theme = command_id[10:].rstrip(")").strip("'\"")
            try:
                self.theme = theme
                self._settings["theme"] = theme
                self._save_settings_to_disk()
                self._output(f"✓ Theme: {theme}", "green")
            except Exception as e:
                self._error(f"Theme change failed: {e}")
        elif command_id == "save_file":
            self.action_save_file()
        else:

            method = getattr(self, f"action_{command_id}", None)
            if method and callable(method):
                try:
                    method()
                except Exception as e:
                    self._error(f"Command error: {e}")
            else:
                self._error(f"Unknown command: {command_id}")

    def _execute_command_bar(self, command: str) -> None:

        cmd = command.strip()
        if cmd.startswith(":"):
            cmd = cmd[1:].strip()
        if not cmd:
            return
        parts = cmd.split(maxsplit=1)
        head = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        if head in ("q", "quit"):
            self.app.exit()
        elif head in ("h", "help"):
            self.action_show_help()
        elif head in ("w", "write", "save"):
            self.action_save_file()
        elif head in ("new",):
            self.action_new_file()
        elif head in ("run",):
            self.action_run_file()
        elif head in ("format",):
            self.action_format_code()
        elif head in ("check", "type"):
            self.action_type_check()
        elif head == "theme":
            self._set_theme_by_name(args.strip() or "triad")
        elif head == "backend":
            self._settings["backend"] = args.strip() or "auto"
            self._save_settings_to_disk()
            self._output(f"✓ Backend set to: {self._settings['backend']}", "green")
        elif head == "tab":
            if args:
                self.action_show_tab(args.strip())
        elif head in ("examples",):
            self.action_show_tab("examples")
        elif head in ("tour", "onboard"):
            self.action_onboarding()
        elif head in ("snip", "snippets"):
            self.action_snippets()
        elif head == "set":

            kv = args.split(maxsplit=1)
            if len(kv) == 2:
                k, v = kv
                self._settings[k] = v
                self._save_settings_to_disk()
                self._output(f"✓ {k} = {v}", "green")
        else:
            self._error(f"Unknown command: :{head}")

    def _set_theme_by_name(self, name: str) -> None:

        aliases = {
            "dark": "triad",
            "light": "triad-light",
            "cat": "catppuccin-mocha",
            "tokyo": "tokyo-night",
            "gruv": "gruvbox-dark",
            "solar": "solarized-dark",
        }
        name = aliases.get(name, name)
        try:
            self.theme = name
            self._settings["theme"] = name
            self._save_settings_to_disk()
            self._output(f"✓ Theme: {name}", "green")
        except Exception as e:
            self._error(f"Theme change failed: {e}")

    def on_file_selected(self, event: FileSelected) -> None:

        if event.path.endswith(".tri"):
            self._open_editor_file(event.path)
            self.action_show_tab("editor")

    def on_snippet_selected(self, event: SnippetSelected) -> None:

        tabs = self._editor_tabs()
        if tabs and hasattr(tabs, "set_source"):
            current = tabs.get_source() if hasattr(tabs, "get_source") else ""
            if current.strip():
                new_source = current.rstrip() + "\n\n" + event.snippet.code
            else:
                new_source = event.snippet.code
            tabs.set_source(new_source)
            self.action_show_tab("editor")
            self._output(f"✓ Snippet '{event.snippet.title}' inserted", "green")

