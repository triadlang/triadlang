from __future__ import annotations

import os
import sys

from rich.syntax import Syntax
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Label, RichLog, Select, Static

PY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

DSL_TEMPLATES = {
    "reg": """// Single region module
module {name} {{
    reg {reg_name} {{
        N   {N}
        T   {T}
        L   {L}
        dt  {dt}
        Lambda {Lambda}
        Gamma {Gamma}
    }}
}}
""",
    "pair": """// Coupled pair module
module {name} {{
    reg A {{
        N {N}  T {T}  L {L}  dt {dt}
        Lambda {Lambda}  Gamma {Gamma}
    }}
    reg B {{
        N {N}  T {T}  L {L}  dt {dt}
        Lambda {Lambda}  Gamma {Gamma}
    }}
    couple A B {{
        kappa {kappa}
    }}
}}
""",
    "ring": """// Ring coupling module
module {name} {{
{regs_text}
    ring {{
        kappa {kappa}
    }}
}}
""",
    "sequence": """// Sequential evolution module
module {name} {{
    reg main {{
        N {N}  T {T}  L {L}  dt {dt}
        Lambda {Lambda}  Gamma {Gamma}
    }}
    sequence main {{
        evolve {regime1} for {T1}
        evolve {regime2} for {T2}
        evolve B0 for {T3}
    }}
}}
""",
}

class DSLWidget(Static):

    DEFAULT_CSS = """
    DSLWidget {
        height: 1fr;
        overflow-y: auto;
    }
    #dsl-main {
        layout: horizontal;
        height: 1fr;
    }
    #dsl-form {
        width: 34;
        padding: 0 1;
    }
    #dsl-preview {
        width: 1fr;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("f5", "generate", "Generate"),
    ]

    def compose(self):
        with Vertical():
            yield Label("[bold cyan]🏗 DSL Builder[/bold cyan]")
            with Horizontal(id="dsl-main"):
                with Vertical(id="dsl-form"):
                    yield Label("[bold]Template[/bold]")
                    yield Select(
                        [("Single Region", "reg"), ("Coupled Pair", "pair"),
                         ("Ring", "ring"), ("Sequence", "sequence")],
                        id="dsl-kind",
                        prompt="Kind",
                        value="reg",
                    )
                    yield Input(placeholder="Module name", value="my_module", id="dsl-name")
                    yield Input(placeholder="Region name", value="main", id="dsl-reg-name")
                    yield Input(placeholder="N (grid)", value="128", id="dsl-n")
                    yield Input(placeholder="T (time)", value="18.0", id="dsl-t")
                    yield Input(placeholder="dt (step)", value="0.005", id="dsl-dt")
                    yield Input(placeholder="L (domain)", value="32.0", id="dsl-l")
                    yield Input(placeholder="Lambda", value="0.5", id="dsl-lambda")
                    yield Input(placeholder="Gamma", value="0.01", id="dsl-gamma")
                    yield Input(placeholder="kappa (coupling)", value="-2.5", id="dsl-kappa")
                    yield Input(placeholder="Ring size", value="3", id="dsl-ring-size")
                    yield Label("")
                    yield Label("[bold]Sequence (if applicable)[/bold]")
                    yield Input(placeholder="Regime 1", value="B0", id="dsl-regime1")
                    yield Input(placeholder="T1", value="5.0", id="dsl-t1")
                    yield Input(placeholder="Regime 2", value="dispersive", id="dsl-regime2")
                    yield Input(placeholder="T2", value="5.0", id="dsl-t2")
                    yield Input(placeholder="T3", value="8.0", id="dsl-t3")
                    yield Label("")
                    with Horizontal():
                        yield Button("▶ Generate", variant="success", id="dsl-gen-btn")
                        yield Button("📋 Copy to Editor", variant="primary", id="dsl-copy-btn")
                        yield Button("🔬 Open in Solver", variant="default", id="dsl-solver-btn")

                with Vertical(id="dsl-preview"):
                    yield Label("[bold]Generated .tri code[/bold]")
                    yield RichLog(id="dsl-code", highlight=True, markup=False, wrap=False)

    def on_button_pressed(self, event):
        bid = event.button.id
        if bid == "dsl-gen-btn":
            self._generate()
        elif bid == "dsl-copy-btn":
            self._copy_to_editor()
        elif bid == "dsl-solver-btn":
            self._open_in_solver()

    def action_generate(self):
        self._generate()

    def _get_params(self):
        return {
            "name": self.query_one("#dsl-name", Input).value or "my_module",
            "reg_name": self.query_one("#dsl-reg-name", Input).value or "main",
            "N": self.query_one("#dsl-n", Input).value or "128",
            "T": self.query_one("#dsl-t", Input).value or "18.0",
            "dt": self.query_one("#dsl-dt", Input).value or "0.005",
            "L": self.query_one("#dsl-l", Input).value or "32.0",
            "Lambda": self.query_one("#dsl-lambda", Input).value or "0.5",
            "Gamma": self.query_one("#dsl-gamma", Input).value or "0.01",
            "kappa": self.query_one("#dsl-kappa", Input).value or "-2.5",
            "regime1": self.query_one("#dsl-regime1", Input).value or "B0",
            "regime2": self.query_one("#dsl-regime2", Input).value or "dispersive",
            "T1": self.query_one("#dsl-t1", Input).value or "5.0",
            "T2": self.query_one("#dsl-t2", Input).value or "5.0",
            "T3": self.query_one("#dsl-t3", Input).value or "8.0",
        }

    def _generate(self):
        kind = self.query_one("#dsl-kind", Select).value or "reg"
        params = self._get_params()

        try:
            if kind == "ring":
                ring_size = int(self.query_one("#dsl-ring-size", Input).value or "3")
                regs_text = "\n".join(
                    f"    reg r{i} {{ N {params['N']}  T {params['T']}  L {params['L']}  dt {params['dt']} }}"
                    for i in range(ring_size)
                )
                code = DSL_TEMPLATES[kind].format(regs_text=regs_text, **params)
            else:
                code = DSL_TEMPLATES[kind].format(**params)
        except KeyError:
            code = "// Unknown template kind"
        except Exception as e:
            code = f"// Error generating: {e}"

        log = self.query_one("#dsl-code", RichLog)
        log.clear()
        log.write(Syntax(code, "triad", theme="monokai", line_numbers=True))
        self._current_code = code

    def _copy_to_editor(self):

        code = getattr(self, "_current_code", "")
        if not code:
            self._generate()
            code = getattr(self, "_current_code", "")
        if not code:
            return
        app = self.app
        if hasattr(app, "_editor_tabs"):
            tabs = app._editor_tabs()
            if tabs:
                tabs.set_source(code)
                app.action_show_tab("editor")
                app._output("✓ DSL code copied to editor", "green")

    def _open_in_solver(self):

        app = self.app
        if hasattr(app, "action_show_tab"):
            app.action_show_tab("solver")

