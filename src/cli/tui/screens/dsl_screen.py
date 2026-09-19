from __future__ import annotations

import os
import sys

from rich.syntax import Syntax
from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Input, Label, RichLog, Select

PY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

def _default_regimes() -> list[tuple[str, str]]:
    try:
        from stdlib.regimes import list_regimes
        return [(r, r) for r in list_regimes()]
    except Exception:
        return [
            ('B0', 'B0'),
            ('dispersive', 'dispersive'),
            ('anti_collapse', 'anti_collapse'),
            ('R5_crystal', 'R5_crystal'),
            ('_pure', '_pure'),
        ]


_REGIMES = _default_regimes()

_OBSERVABLES = ['k_star', 'crystallinity', 'peak', 'atom_count']

class DSLScreen(Screen):
    BINDINGS = [
        Binding("f5", "generate_and_run", "Run"),
    ]

    def compose(self):
        with Vertical():
            yield Label(" [bold cyan]DSL Builder[/bold cyan]  —  build reg/pair/ring/OBSERVE visually")
            with Horizontal():
                with Vertical(classes="dsl-col"):
                    yield Label("Regions:")
                    yield Input(placeholder="Region name", id="dsl-reg-name")
                    yield Select(
                        [(label, regime) for regime, label in _REGIMES],
                        prompt="Select regime",
                        id="dsl-reg-regime",
                    )
                    yield Input(placeholder="N (grid size)", value="128", id="dsl-reg-n")
                    yield Button("Add Region", variant="success", id="dsl-add-reg")
                with Vertical(classes="dsl-col"):
                    yield Label("Coupling:")
                    yield Input(placeholder="Region names (comma-sep)", id="dsl-couple-names")
                    yield Input(placeholder="kappa", value="-2.5", id="dsl-couple-kappa")
                    yield Select(
                        [("Pair (2 regions)", "pair"), ("Ring (3+ regions)", "ring")],
                        prompt="Coupling type",
                        id="dsl-couple-type",
                    )
                    yield Button("Add Coupling", variant="success", id="dsl-add-couple")
                with Vertical(classes="dsl-col"):
                    yield Label("Config:")
                    yield Input(placeholder="Temperature", value="18.0", id="dsl-temp")
                    yield Label("Observables:")
                    for obs in _OBSERVABLES:
                        yield Checkbox(obs, id=f"dsl-obs-{obs}")
                    yield Button("Generate Code", variant="primary", id="dsl-generate")
                    yield Button("Run", variant="success", id="dsl-run")
            yield Label(" [bold]Generated Code:[/bold]")
            yield RichLog(id="dsl-code", highlight=True, markup=True)
            yield Label(" [bold]Output:[/bold]")
            yield RichLog(id="dsl-output", highlight=True, markup=True)

    def on_mount(self):
        self._regions = []
        self._couplings = []
        self._update_code()

    def on_button_pressed(self, event: Button.Pressed):
        bid = event.button.id
        if bid == "dsl-add-reg":
            self._add_region()
        elif bid == "dsl-add-couple":
            self._add_coupling()
        elif bid == "dsl-generate":
            self._update_code()
        elif bid == "dsl-run":
            self.action_generate_and_run()

    def _add_region(self):
        name = self.query_one("#dsl-reg-name", Input).value.strip()
        regime = self.query_one("#dsl-reg-regime", Select).value
        n = self.query_one("#dsl-reg-n", Input).value.strip() or "128"
        if not name or not regime:
            return
        self._regions.append({"name": name, "regime": regime, "n": n})
        self.query_one("#dsl-reg-name", Input).value = ""
        self._update_code()

    def _add_coupling(self):
        names = self.query_one("#dsl-couple-names", Input).value.strip()
        kappa = self.query_one("#dsl-couple-kappa", Input).value.strip() or "-2.5"
        ctype = self.query_one("#dsl-couple-type", Select).value or "pair"
        if not names:
            return
        self._couplings.append({"names": names, "kappa": kappa, "type": ctype})
        self.query_one("#dsl-couple-names", Input).value = ""
        self._update_code()

    def _update_code(self):
        temp = self.query_one("#dsl-temp", Input).value.strip() or "18.0"
        obs_selected = []
        for obs in _OBSERVABLES:
            try:
                cb = self.query_one(f"#dsl-obs-{obs}", Checkbox)
                if cb.value:
                    obs_selected.append(obs)
            except (LookupError, AttributeError):
                continue

        lines = [f"@T({temp})"]
        for r in self._regions:
            lines.append(f"reg {r['name']} : {r['regime']} = {r['n']};")
        for c in self._couplings:
            names = c['names']
            kappa = c['kappa']
            ctype = c['type']
            lines.append(f"{ctype}({names}) kappa={kappa} for T={temp};")
        for r in self._regions:
            if obs_selected:
                lines.append(f"OBSERVE {r['name']} {', '.join(obs_selected)};")
        code = '\n'.join(lines)

        code_log = self.query_one("#dsl-code", RichLog)
        code_log.clear()
        syntax = Syntax(code, "tri", theme="monokai", line_numbers=True)
        code_log.write(syntax)

    def action_generate_and_run(self):
        self._update_code()
        code_log = self.query_one("#dsl-code", RichLog)
        output_log = self.query_one("#dsl-output", RichLog)
        output_log.clear()

        temp = self.query_one("#dsl-temp", Input).value.strip() or "18.0"
        lines = [f"@T({temp})"]
        for r in self._regions:
            lines.append(f"reg {r['name']} : {r['regime']} = {r['n']};")
        for c in self._couplings:
            lines.append(f"{c['type']}({c['names']}) kappa={c['kappa']} for T={temp};")
        for r in self._regions:
            lines.append(f"OBSERVE {r['name']} k_star, peak;")
        code = '\n'.join(lines)

        import io

        from frontend.parser_universal import parse
        from runtime.compiler_runtime import TriadCompiler
        try:
            mod = parse(code, '<dsl>')
            compiler = TriadCompiler()
            old_stdout = sys.stdout
            sys.stdout = buf = io.StringIO()
            try:
                compiler.compile_and_run(mod)
            finally:
                captured = buf.getvalue()
                sys.stdout = old_stdout
            if captured.strip():
                for line in captured.strip().split('\n'):
                    output_log.write(Text(line))
            output_log.write(Text("DSL program executed OK", style="bold green"))
        except Exception as e:
            output_log.write(Text(f"{type(e).__name__}: {e}", style="red"))

