from __future__ import annotations

import os
import sys

from rich.table import Table
from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Label, RichLog, Select

PY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

def _default_regimes() -> list[str]:
    try:
        from stdlib.regimes import list_regimes
        return list_regimes()
    except Exception:
        return [
            'B0', 'dispersive', 'anti_collapse', 'R5_crystal', '_pure',
            'B0_3d', 'HodgkinHuxley', 'ENSO_recharge',
            'England_autopoietic', 'Eigen_hypercycle', 'Belousov_Zhabotinsky',
            'LSV_market', 'MaxwellWiechert', 'Cepheid_pulsator',
            'DarkMatter_halo', 'Cosmological_inflation',
        ]


_REGIMES = _default_regimes()

_OBSERVABLES = ['k_star', 'crystallinity', 'peak', 'atom_count', 'norm', 'ipr', 'fwhm']

_COUPLING_DIAGRAMS = {
    'pair': lambda a, b: f"  ┌─────┐   κ   ┌─────┐\n  │ {a:^3s} │ ◄──► │ {b:^3s} │\n  └─────┘       └─────┘",
    'ring': lambda *regs: '\n'.join(
        [f"  {'  →  '.join(r.center(5) for r in regs)}",
         f"  {'  ←  '.join('κ'.center(5) for _ in regs)}"]
    ),
}

class SolverScreen(Screen):
    BINDINGS = [
        Binding("f5", "run_solver", "Run"),
        Binding("space", "toggle_play", "Play/Pause"),
        Binding("n", "step_forward", "Step"),
        Binding("ctrl+r", "reset", "Reset"),
    ]

    def compose(self):
        with Vertical():
            yield Label(" [bold cyan]Solver Visualization[/bold cyan]")
            with Horizontal():
                with Vertical(classes="dsl-col"):
                    yield Label("Regions:")
                    yield Input(placeholder="Region name", id="solver-reg-name")
                    yield Select(
                        [(r, r) for r in _REGIMES],
                        prompt="Regime",
                        id="solver-regime",
                    )
                    yield Input(placeholder="N", value="128", id="solver-n")
                    yield Input(placeholder="T", value="18.0", id="solver-t")
                    yield Input(placeholder="dt", value="0.005", id="solver-dt")
                    yield Input(placeholder="L", value="32.0", id="solver-l")
                    yield Input(placeholder="Lambda (manual mode only)", value="", id="solver-lambda")
                    yield Input(placeholder="Gamma (manual mode only)", value="", id="solver-gamma")
                    with Horizontal():
                        yield Button("+ Region", variant="success", id="solver-add-reg")
                        yield Button("Remove", variant="error", id="solver-rm-reg")
                with Vertical(classes="dsl-col"):
                    yield Label("Controls:")
                    with Horizontal():
                        yield Button("▶ Run", variant="success", id="solver-run-btn")
                        yield Button("⏸ Pause", variant="default", id="solver-pause-btn")
                        yield Button("⏭ Step", variant="primary", id="solver-step-btn")
                        yield Button("🔄 Reset", variant="default", id="solver-reset-btn")
                    yield Label("Coupling:")
                    yield Input(placeholder="Regions (comma-sep)", id="solver-couple-regs")
                    yield Select(
                        [("Pair", "pair"), ("Ring", "ring")],
                        prompt="Type",
                        id="solver-couple-type",
                    )
                    yield Input(placeholder="kappa", value="-2.5", id="solver-couple-kappa")
                    yield Button("Add Coupling", variant="success", id="solver-add-couple")
            yield Label("")
            with Horizontal():
                with Vertical(classes="dsl-col"):
                    yield Label(" [b]Regions[/b]")
                    yield RichLog(id="solver-regions-log", highlight=True, markup=True, classes="dsl-col")
                with Vertical(classes="dsl-col"):
                    yield Label(" [b]Field Viz[/b]")
                    yield RichLog(id="solver-output", highlight=True, markup=True, classes="dsl-col")
                with Vertical(classes="dsl-col"):
                    yield Label(" [b]Observables[/b]")
                    yield RichLog(id="solver-obs-log", highlight=True, markup=True, classes="dsl-col")

    def on_mount(self):
        self._regions = []
        self._couplings = []
        self._playing = False
        self._step_count = 0
        self._obs_history: dict[str, list[float]] = {o: [] for o in _OBSERVABLES}

    def on_button_pressed(self, event: Button.Pressed):
        bid = event.button.id
        if bid == "solver-add-reg":
            self._add_region()
        elif bid == "solver-rm-reg":
            self._rm_region()
        elif bid == "solver-run-btn":
            self._playing = True
            self.action_run_solver()
        elif bid == "solver-pause-btn":
            self._playing = False
        elif bid == "solver-step-btn":
            self._step_forward()
        elif bid == "solver-reset-btn":
            self.action_reset()
        elif bid == "solver-add-couple":
            self._add_coupling()

    def _add_region(self):
        name = self.query_one("#solver-reg-name", Input).value.strip()
        regime = self.query_one("#solver-regime", Select).value
        n = self.query_one("#solver-n", Input).value.strip() or "128"
        if not name or not regime:
            return
        self._regions.append({"name": name, "regime": regime, "n": n})
        self.query_one("#solver-reg-name", Input).value = ""
        self._update_regions_log()

    def _rm_region(self):
        if self._regions:
            self._regions.pop()
            self._update_regions_log()

    def _add_coupling(self):
        regs_str = self.query_one("#solver-couple-regs", Input).value.strip()
        ctype = self.query_one("#solver-couple-type", Select).value or "pair"
        kappa = self.query_one("#solver-couple-kappa", Input).value.strip() or "-2.5"
        if not regs_str:
            return
        self._couplings.append({"regs": regs_str, "type": ctype, "kappa": kappa})
        self.query_one("#solver-couple-regs", Input).value = ""
        self._update_regions_log()

    def _update_regions_log(self):
        log = self.query_one("#solver-regions-log", RichLog)
        log.clear()
        if self._regions:
            log.write(Text("Regions:", style="bold cyan"))
            for r in self._regions:
                log.write(Text(f"  {r['name']}: {r['regime']} N={r['n']}", style="green"))
        if self._couplings:
            log.write(Text("\nCouplings:", style="bold yellow"))
            for c in self._couplings:
                log.write(Text(f"  {c['type']}({c['regs']}) κ={c['kappa']}", style="yellow"))
                regs = [x.strip() for x in c['regs'].split(',')]
                if c['type'] == 'pair' and len(regs) >= 2:
                    diagram = _COUPLING_DIAGRAMS['pair'](regs[0][:3], regs[1][:3])
                    log.write(Text(diagram, style="dim"))
                elif c['type'] == 'ring' and len(regs) >= 3:
                    diagram = _COUPLING_DIAGRAMS['ring'](*[r[:5] for r in regs])
                    log.write(Text(diagram, style="dim"))
        if not self._regions and not self._couplings:
            log.write(Text("  (none — add regions above)", style="dim"))

    def action_run_solver(self):
        output = self.query_one("#solver-output", RichLog)
        output.clear()
        regime = self.query_one("#solver-regime", Select).value
        n_str = self.query_one("#solver-n", Input).value
        t_str = self.query_one("#solver-t", Input).value
        dt_str = self.query_one("#solver-dt", Input).value
        l_str = self.query_one("#solver-l", Input).value

        try:
            N = int(n_str) if n_str else 128
            T = float(t_str) if t_str else 18.0
            dt = float(dt_str) if dt_str else 0.005
            L = float(l_str) if l_str else 32.0
        except ValueError:
            output.write(Text("Invalid parameter values", style="red"))
            return

        output.write(Text(f"Solving: regime={regime}, N={N}, T={T}, dt={dt}, L={L}", style="cyan"))

        try:
            from runtime.core.solver import TriadParams, integrate
            from triad import ntri as np
            if regime:
                from stdlib.regimes import resolve_regime
                params = resolve_regime(regime, N=N, L=L, dt=dt)
                params.T = T
            else:
                params = TriadParams(N=N, T=T, L=L, dt=dt)
                lam_str = self.query_one("#solver-lambda", Input).value.strip()
                gam_str = self.query_one("#solver-gamma", Input).value.strip()
                if lam_str:
                    params.Lambda = float(lam_str)
                if gam_str:
                    params.Gamma = float(gam_str)
            result = integrate(params)
            psi = result['psi_final'].ravel()
            dx = float(result.get('dx', L / N))

            from cli.tui.widgets.viz_canvas import field_ascii, sparkline
            from runtime.physics.observables import (
                crystallinity,
                dominant_wavenumber,
                norm,
                peak_density,
            )

            k_star = float(dominant_wavenumber(psi, dx))
            crys = float(crystallinity(psi, dx))
            pk = float(peak_density(psi))
            nm = float(norm(psi, dx))

            self._obs_history['k_star'].append(k_star)
            self._obs_history['crystallinity'].append(crys)
            self._obs_history['peak'].append(pk)
            self._obs_history['norm'].append(nm)
            self._step_count += 1

            output.write(Text("\n|psi(x)|:", style="bold"))
            output.write(Text(field_ascii(list(np.abs(psi)), width=70)))

            obs_log = self.query_one("#solver-obs-log", RichLog)
            obs_log.clear()
            obs_table = Table(title=f"Step {self._step_count}")
            obs_table.add_column("Observable", style="cyan")
            obs_table.add_column("Value", style="green", justify="right")
            obs_table.add_column("History", style="yellow")
            for o in _OBSERVABLES:
                hist = self._obs_history.get(o, [])
                if hist:
                    obs_table.add_row(o, f"{hist[-1]:.6f}", sparkline(hist, 20))
            obs_log.write(obs_table)

            status = "▶ RUNNING" if self._playing else "⏸ PAUSED"
            output.write(Text(f"\n{status} │ Step {self._step_count}", style="bold cyan"))

        except Exception as e:
            output.write(Text(f"Solver error: {e}", style="red"))

    def action_toggle_play(self):
        self._playing = not self._playing
        if self._playing:
            self.action_run_solver()

    def action_step_forward(self):
        self._playing = False
        self.action_run_solver()

    def action_reset(self):
        self._regions = []
        self._couplings = []
        self._playing = False
        self._step_count = 0
        self._obs_history = {o: [] for o in _OBSERVABLES}
        self.query_one("#solver-regime", Select).value = None
        self.query_one("#solver-n", Input).value = "128"
        self.query_one("#solver-t", Input).value = "18.0"
        self.query_one("#solver-dt", Input).value = "0.005"
        self.query_one("#solver-l", Input).value = "32.0"
        self.query_one("#solver-output", RichLog).clear()
        self.query_one("#solver-obs-log", RichLog).clear()
        self._update_regions_log()

