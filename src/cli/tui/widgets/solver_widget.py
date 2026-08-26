from __future__ import annotations

import os
import sys

from rich.table import Table
from rich.text import Text
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical
from textual.widgets import Button, Input, Label, RichLog, Select, Static, Switch

PY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

_REGIMES = [
    'B0', 'dispersive', 'anti_collapse', 'R5_crystal', '_pure',
    'B0_3d', 'B0_2d', 'memory_heavy', 'HodgkinHuxley', 'ENSO_recharge',
    'England_autopoietic', 'Eigen_hypercycle', 'Belousov_Zhabotinsky',
    'LSV_market', 'MaxwellWiechert', 'Cepheid_pulsator',
    'DarkMatter_halo', 'Cosmological_inflation',
]

_OBSERVABLES = ['k_star', 'crystallinity', 'peak', 'norm', 'ipr', 'fwhm']

class SolverWidget(Static):

    DEFAULT_CSS = """
    SolverWidget {
        height: 1fr;
        overflow-y: auto;
    }
    #solver-grid {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 1fr;
        height: auto;
    }
    #solver-params {
        padding: 0 1;
    }
    #solver-controls {
        padding: 0 1;
    }
    #solver-viz {
        height: 1fr;
        padding: 0 1;
    }
    #solver-obs {
        height: 1fr;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("f5", "run_solver", "Run"),
        Binding("space", "toggle_play", "Play/Pause"),
        Binding("n", "step_forward", "Step"),
        Binding("ctrl+r", "reset", "Reset"),
    ]

    def compose(self):
        with Vertical():
            yield Label("[bold cyan]🔬 Solver[/bold cyan]")
            with Grid(id="solver-grid"):
                with Vertical(id="solver-params"):
                    yield Label("[bold]Parameters[/bold]")
                    yield Select([(r, r) for r in _REGIMES], prompt="Regime", id="solver-regime", value="B0")
                    yield Input(placeholder="N (grid points)", value="128", id="solver-n")
                    yield Input(placeholder="T (total time)", value="18.0", id="solver-t")
                    yield Input(placeholder="dt (time step)", value="0.005", id="solver-dt")
                    yield Input(placeholder="L (domain size)", value="32.0", id="solver-l")
                    yield Input(placeholder="Lambda", value="0.5", id="solver-lambda")
                    yield Input(placeholder="Gamma", value="0.01", id="solver-gamma")
                    yield Label("[dim]Live update:[/dim]")
                    yield Switch(id="solver-live", value=False)

                with Vertical(id="solver-controls"):
                    yield Label("[bold]Controls[/bold]")
                    with Horizontal():
                        yield Button("▶ Run", variant="success", id="solver-run-btn")
                        yield Button("⏸ Pause", variant="warning", id="solver-pause-btn")
                        yield Button("⏭ Step", variant="primary", id="solver-step-btn")
                        yield Button("↺ Reset", variant="default", id="solver-reset-btn")
                    yield Label("")
                    yield Label("[bold]Multi-Region[/bold] [dim](advanced)[/dim]")
                    yield Input(placeholder="Region name", id="solver-reg-name")
                    yield Button("+ Add Region", variant="success", id="solver-add-reg")
                    yield Button("− Remove Last", variant="error", id="solver-rm-reg")
                    yield Label("")
                    yield Label("[bold]Coupling[/bold] [dim](advanced)[/dim]")
                    yield Input(placeholder="Regions (comma-sep)", id="solver-couple-regs")
                    yield Select([("Pair", "pair"), ("Ring", "ring")], prompt="Type", id="solver-couple-type", value="pair")
                    yield Input(placeholder="kappa", value="-2.5", id="solver-couple-kappa")
                    yield Button("+ Add Coupling", variant="success", id="solver-add-couple")

            with Horizontal():
                with Vertical(id="solver-viz"):
                    yield Label("[bold]|ψ(x)|²[/bold]")
                    yield RichLog(id="solver-output", highlight=True, markup=True, wrap=False)
                with Vertical(id="solver-obs"):
                    yield Label("[bold]Observables[/bold]")
                    yield RichLog(id="solver-obs-log", highlight=True, markup=True)

    def on_mount(self):
        self._regions: list[dict] = []
        self._couplings: list[dict] = []
        self._playing: bool = False
        self._step_count: int = 0
        self._obs_history: dict[str, list[float]] = {o: [] for o in _OBSERVABLES}
        self._psi_current = None
        self._params_current = None

    def on_button_pressed(self, event):
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
        output = self.query_one("#solver-output", RichLog)
        if self._regions:
            output.write(Text("Regions:", style="bold cyan"))
            for r in self._regions:
                output.write(Text(f"  {r['name']}: {r['regime']} N={r['n']}", style="green"))
        if self._couplings:
            output.write(Text("Couplings:", style="bold yellow"))
            for c in self._couplings:
                output.write(Text(f"  {c['type']}({c['regs']}) κ={c['kappa']}", style="yellow"))

    def action_run_solver(self):
        output = self.query_one("#solver-output", RichLog)
        output.clear()
        regime = self.query_one("#solver-regime", Select).value
        n_str = self.query_one("#solver-n", Input).value
        t_str = self.query_one("#solver-t", Input).value
        dt_str = self.query_one("#solver-dt", Input).value
        l_str = self.query_one("#solver-l", Input).value
        lam_str = self.query_one("#solver-lambda", Input).value
        gam_str = self.query_one("#solver-gamma", Input).value

        try:
            N = int(n_str) if n_str else 128
            T = float(t_str) if t_str else 18.0
            dt = float(dt_str) if dt_str else 0.005
            L = float(l_str) if l_str else 32.0
            Lambda = float(lam_str) if lam_str else 0.5
            Gamma = float(gam_str) if gam_str else 0.01
        except ValueError:
            output.write(Text("✗ Invalid parameter values", style="red"))
            return

        output.write(Text(
            f"Solving: regime={regime}, N={N}, T={T}, dt={dt}, L={L}, Λ={Lambda}, Γ={Gamma}",
            style="bold cyan",
        ))

        try:
            from runtime.core.solver import TriadParams, integrate
            from triad import ntri as np

            if regime:
                from stdlib.regimes import resolve_regime
                params = resolve_regime(regime, N=N, L=L, dt=dt)
                params.T = T
                params.Lambda = Lambda
                params.Gamma = Gamma
            else:
                params = TriadParams(N=N, T=T, L=L, dt=dt, Lambda=Lambda, Gamma=Gamma)

            self._params_current = params
            result = integrate(params)
            psi = result["psi_final"].ravel()
            dx = float(result.get("dx", L / N))
            self._psi_current = psi

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

            self._obs_history["k_star"].append(k_star)
            self._obs_history["crystallinity"].append(crys)
            self._obs_history["peak"].append(pk)
            self._obs_history["norm"].append(nm)
            self._step_count += 1

            abs_psi = list(np.abs(psi))
            output.write(Text("\n|ψ(x)|:", style="bold"))
            output.write(Text(field_ascii(abs_psi, width=70)))

            self._send_heatmap_debug(psi, dx, k_star, crys, pk, nm)

            obs_log = self.query_one("#solver-obs-log", RichLog)
            obs_log.clear()
            t = Table(title=f"Step {self._step_count}", border_style="cyan")
            t.add_column("Observable", style="cyan")
            t.add_column("Value", style="green", justify="right")
            t.add_column("Trend", style="yellow")
            for o in _OBSERVABLES:
                hist = self._obs_history.get(o, [])
                if hist:
                    t.add_row(o, f"{hist[-1]:.6f}", sparkline(hist, 25))
            obs_log.write(t)

            status = "▶ RUNNING" if self._playing else "⏸ PAUSED"
            output.write(Text(f"\n{status} | Step {self._step_count} | dt={dt}", style="bold cyan"))

        except Exception as e:
            output.write(Text(f"✗ Solver error: {e}", style="red"))
            import traceback
            output.write(Text(traceback.format_exc(), style="dim red"))

    def _send_heatmap_debug(self, psi, dx, k_star, crys, pk, nm):


        from triad import ntri as np
        rho = np.abs(psi) ** 2

        target_w, target_h = 60, 12
        if len(rho) > target_w:
            block = rho[:target_w * (len(rho) // target_w)].reshape(-1, len(rho) // target_w).mean(axis=1)
        else:
            block = rho
        block = np.interp(np.linspace(0, len(block) - 1, target_w), np.arange(len(block)), block)
        lines = []
        for val in block:
            idx = min(int(val / max(block.max(), 1e-12) * 7), 7)
            ch = [" ", "░", "░", "▒", "▒", "▓", "▓", "█"][idx]
            lines.append(ch * target_h)
        heatmap = "\n".join(lines)
        msg = (
            f"[bold cyan]Solver Debug @ step {self._step_count}[/bold cyan]\n"
            f"k*={k_star:.4f}  crys={crys:.4f}  peak={pk:.4f}  norm={nm:.4f}\n"
            f"{heatmap}"
        )
        from textual.message import Message
        class _Dbg(Message):
            def __init__(self, text):
                super().__init__()
                self.text = text

        try:
            from cli.tui.app import OutputMessage
            self.post_message(OutputMessage(msg, target="debug"))
        except (LookupError, RuntimeError):
            pass

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
        self._psi_current = None
        self._params_current = None
        self.query_one("#solver-regime", Select).value = "B0"
        self.query_one("#solver-n", Input).value = "128"
        self.query_one("#solver-t", Input).value = "18.0"
        self.query_one("#solver-dt", Input).value = "0.005"
        self.query_one("#solver-l", Input).value = "32.0"
        self.query_one("#solver-lambda", Input).value = "0.5"
        self.query_one("#solver-gamma", Input).value = "0.01"
        self.query_one("#solver-output", RichLog).clear()
        self.query_one("#solver-obs-log", RichLog).clear()

