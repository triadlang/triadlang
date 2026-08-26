from __future__ import annotations

from rich.text import Text
from textual.reactive import reactive
from textual.widget import Widget

from triad import ntri as np

_HEAT_RAMP = [
    ' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█',
]

_HEAT_COLORS = [
    '#000000', '#1a0a3b', '#2a0a5b', '#3b0a7b', '#4a0a9b',
    '#5a0abb', '#6a0adb', '#7a0afb', '#8a2af0', '#9a4ae0',
]

def _density_to_blocks(density: np.ndarray, width: int = 40, height: int = 12) -> str:

    rho = np.abs(density)
    if rho.size == 0:
        return "(empty field)"

    mx = rho.max()
    if mx < 1e-15:
        mx = 1.0
    rho_norm = rho / mx

    n = len(rho_norm)
    if n > width:

        bins = np.array_split(rho_norm, width)
        vals = np.array([b.mean() for b in bins])
    else:
        vals = rho_norm

    lines = []
    for h in range(height, 0, -1):
        thresh = h / height
        row = ''
        for v in vals:
            if v >= thresh:
                idx = min(int(v * (len(_HEAT_RAMP) - 1)), len(_HEAT_RAMP) - 1)
                row += _HEAT_RAMP[idx]
            else:
                row += ' '
        lines.append(row)
    return '\n'.join(lines)

def _density_to_heatmap_text(density: np.ndarray, width: int = 40, height: int = 12) -> Text:

    rho = np.abs(density)
    if rho.size == 0:
        return Text("(empty field)")
    mx = rho.max()
    if mx < 1e-15:
        mx = 1.0
    rho_norm = rho / mx
    n = len(rho_norm)
    if n > width:
        bins = np.array_split(rho_norm, width)
        vals = np.array([b.mean() for b in bins])
    else:
        vals = rho_norm
    text = Text()
    for h in range(height, 0, -1):
        thresh = h / height
        row = Text()
        for v in vals:
            if v >= thresh:
                ci = min(int(v * (len(_HEAT_COLORS) - 1)), len(_HEAT_COLORS) - 1)
                row += Text('█', style=f"{_HEAT_COLORS[ci]}")
            else:
                row += Text(' ')
        text.append_text(row)
        text.append('\n')
    return text

class HeatmapWidget(Widget):

    DEFAULT_CSS = """
    HeatmapWidget {
        width: 1fr;
        height: 1fr;
        background: $surface-darken-1;
        padding: 1;
        border: solid $primary;
    }
    """

    density: reactive[np.ndarray | None] = reactive(None)
    subtitle: reactive[str] = reactive("")

    def __init__(self, width: int = 40, height: int = 12, **kwargs):
        super().__init__(**kwargs)
        self._w = width
        self._h = height

    def watch_density(self, density: np.ndarray | None):
        if density is not None:
            self.update(_density_to_heatmap_text(density, self._w, self._h))

    def render(self):
        if self.density is None:
            return Text(f"  {self.subtitle}\n  (no solver data — run a .tri file to see |\u03c8|²)", style="dim")
        return _density_to_heatmap_text(self.density, self._w, self._h)

    def set_data(self, density: np.ndarray, subtitle: str = ""):
        self.density = density
        self.subtitle = subtitle
        self.refresh()

