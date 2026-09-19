from __future__ import annotations

import math as _math
import os

from stdlib._canvas import Image, ImageDraw, ImageFont

_DEFAULT_FONT_CANDIDATES = [
    '/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf',
    '/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf',
    '/usr/share/fonts/TTF/DejaVuSans.ttf',
    '/usr/share/fonts/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]

def _find_default_font() -> str | None:
    for p in _DEFAULT_FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    try:
        import glob as _glob
        for ext in ('ttf', 'otf'):
            matches = _glob.glob(f'/usr/share/fonts/**/*.{ext}', recursive=True)
            if matches:
                return matches[0]
    except (ImportError, OSError):
        return None
    return None

_DEFAULT_FONT_PATH = _find_default_font()

def _get_font(size: float):
    if _DEFAULT_FONT_PATH:
        try:
            return ImageFont.truetype(_DEFAULT_FONT_PATH, int(size))
        except (OSError, ValueError, AttributeError):
            pass
    return ImageFont.load_default()

def _measure(font, text):
    try:
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return font.getsize(text)

class _RcParams(dict):
    def __init__(self):
        super().__init__()
        self.update({
            'figure.figsize': (6.4, 4.8),
            'figure.dpi': 100.0,
            'figure.facecolor': 'white',
            'figure.color': 'white',
            'savefig.dpi': 100.0,
            'savefig.bbox': 'standard',
            'savefig.pad_inches': 0.1,
            'axes.facecolor': 'white',
            'axes.color': 'black',
            'axes.linewidth': 1.0,
            'axes.grid': False,
            'axes.titlesize': 12.0,
            'axes.labelsize': 10.0,
            'axes.unicode_minus': True,
            'xtick.labelsize': 9.0,
            'ytick.labelsize': 9.0,
            'xtick.color': 'black',
            'ytick.color': 'black',
            'lines.linewidth': 1.5,
            'lines.color': 'C0',
            'lines.marker': 'None',
            'lines.markersize': 6.0,
            'patch.color': 'black',
            'patch.facecolor': 'C0',
            'font.family': ['sans-serif'],
            'font.size': 10.0,
            'legend.fontsize': 9.0,
            'legend.frameon': True,
            'grid.color': '#b0b0b0',
            'grid.linestyle': '-',
            'grid.linewidth': 0.5,
            'grid.alpha': 0.7,
        })

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

rcParams = _RcParams()
rcParamsDefault = _RcParams()

def _rc(name, default=None):
    return rcParams.get(name, default)

_CYCLE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

_NAMED_COLORS = {
    'white': (1.0, 1.0, 1.0),
    'black': (0.0, 0.0, 0.0),
    'red': (0.86, 0.20, 0.20),
    'green': (0.20, 0.65, 0.30),
    'blue': (0.20, 0.40, 0.85),
    'yellow': (0.95, 0.85, 0.20),
    'orange': (1.0, 0.55, 0.10),
    'purple': (0.55, 0.30, 0.75),
    'pink': (0.95, 0.55, 0.75),
    'gray': (0.55, 0.55, 0.55),
    'grey': (0.55, 0.55, 0.55),
    'cyan': (0.20, 0.75, 0.85),
    'brown': (0.55, 0.30, 0.10),
    'none': None,
    'C0': '#1f77b4', 'C1': '#ff7f0e', 'C2': '#2ca02c', 'C3': '#d62728',
    'C4': '#9467bd', 'C5': '#8c564b', 'C6': '#e377c2', 'C7': '#7f7f7f',
    'C8': '#bcbd22', 'C9': '#17becf',
}

def _clamp_rgb(v: int) -> int:
    return 0 if v < 0 else (255 if v > 255 else v)


def _to_rgb(value):
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) in (3, 4):
        if len(value) == 4 and value[3] == 0.0:
            return None
        return (_clamp_rgb(int(round(value[0] * 255)) if value[0] <= 1.0 else int(value[0])),
                _clamp_rgb(int(round(value[1] * 255)) if value[1] <= 1.0 else int(value[1])),
                _clamp_rgb(int(round(value[2] * 255)) if value[2] <= 1.0 else int(value[2])))
    if isinstance(value, str):
        s = value.strip()
        if s in _NAMED_COLORS:
            v = _NAMED_COLORS[s]
            if v is None:
                return None
            if isinstance(v, str):
                s = v
            else:
                return (int(round(v[0] * 255)),
                        int(round(v[1] * 255)),
                        int(round(v[2] * 255)))
        if s.startswith('#'):
            s = s[1:]
            if len(s) == 3:
                s = ''.join(ch * 2 for ch in s)
            if len(s) == 6:
                try:
                    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
                except ValueError:
                    return None
        if s.startswith('rgb('):
            inside = s[4:].rstrip(')').split(',')
            try:
                vals = [_clamp_rgb(int(float(x.strip()))) for x in inside[:3]]
                return tuple(vals)
            except ValueError:
                return None
    return None

def _rgba(value, alpha: float = 1.0):
    base = _to_rgb(value)
    if base is None:
        return None
    return (base[0], base[1], base[2], int(round(alpha * 255)))

def _next_color(index: int) -> str:
    return _CYCLE[index % len(_CYCLE)]

def _stops_to_cmap(name, stops):
    def cmap(n=256):
        out = []
        for i in range(int(n)):
            t = i / max(1, n - 1)
            for j in range(len(stops) - 1):
                t0, c0 = stops[j]
                t1, c1 = stops[j + 1]
                if t0 <= t <= t1:
                    f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                    r = c0[0] + (c1[0] - c0[0]) * f
                    g = c0[1] + (c1[1] - c0[1]) * f
                    b = c0[2] + (c1[2] - c0[2]) * f
                    out.append((int(round(r * 255)),
                                int(round(g * 255)),
                                int(round(b * 255))))
                    break
        return out
    cmap.__name__ = name
    return cmap

_VIRIDIS = _stops_to_cmap('viridis', [
    (0.0, (0.267, 0.005, 0.329)),
    (0.25, (0.229, 0.322, 0.546)),
    (0.5, (0.127, 0.567, 0.551)),
    (0.75, (0.369, 0.789, 0.383)),
    (1.0, (0.993, 0.906, 0.144)),
])

_INFERNO = _stops_to_cmap('inferno', [
    (0.0, (0.001, 0.000, 0.014)),
    (0.25, (0.310, 0.058, 0.422)),
    (0.5, (0.717, 0.215, 0.475)),
    (0.75, (0.989, 0.498, 0.146)),
    (1.0, (0.988, 1.000, 0.645)),
])

_PLASMA = _stops_to_cmap('plasma', [
    (0.0, (0.058, 0.030, 0.527)),
    (0.25, (0.494, 0.012, 0.658)),
    (0.5, (0.798, 0.281, 0.469)),
    (0.75, (0.973, 0.555, 0.232)),
    (1.0, (0.940, 0.975, 0.131)),
])

_MAGMA = _stops_to_cmap('magma', [
    (0.0, (0.001, 0.000, 0.014)),
    (0.25, (0.290, 0.090, 0.426)),
    (0.5, (0.712, 0.198, 0.469)),
    (0.75, (0.984, 0.524, 0.327)),
    (1.0, (0.987, 0.991, 0.749)),
])

_GRAY = _stops_to_cmap('gray', [
    (0.0, (0.0, 0.0, 0.0)),
    (1.0, (1.0, 1.0, 1.0)),
])

_HOT = _stops_to_cmap('hot', [
    (0.0, (0.04, 0.0, 0.0)),
    (0.33, (1.0, 0.0, 0.0)),
    (0.66, (1.0, 1.0, 0.0)),
    (1.0, (1.0, 1.0, 1.0)),
])

_COOL = _stops_to_cmap('cool', [
    (0.0, (0.0, 1.0, 1.0)),
    (1.0, (1.0, 0.0, 1.0)),
])

_CIVIDIS = _stops_to_cmap('cividis', [
    (0.0, (0.000, 0.135, 0.305)),
    (0.5, (0.439, 0.471, 0.272)),
    (1.0, (0.940, 0.875, 0.156)),
])

_CMAPS = {
    'viridis': _VIRIDIS, 'inferno': _INFERNO, 'plasma': _PLASMA,
    'magma': _MAGMA, 'gray': _GRAY, 'grey': _GRAY,
    'hot': _HOT, 'cool': _COOL, 'cividis': _CIVIDIS,
}

def get_cmap(name):
    if callable(name):
        return name
    if isinstance(name, str):
        key = name.lower()
        if key in _CMAPS:
            return _CMAPS[key]
    raise ValueError(f'unknown colormap: {name!r}')

class Normalize:
    def __init__(self, vmin=None, vmax=None, clip=False):
        self.vmin = vmin
        self.vmax = vmax
        self.clip = clip

    def __call__(self, value):
        v = float(value)
        if self.vmin is None or self.vmax is None:
            return v
        if self.vmax == self.vmin:
            return 0.0
        t = (v - self.vmin) / (self.vmax - self.vmin)
        if self.clip:
            t = max(0.0, min(1.0, t))
        return t

    def inverse(self, t):
        if self.vmin is None or self.vmax is None:
            return t
        return self.vmin + t * (self.vmax - self.vmin)

def _nice_ticks(lo, hi, count=6, integer=False, scale='triad'):
    if lo is None or hi is None or not _math.isfinite(lo) or not _math.isfinite(hi) or lo == hi:
        return [lo] if lo is not None else []
    if scale == 'log':
        if lo <= 0 or hi <= 0:
            return []
        llo, lhi = _math.log10(lo), _math.log10(hi)
        step = max(1.0, (lhi - llo) / count)
        ticks = []
        v = llo
        while v <= lhi + 1e-9:
            ticks.append(10 ** v)
            v += step
        return ticks
    span = hi - lo
    raw_step = span / max(1, count - 1)
    if integer:
        step = max(1, int(raw_step))
        start = int(lo) - (int(lo) % step)
        end = int(hi) + step
        return list(range(start, end + 1, step))
    magnitude = 10 ** _math.floor(_math.log10(abs(raw_step)))
    candidates = [1, 2, 2.5, 5, 10]
    step = magnitude
    for c in candidates:
        if magnitude * c >= raw_step:
            step = magnitude * c
            break
    start = _math.floor(lo / step) * step
    end = _math.ceil(hi / step) * step
    n = int(round((end - start) / step)) + 1
    return [start + i * step for i in range(n)]

def _format_tick(v, scale='triad'):
    if scale == 'log':
        if v <= 0:
            return ''
        e = _math.log10(v)
        if abs(e - round(e)) < 1e-9:
            return f'1e{int(round(e))}'
        return f'{v:.3g}'
    av = abs(v)
    if av == 0:
        return '0'
    if av >= 1e4 or av < 1e-3:
        return f'{v:.2e}'
    if av >= 100:
        return f'{v:.0f}'
    if av >= 10:
        return f'{v:.1f}'
    return f'{v:.2f}'

def _log_ticks(lo, hi):
    if lo <= 0 or hi <= 0:
        return []
    llo, lhi = _math.log10(lo), _math.log10(hi)
    ticks = []
    for e in range(int(_math.floor(llo)), int(_math.ceil(lhi)) + 1):
        v = 10 ** e
        if lo <= v <= hi:
            ticks.append(v)
    return ticks

class _Artist:
    def __init__(self):
        self.visible = True
        self.label = None
        self.zorder = 1

    def draw(self, axes, draw_ctx):
        raise NotImplementedError(f'{type(self).__name__}.draw not implemented')

class Line2D(_Artist):
    def __init__(self, xdata, ydata, color=None, linewidth=None, linestyle='-',
                 marker='None', markersize=None, alpha=1.0, label=None, zorder=2):
        super().__init__()
        self.xdata = list(xdata)
        self.ydata = list(ydata)
        self.color = color
        self.linewidth = linewidth
        self.linestyle = linestyle
        self.marker = marker
        self.markersize = markersize
        self.alpha = alpha
        self.label = label
        self.zorder = zorder

    def draw(self, axes, draw_ctx):
        draw = draw_ctx['draw']
        color = self.color or _rc('lines.color')
        lw = self.linewidth or _rc('lines.linewidth')
        rgba = _rgba(color, self.alpha)
        if rgba is None:
            return
        n = min(len(self.xdata), len(self.ydata))
        if n < 1:
            return
        if self.linestyle not in ('None', None, ' ', ''):
            xs = [axes._proj_x(self.xdata[i]) for i in range(n)]
            ys = [axes._proj_y(self.ydata[i]) for i in range(n)]
            if self.linestyle == '-':
                for i in range(n - 1):
                    draw.line([(xs[i], ys[i]), (xs[i + 1], ys[i + 1])],
                              fill=rgba, width=max(1, int(round(lw))))
            elif self.linestyle == '--':
                _draw_dashed(draw, list(zip(xs, ys)), rgba, lw)
            elif self.linestyle == '-.':
                _draw_dashdot(draw, list(zip(xs, ys)), rgba, lw)
            elif self.linestyle == ':':
                _draw_dotted(draw, list(zip(xs, ys)), rgba, lw)
        if self.marker not in ('None', None, ' ', ''):
            ms = self.markersize or _rc('lines.markersize')
            r = max(1, int(round(ms / 2)))
            for i in range(n):
                cx, cy = axes._proj_x(self.xdata[i]), axes._proj_y(self.ydata[i])
                _draw_marker(draw, self.marker, cx, cy, r, rgba)

def _draw_dashed(draw, pts, fill, lw):
    on = max(2, int(round(lw * 4)))
    off = max(2, int(round(lw * 2)))
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        dx, dy = x1 - x0, y1 - y0
        length = _math.hypot(dx, dy)
        if length == 0:
            continue
        ux, uy = dx / length, dy / length
        pos = 0.0
        drawing = True
        while pos < length:
            seg = on if drawing else off
            seg_end = min(pos + seg, length)
            if drawing:
                sx = x0 + ux * pos
                sy = y0 + uy * pos
                ex = x0 + ux * seg_end
                ey = y0 + uy * seg_end
                draw.line([(sx, sy), (ex, ey)], fill=fill, width=max(1, int(round(lw))))
            pos = seg_end
            drawing = not drawing

def _draw_dashdot(draw, pts, fill, lw):
    on = max(2, int(round(lw * 4)))
    off = max(2, int(round(lw * 2)))
    dot = max(1, int(round(lw)))
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        dx, dy = x1 - x0, y1 - y0
        length = _math.hypot(dx, dy)
        if length == 0:
            continue
        ux, uy = dx / length, dy / length
        pos = 0.0
        seg_idx = 0
        while pos < length:
            if seg_idx % 2 == 0:
                seg = on
            elif seg_idx % 4 == 1:
                seg = dot
            else:
                seg = off
            seg_end = min(pos + seg, length)
            if seg_idx % 4 in (0, 2):
                sx = x0 + ux * pos
                sy = y0 + uy * pos
                ex = x0 + ux * seg_end
                ey = y0 + uy * seg_end
                draw.line([(sx, sy), (ex, ey)], fill=fill, width=max(1, int(round(lw))))
            elif seg_idx % 4 == 1:
                cx = x0 + ux * (pos + seg / 2)
                cy = y0 + uy * (pos + seg / 2)
                draw.ellipse((cx - 0.5, cy - 0.5, cx + 0.5, cy + 0.5), fill=fill)
            pos = seg_end
            seg_idx += 1

def _draw_dotted(draw, pts, fill, lw):
    gap = max(2, int(round(lw * 2)))
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        dx, dy = x1 - x0, y1 - y0
        length = _math.hypot(dx, dy)
        if length == 0:
            continue
        ux, uy = dx / length, dy / length
        pos = 0.0
        while pos < length:
            cx = x0 + ux * (pos + 1)
            cy = y0 + uy * (pos + 1)
            draw.ellipse((cx - 0.5, cy - 0.5, cx + 0.5, cy + 0.5), fill=fill)
            pos += gap

def _draw_marker(draw, marker, cx, cy, r, fill):
    if marker in ('o', '.'):
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=fill, width=max(1, r // 2))
    elif marker == 's':
        draw.rectangle((cx - r, cy - r, cx + r, cy + r), outline=fill, width=max(1, r // 2))
    elif marker in ('D', 'd'):
        draw.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], outline=fill)
    elif marker == '^':
        draw.polygon([(cx, cy - r), (cx + r, cy + r), (cx - r, cy + r)], outline=fill)
    elif marker == 'v':
        draw.polygon([(cx, cy + r), (cx + r, cy - r), (cx - r, cy - r)], outline=fill)
    elif marker == '<':
        draw.polygon([(cx - r, cy), (cx + r, cy - r), (cx + r, cy + r)], outline=fill)
    elif marker == '>':
        draw.polygon([(cx + r, cy), (cx - r, cy - r), (cx - r, cy + r)], outline=fill)
    elif marker in ('x', 'X'):
        draw.line([(cx - r, cy - r), (cx + r, cy + r)], fill=fill, width=max(1, r // 2))
        draw.line([(cx - r, cy + r), (cx + r, cy - r)], fill=fill, width=max(1, r // 2))
    elif marker == '+':
        draw.line([(cx - r, cy), (cx + r, cy)], fill=fill, width=max(1, r // 2))
        draw.line([(cx, cy - r), (cx, cy + r)], fill=fill, width=max(1, r // 2))
    elif marker == '*':
        draw.line([(cx - r, cy), (cx + r, cy)], fill=fill, width=max(1, r // 2))
        draw.line([(cx, cy - r), (cx, cy + r)], fill=fill, width=max(1, r // 2))
        draw.line([(cx - r * 0.7, cy - r * 0.7), (cx + r * 0.7, cy + r * 0.7)],
                  fill=fill, width=max(1, r // 2))
        draw.line([(cx - r * 0.7, cy + r * 0.7), (cx + r * 0.7, cy - r * 0.7)],
                  fill=fill, width=max(1, r // 2))
    elif marker in (',', '|'):
        draw.line([(cx, cy), (cx, cy + r)], fill=fill, width=max(1, r // 2))
    elif marker == '_':
        draw.line([(cx - r, cy), (cx + r, cy)], fill=fill, width=max(1, r // 2))
    else:
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=fill, width=max(1, r // 2))

class Patch(_Artist):
    def __init__(self, xy, width, height, facecolor=None,
                 color=None, linewidth=None, alpha=1.0, label=None, zorder=1):
        super().__init__()
        self.xy = xy
        self.width = width
        self.height = height
        self.facecolor = facecolor
        self.color = color
        self.linewidth = linewidth
        self.alpha = alpha
        self.label = label
        self.zorder = zorder

    def draw(self, axes, draw_ctx):
        draw = draw_ctx['draw']
        x0, y0 = axes._proj_x(self.xy[0]), axes._proj_y(self.xy[1])
        x1, y1 = axes._proj_x(self.xy[0] + self.width), axes._proj_y(self.xy[1] + self.height)
        xa, xb = (min(x0, x1), max(x0, x1))
        ya, yb = (min(y0, y1), max(y0, y1))
        fc = _rgba(self.facecolor or _rc('patch.facecolor'), self.alpha)
        ec = _rgba(self.color or _rc('patch.color'))
        if fc is not None:
            draw.rectangle((xa, ya, xb, yb), fill=fc)
        if ec is not None and self.linewidth != 0:
            draw.rectangle((xa, ya, xb, yb), outline=ec, width=max(1, int(self.linewidth or 1)))

class Polygon(_Artist):
    def __init__(self, points, facecolor=None, color=None, linewidth=None,
                 alpha=1.0, closed=True, zorder=1, label=None):
        super().__init__()
        self.points = list(points)
        self.facecolor = facecolor
        self.color = color
        self.linewidth = linewidth
        self.alpha = alpha
        self.closed = closed
        self.label = label
        self.zorder = zorder

    def draw(self, axes, draw_ctx):
        if not self.points:
            return
        draw = draw_ctx['draw']
        pts = [(axes._proj_x(x), axes._proj_y(y)) for (x, y) in self.points]
        fc = _rgba(self.facecolor, self.alpha)
        ec = _rgba(self.color)
        if fc is not None and len(pts) >= 3:
            draw.polygon(pts, fill=fc)
        if ec is not None and len(pts) >= 2:
            closed = list(pts)
            if self.closed:
                closed.append(pts[0])
            draw.line(closed, fill=ec, width=max(1, int(self.linewidth or 1)))

class Circle(_Artist):
    def __init__(self, xy, radius, facecolor=None, edgecolor=None, linewidth=None,
                 alpha=1.0, label=None, zorder=1):
        super().__init__()
        self.xy = xy
        self.radius = radius
        self.facecolor = facecolor
        self.color = edgecolor
        self.linewidth = linewidth
        self.alpha = alpha
        self.label = label
        self.zorder = zorder

    def draw(self, axes, draw_ctx):
        draw = draw_ctx['draw']
        cx, cy = axes._proj_x(self.xy[0]), axes._proj_y(self.xy[1])
        rx = max(1, axes._proj_dx(self.radius))
        ry = max(1, axes._proj_dy(self.radius))
        fc = _rgba(self.facecolor, self.alpha)
        ec = _rgba(self.color)
        if fc is not None:
            draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=fc)
        if ec is not None and self.linewidth != 0:
            draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=ec,
                         width=max(1, int(self.linewidth or 1)))

class W(_Artist):
    def __init__(self, center, radius, theta1, theta2, facecolor=None,
                 color=None, linewidth=None, alpha=1.0, zorder=1, label=None):
        super().__init__()
        self.center = center
        self.radius = radius
        self.theta1 = theta1
        self.theta2 = theta2
        self.facecolor = facecolor
        self.color = color
        self.linewidth = linewidth
        self.alpha = alpha
        self.label = label
        self.zorder = zorder

    def draw(self, axes, draw_ctx):
        draw = draw_ctx['draw']
        cx, cy = axes._proj_x(self.center[0]), axes._proj_y(self.center[1])
        rx = max(1, axes._proj_dx(self.radius))
        ry = max(1, axes._proj_dy(self.radius))
        a1 = _math.radians(self.theta1)
        a2 = _math.radians(self.theta2)
        steps = max(8, int(abs(a2 - a1) / _math.pi * 36))
        if steps < 1:
            steps = 1
        pts = []
        for i in range(steps + 1):
            t = a1 + (a2 - a1) * i / steps
            x = cx + rx * _math.cos(t)
            y = cy - ry * _math.sin(t)
            pts.append((x, y))
        fc = _rgba(self.facecolor, self.alpha)
        ec = _rgba(self.color)
        if fc is not None and len(pts) >= 3:
            triad = [(cx, cy)] + pts
            draw.polygon(triad, fill=fc)
        if ec is not None and len(pts) >= 2:
            draw.line(pts, fill=ec, width=max(1, int(self.linewidth or 1)))

class AxesImage(_Artist):
    def __init__(self, data, extent=None, cmap='viridis', norm=None,
                 origin='upper', aspect='auto', interpolation='nearest',
                 alpha=1.0, zorder=0, label=None):
        super().__init__()
        from triad import ntri as _np
        arr = _np.asarray(data)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim > 2:
            if arr.shape[-1] in (3, 4):
                arr = arr[..., 0]
            else:
                arr = arr.reshape(arr.shape[0], -1)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        arr = _np.nan_to_num(arr.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        self._np = arr
        self.extent = extent
        self.cmap = get_cmap(cmap)
        self.norm = norm or Normalize()
        self.origin = origin
        self.aspect = aspect
        self.interpolation = interpolation
        self.alpha = alpha
        self.label = label
        self.zorder = zorder

    def draw(self, axes, draw_ctx):
        from triad import ntri as _np
        arr = self._np
        h, w = arr.shape
        vmin = self.norm.vmin if self.norm.vmin is not None else float(arr.min())
        vmax = self.norm.vmax if self.norm.vmax is not None else float(arr.max())
        if vmin == vmax:
            vmax = vmin + 1.0
        self.norm.vmin = vmin
        self.norm.vmax = vmax
        n_colors = 256
        lut = self.cmap(n_colors)
        idx = ((arr - vmin) / (vmax - vmin) * (n_colors - 1)).clip(0, n_colors - 1).astype(int)
        rgb = _np.array(lut, dtype=_np.uint8)[idx]
        if self.alpha < 1.0:
            a = int(round(self.alpha * 255))
            alpha = _np.triad(rgb.shape[:2], a, dtype=_np.uint8)
            rgba = _np.dstack([rgb, alpha])
        else:
            rgba = _np.dstack([rgb, _np.triad(rgb.shape[:2], 255, dtype=_np.uint8)])
        img = Image.fromarray(rgba, 'RGBA')

        if self.extent is None:
            x0, x1, y0, y1 = 0.0, float(w), 0.0, float(h)
        else:
            x0, x1, y0, y1 = self.extent

        if self.origin == 'lower':
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        axes_x0 = axes._proj_x(min(x0, x1))
        axes_x1 = axes._proj_x(max(x0, x1))
        axes_y0 = axes._proj_y(min(y0, y1))
        axes_y1 = axes._proj_y(max(y0, y1))
        target_w = max(1, int(abs(axes_x1 - axes_x0)))
        target_h = max(1, int(abs(axes_y1 - axes_y0)))
        if self.aspect == 'equal':
            cell_w = max(1, target_w // w)
            cell_h = max(1, target_h // h)
            cell = min(cell_w, cell_h)
            target_w = cell * w
            target_h = cell * h
            axes_x0 = (axes_x0 + axes_x1) / 2 - target_w / 2
            axes_x1 = axes_x0 + target_w
            axes_y0 = (axes_y0 + axes_y1) / 2 - target_h / 2
            axes_y1 = axes_y0 + target_h
        img = img.resize((target_w, target_h), Image.NEAREST)
        paste_x = int(min(axes_x0, axes_x1))
        paste_y = int(min(axes_y0, axes_y1))
        draw_ctx['image'].paste(img, (paste_x, paste_y), img)

class QuadMesh(_Artist):
    def __init__(self, X, Y, C, cmap='viridis', norm=None, colors='none',
                 linewidth=0.0, alpha=1.0, zorder=0, label=None):
        super().__init__()
        self.X = X
        self.Y = Y
        self.C = C
        self.cmap = get_cmap(cmap)
        self.norm = norm or Normalize()
        self.colors = colors
        self.linewidth = linewidth
        self.alpha = alpha
        self.label = label
        self.zorder = zorder

    def draw(self, axes, draw_ctx):
        from triad import ntri as _np
        X = _np.asarray(self.X, dtype=float)
        Y = _np.asarray(self.Y, dtype=float)
        C = _np.asarray(self.C, dtype=float)
        ny, nx = C.shape
        vmin = self.norm.vmin if self.norm.vmin is not None else float(C.min())
        vmax = self.norm.vmax if self.norm.vmax is not None else float(C.max())
        if vmin == vmax:
            vmax = vmin + 1.0
        self.norm.vmin = vmin
        self.norm.vmax = vmax
        lut = self.cmap(256)
        draw = draw_ctx['draw']
        for j in range(ny):
            for i in range(nx):
                x0, x1 = float(X[j, i]), float(X[j, i + 1])
                y0, y1 = float(Y[j, i]), float(Y[j + 1, i])
                t = (float(C[j, i]) - vmin) / (vmax - vmin)
                t = max(0.0, min(1.0, t))
                idx = int(round(t * 255))
                rgb = lut[idx]
                rgba = (rgb[0], rgb[1], rgb[2], int(self.alpha * 255))
                px0 = axes._proj_x(x0)
                px1 = axes._proj_x(x1)
                py0 = axes._proj_y(y0)
                py1 = axes._proj_y(y1)
                xa, xb = (min(px0, px1), max(px0, px1))
                ya, yb = (min(py0, py1), max(py0, py1))
                draw.rectangle((xa, ya, xb, yb), fill=rgba)
                if self.colors not in (None, 'none', 'None', '') and self.linewidth:
                    ec = _rgba(self.colors)
                    if ec is not None:
                        draw.rectangle((xa, ya, xb, yb), outline=ec,
                                       width=max(1, int(self.linewidth)))

class Collection(_Artist):
    def __init__(self, segments, faces=None, s=None, sizes=None,
                 marker='o', alpha=1.0, zorder=2, label=None, cmap=None, norm=None):
        super().__init__()
        self.segments = segments
        self.faces = faces
        self.s = s
        self.sizes = sizes
        self.marker = marker
        self.alpha = alpha
        self.label = label
        self.zorder = zorder
        self.cmap = cmap
        self.norm = norm

    def draw(self, axes, draw_ctx):
        draw = draw_ctx['draw']
        for idx, seg in enumerate(self.segments):
            if len(seg) >= 2:
                x, y = seg[0], seg[1]
                cx = axes._proj_x(x)
                cy = axes._proj_y(y)
                size = self.sizes[idx] if self.sizes else 4.0
                r = max(1, int(round(size)))
                fc = _rgba(self.faces[idx] if self.faces else 'C0', self.alpha)
                ec = _rgba(self.s[idx] if self.s else fc, self.alpha)
                if fc is not None:
                    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fc)
                if ec is not None:
                    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=ec, width=1)

class Text(_Artist):
    def __init__(self, x, y, text, color='black', fontsize=None, ha='left',
                 va='baseline', rotation=0.0, alpha=1.0, zorder=3, weight='normal'):
        super().__init__()
        self.x = x
        self.y = y
        self.text = text
        self.color = color
        self.fontsize = fontsize or _rc('font.size')
        self.ha = ha
        self.va = va
        self.rotation = rotation
        self.alpha = alpha
        self.weight = weight
        self.zorder = zorder
        self._size = None
        self._cached_font = _get_font(self.fontsize)

    def measure(self):
        return _measure(self._cached_font, self.text)

    def draw(self, axes, draw_ctx):
        draw = draw_ctx['draw']
        rgba = _rgba(self.color, self.alpha)
        if rgba is None or self.text is None or str(self.text) == '':
            return
        x = axes._proj_x(self.x)
        y = axes._proj_y(self.y)
        if self.rotation:
            self._draw_rotated(axes, draw_ctx, str(self.text), rgba, x, y)
            return
        anchor = {'left': 'l', 'center': 'm', 'right': 'r'}.get(self.ha, 'l') + \
            {'baseline': 'a', 'center': 'm', 'top': 'a', 'bottom': 'b'}.get(self.va, 'a')
        try:
            draw.text((x, y), str(self.text), fill=rgba, font=self._cached_font, anchor=anchor)
        except (TypeError, AttributeError):
            draw.text((x, y), str(self.text), fill=rgba, font=self._cached_font)

    def _draw_rotated(self, axes, draw_ctx, text, rgba, x, y):
        try:
            from stdlib._canvas import Image as _PILImage
            tw, th = _measure(self._cached_font, text)
            try:
                ascent = self._cached_font.getmetrics()[0]
            except (AttributeError, TypeError):
                ascent = int(th * 0.8)
            ax = {'left': 0.0, 'center': tw / 2.0, 'right': float(tw)}.get(self.ha, 0.0)
            ay = {'top': 0.0, 'center': th / 2.0, 'bottom': float(th),
                  'baseline': float(ascent)}.get(self.va, float(ascent))
            import math as _math
            side = int(2 * _math.ceil(_math.hypot(tw or 1, th or 1)) + 8)
            c = side // 2
            layer = _PILImage.new('RGBA', (side, side), (0, 0, 0, 0))
            d = _PILImage.Draw(layer)
            d.text((c - ax, c - ay), text, fill=rgba, font=self._cached_font)
            try:
                rs = _PILImage.Resampling.BICUBIC
            except AttributeError:
                rs = _PILImage.BICUBIC
            rot = layer.rotate(self.rotation, resample=rs)
            img = draw_ctx.get('image') if isinstance(draw_ctx, dict) else None
            if img is None:
                return
            img.paste(rot, (int(x - c), int(y - c)), rot)
        except Exception:
            draw = draw_ctx['draw']
            draw.text((x, y), text, fill=rgba, font=self._cached_font)

class _ErrorBarV(_Artist):
    def __init__(self, x, ylo, yhi, color='black', elinewidth=1.0, capsize=3):
        super().__init__()
        self.x = x
        self.ylo = ylo
        self.yhi = yhi
        self.color = color
        self.elinewidth = elinewidth
        self.capsize = capsize
        self.zorder = 2

    def draw(self, axes, draw_ctx):
        draw = draw_ctx['draw']
        c = _rgba(self.color)
        if c is None:
            return
        cx = axes._proj_x(self.x)
        y0 = axes._proj_y(self.ylo)
        y1 = axes._proj_y(self.yhi)
        draw.line([(cx, y0), (cx, y1)], fill=c, width=max(1, int(self.elinewidth)))
        cap = max(1, int(self.capsize))
        draw.line([(cx - cap, y0), (cx + cap, y0)], fill=c, width=max(1, int(self.elinewidth)))
        draw.line([(cx - cap, y1), (cx + cap, y1)], fill=c, width=max(1, int(self.elinewidth)))

class _ArrowHead(_Artist):
    def __init__(self, x, y, dx, dy, color='black', head_width=0.1, head_length=0.15):
        super().__init__()
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.color = color
        self.head_width = head_width
        self.head_length = head_length
        self.zorder = 3

    def draw(self, axes, draw_ctx):
        draw = draw_ctx['draw']
        x1, y1 = self.x + self.dx, self.y + self.dy
        ang = _math.atan2(self.dy, self.dx)
        L = _math.hypot(self.dx, self.dy)
        if L == 0:
            return
        hl = max(0.01, min(self.head_length, L * 0.5))
        hw = hl
        c = _rgba(self.color)
        if c is None:
            return
        bx = x1 - hl * _math.cos(ang)
        by = y1 - hl * _math.sin(ang)
        p1 = (x1, y1)
        p2 = (bx + hw / 2 * _math.cos(ang + _math.pi / 2),
              by + hw / 2 * _math.sin(ang + _math.pi / 2))
        p3 = (bx - hw / 2 * _math.cos(ang + _math.pi / 2),
              by - hw / 2 * _math.sin(ang + _math.pi / 2))
        pts = [(axes._proj_x(p[0]), axes._proj_y(p[1])) for p in (p1, p2, p3)]
        draw.polygon(pts, fill=c)

class Axes:
    def __init__(self, fig, rect, **kwargs):
        self.figure = fig
        self._rect = rect
        self._xlim = kwargs.pop('xlim', None)
        self._ylim = kwargs.pop('ylim', None)
        self._xscale = kwargs.pop('xscale', 'triad')
        self._yscale = kwargs.pop('yscale', 'triad')
        self._aspect = kwargs.pop('aspect', 'auto')
        self._title = kwargs.pop('title', None)
        self._xlabel = kwargs.pop('xlabel', None)
        self._ylabel = kwargs.pop('ylabel', None)
        self._facecolor = kwargs.pop('facecolor', _rc('axes.facecolor'))
        self._grid = bool(_rc('axes.grid'))
        self._xticks_pos = None
        self._xticks_lab = None
        self._yticks_pos = None
        self._yticks_lab = None
        self._spines_visible = {'top': False, 'right': False, 'left': True, 'bottom': True}
        self._artists = []
        self._texts = []
        self._colorbars = []
        self._xmargin = 0.05
        self._ymargin = 0.05
        self._autoscale = True
        self._title_color = None
        self._xlabel_color = None
        self._ylabel_color = None
        self._title_fontsize = None
        self._xlabel_fontsize = None
        self._ylabel_fontsize = None
        self._explicit_legend = None
        self._legend_frame = _rc('legend.frameon')
        self._legend_fontsize = _rc('legend.fontsize')
        self._legend_loc = 'best'
        self._spines_color = {k: _rc('axes.color') for k in self._spines_visible}
        self._twinx = None
        self._twiny = None
        self._xaxis_side = 'bottom'
        self._yaxis_side = 'left'
        self._sharex_twin = False
        self._sharey_twin = False

    def _box(self):
        fig = self.figure
        W, H = fig._canvas_size
        dpi = fig.dpi
        left = int(self._rect[0] * W)
        width = int(self._rect[2] * W)
        height = int(self._rect[3] * H)
        px_bottom = int((1.0 - self._rect[1]) * H)
        px_top = px_bottom - height
        tick_pad = int(0.10 * dpi)
        xlabel_pad = int(0.30 * dpi)
        ylabel_pad = int(0.30 * dpi)
        title_pad = int(0.30 * dpi)
        return {
            'left': left + ylabel_pad,
            'right': left + width - tick_pad,
            'bottom': px_bottom - xlabel_pad,
            'top': px_top + title_pad,
        }

    def _ensure_limits(self):
        if self._xlim is None:
            xs = []
            for art in self._artists:
                if isinstance(art, Line2D):
                    xs.extend(art.xdata)
                elif isinstance(art, Patch):
                    xs.extend([art.xy[0], art.xy[0] + art.width])
                elif isinstance(art, Polygon):
                    xs.extend(p[0] for p in art.points)
                elif isinstance(art, Circle):
                    xs.extend([art.xy[0] - art.radius, art.xy[0] + art.radius])
                elif isinstance(art, W):
                    xs.extend([art.center[0] - art.radius, art.center[0] + art.radius])
                elif isinstance(art, Collection):
                    for s in art.segments:
                        if len(s) >= 1:
                            xs.append(s[0])
                elif isinstance(art, AxesImage) and art.extent is None:
                    xs.extend([0, art._np.shape[1]])
                elif isinstance(art, _ErrorBarV):
                    xs.append(art.x)
            if xs:
                lo, hi = min(xs), max(xs)
                if lo == hi:
                    lo, hi = lo - 1, hi + 1
                margin = (hi - lo) * self._xmargin
                self._xlim = (lo - margin, hi + margin)
            else:
                self._xlim = (0.0, 1.0)
        if self._ylim is None:
            ys = []
            for art in self._artists:
                if isinstance(art, Line2D):
                    ys.extend(art.ydata)
                elif isinstance(art, Patch):
                    ys.extend([art.xy[1], art.xy[1] + art.height])
                elif isinstance(art, Polygon):
                    ys.extend(p[1] for p in art.points)
                elif isinstance(art, Circle):
                    ys.extend([art.xy[1] - art.radius, art.xy[1] + art.radius])
                elif isinstance(art, W):
                    ys.extend([art.center[1] - art.radius, art.center[1] + art.radius])
                elif isinstance(art, Collection):
                    for s in art.segments:
                        if len(s) >= 2:
                            ys.append(s[1])
                elif isinstance(art, AxesImage) and art.extent is None:
                    ys.extend([0, art._np.shape[0]])
                elif isinstance(art, _ErrorBarV):
                    ys.extend([art.ylo, art.yhi])
            if ys:
                lo, hi = min(ys), max(ys)
                if lo == hi:
                    lo, hi = lo - 1, hi + 1
                margin = (hi - lo) * self._ymargin
                self._ylim = (lo - margin, hi + margin)
            else:
                self._ylim = (0.0, 1.0)

    def _proj_x(self, x):
        self._ensure_limits()
        lo, hi = self._xlim
        b = self._box()
        if self._xscale == 'log':
            if x <= 0 or lo <= 0 or hi <= 0:
                return b['left']
            xn = _math.log10(x)
            ln, lh = _math.log10(lo), _math.log10(hi)
        else:
            xn = x
            ln, lh = lo, hi
        if lh == ln:
            return (b['left'] + b['right']) / 2
        f = (xn - ln) / (lh - ln)
        return b['left'] + f * (b['right'] - b['left'])

    def _proj_y(self, y):
        self._ensure_limits()
        lo, hi = self._ylim
        b = self._box()
        if self._yscale == 'log':
            if y <= 0 or lo <= 0 or hi <= 0:
                return b['bottom']
            yn = _math.log10(y)
            ln, lh = _math.log10(lo), _math.log10(hi)
        else:
            yn = y
            ln, lh = lo, hi
        if lh == ln:
            return (b['bottom'] + b['top']) / 2
        f = (yn - ln) / (lh - ln)
        return b['bottom'] - f * (b['bottom'] - b['top'])

    def _proj_dx(self, dx):
        self._ensure_limits()
        lo, hi = self._xlim
        b = self._box()
        if hi == lo:
            return 0
        return abs(dx) / (hi - lo) * (b['right'] - b['left'])

    def _proj_dy(self, dy):
        self._ensure_limits()
        lo, hi = self._ylim
        b = self._box()
        if hi == lo:
            return 0
        return abs(dy) / (hi - lo) * (b['top'] - b['bottom'])

    def set_xlim(self, left=None, right=None, **kw):
        if isinstance(left, (list, tuple)):
            right = left[1] if len(left) > 1 else self._xlim[1]
            left = left[0]
        if left is None and right is None:
            return self._xlim
        if isinstance(left, str):
            return self._xlim
        self._xlim = (left if left is not None else self._xlim[0],
                      right if right is not None else self._xlim[1])
        return self._xlim

    def set_ylim(self, bottom=None, top=None, **kw):
        if isinstance(bottom, (list, tuple)):
            top = bottom[1] if len(bottom) > 1 else self._ylim[1]
            bottom = bottom[0]
        self._ylim = (bottom if bottom is not None else self._ylim[0],
                      top if top is not None else self._ylim[1])
        return self._ylim

    def get_xlim(self):
        return self._xlim

    def get_ylim(self):
        return self._ylim

    def set_xscale(self, value, **kw):
        self._xscale = value
        return self._xscale

    def set_yscale(self, value, **kw):
        self._yscale = value
        return self._yscale

    def get_xscale(self):
        return self._xscale

    def get_yscale(self):
        return self._yscale

    def set_aspect(self, value, **kw):
        self._aspect = value
        return self._aspect

    def set_title(self, label, **kw):
        self._title = label
        self._title_fontsize = kw.get('fontsize')
        self._title_color = kw.get('color')
        return self._title

    def set_xlabel(self, label, **kw):
        self._xlabel = label
        self._xlabel_fontsize = kw.get('fontsize')
        self._xlabel_color = kw.get('color')
        return self._xlabel

    def set_ylabel(self, label, **kw):
        self._ylabel = label
        self._ylabel_fontsize = kw.get('fontsize')
        self._ylabel_color = kw.get('color')
        return self._ylabel

    def set_xticks(self, ticks, labels=None, **kw):
        self._xticks_pos = list(ticks) if ticks is not None else None
        self._xticks_lab = list(labels) if labels is not None else None
        return self._xticks_pos

    def set_yticks(self, ticks, labels=None, **kw):
        self._yticks_pos = list(ticks) if ticks is not None else None
        self._yticks_lab = list(labels) if labels is not None else None
        return self._yticks_pos

    def grid(self, visible=None, which='major', axis='both', **kw):
        if visible is None:
            visible = not self._grid
        self._grid = bool(visible)
        return self._grid

    def set_facecolor(self, color):
        self._facecolor = color
        return self._facecolor

    def get_facecolor(self):
        return self._facecolor

    def tick_params(self, **kw):
        return None

    def set_xticklabels(self, labels, **kw):
        self._xticks_lab = list(labels)
        return self._xticks_lab

    def set_yticklabels(self, labels, **kw):
        self._yticks_lab = list(labels)
        return self._yticks_lab

    def get_xticklabels(self):
        return list(self._xticks_lab) if self._xticks_lab else []

    def get_yticklabels(self):
        return list(self._yticks_lab) if self._yticks_lab else []

    def legend(self, *args, **kwargs):
        if args and not kwargs.get('handles'):
            if isinstance(args[0], list):
                kwargs['handles'] = args[0]
            else:
                kwargs['handles'] = args[0]
        if 'labels' in kwargs and 'handles' in kwargs:
            self._explicit_legend = list(zip(kwargs['handles'], kwargs['labels']))
        else:
            handles = []
            labels = []
            for art in self._artists:
                if isinstance(art, Line2D) and art.label:
                    handles.append(art)
                    labels.append(art.label)
                elif isinstance(art, (Patch, Polygon, Circle, W, Collection)) and art.label:
                    handles.append(art)
                    labels.append(art.label)
            self._explicit_legend = list(zip(handles, labels))
        self._legend_loc = kwargs.get('loc', 'best')
        self._legend_frame = kwargs.get('frameon', _rc('legend.frameon'))
        self._legend_fontsize = kwargs.get('fontsize', _rc('legend.fontsize'))
        return self._explicit_legend

    def plot(self, *args, scalex=True, scaley=True, **kwargs):
        lines = []
        if not args:
            return lines
        if len(args) == 1:
            y = args[0]
            x = list(range(len(y)))
            data = [(x, y, None)]
        elif len(args) == 2:
            x, y = args
            data = [(x, y, None)]
        else:
            data = []
            for i in range(0, len(args), 3):
                x = args[i]
                y = args[i + 1] if i + 1 < len(args) else []
                fmt = args[i + 2] if i + 2 < len(args) else None
                data.append((x, y, fmt))
        for i, (xs, ys, fmt) in enumerate(data):
            label = kwargs.get('label')
            color = kwargs.get('color', _next_color(i))
            linestyle = kwargs.get('linestyle', '-')
            marker = kwargs.get('marker', 'None')
            if fmt is not None:
                color_letter = None
                marker_letter = None
                line_letter = None
                for ch in fmt:
                    if ch in '-.':
                        line_letter = ch
                    elif ch in 'osDxX+*v^<>|_ ,':
                        marker_letter = ch
                    elif ch in 'rgbcmykw':
                        color_letter = ch
                if line_letter:
                    linestyle = line_letter * 2 if line_letter == '-' else line_letter
                if marker_letter:
                    marker = marker_letter
                if color_letter:
                    color = {'r': 'C3', 'g': 'C2', 'b': 'C0',
                             'c': 'cyan', 'm': 'purple', 'y': 'yellow',
                             'k': 'black', 'w': 'white'}[color_letter]
            line = Line2D(xs, ys,
                          color=color,
                          linewidth=kwargs.get('linewidth'),
                          linestyle=linestyle,
                          marker=marker,
                          markersize=kwargs.get('markersize'),
                          alpha=kwargs.get('alpha', 1.0),
                          label=label if i == 0 else None)
            self._artists.append(line)
            lines.append(line)
        return lines

    def scatter(self, x, y, s=None, c=None, marker='o', cmap=None, norm=None,
                vmin=None, vmax=None, alpha=1.0, colors=None, linewidths=None,
                label=None, **kw):
        cmap = get_cmap(cmap or 'viridis')
        norm = norm or Normalize(vmin=vmin, vmax=vmax)
        faces = []
        dg_colors = []
        sizes = []
        for i, (xi, yi) in enumerate(zip(x, y)):
            if c is not None:
                col = c if isinstance(c, str) else c[i]
                if isinstance(col, (int, float)):
                    t = norm(col)
                    idx = int(round(t * 255))
                    idx = max(0, min(255, idx))
                    rgb = cmap(256)[idx]
                    faces.append(rgb)
                else:
                    faces.append(col)
            else:
                faces.append(_next_color(i))
            dg_colors.append(colors if colors is not None else faces[-1])
            sizes.append(s if s is not None else 6.0)
        coll = Collection(list(zip(x, y)),
                          faces=faces, s=dg_colors, sizes=sizes,
                          marker=marker, alpha=alpha, zorder=2, label=label,
                          cmap=cmap, norm=norm)
        self._artists.append(coll)
        return coll

    def bar(self, x, height, width=0.8, bottom=None, color=None,
            linewidth=None, alpha=1.0, label=None, align='center', **kw):
        bars = []
        color_iter = color is None
        for i, (xi, hi) in enumerate(zip(x, height)):
            c = _next_color(i) if color_iter else color
            if isinstance(xi, str):
                offset = (i - width / 2) if align == 'center' else i
            else:
                offset = (xi - width / 2) if align == 'center' else xi
            bot = bottom[i] if bottom is not None else 0
            patch = Patch(xy=(offset, bot), width=width, height=hi,
                          facecolor=c, color=color, linewidth=linewidth,
                          alpha=alpha, label=label if i == 0 else None, zorder=1)
            self._artists.append(patch)
            bars.append(patch)
        return bars

    def barh(self, y, width, height=0.8, left=None, color=None, **kw):
        bars = []
        for i, (yi, wi) in enumerate(zip(y, width)):
            c = color if color else _next_color(i)
            offset = (yi - height / 2)
            lf = left[i] if left is not None else 0
            patch = Patch(xy=(lf, offset), width=wi, height=height,
                          facecolor=c, color=kw.get('color'),
                          linewidth=kw.get('linewidth'), alpha=kw.get('alpha', 1.0),
                          label=kw.get('label') if i == 0 else None)
            self._artists.append(patch)
            bars.append(patch)
        return bars

    def hist(self, x, bins=10, range=None, density=False, color='black',
             alpha=1.0, label=None, **kw):
        from triad import ntri as _np
        arr = _np.asarray(x, dtype=float)
        if range is not None:
            lo, hi = range
        else:
            lo, hi = float(arr.min()), float(arr.max())
        if lo == hi:
            hi = lo + 1
        s = _np.linspace(lo, hi, int(bins) + 1)
        counts, _ = _np.histogram(arr, bins=s)
        if density:
            total = counts.sum()
            if total > 0:
                counts = counts.astype(float) / total * (int(bins) / (hi - lo))
        bars = []
        for i, c in enumerate(counts):
            col = color if color else _next_color(0)
            patch = Patch(xy=(float(s[i]), 0),
                          width=float(s[i + 1] - s[i]),
                          height=float(c),
                          facecolor=col, color=color,
                          alpha=alpha, label=label if i == 0 else None)
            self._artists.append(patch)
            bars.append(patch)
        return bars

    def imshow(self, X, cmap=None, norm=None, aspect=None, interpolation='nearest',
               alpha=1.0, origin='upper', extent=None, vmin=None, vmax=None, **kw):
        norm = norm or Normalize(vmin=vmin, vmax=vmax)
        img = AxesImage(X, extent=extent, cmap=cmap or 'viridis', norm=norm,
                    origin=origin, aspect=aspect or 'auto', alpha=alpha)
        self._artists.append(img)
        if extent is None:
            arr = img._np
            self._xlim = (0.0, float(arr.shape[1]))
            self._ylim = (0.0, float(arr.shape[0]))
        return img

    def pcolormesh(self, X, Y, C, cmap=None, norm=None, vmin=None, vmax=None,
                   colors='none', linewidth=0.0, alpha=1.0, **kw):
        norm = norm or Normalize(vmin=vmin, vmax=vmax)
        m = QuadMesh(X, Y, C, cmap=cmap or 'viridis', norm=norm,
                     colors=colors, linewidth=linewidth, alpha=alpha)
        self._artists.append(m)
        return m

    def axhline(self, y=0, xmin=0, xmax=1, color=None, linewidth=None,
                linestyle='--', alpha=1.0, **kw):
        self._ensure_limits()
        x0 = self._xlim[0] + (self._xlim[1] - self._xlim[0]) * xmin
        x1 = self._xlim[0] + (self._xlim[1] - self._xlim[0]) * xmax
        line = Line2D([x0, x1], [y, y],
                      color=color or '#888888', linewidth=linewidth or 1.0,
                      linestyle=linestyle, alpha=alpha, zorder=1)
        self._artists.append(line)
        return line

    def axvline(self, x=0, ymin=0, ymax=1, **kw):
        self._ensure_limits()
        y0 = self._ylim[0] + (self._ylim[1] - self._ylim[0]) * ymin
        y1 = self._ylim[0] + (self._ylim[1] - self._ylim[0]) * ymax
        line = Line2D([x, x], [y0, y1],
                      color=kw.get('color', '#888888'),
                      linewidth=kw.get('linewidth', 1.0),
                      linestyle=kw.get('linestyle', '--'),
                      alpha=kw.get('alpha', 1.0), zorder=1)
        self._artists.append(line)
        return line

    def axhspan(self, ymin, ymax, xmin=0, xmax=1, facecolor=None, alpha=0.3, **kw):
        self._ensure_limits()
        x0 = self._xlim[0] + (self._xlim[1] - self._xlim[0]) * xmin
        x1 = self._xlim[0] + (self._xlim[1] - self._xlim[0]) * xmax
        patch = Patch(xy=(x0, ymin), width=(x1 - x0), height=(ymax - ymin),
                      facecolor=facecolor or '#cccccc', color=None,
                      alpha=alpha, zorder=0)
        self._artists.append(patch)
        return patch

    def axvspan(self, xmin, xmax, ymin=None, ymax=None, **kw):
        if ymin is None:
            ymin = self._ylim[0]
        if ymax is None:
            ymax = self._ylim[1]
        return self.axhspan(ymin, ymax, xmin=xmin, xmax=xmax,
                            facecolor=kw.get('facecolor'),
                            alpha=kw.get('alpha', 0.3))

    def fill_between(self, x, y1, y2=0, where=None, color=None, alpha=0.3,
                     linewidth=0.0, **kw):
        if where is None:
            where = [True] * len(x)
        pts = []
        for i, (xi, yi) in enumerate(zip(x, y1)):
            if where[i]:
                pts.append((xi, yi))
        for i in range(len(x) - 1, -1, -1):
            if where[i]:
                pts.append((x[i], y2[i] if hasattr(y2, '__getitem__') else y2))
        poly = Polygon(pts, facecolor=color or 'C0', color='none',
                       alpha=alpha, closed=True, zorder=0)
        self._artists.append(poly)
        return poly

    def fill_betweenx(self, y, x1, x2=0, where=None, color=None, alpha=0.3, **kw):
        if where is None:
            where = [True] * len(y)
        pts = []
        for i, yi in enumerate(y):
            if where[i]:
                pts.append((x1[i] if hasattr(x1, '__getitem__') else x1, yi))
        for i in range(len(y) - 1, -1, -1):
            if where[i]:
                pts.append((x2[i] if hasattr(x2, '__getitem__') else x2, y[i]))
        poly = Polygon(pts, facecolor=color or 'C0', color='none',
                       alpha=alpha, closed=True, zorder=0)
        self._artists.append(poly)
        return poly

    def errorbar(self, x, y, yerr=None, xerr=None, fmt='-', color=None,
                 ecolor=None, elinewidth=None, capsize=3, capthick=None,
                 alpha=1.0, label=None, **kw):
        color = color or _next_color(0)
        ecolor = ecolor or color
        if len(x) and isinstance(x[0], str):
            x_plot = list(range(len(x)))
        else:
            x_plot = list(x)
        line = Line2D(x_plot, y, color=color,
                      linestyle=fmt if '-' in fmt or '--' in fmt else '-',
                      marker=fmt if fmt not in '-o' else 'None', label=label)
        self._artists.append(line)
        if yerr is not None:
            for xi, yi, ye in zip(x_plot, y, yerr):
                if not isinstance(ye, (list, tuple)):
                    ye = (ye, ye)
                self._artists.append(_ErrorBarV(xi, yi - ye[0], yi + ye[1],
                                                color=ecolor,
                                                elinewidth=elinewidth or 1.0,
                                                capsize=capsize))
        return line

    def text(self, x, y, s, **kw):
        t = Text(x, y, s,
                 color=kw.get('color', 'black'),
                 fontsize=kw.get('fontsize', _rc('font.size')),
                 ha=kw.get('ha', 'left'),
                 va=kw.get('va', 'baseline'),
                 rotation=kw.get('rotation', 0.0),
                 alpha=kw.get('alpha', 1.0))
        self._artists.append(t)
        self._texts.append(t)
        return t

    def annotate(self, text, xy, xytext=None, arrowprops=None,
                 color='black', fontsize=None, ha='center', va='center', **kw):
        if xytext is None:
            xytext = xy
        t = self.text(xytext[0], xytext[1], text, color=color,
                      fontsize=fontsize, ha=ha, va=va)
        if arrowprops and xy != xytext:
            line = Line2D([xy[0], xytext[0]], [xy[1], xytext[1]],
                          color=arrowprops.get('color', 'black'),
                          linewidth=arrowprops.get('linewidth', 1.0),
                          linestyle=arrowprops.get('linestyle', '-'),
                          alpha=arrowprops.get('alpha', 1.0), zorder=2)
            self._artists.append(line)
        return t

    def arrow(self, x, y, dx, dy, **kw):
        line = Line2D([x, x + dx], [y, y + dy],
                      color=kw.get('color', 'black'),
                      linewidth=kw.get('linewidth', 1.0),
                      linestyle=kw.get('linestyle', '-'), zorder=2)
        self._artists.append(line)
        self._artists.append(_ArrowHead(x, y, dx, dy,
                                        color=kw.get('color', 'black'),
                                        head_width=kw.get('head_width', 0.1),
                                        head_length=kw.get('head_length', 0.15)))
        return line

    def twinx(self):
        ax2 = self._twin_axes(sharex=True, sharey=False)
        self._twinx = ax2
        return ax2

    def twiny(self):
        ax2 = self._twin_axes(sharex=False, sharey=True)
        self._twiny = ax2
        return ax2

    def _twin_axes(self, sharex=False, sharey=False):
        ax2 = Axes(self.figure, self._rect)
        if sharex:
            ax2._xlim = self._xlim
            ax2._xscale = self._xscale
            ax2._sharex_twin = True
            ax2._yaxis_side = 'right'
            ax2._spines_visible = {'top': False, 'right': True,
                                   'left': False, 'bottom': False}
        if sharey:
            ax2._ylim = self._ylim
            ax2._yscale = self._yscale
            ax2._sharey_twin = True
            ax2._xaxis_side = 'top'
            ax2._spines_visible = {'top': True, 'right': False,
                                   'left': False, 'bottom': False}
        return ax2

    def add_colorbar(self, mappable=None, **kw):
        if mappable is None:
            for art in self._artists:
                if isinstance(art, (AxesImage, QuadMesh)):
                    mappable = art
                    break
        cb = _Colorbar(self, mappable, **kw)
        self._colorbars.append(cb)
        return cb

    def add_patch(self, patch):
        self._artists.append(patch)
        return patch

    def add_line(self, line):
        self._artists.append(line)
        return line

    def add_image(self, image):
        self._artists.append(image)
        return image

    def add_collection(self, coll):
        self._artists.append(coll)
        return coll

    def add_artist(self, art):
        self._artists.append(art)
        return art

    def clear(self):
        self._artists = []
        self._texts = []
        self._colorbars = []

    def cla(self):
        self.clear()

class _Colorbar:
    def __init__(self, axes, mappable, orientation='vertical', shrink=0.8,
                 pad=0.05, label=None):
        self.axes = axes
        self.mappable = mappable
        self.orientation = orientation
        self.shrink = shrink
        self.pad = pad
        self.label = label
        self.figure = axes.figure

    def render(self, image, draw):
        fig = self.figure
        W, H = fig._canvas_size
        dpi = fig.dpi
        rect = self.axes._rect
        cb_w = int(0.04 * dpi)
        cb_h = int(rect[3] * H * self.shrink)
        cb_left = int((rect[0] + rect[2]) * W + 0.02 * dpi)
        cb_bottom = int((rect[1] + (rect[3] - self.shrink * rect[3]) / 2) * H)
        n = 256
        cmap_obj = self.mappable.cmap
        if callable(cmap_obj):
            cmap = cmap_obj
        elif isinstance(cmap_obj, str):
            cmap = get_cmap(cmap_obj)
        else:
            cmap = get_cmap('viridis')
        ramp = cmap(n)
        for i in range(n):
            t = i / (n - 1)
            y = cb_bottom + t * cb_h
            r, g, b = ramp[i]
            draw.rectangle((cb_left, y, cb_left + cb_w, y + 1), fill=(r, g, b))
        ec = _to_rgb(_rc('axes.color'))
        draw.rectangle((cb_left, cb_bottom, cb_left + cb_w, cb_bottom + cb_h),
                       outline=ec)
        norm = self.mappable.norm
        n_ticks = 5
        for i in range(n_ticks):
            t = i / (n_ticks - 1)
            value = norm.inverse(t)
            y = cb_bottom + t * cb_h
            draw.line([(cb_left + cb_w, y), (cb_left + cb_w + 4, y)],
                      fill=ec)
            label = _format_tick(value, scale='triad')
            font = _get_font(_rc('font.size'))
            tw, th = _measure(font, label)
            draw.text((cb_left + cb_w + 6, y - th / 2), label, fill=ec, font=font)
        if self.label:
            font = _get_font(_rc('font.size'))
            tw, th = _measure(font, self.label)
            draw.text((cb_left + cb_w / 2 - tw / 2, cb_bottom + cb_h + 4),
                      self.label, fill=ec, font=font)

class _AxesGrid:
    def __init__(self, axes_list, nrows, ncols):
        self._list = axes_list
        self._nr = nrows
        self._nc = ncols
    def __getitem__(self, key):
        if isinstance(key, tuple):
            r, c = key
            return self._list[r * self._nc + c]
        return self._list[key]
    def __len__(self):
        return len(self._list)
    def __iter__(self):
        return iter(self._list)
    def __repr__(self):
        return f"<AxesGrid {self._nr}x{self._nc}>"
    @property
    def shape(self):
        return (self._nr, self._nc)
    def flatten(self):
        return list(self._list)
    def flat(self):
        return iter(self._list)

class Figure:
    def __init__(self, figsize=None, dpi=None, facecolor=None, color=None, **kw):
        figsize = figsize or _rc('figure.figsize')
        dpi = dpi or _rc('figure.dpi')
        self.figsize = figsize
        self.dpi = float(dpi)
        self.facecolor = facecolor or _rc('figure.facecolor')
        self.color = color or _rc('figure.color')
        self._axes = []
        self._suptitle = None
        self._suptitle_fontsize = None
        self._subplotpars = {'left': 0.125, 'right': 0.9, 'bottom': 0.11, 'top': 0.88,
                            'wspace': 0.2, 'hspace': 0.2}
        self._canvas_size = (int(self.figsize[0] * self.dpi),
                             int(self.figsize[1] * self.dpi))

    @property
    def axes(self):
        return list(self._axes)

    def add_axes(self, rect, **kw):
        ax = Axes(self, rect, **kw)
        self._axes.append(ax)
        return ax

    def add_subplot(self, *args, **kwargs):
        nrows, ncols, index = _parse_subplot_spec(*args)
        left, bottom, width, height = _gs_rect(nrows, ncols, index, self._subplotpars)
        ax = self.add_axes((left, bottom, width, height), **kwargs)
        return ax

    def suptitle(self, t, **kw):
        self._suptitle = t
        if 'fontsize' in kw:
            self._suptitle_fontsize = kw['fontsize']
        return self._suptitle

    def subplots_adjust(self, left=None, right=None, bottom=None, top=None,
                         wspace=None, hspace=None):
        if left is not None: self._subplotpars['left'] = left
        if right is not None: self._subplotpars['right'] = right
        if bottom is not None: self._subplotpars['bottom'] = bottom
        if top is not None: self._subplotpars['top'] = top
        if wspace is not None: self._subplotpars['wspace'] = wspace
        if hspace is not None: self._subplotpars['hspace'] = hspace
        return self._subplotpars

    def tight_layout(self, **kw):
        return None

    def colorbar(self, mappable, ax=None, **kw):
        if ax is None:
            ax = self._axes[0] if self._axes else None
        return ax.add_colorbar(mappable, **kw)

    def gca(self):
        if not self._axes:
            return self.add_subplot(1, 1, 1)
        return self._axes[-1]

    def savefig(self, fname, dpi=None, bbox_inches=None, pad_inches=None, **kw):
        dpi = dpi or _rc('savefig.dpi')
        canvas = self._render(dpi=float(dpi))
        ext = os.path.splitext(fname)[1].lower()
        if not ext:
            fname = fname + '.png'
            ext = '.png'
        parent = os.path.dirname(os.path.abspath(fname))
        if parent:
            os.makedirs(parent, exist_ok=True)
        if ext == '.png':
            canvas.save(fname, format='PNG')
        elif ext in ('.jpg', '.jpeg'):
            canvas.save(fname, format='JPEG', quality=kw.get('quality', 95))
        elif ext == '.bmp':
            canvas.save(fname, format='BMP')
        elif ext == '.gif':
            canvas.save(fname, format='GIF')
        elif ext in ('.tiff', '.tif'):
            canvas.save(fname, format='TIFF')
        elif ext == '.pdf':
            canvas.save(fname + '.png', format='PNG')
            fname = fname + '.png'
        else:
            canvas.save(fname, format='PNG')
        return fname

    def show(self, **kw):
        import tempfile
        try:
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            tmp.close()
            self.savefig(tmp.name)
            return tmp.name
        except (OSError, ValueError, RuntimeError):
            return None

    def clear(self):
        self._axes = []

    def clf(self):
        self.clear()

    def close(self):
        global _FIGURES, _CURRENT_FIGURE, _CURRENT_AXES
        if self in _FIGURES:
            _FIGURES.remove(self)
        if _CURRENT_FIGURE is self:
            _CURRENT_FIGURE = _FIGURES[-1] if _FIGURES else None
            _CURRENT_AXES = _CURRENT_FIGURE._axes[-1] if _CURRENT_FIGURE and _CURRENT_FIGURE._axes else None

    def _render(self, dpi=None):
        dpi = dpi or self.dpi
        self._canvas_size = (int(self.figsize[0] * dpi), int(self.figsize[1] * dpi))
        W, H = self._canvas_size
        bg = _to_rgb(self.facecolor) or (255, 255, 255)
        image = Image.new('RGBA', (W, H), (*bg, 255))
        for ax in self._axes:
            self._draw_axes(ax, image)
        if self._suptitle:
            font = _get_font(self._suptitle_fontsize or (_rc('axes.titlesize') + 2))
            tw, th = _measure(font, self._suptitle)
            overlay = ImageDraw.Draw(image)
            overlay.text(((W - tw) / 2, max(2, int(0.02 * dpi))),
                         self._suptitle, fill=_to_rgb('black'), font=font)
        return image.convert('RGB')

    def _draw_axes(self, ax, image):
        ax._ensure_limits()
        b = ax._box()
        bg = _to_rgb(ax._facecolor) or (255, 255, 255)
        ax_overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(ax_overlay)
        draw.rectangle((b['left'], min(b['top'], b['bottom']),
                        b['right'], max(b['top'], b['bottom'])), fill=bg)
        ctx = {'draw': draw, 'image': ax_overlay, 'dpi': ax.figure.dpi}
        for art in sorted(ax._artists, key=lambda a: a.zorder):
            if not art.visible:
                continue
            if isinstance(art, Text):
                self._draw_text(ax, art, draw)
            else:
                art.draw(ax, ctx)
        if ax._grid:
            self._draw_grid(ax, draw)
        ec = _to_rgb(_rc('axes.color'))
        lw = max(1, int(_rc('axes.linewidth')))
        if ax._spines_visible.get('left', True):
            draw.line([(b['left'], b['bottom']), (b['left'], b['top'])], fill=ec, width=lw)
        if ax._spines_visible.get('bottom', True):
            draw.line([(b['left'], b['bottom']), (b['right'], b['bottom'])], fill=ec, width=lw)
        if ax._spines_visible.get('top', False):
            draw.line([(b['left'], b['top']), (b['right'], b['top'])], fill=ec, width=lw)
        if ax._spines_visible.get('right', False):
            draw.line([(b['right'], b['bottom']), (b['right'], b['top'])], fill=ec, width=lw)
        if not getattr(ax, '_sharex_twin', False):
            x_side = getattr(ax, '_xaxis_side', 'bottom')
            x_base = b['bottom'] if x_side == 'bottom' else b['top']
            x_dir = 1 if x_side == 'bottom' else -1
            if ax._xscale == 'log':
                xticks = _log_ticks(ax._xlim[0], ax._xlim[1])
            else:
                xticks = _nice_ticks(ax._xlim[0], ax._xlim[1], count=6)
            xlo, xhi = min(ax._xlim), max(ax._xlim)
            for t in xticks:
                if t < xlo - 1e-9 or t > xhi + 1e-9:
                    continue
                x = ax._proj_x(t)
                draw.line([(x, x_base), (x, x_base + 4 * x_dir)], fill=ec, width=lw)
                label = _format_tick(t, scale=ax._xscale)
                if label:
                    font = _get_font(_rc('xtick.labelsize'))
                    tw, th = _measure(font, label)
                    ly = x_base + 6 if x_side == 'bottom' else x_base - th - 6
                    draw.text((x - tw / 2, ly), label, fill=ec, font=font)
        if not getattr(ax, '_sharey_twin', False):
            y_side = getattr(ax, '_yaxis_side', 'left')
            y_base = b['left'] if y_side == 'left' else b['right']
            y_dir = -1 if y_side == 'left' else 1
            if ax._yscale == 'log':
                yticks = _log_ticks(ax._ylim[0], ax._ylim[1])
            else:
                yticks = _nice_ticks(ax._ylim[0], ax._ylim[1], count=6)
            ylo, yhi = min(ax._ylim), max(ax._ylim)
            for t in yticks:
                if t < ylo - 1e-9 or t > yhi + 1e-9:
                    continue
                y = ax._proj_y(t)
                draw.line([(y_base + 4 * y_dir, y), (y_base, y)], fill=ec, width=lw)
                label = _format_tick(t, scale=ax._yscale)
                font = _get_font(_rc('ytick.labelsize'))
                tw, th = _measure(font, label)
                lx = y_base - 6 - tw if y_side == 'left' else y_base + 6
                draw.text((lx, y - th / 2), label, fill=ec, font=font)
        if ax._xlabel:
            font = _get_font(ax._xlabel_fontsize or _rc('axes.labelsize'))
            tw, th = _measure(font, ax._xlabel)
            color = _to_rgb(ax._xlabel_color) or _to_rgb('black')
            draw.text(((b['left'] + b['right']) / 2 - tw / 2, b['bottom'] + th + 6),
                      ax._xlabel, fill=color, font=font)
        if ax._ylabel:
            font = _get_font(ax._ylabel_fontsize or _rc('axes.labelsize'))
            tw, th = _measure(font, ax._ylabel)
            color = _to_rgb(ax._ylabel_color) or _to_rgb('black')
            y_pos = (b['bottom'] + b['top']) / 2
            label_img = Image.new('RGBA', (tw + 4, th + 4), (0, 0, 0, 0))
            ld = ImageDraw.Draw(label_img)
            ld.text((2, 2), ax._ylabel, fill=color, font=font)
            label_img = label_img.rotate(90, expand=True)
            ax_overlay.paste(label_img,
                             (int(b['left'] - label_img.size[0] - 8),
                              int(y_pos - label_img.size[1] / 2)),
                             label_img)
        if ax._title:
            font = _get_font(ax._title_fontsize or _rc('axes.titlesize'))
            tw, th = _measure(font, ax._title)
            color = _to_rgb(ax._title_color) or _to_rgb('black')
            draw.text(((b['left'] + b['right']) / 2 - tw / 2, max(0, b['top'] - th - 4)),
                      ax._title, fill=color, font=font)
        if getattr(ax, '_explicit_legend', None):
            self._draw_legend(ax, draw, ax._explicit_legend)
        for cb in ax._colorbars:
            cb.render(image, draw)
        image.alpha_composite(ax_overlay)

    def _draw_text(self, ax, t, draw):
        cx = ax._proj_x(t.x)
        cy = ax._proj_y(t.y)
        font = _get_font(t.fontsize)
        text = t.text
        if not text:
            return
        tw, th = _measure(font, text)
        if t.ha == 'center':
            x = cx - tw / 2
        elif t.ha == 'right':
            x = cx - tw
        else:
            x = cx
        if t.va == 'center':
            y = cy - th / 2
        elif t.va == 'top':
            y = cy - th
        else:
            y = cy
        c = _rgba(t.color, t.alpha)
        if c is None:
            return
        if t.rotation:
            try:
                img = Image.new('RGBA', (tw + 8, th + 8), (0, 0, 0, 0))
                d = ImageDraw.Draw(img)
                d.text((4, 4), text, fill=c, font=font)
                img = img.rotate(-t.rotation, expand=True)
                draw._image.alpha_composite(img, (int(x) - 4, int(y) - 4))
            except (OSError, ValueError, RuntimeError):
                draw.text((int(x), int(y)), text, fill=c, font=font)
        else:
            draw.text((int(x), int(y)), text, fill=c, font=font)

    def _draw_legend(self, ax, draw, items):
        if not items:
            return
        b = ax._box()
        max_tw = 0
        total_th = 0
        rows = []
        for handle, label in items:
            text = str(label) if label is not None else '_nolegend_'
            font = _get_font(_rc('legend.fontsize'))
            tw, th = _measure(font, text)
            max_tw = max(max_tw, tw)
            total_th += th + 4
            rows.append((handle, text, tw, th))
        pad = 6
        w = max_tw + 60 + pad
        h = total_th + pad
        x0 = b['right'] - w - 8
        y0 = b['top'] - h - 8
        if getattr(ax, '_legend_frame', True):
            fc = _to_rgb('#FFFFFF') or (255, 255, 255)
            ec = _to_rgb('#999999') or (153, 153, 153)
            draw.rectangle((x0, y0, x0 + w, y0 + h), fill=(*fc, 220), outline=ec)
        cy = y0 + pad
        for handle, text, tw, th in rows:
            if isinstance(handle, Line2D):
                color = handle.color or _next_color(0)
                swatch_color = _rgba(color, handle.alpha)
                if swatch_color is not None:
                    draw.line([(x0 + 6, cy + th / 2), (x0 + 30, cy + th / 2)],
                              fill=swatch_color, width=max(1, int(handle.linewidth or 1)))
            else:
                color = _rgba(_next_color(0))
                if color is not None:
                    draw.rectangle((x0 + 6, cy + 2, x0 + 30, cy + th - 2), fill=color)
            font = _get_font(_rc('legend.fontsize'))
            draw.text((x0 + 36, cy), text, fill=_to_rgb('black'), font=font)
            cy += th + 4

    def _draw_grid(self, ax, draw):
        b = ax._box()
        gc = _to_rgb(_rc('grid.color')) or (176, 176, 176)
        glw = max(1, int(round(_rc('grid.linewidth'))))
        if ax._xscale == 'log':
            xticks = _log_ticks(ax._xlim[0], ax._xlim[1])
        else:
            xticks = _nice_ticks(ax._xlim[0], ax._xlim[1], count=6)
        if ax._yscale == 'log':
            yticks = _log_ticks(ax._ylim[0], ax._ylim[1])
        else:
            yticks = _nice_ticks(ax._ylim[0], ax._ylim[1], count=6)
        xlo, xhi = min(ax._xlim), max(ax._xlim)
        ylo, yhi = min(ax._ylim), max(ax._ylim)
        for t in xticks:
            if t < xlo - 1e-9 or t > xhi + 1e-9:
                continue
            x = ax._proj_x(t)
            draw.line([(x, b['bottom']), (x, b['top'])], fill=gc, width=glw)
        for t in yticks:
            if t < ylo - 1e-9 or t > yhi + 1e-9:
                continue
            y = ax._proj_y(t)
            draw.line([(b['left'], y), (b['right'], y)], fill=gc, width=glw)

def _parse_subplot_spec(*args):
    if len(args) == 1:
        spec = str(args[0])
        if ',' in spec:
            parts = [int(x) for x in spec.split(',')]
            return parts[0], parts[1], parts[2]
        if len(spec) == 3:
            return int(spec[0]), int(spec[1]), int(spec[2])
    if len(args) == 3:
        return args[0], args[1], args[2]
    raise ValueError(f"cannot parse subplot spec: {args}")

def _gs_rect(nrows, ncols, index, sp):
    i = (index - 1) % ncols
    j = (nrows - 1) - ((index - 1) // ncols)
    avail_w = sp['right'] - sp['left']
    avail_h = sp['top'] - sp['bottom']
    cell_w = avail_w / ncols
    cell_h = avail_h / nrows
    if ncols > 1 and sp['wspace']:
        cell_w = cell_w * (1 - sp['wspace'])
    if nrows > 1 and sp['hspace']:
        cell_h = cell_h * (1 - sp['hspace'])
    left = sp['left'] + i * (avail_w / ncols)
    if ncols > 1 and sp['wspace']:
        left = sp['left'] + i * (avail_w / ncols) + sp['wspace'] * cell_w * 0.5
    bottom = sp['bottom'] + j * (avail_h / nrows)
    if nrows > 1 and sp['hspace']:
        bottom = sp['bottom'] + j * (avail_h / nrows) + sp['hspace'] * cell_h * 0.5
    return left, bottom, cell_w, cell_h

_FIGURES = []
_CURRENT_FIGURE = None
_CURRENT_AXES = None

def figure(num=None, figsize=None, dpi=None, facecolor=None,
           color=None, **kw):
    global _CURRENT_FIGURE, _CURRENT_AXES
    if num is not None and isinstance(num, int):
        for f in _FIGURES:
            if getattr(f, '_num', None) == num:
                _CURRENT_FIGURE = f
                _CURRENT_AXES = f._axes[-1] if f._axes else None
                return f
    f = Figure(figsize=figsize, dpi=dpi, facecolor=facecolor, color=color, **kw)
    f._num = num if num is not None else len(_FIGURES) + 1
    _FIGURES.append(f)
    _CURRENT_FIGURE = f
    _CURRENT_AXES = None
    return f

def subplots(nrows=1, ncols=1, *, sharex=False, sharey=False, squeeze=True,
             figsize=None, dpi=None, facecolor=None, **kw):
    global _CURRENT_FIGURE, _CURRENT_AXES
    f = figure(figsize=figsize, dpi=dpi, facecolor=facecolor)
    if nrows * ncols == 1:
        ax = f.add_subplot(1, 1, 1, **kw)
        _CURRENT_FIGURE = f
        _CURRENT_AXES = ax
        if squeeze:
            return f, ax
        return f, [ax]
    axes = []
    for i in range(1, nrows * ncols + 1):
        axes.append(f.add_subplot(nrows, ncols, i, **kw))
    _CURRENT_FIGURE = f
    _CURRENT_AXES = axes[-1]
    if squeeze and (nrows == 1 or ncols == 1):
        return f, axes
    return f, _AxesGrid(axes, nrows, ncols)

def gcf():
    return _CURRENT_FIGURE or figure()

def gca():
    global _CURRENT_AXES
    f = gcf()
    if _CURRENT_AXES is None or _CURRENT_AXES not in f._axes:
        _CURRENT_AXES = f.add_subplot(1, 1, 1)
    return _CURRENT_AXES

def sca(ax):
    global _CURRENT_AXES
    _CURRENT_AXES = ax
    return ax

def clf():
    gcf().clf()

def close(fig=None):
    global _CURRENT_FIGURE, _CURRENT_AXES
    if fig is None:
        fig = _CURRENT_FIGURE
    if fig in _FIGURES:
        _FIGURES.remove(fig)
    if _CURRENT_FIGURE is fig:
        _CURRENT_FIGURE = _FIGURES[-1] if _FIGURES else None
        _CURRENT_AXES = _CURRENT_FIGURE._axes[-1] if _CURRENT_FIGURE and _CURRENT_FIGURE._axes else None

def show(*args, **kw):
    f = gcf()
    if f is not None:
        return f.show(*args, **kw)

def savefig(fname, **kw):
    f = gcf()
    if f is not None:
        return f.savefig(fname, **kw)

def title(t, **kw):
    return gca().set_title(t, **kw)

def xlabel(t, **kw):
    return gca().set_xlabel(t, **kw)

def ylabel(t, **kw):
    return gca().set_ylabel(t, **kw)

def xlim(*args):
    return gca().set_xlim(*args)

def ylim(*args):
    return gca().set_ylim(*args)

def xticks(*args, **kw):
    if not args:
        return gca()._xticks_pos
    return gca().set_xticks(args[0], labels=args[1] if len(args) > 1 else None, **kw)

def yticks(*args, **kw):
    if not args:
        return gca()._yticks_pos
    return gca().set_yticks(args[0], labels=args[1] if len(args) > 1 else None, **kw)

def plot(*args, **kw):
    return gca().plot(*args, **kw)

def scatter(*args, **kw):
    return gca().scatter(*args, **kw)

def bar(*args, **kw):
    return gca().bar(*args, **kw)

def hist(*args, **kw):
    return gca().hist(*args, **kw)

def imshow(*args, **kw):
    return gca().imshow(*args, **kw)

def legend(*args, **kw):
    return gca().legend(*args, **kw)

def grid(b=None, **kw):
    if b is None:
        return gca()._grid
    return gca().grid(b, **kw)

def suptitle(t, **kw):
    return gcf().suptitle(t, **kw)

def subplot(*args, **kw):
    return gcf().add_subplot(*args, **kw)

def axes(rect, **kw):
    return gcf().add_axes(rect, **kw)

def text(x, y, s, **kw):
    return gca().text(x, y, s, **kw)

def annotate(*args, **kw):
    return gca().annotate(*args, **kw)

def arrow(*args, **kw):
    return gca().arrow(*args, **kw)

def axhline(*args, **kw):
    return gca().axhline(*args, **kw)

def axvline(*args, **kw):
    return gca().axvline(*args, **kw)

def axhspan(*args, **kw):
    return gca().axhspan(*args, **kw)

def axvspan(*args, **kw):
    return gca().axvspan(*args, **kw)

def fill_between(*args, **kw):
    return gca().fill_between(*args, **kw)

def errorbar(*args, **kw):
    return gca().errorbar(*args, **kw)

def pcolormesh(*args, **kw):
    return gca().pcolormesh(*args, **kw)

def colorbar(mappable=None, **kw):
    return gcf().colorbar(mappable, **kw)

def subplots_adjust(**kw):
    return gcf().subplots_adjust(**kw)

def tight_layout(**kw):
    return gcf().tight_layout(**kw)

def line(data, title='', xlabel='', ylabel='', label='', **kw):
    f, ax = subplots()
    ax.plot(data, label=label or None)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if label:
        ax.legend()
    return f, ax

def scatter_xy(x, y, title='', xlabel='', ylabel='', label='', **kw):
    f, ax = subplots()
    ax.scatter(x, y, label=label or None)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if label:
        ax.legend()
    return f, ax

def heatmap(matrix, title='', cmap='viridis', **kw):
    f, ax = subplots()
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', origin='lower')
    f.colorbar(im, ax=ax)
    if title:
        ax.set_title(title)
    return f, ax

def histogram(data, bins=10, title='', **kw):
    f, ax = subplots()
    ax.hist(data, bins=bins)
    if title:
        ax.set_title(title)
    return f, ax

def bar_chart(labels, values, title='', **kw):
    f, ax = subplots()
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    if title:
        ax.set_title(title)
    return f, ax

def save(path, dpi=150):
    f = gcf()
    if f is not None:
        return f.savefig(path, dpi=dpi)

def imread(path):
    return Image.open(path)

__all__ = [
    'figure', 'subplots', 'gcf', 'gca', 'sca', 'clf', 'close', 'show', 'savefig',
    'title', 'xlabel', 'ylabel', 'xlim', 'ylim', 'xticks', 'yticks',
    'plot', 'scatter', 'bar', 'hist', 'imshow', 'legend', 'grid', 'suptitle',
    'subplot', 'axes', 'text', 'annotate', 'arrow', 'axhline', 'axvline',
    'axhspan', 'axvspan', 'fill_between', 'errorbar', 'pcolormesh', 'colorbar',
    'subplots_adjust', 'tight_layout', 'get_cmap', 'Normalize', 'rcParams',
    'line', 'scatter_xy', 'heatmap', 'histogram', 'bar_chart', 'save',
    'Figure', 'Axes', 'Line2D', 'Patch', 'Polygon', 'Circle', 'W',
    'Collection', 'Image', 'QuadMesh', 'Text',
]
