from __future__ import annotations
from typing import Optional, Sequence
import sys
_mpl = None
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as _plt
    _mpl = _plt
except ImportError:
    _plt = None
Figures: list = []
_current: Optional[object] = None

def _ensure_fig():
    global _current
    if _mpl:
        _current, _ = _plt.subplots()
    return _current

def line(data, title: str='', xlabel: str='', ylabel: str='', label: str='', **kw):
    fig = _ensure_fig()
    if _mpl:
        _current.plot(data, label=label or None, **kw)
        if title:
            _current.set_title(title)
        if xlabel:
            _current.set_xlabel(xlabel)
        if ylabel:
            _current.set_ylabel(ylabel)
        if label:
            _current.legend()
    else:
        _ascii_line(data, title)

def scatter(x, y, title: str='', xlabel: str='', ylabel: str='', label: str='', **kw):
    fig = _ensure_fig()
    if _mpl:
        _current.scatter(x, y, label=label or None, **kw)
        if title:
            _current.set_title(title)
        if xlabel:
            _current.set_xlabel(xlabel)
        if ylabel:
            _current.set_ylabel(ylabel)
        if label:
            _current.legend()
    else:
        _ascii_scatter(x, y, title)

def heatmap(matrix, title: str='', **kw):
    fig = _ensure_fig()
    if _mpl:
        im = _current.imshow(matrix, aspect='auto', cmap='inferno', **kw)
        _plt.colorbar(im, ax=_current)
        if title:
            _current.set_title(title)
    else:
        _ascii_heatmap(matrix, title)

def histogram(data, bins: int=10, title: str='', **kw):
    fig = _ensure_fig()
    if _mpl:
        _current.hist(data, bins=bins, **kw)
        if title:
            _current.set_title(title)
    else:
        _ascii_histogram(data, bins, title)

def bar(labels, values, title: str='', **kw):
    fig = _ensure_fig()
    if _mpl:
        _current.bar(range(len(labels)), values, **kw)
        _current.set_xticks(range(len(labels)))
        _current.set_xticklabels(labels)
        if title:
            _current.set_title(title)
    else:
        _ascii_bar(labels, values, title)

def save(path: str, dpi: int=150):
    if _mpl and _current:
        _current.savefig(path, dpi=dpi, bbox_inches='tight')
        _close_current()

def show():
    global _current
    if _mpl and _current:
        Figures.append(_current)
        _close_current()

def _close_current():
    global _current
    if _mpl and _current is not None:
        _plt.close(_current)
        _current = None

def _ascii_line(data, title: str=''):
    if title:
        print(f'  {title}')
    nums = list(data)
    if not nums:
        print('  (empty)')
        return
    lo, hi = (min(nums), max(nums))
    span = hi - lo if hi != lo else 1.0
    w = 60
    for v in nums:
        pos = int((v - lo) / span * (w - 1))
        print('  ' + ' ' * pos + '*')
    print(f'  lo={lo:.4g}  hi={hi:.4g}')

def _ascii_scatter(x, y, title: str=''):
    if title:
        print(f'  {title}')
    if not x or not y:
        print('  (empty)')
        return
    n = min(len(x), len(y), 200)
    print(f'  {n} points  x=[{min(x):.3g}, {max(x):.3g}]  y=[{min(y):.3g}, {max(y):.3g}]')

def _ascii_heatmap(matrix, title: str=''):
    if title:
        print(f'  {title}')
    rows = matrix
    if hasattr(matrix, 'shape'):
        rows = matrix
    if not rows:
        print('  (empty)')
        return
    chars = ' .:-=+*#%@'
    n = len(chars)
    flat = []
    for row in rows:
        flat.extend(row)
    lo, hi = (min(flat), max(flat))
    span = hi - lo if hi != lo else 1.0
    for row in rows:
        line_chars = []
        for v in row:
            idx = int((v - lo) / span * (n - 1))
            line_chars.append(chars[min(idx, n - 1)])
        print('  ' + ''.join(line_chars))

def _ascii_histogram(data, bins: int, title: str=''):
    if title:
        print(f'  {title}')
    nums = list(data)
    if not nums:
        print('  (empty)')
        return
    lo, hi = (min(nums), max(nums))
    span = hi - lo if hi != lo else 1.0
    counts = [0] * bins
    for v in nums:
        idx = min(int((v - lo) / span * bins), bins - 1)
        counts[idx] += 1
    mx = max(counts) if counts else 1
    w = 40
    for i, c in enumerate(counts):
        lo_b = lo + i * span / bins
        bar_len = int(c / mx * w) if mx else 0
        print(f"  [{lo_b:8.3g}] {'#' * bar_len} ({c})")

def _ascii_bar(labels, values, title: str=''):
    if title:
        print(f'  {title}')
    mx = max(values) if values else 1
    w = 40
    for label, val in zip(labels, values):
        bar_len = int(val / mx * w) if mx else 0
        print(f"  {str(label):>10s} | {'#' * bar_len} {val:.3g}")