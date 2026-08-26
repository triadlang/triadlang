from __future__ import annotations

_SPARK_CHARS = "▁▂▃▄▅▆▇█"
_FIELD_CHARS = " ░▒▓█"

def sparkline(values: list[float], width: int = 30) -> str:
    if not values:
        return ""
    n = len(values)
    if n <= width:
        sampled = values
    else:
        step = n / width
        sampled = [values[int(i * step)] for i in range(width)]
    lo = min(sampled)
    hi = max(sampled)
    rng = hi - lo if hi != lo else 1.0
    chars = []
    for v in sampled:
        idx = int((v - lo) / rng * (len(_SPARK_CHARS) - 1))
        chars.append(_SPARK_CHARS[max(0, min(idx, len(_SPARK_CHARS) - 1))])
    return "".join(chars)

def bar(value: float, max_val: float, width: int = 20) -> str:
    if max_val == 0:
        return "[" + " " * width + "]"
    ratio = min(value / max_val, 1.0)
    filled = int(ratio * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"

def field_ascii(data: list[float], width: int = 60) -> str:
    if not data:
        return ""
    n = len(data)
    step = max(1, n // width)
    sampled = [data[i] for i in range(0, n, step)][:width]
    lo = min(sampled)
    hi = max(sampled)
    rng = hi - lo if hi != lo else 1.0
    chars = []
    for v in sampled:
        idx = int((v - lo) / rng * (len(_FIELD_CHARS) - 1))
        chars.append(_FIELD_CHARS[max(0, min(idx, len(_FIELD_CHARS) - 1))])
    return "".join(chars)

def field_2d_ascii(data_2d: list[list[float]], width: int = 40, height: int = 15) -> str:
    if not data_2d or not data_2d[0]:
        return ""
    rows = len(data_2d)
    cols = len(data_2d[0])
    flat = [data_2d[r][c] for r in range(rows) for c in range(cols)]
    lo = min(flat)
    hi = max(flat)
    rng = hi - lo if hi != lo else 1.0
    lines = []
    row_step = max(1, rows // height)
    col_step = max(1, cols // width)
    for r in range(0, rows, row_step)[:height]:
        line = []
        for c in range(0, cols, col_step)[:width]:
            v = data_2d[r][c]
            idx = int((v - lo) / rng * (len(_FIELD_CHARS) - 1))
            line.append(_FIELD_CHARS[max(0, min(idx, len(_FIELD_CHARS) - 1))])
        lines.append("".join(line))
    return "\n".join(lines)

def model_arch_ascii(layers: list[dict]) -> str:
    if not layers:
        return ""
    boxes = []
    for layer in layers:
        name = layer.get("name", "?")
        detail = layer.get("detail", "")
        w = max(len(name), len(detail)) + 4
        top = "┌" + "─" * w + "┐"
        mid_name = "│ " + name.center(w - 2) + " │"
        mid_detail = "│ " + detail.center(w - 2) + " │"
        bot = "└" + "─" * w + "┘"
        boxes.append((top, mid_name, mid_detail, bot, w))
    arrow = " → "
    result_lines = [""] * 4
    for i, (top, mid_name, mid_detail, bot, w) in enumerate(boxes):
        if i > 0:
            for j in range(4):
                result_lines[j] += arrow
        result_lines[0] += top
        result_lines[1] += mid_name
        result_lines[2] += mid_detail
        result_lines[3] += bot
    return "\n".join(result_lines)

def format_observable(name: str, history: list[float], current: float, width: int = 30) -> str:
    spark = sparkline(history, width)
    return f"{name:14s} {current:10.6f}  {spark}"
