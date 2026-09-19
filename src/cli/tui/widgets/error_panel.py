from __future__ import annotations

import os
import re

from rich.panel import Panel
from rich.text import Text
from textual.widgets import RichLog


def format_error(msg: str, source: str = "", file_path: str = "") -> Panel:
    lines = []
    err_type = "Error"
    err_line = -1
    err_col = -1

    lex_match = re.search(r'error\[LEX\]:\s*(.*)', msg, re.IGNORECASE)
    parse_match = re.search(r'ERROR\[PARSE\]:\s*(.*)', msg, re.IGNORECASE)
    pos_match = re.search(r'L(\d+):C(\d+)', msg)
    line_match = re.search(r'line:\s*(\d+)', msg, re.IGNORECASE)
    col_match = re.search(r'col:\s*(\d+)', msg, re.IGNORECASE)

    if pos_match:
        err_line = int(pos_match.group(1))
        err_col = int(pos_match.group(2))
    else:
        if line_match:
            err_line = int(line_match.group(1))
        if col_match:
            err_col = int(col_match.group(1))

    if lex_match:
        err_type = "LexError"
        detail = lex_match.group(1)
    elif parse_match:
        err_type = "ParseError"
        detail = parse_match.group(1)
    else:
        detail = msg

    content = Text()
    content.append(f"{err_type}", style="bold red")
    if file_path:
        content.append(f" in {os.path.basename(file_path)}")
    if err_line > 0:
        content.append(f":{err_line}")
    if err_col > 0:
        content.append(f":{err_col}")
    content.append("\n")
    content.append(f"{detail}\n\n")

    if source and err_line > 0:
        src_lines = source.split('\n')
        start = max(0, err_line - 3)
        end = min(len(src_lines), err_line + 2)
        for i in range(start, end):
            marker = " >" if i == err_line - 1 else "  "
            line_text = f"{marker} {i+1:4d}│ {src_lines[i]}\n"
            if i == err_line - 1:
                content.append(line_text, style="bold red")
            else:
                content.append(line_text, style="dim")

    return Panel(content, title="Error", border_style="red")

def write_error(log: RichLog, msg: str, source: str = "", file_path: str = ""):
    panel = format_error(msg, source, file_path)
    log.write(panel)
