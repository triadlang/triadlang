
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import IntEnum


class Level(IntEnum):
    ERROR = 0
    WARNING = 1
    NOTE = 2
    HELP = 3

    @property
    def label(self) -> str:
        return {Level.ERROR: 'error', Level.WARNING: 'warning', Level.NOTE: 'note', Level.HELP: 'help'}[self]

    @property
    def color(self) -> str:
        return {
            Level.ERROR: '\033[1;31m',
            Level.WARNING: '\033[1;33m',
            Level.NOTE: '\033[1;36m',
            Level.HELP: '\033[1;32m',
        }[self]

@dataclass
class Span:

    line_start: int
    col_start: int
    line_end: int = 0
    col_end: int = 0

    def __post_init__(self):
        if self.line_end == 0:
            self.line_end = self.line_start
        if self.col_end == 0:
            self.col_end = self.col_start + 1

    @property
    def is_single_line(self) -> bool:
        return self.line_start == self.line_end

@dataclass
class Diagnostic:

    level: Level
    span: Span
    message: str
    code: str = ''
    hint: str = ''
    children: list[Diagnostic] = field(default_factory=list)

    def format(self, source_lines: list[str], filename: str, use_color: bool = True) -> str:

        parts = []
        reset = '\033[0m' if use_color else ''
        bold = '\033[1m' if use_color else ''
        dim = '\033[2m' if use_color else ''
        color = self.level.color if use_color else ''
        level_str = self.level.label.upper()

        loc = f"{filename}:{self.span.line_start}:{self.span.col_start}"
        code_part = f"[{self.code}]: " if self.code else ": "
        header = f"{color}{level_str}{reset}{bold}{code_part}{self.message}{reset}"
        parts.append(f"{bold}{loc}{reset}: {header}")

        if 0 < self.span.line_start <= len(source_lines):
            line_content = source_lines[self.span.line_start - 1]
            line_num_str = str(self.span.line_start)
            gutter = ' ' * len(line_num_str)

            parts.append(f"{dim}{gutter} |{reset}")
            parts.append(f"{dim}{line_num_str} |{reset} {line_content}")

            col_start = max(0, self.span.col_start - 1)
            col_end = self.span.col_end - 1 if self.span.is_single_line else len(line_content)
            pointer_len = max(1, col_end - col_start)
            pointer = ' ' * col_start + color + '^' * pointer_len + reset
            parts.append(f"{dim}{gutter} |{reset} {pointer}")

        if self.hint:
            parts.append(f"{Level.HELP.color if use_color else ''}  help: {self.hint}{reset}")

        for child in self.children:
            child_lines = child.format(source_lines, filename, use_color)
            for cl in child_lines.split('\n'):
                if cl.strip():
                    parts.append(f"  {cl}")

        return '\n'.join(parts)

class DiagnosticSink:

    def __init__(self, source: str = '', filename: str = '<triad>'):
        self.source = source
        self.filename = filename
        self.source_lines = source.split('\n') if source else []
        self.diagnostics: list[Diagnostic] = []
        self.error_count = 0
        self.warning_count = 0

    def _add(self, level: Level, line: int, col: int, message: str, code: str = '', hint: str = '', end_col: int = 0):
        span = Span(line, col, line, end_col if end_col > 0 else col + 1)
        d = Diagnostic(level=level, span=span, message=message, code=code, hint=hint)
        self.diagnostics.append(d)
        if level == Level.ERROR:
            self.error_count += 1
        elif level == Level.WARNING:
            self.warning_count += 1
        return d

    def error(self, line: int, col: int, message: str, code: str = '', hint: str = '', end_col: int = 0) -> Diagnostic:
        return self._add(Level.ERROR, line, col, message, code, hint, end_col)

    def warning(self, line: int, col: int, message: str, code: str = '', hint: str = '', end_col: int = 0) -> Diagnostic:
        return self._add(Level.WARNING, line, col, message, code, hint, end_col)

    def note(self, line: int, col: int, message: str, code: str = '', hint: str = ''):
        return self._add(Level.NOTE, line, col, message, code, hint)

    def has_errors(self) -> bool:
        return self.error_count > 0

    def format_all(self, use_color: bool = None) -> str:

        if use_color is None:
            use_color = hasattr(sys.stderr, 'isatty') and sys.stderr.isatty()
        parts = []
        for d in self.diagnostics:
            parts.append(d.format(self.source_lines, self.filename, use_color))
        return '\n\n'.join(parts)

    def report(self, file=None, use_color: bool = None):

        if file is None:
            file = sys.stderr
        output = self.format_all(use_color)
        if output:
            print(output, file=file)
            if self.error_count > 0:
                summary = f"aborting due to {self.error_count} previous error"
                if self.error_count > 1:
                    summary += 's'
                print(f"\033[1;31m{summary}\033[0m" if (use_color if use_color is not None else hasattr(file, 'isatty') and file.isatty()) else summary, file=file)

    def raise_if_errors(self):

        if self.error_count > 0:
            raise DiagnosticError(self)

class DiagnosticError(Exception):

    def __init__(self, sink: DiagnosticSink):
        self.sink = sink
        self.diagnostics = sink.diagnostics
        self.error_count = sink.error_count
        super().__init__(sink.format_all(use_color=False))

def format_code_error(filename: str, source: str, line: int, col: int, message: str, hint: str = '', code: str = '') -> str:

    sink = DiagnosticSink(source, filename)
    sink.error(line, col, message, code=code, hint=hint)
    return sink.format_all(use_color=False)
