from __future__ import annotations

import re

from rich.text import Text
from textual.widgets import Static

_KIND_STYLES = {
    'fn': 'bold cyan',
    'class': 'bold purple',
    'type': 'bold yellow',
    'let': 'green',
    'const': 'green',
    'reg': 'bold magenta',
    '@': 'dim cyan',
}

_KIND_ICONS = {
    'fn': 'ƒ',
    'class': 'C',
    'type': 'T',
    'let': 'v',
    'const': 'c',
    'reg': 'R',
    '@': '@',
}

class SymbolOutline(Static):
    DEFAULT_CSS = """
    SymbolOutline {
        height: auto;
        max-height: 16;
        border-top: solid $primary;
        padding: 0 1;
        overflow-y: auto;
    }
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self._symbols: list[tuple[str, str, int]] = []

    def update_from_source(self, source: str):
        self._symbols = []
        for i, line in enumerate(source.split('\n'), 1):
            s = line.strip()
            if not s or s.startswith('//') or s.startswith('#'):
                continue
            m = re.match(r'fn\s+(\w+)', s)
            if m:
                self._symbols.append(('fn', m.group(1), i))
                continue
            m = re.match(r'class\s+(\w+)', s)
            if m:
                self._symbols.append(('class', m.group(1), i))
                continue
            m = re.match(r'type\s+(\w+)', s)
            if m:
                self._symbols.append(('type', m.group(1), i))
                continue
            m = re.match(r'const\s+(\w+)', s)
            if m:
                self._symbols.append(('const', m.group(1), i))
                continue
            m = re.match(r'reg\s+(\w+)', s)
            if m:
                self._symbols.append(('reg', m.group(1), i))
                continue
            m = re.match(r'let\s+(\w+)', s)
            if m:
                name = m.group(1)
                if name.isidentifier():
                    self._symbols.append(('let', name, i))
                    continue
            if s.startswith('@') and not s.startswith('@T') and not s.startswith('@par'):
                dec = s.split('(')[0].strip() if '(' in s else s.split()[0].strip()
                self._symbols.append(('@', dec, i))

        self._render_symbols()

    def _render_symbols(self):
        if not self._symbols:
            self.update(Text("  (no symbols)", style="dim"))
            return
        lines = []
        for kind, name, lineno in self._symbols[:25]:
            icon = _KIND_ICONS.get(kind, '?')
            style = _KIND_STYLES.get(kind, '')
            lines.append(f"  {icon} [{style}]{name}[/{style}] [dim]L{lineno}[/dim]")
        self.update(Text.from_markup('\n'.join(lines)))

    def get_symbols(self) -> list[tuple[str, str, int]]:
        return list(self._symbols)

