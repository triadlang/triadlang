"""Source map for TriadLang: maps compiled Python line numbers back to .tri source lines.

When TriadLang compiles .tri -> Python, each emitted Python line may originate
from a different .tri source line. This module tracks that mapping so the
debugger and error reporting can show the original .tri location.

The source map is stored as:
    { python_line_number: (tri_line_number, tri_file) }

Usage in compiler:
    self._sourcemap = SourceMap(source_file)
    self._sourcemap.mark(tri_line=5)
    self._emit('x = 10')  # python line N maps to tri line 5

Usage in debugger/error handler:
    tri_loc = sourcemap.lookup(python_line=42)
    # -> SourceLoc(line=5, col=0, file='test.tri')
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field

@dataclass
class SourceLoc:
    line: int = 0
    col: int = 0
    file: str = ''

class SourceMap:
    """Tracks mapping from compiled output lines to original source lines."""

    def __init__(self, source_file: str = '<triad>'):
        self.source_file = source_file
        
        self._entries: dict[int, SourceLoc] = {}
        self._pending_loc: SourceLoc | None = None
        self._lines_emitted: int = 0

    def set_source_file(self, filepath: str):
        self.source_file = filepath

    def mark(self, tri_line: int, tri_col: int = 0, tri_file: str = ''):
        """Mark the next emitted line as originating from tri_line:tri_col.

        Call this BEFORE the corresponding _emit(). The mark applies to
        the next line emitted. If multiple _emit() calls happen after one
        mark(), only the first gets mapped; subsequent ones inherit.
        """
        f = tri_file or self.source_file
        self._pending_loc = SourceLoc(line=tri_line, col=tri_col, file=f)

    def mark_stmt(self, stmt) -> None:
        """Mark from an AST statement's pos field."""
        if hasattr(stmt, 'pos') and stmt.pos.line > 0:
            self.mark(stmt.pos.line, stmt.pos.col, stmt.pos.file)

    def record_line(self) -> None:
        """Called by the compiler each time a Python line is emitted.

        Records the current pending location for the next Python line number.
        """
        self._lines_emitted += 1
        if self._pending_loc is not None:
            self._entries[self._lines_emitted] = self._pending_loc
            
        elif self._lines_emitted > 1 and (self._lines_emitted - 1) in self._entries:
            
            prev = self._entries[self._lines_emitted - 1]
            self._entries[self._lines_emitted] = prev

    def lookup(self, python_line: int) -> SourceLoc:
        """Look up the .tri source location for a compiled Python line number."""
        if python_line in self._entries:
            return self._entries[python_line]
        
        best = SourceLoc(file=self.source_file)
        for pl in sorted(self._entries.keys()):
            if pl <= python_line:
                best = self._entries[pl]
            else:
                break
        return best

    def lookup_traceback(self, tb) -> list[dict]:
        """Convert a Python traceback to .tri source locations.

        Returns list of dicts with keys: py_file, py_line, tri_file, tri_line, tri_col
        """
        frames = []
        while tb is not None:
            frame = tb.tb_frame
            py_file = frame.f_code.co_filename
            py_line = tb.tb_lineno
            entry = {
                'py_file': py_file,
                'py_line': py_line,
            }
            if py_file == self.source_file or os.path.basename(py_file) == os.path.basename(self.source_file):
                loc = self.lookup(py_line)
                entry['tri_file'] = loc.file
                entry['tri_line'] = loc.line
                entry['tri_col'] = loc.col
            frames.append(entry)
            tb = tb.tb_next
        return frames

    def format_traceback(self, tb, source_text: str = '') -> str:
        """Format a Python traceback using .tri source locations."""
        frames = self.lookup_traceback(tb)
        source_lines = source_text.split('\n') if source_text else []
        parts = ['Traceback (most recent call last):']
        for f in frames:
            if 'tri_file' in f and 'tri_line' in f:
                tline = f['tri_line']
                parts.append(f'  File "{f["tri_file"]}", line {tline}')
                if tline > 0 and tline <= len(source_lines):
                    parts.append(f'    {source_lines[tline - 1]}')
            else:
                parts.append(f'  File "{f["py_file"]}", line {f["py_line"]}')
        return '\n'.join(parts)

    def to_json(self) -> str:
        """Serialize the source map to JSON."""
        entries = {}
        for k, v in self._entries.items():
            entries[str(k)] = {'line': v.line, 'col': v.col, 'file': v.file}
        return json.dumps({
            'version': 1,
            'source_file': self.source_file,
            'entries': entries,
            'lines_emitted': self._lines_emitted,
        })

    @classmethod
    def from_json(cls, data: str) -> SourceMap:
        """Deserialize a source map from JSON."""
        obj = json.loads(data)
        sm = cls(obj.get('source_file', '<triad>'))
        sm._lines_emitted = obj.get('lines_emitted', 0)
        for k, v in obj.get('entries', {}).items():
            sm._entries[int(k)] = SourceLoc(line=v['line'], col=v['col'], file=v['file'])
        return sm

    def breakpoint_translate(self, tri_lines: list[int]) -> dict[int, list[int]]:
        """Given .tri line numbers, find which Python line numbers they map to.

        Returns: { tri_line: [py_line_1, py_line_2, ...] }
        """
        result: dict[int, list[int]] = {}
        for tri_line in tri_lines:
            result[tri_line] = []
        for py_line, loc in self._entries.items():
            if loc.line in result:
                result[loc.line].append(py_line)
        return result

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f'SourceMap(file={self.source_file!r}, entries={len(self._entries)}, lines={self._lines_emitted})'
