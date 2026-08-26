
from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass
class SourceLoc:
    line: int = 0
    col: int = 0
    file: str = ''

class SourceMap:

    def __init__(self, source_file: str = '<triad>'):
        self.source_file = source_file

        self._entries: dict[int, SourceLoc] = {}
        self._pending_loc: SourceLoc | None = None
        self._lines_emitted: int = 0

    def set_source_file(self, filepath: str):
        self.source_file = filepath

    def mark(self, tri_line: int, tri_col: int = 0, tri_file: str = ''):

        f = tri_file or self.source_file
        self._pending_loc = SourceLoc(line=tri_line, col=tri_col, file=f)

    def mark_stmt(self, stmt) -> None:

        if hasattr(stmt, 'pos') and stmt.pos.line > 0:
            self.mark(stmt.pos.line, stmt.pos.col, stmt.pos.file)

    def record_line(self) -> None:

        self._lines_emitted += 1
        if self._pending_loc is not None:
            self._entries[self._lines_emitted] = self._pending_loc

        elif self._lines_emitted > 1 and (self._lines_emitted - 1) in self._entries:

            prev = self._entries[self._lines_emitted - 1]
            self._entries[self._lines_emitted] = prev

    def lookup(self, python_line: int) -> SourceLoc:

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

        obj = json.loads(data)
        sm = cls(obj.get('source_file', '<triad>'))
        sm._lines_emitted = obj.get('lines_emitted', 0)
        for k, v in obj.get('entries', {}).items():
            sm._entries[int(k)] = SourceLoc(line=v['line'], col=v['col'], file=v['file'])
        return sm

    def breakpoint_translate(self, tri_lines: list[int]) -> dict[int, list[int]]:

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
