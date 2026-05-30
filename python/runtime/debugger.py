from __future__ import annotations
import sys
import os
import linecache
import traceback
from typing import Optional

class DebugBreak(Exception):
    pass

class TriadDebugger:

    def __init__(self):
        self._breakpoints: set[int] = set()
        self._step_mode: bool = False
        self._current_frame = None
        self._filename: str = ''
        self._source_lines: list[str] = []
        self._hit_count: dict[int, int] = {}

    def set_breakpoints(self, lines: list[int]):
        self._breakpoints = set(lines)

    def add_breakpoint(self, line: int):
        self._breakpoints.add(line)

    def remove_breakpoint(self, line: int):
        self._breakpoints.discard(line)

    def clear_breakpoints(self):
        self._breakpoints.clear()

    def _bp_hook(self, line: int, locals_dict: dict):
        self._hit_count[line] = self._hit_count.get(line, 0) + 1
        if line in self._breakpoints or self._step_mode:
            self._current_frame = locals_dict
            self._interact(line)

    def _interact(self, line: int):
        src_line = self._source_lines[line - 1] if 0 < line <= len(self._source_lines) else ''
        print(f'\n  \x1b[1;33m-- Break at line {line} --\x1b[0m')
        print(f'  \x1b[2m{line}: {src_line.strip()}\x1b[0m')
        print(f"  Type 'h' for commands, 'c' to continue, 'q' to quit")
        while True:
            try:
                cmd = input('\x1b[1;36mtriad-dbg> \x1b[0m').strip()
            except (EOFError, KeyboardInterrupt):
                print('c')
                cmd = 'c'
            if not cmd:
                continue
            parts = cmd.split(None, 1)
            action = parts[0]
            arg = parts[1] if len(parts) > 1 else ''
            if action == 'c' or action == 'continue':
                self._step_mode = False
                break
            elif action == 'n' or action == 'next':
                self._step_mode = True
                break
            elif action == 'q' or action == 'quit':
                raise SystemExit(0)
            elif action == 'l' or action == 'list':
                start = max(1, line - 3)
                end = min(len(self._source_lines), line + 3)
                for i in range(start, end + 1):
                    marker = '>>>' if i == line else '   '
                    print(f'  {marker} {i}: {self._source_lines[i - 1].rstrip()}')
            elif action == 'p' or action == 'print':
                if not arg:
                    print('  usage: p <expr>')
                    continue
                try:
                    result = eval(arg, {}, self._current_frame or {})
                    print(f'  {result}')
                except Exception as e:
                    print(f'  error: {e}')
            elif action == 'w' or action == 'where':
                print(f'  at {self._filename}:{line}')
            elif action == 'v' or action == 'vars':
                if self._current_frame:
                    for k, v in sorted(self._current_frame.items()):
                        if not k.startswith('_') and (not callable(v)):
                            vstr = repr(v)
                            if len(vstr) > 80:
                                vstr = vstr[:77] + '...'
                            print(f'  {k} = {vstr}')
                else:
                    print('  no frame available')
            elif action == 'b' or action == 'break':
                if arg:
                    try:
                        bl = int(arg)
                        self.add_breakpoint(bl)
                        print(f'  breakpoint set at line {bl}')
                    except ValueError:
                        print('  usage: b <line>')
                else:
                    for bp in sorted(self._breakpoints):
                        hits = self._hit_count.get(bp, 0)
                        print(f'  line {bp} (hits: {hits})')
            elif action == 'd' or action == 'delete':
                if arg:
                    try:
                        self.remove_breakpoint(int(arg))
                        print(f'  breakpoint removed at line {arg}')
                    except ValueError:
                        print('  usage: d <line>')
            elif action == 'h' or action == 'help':
                print('  Commands:')
                print('    c, continue     Continue execution')
                print('    n, next         Step to next line')
                print('    l, list         Show source around current line')
                print('    p <expr>        Print expression value')
                print('    v, vars         Show local variables')
                print('    w, where        Show current location')
                print('    b [line]        Set/list breakpoints')
                print('    d <line>        Delete breakpoint')
                print('    q, quit         Exit program')
            else:
                try:
                    result = eval(cmd, {}, self._current_frame or {})
                    print(f'  {result}')
                except Exception as e:
                    print(f'  unknown command: {action} ({e})')

    def inject_hooks(self, source: str, filename: str) -> str:
        self._filename = filename
        self._source_lines = source.split('\n')
        lines = source.split('\n')
        result = []
        for i, line in enumerate(lines):
            line_num = i + 1
            stripped = line.lstrip()
            result.append(line)
            if line_num in self._breakpoints:
                indent = '    ' * (len(line) - len(stripped))
                indent_spaces = len(line) - len(stripped)
                indent_str = ' ' * indent_spaces
                result.append(f'{indent_str}_triad_dbg._bp_hook({line_num}, dict(locals()))')
        return '\n'.join(result)

    def run_source(self, source: str, filename: str='<triad>'):
        from frontend.parser_universal import parse
        from runtime.compiler_runtime import TriadCompiler
        mod = parse(source, filename)
        compiler = TriadCompiler()
        code = compiler.compile_to_source(mod)
        code = self.inject_hooks(code, filename)
        env = compiler._make_globals(filename)
        env['_triad_dbg'] = self
        try:
            compiled = compile(code, filename, 'exec')
            exec(compiled, env)
        except DebugBreak:
            pass
        except SystemExit:
            raise
        except Exception as e:
            print(f'runtime error: {e}', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    def run_file(self, path: str):
        with open(path) as f:
            source = f.read()
        self.run_source(source, path)

def cmd_debug(args):
    from frontend.parser_universal import parse
    import argparse as _ap
    p = _ap.ArgumentParser(prog='triad debug')
    p.add_argument('file', help='.tri file to debug')
    p.add_argument('--break', '-b', dest='breaks', default='', help='comma-separated line numbers')
    parsed = p.parse_args(args)
    path = parsed.file
    if not os.path.exists(path):
        print(f'error: file not found: {path}', file=sys.stderr)
        return 1
    dbg = TriadDebugger()
    if parsed.breaks:
        for b in parsed.breaks.split(','):
            b = b.strip()
            if b:
                try:
                    dbg.add_breakpoint(int(b))
                except ValueError:
                    print(f'invalid breakpoint line: {b}', file=sys.stderr)
    try:
        dbg.run_file(path)
    except SystemExit:
        pass
    return 0