from __future__ import annotations

import json
import os
import sys
import threading
import traceback

from triad import ntri as np


class DebugBreak(Exception):
    pass

class DAPProtocol:

    def __init__(self, infile=None, outfile=None):
        self._in = infile or sys.stdin.buffer
        self._out = outfile or sys.stdout.buffer
        self._seq = 1

    def _next_seq(self) -> int:
        seq = self._seq
        self._seq += 1
        return seq

    def read_message(self) -> dict | None:

        header = b''
        while True:
            byte = self._in.read(1)
            if not byte:
                return None
            header += byte
            if header.endswith(b'\r\n\r\n'):
                break
        header_str = header.decode('ascii')
        content_length = 0
        for line in header_str.split('\r\n'):
            if line.startswith('Content-Length:'):
                content_length = int(line.split(':')[1].strip())
        if content_length == 0:
            return None
        body = self._in.read(content_length)
        return json.loads(body.decode('utf-8'))

    def send_message(self, msg: dict[str, object]):

        body = json.dumps(msg).encode('utf-8')
        header = f'Content-Length: {len(body)}\r\n\r\n'.encode('ascii')
        self._out.write(header + body)
        self._out.flush()

    def send_response(self, request_seq: int, command: str, success: bool, body: dict = None, message: str = None):
        resp = {
            'seq': self._next_seq(),
            'type': 'response',
            'request_seq': request_seq,
            'success': success,
            'command': command,
        }
        if body is not None:
            resp['body'] = body
        if message is not None:
            resp['message'] = message
        self.send_message(resp)

    def send_event(self, event: str, body: dict = None):
        ev = {
            'seq': self._next_seq(),
            'type': 'event',
            'event': event,
        }
        if body is not None:
            ev['body'] = body
        self.send_message(ev)

class TriadDebugger:

    def __init__(self):
        self._breakpoints: dict[str, set[int]] = {}
        self._step_mode: bool = False
        self._current_frame = None
        self._filename: str = ''
        self._source_lines: list[str] = []
        self._hit_count: dict[str, dict[int, int]] = {}
        self._call_stack: list[tuple[str, int, str]] = []
        self._dap: DAPProtocol | None = None
        self._dap_lock = threading.Lock()
        self._stopped = threading.Event()
        self._thread_id = 1
        self._non_interactive: bool = False
        self._sourcemap = None
        self._solver_frame: bool = False

    def set_breakpoints(self, lines: list[int], source_path: str = None):
        key = source_path or self._filename or '<triad>'
        self._breakpoints[key] = set(lines)

    def add_breakpoint(self, line: int, source_path: str = None):
        key = source_path or self._filename or '<triad>'
        self._breakpoints.setdefault(key, set()).add(line)

    def remove_breakpoint(self, line: int, source_path: str = None):
        key = source_path or self._filename or '<triad>'
        if key in self._breakpoints:
            self._breakpoints[key].discard(line)

    def clear_breakpoints(self, source_path: str = None):
        if source_path:
            self._breakpoints.pop(source_path, None)
        else:
            self._breakpoints.clear()

    def _solver_bp_hook(self, substrate_name: str, t: float, psi: np.ndarray, y: np.ndarray | None, metrics: dict):

        self._solver_frame = True
        self._current_frame = {
            '_substrate': substrate_name,
            '_t': t,
            '_psi_shape': psi.shape if psi is not None else None,
            '_psi_preview': psi[:4] if psi is not None else None,
            '_y_shape': y.shape if y is not None else None,
            **(metrics or {}),
        }
        self._call_stack.append((self._filename, 0, f'solver:{substrate_name}'))
        if self._dap:
            self._dap_stopped(0, reason='solver_break')
        elif not self._non_interactive:
            self._interact_solver(substrate_name, t, metrics)
        self._call_stack.pop()
        self._solver_frame = False

    def _interact_solver(self, substrate_name: str, t: float, metrics: dict):
        print(f'\n  \x1b[1;33m-- Solver break: {substrate_name} @ t={t:.3f} --\x1b[0m')
        if metrics:
            for k, v in metrics.items():
                print(f'  \x1b[2m{k} = {v:.4f}\x1b[0m')
        print("  Type 'c' to continue, 'q' to quit")
        while True:
            try:
                cmd = input('\x1b[1;36mtriad-solver-dbg> \x1b[0m').strip()
            except (EOFError, KeyboardInterrupt):
                print('c')
                cmd = 'c'
            if cmd in ('c', 'continue'):
                break
            elif cmd in ('q', 'quit'):
                raise SystemExit(0)
            elif cmd in ('h', 'help'):
                print('  c, continue   Continue solver integration')
                print('  q, quit       Exit program')
            else:
                print(f'  unknown command: {cmd}')

    def set_solver_breakpoint(self, substrate_name: str, metric: str | None = None, threshold: float | None = None):

        key = f'__solver__:{substrate_name}'
        self._breakpoints.setdefault(key, set()).add((metric, threshold))

    def _bp_hook(self, line: int, locals_dict: dict[str, object]):
        key = self._filename
        self._hit_count.setdefault(key, {})
        self._hit_count[key][line] = self._hit_count[key].get(line, 0) + 1
        bp_lines = set()
        for k in (key, '<triad>'):
            if k in self._breakpoints:
                bp_lines.update(self._breakpoints[k])
        should_break = line in bp_lines or self._step_mode
        if should_break:
            self._solver_frame = False
            self._current_frame = locals_dict
            self._call_stack.append((self._filename, line, '<module>'))
            if self._dap:
                self._dap_stopped(line, reason='breakpoint' if line in bp_lines else 'step')
            elif not self._non_interactive:
                self._interact(line)
            self._call_stack.pop()

    def _safe_eval(self, expr: str, frame: dict) -> object:
        allowed_builtins = {
            'len': len, 'range': range, 'str': str, 'int': int, 'float': float,
            'bool': bool, 'type': type, 'abs': abs, 'min': min, 'max': max,
            'sum': sum, 'sorted': sorted, 'reversed': reversed, 'enumerate': enumerate,
            'zip': zip, 'map': map, 'filter': filter, 'round': round, 'divmod': divmod,
            'pow': pow, 'chr': chr, 'ord': ord, 'hash': hash, 'repr': repr,
            'isinstance': isinstance, 'issubclass': issubclass,
            'True': True, 'False': False, 'None': None,
        }
        for token in ('__import__', 'eval', 'exec', 'compile', 'open', 'globals', 'locals'):
            if token in expr:
                raise NameError(f'forbidden token in debugger expression: {token}')
        return eval(expr, {'__builtins__': allowed_builtins}, frame)

    def _interact(self, line: int):
        src_line = self._source_lines[line - 1] if 0 < line <= len(self._source_lines) else ''
        print(f'\n  \x1b[1;33m-- Break at line {line} --\x1b[0m')
        print(f'  \x1b[2m{line}: {src_line.strip()}\x1b[0m')
        print("  Type 'h' for commands, 'c' to continue, 'q' to quit")
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
            if action in ('c', 'continue'):
                self._step_mode = False
                break
            elif action in ('n', 'next', 's', 'step'):
                self._step_mode = True
                break
            elif action in ('q', 'quit'):
                raise SystemExit(0)
            elif action in ('l', 'list'):
                start = max(1, line - 3)
                end = min(len(self._source_lines), line + 3)
                for i in range(start, end + 1):
                    marker = '>>>' if i == line else '   '
                    print(f'  {marker} {i}: {self._source_lines[i - 1].rstrip()}')
            elif action in ('p', 'print'):
                if not arg:
                    print('  usage: p <expr>')
                    continue
                try:
                    result = self._safe_eval(arg, self._current_frame or {})
                    print(f'  {result}')
                except Exception as e:
                    print(f'  error: {e}')
            elif action in ('w', 'where', 'bt', 'backtrace'):
                for i, (fname, lnum, name) in enumerate(reversed(self._call_stack)):
                    print(f'  #{i} {name} at {fname}:{lnum}')
                if not self._call_stack:
                    print(f'  at {self._filename}:{line}')
            elif action in ('v', 'vars'):
                if self._current_frame:
                    for k, v in sorted(self._current_frame.items()):
                        if not k.startswith('_') and not callable(v):
                            vstr = repr(v)
                            if len(vstr) > 80:
                                vstr = vstr[:77] + '...'
                            print(f'  {k} = {vstr}')
                else:
                    print('  no frame available')
            elif action in ('b', 'break'):
                if arg:
                    try:
                        bl = int(arg)
                        self.add_breakpoint(bl)
                        print(f'  breakpoint set at line {bl}')
                    except ValueError:
                        print('  usage: b <line>')
                else:
                    for key in self._breakpoints:
                        for bp in sorted(self._breakpoints[key]):
                            hits = self._hit_count.get(key, {}).get(bp, 0)
                            print(f'  {key}:{bp} (hits: {hits})')
            elif action in ('d', 'delete'):
                if arg:
                    try:
                        self.remove_breakpoint(int(arg))
                        print(f'  breakpoint removed at line {arg}')
                    except ValueError:
                        print('  usage: d <line>')
            elif action in ('h', 'help'):
                print('  Commands:')
                print('    c, continue     Continue execution')
                print('    n, next         Step to next line')
                print('    s, step         Step into (same as next in TriadLang)')
                print('    l, list         Show source around current line')
                print('    p <expr>        Print expression value')
                print('    v, vars         Show local variables')
                print('    w, where        Show current location / backtrace')
                print('    bt, backtrace   Show call stack')
                print('    b [line]        Set/list breakpoints')
                print('    d <line>        Delete breakpoint')
                print('    q, quit         Exit program')
            else:
                try:
                    result = self._safe_eval(cmd, self._current_frame or {})
                    print(f'  {result}')
                except Exception as e:
                    print(f'  unknown command: {action} ({e})')

    def _dap_stopped(self, line: int, reason: str = 'breakpoint'):

        with self._dap_lock:
            self._dap.send_event('stopped', {
                'reason': reason,
                'threadId': self._thread_id,
                'frames': {
                    'totalFrames': len(self._call_stack),
                },
            })
        self._stopped.wait()
        self._stopped.clear()

    def _dap_continue(self):
        self._step_mode = False
        self._stopped.set()

    def _dap_step(self):
        self._step_mode = True
        self._stopped.set()

    def run_dap(self):

        self._dap = DAPProtocol()
        initialized = False
        launched = False

        while True:
            msg = self._dap.read_message()
            if msg is None:
                break

            msg_type = msg.get('type')
            command = msg.get('command', '')
            args = msg.get('arguments', {})
            seq = msg.get('seq', 0)

            if msg_type == 'request':
                if command == 'initialize':
                    self._dap.send_response(seq, 'initialize', True, {
                        'supportsConfigurationDoneRequest': True,
                        'supportsStepBack': False,
                        'supportsGotoTargetsRequest': False,
                        'supportsFunctionBreakpoints': False,
                        'supportsEvaluateForHovers': True,
                        'supportsSetVariable': True,
                        'supportsConditionalBreakpoints': False,
                        'supportsHitConditionalBreakpoints': False,
                        'supportsLogPoints': False,
                        'supportsThreads': True,
                    })
                    self._dap.send_event('initialized')
                    initialized = True

                elif command == 'launch' and initialized:
                    program = args.get('program', '')
                    if not program or not os.path.exists(program):
                        self._dap.send_response(seq, 'launch', False, message=f'file not found: {program}')
                        continue
                    self._dap.send_response(seq, 'launch', True)
                    launched = True

                    def _run():
                        try:
                            self.run_file(program)
                        except SystemExit:
                            pass
                        except Exception as exc:
                            import logging
                            logging.getLogger(__name__).warning('debugger launch failed: %s', exc)
                        finally:
                            self._dap.send_event('terminated')
                    t = threading.Thread(target=_run, daemon=True)
                    t.start()

                elif command == 'setBreakpoints' and initialized:
                    source = args.get('source', {})
                    path = source.get('path', self._filename)
                    bp_lines = [bp.get('line', 0) for bp in args.get('breakpoints', [])]
                    self._breakpoints[path] = set(bp_lines)
                    bp_response = []
                    for bl in bp_lines:
                        bp_response.append({
                            'id': bl,
                            'verified': True,
                            'line': bl,
                        })
                    self._dap.send_response(seq, 'setBreakpoints', True, {
                        'breakpoints': bp_response,
                    })

                elif command == 'configurationDone' and initialized:
                    self._dap.send_response(seq, 'configurationDone', True)

                elif command == 'threads' and launched:
                    self._dap.send_response(seq, 'threads', True, {
                        'threads': [{'id': self._thread_id, 'name': 'main'}],
                    })

                elif command == 'stackTrace' and launched:
                    frames = []
                    for i, (fname, lnum, name) in enumerate(reversed(self._call_stack)):
                        frames.append({
                            'id': i,
                            'name': name,
                            'source': {'path': fname, 'name': os.path.basename(fname)},
                            'line': lnum,
                            'column': 1,
                        })
                    if not frames:
                        frames.append({
                            'id': 0,
                            'name': '<module>',
                            'source': {'path': self._filename, 'name': os.path.basename(self._filename)},
                            'line': 1,
                            'column': 1,
                        })
                    self._dap.send_response(seq, 'stackTrace', True, {
                        'stackFrames': frames,
                        'totalFrames': len(frames),
                    })

                elif command == 'scopes' and launched:
                    self._dap.send_response(seq, 'scopes', True, {
                        'scopes': [{
                            'name': 'Locals',
                            'variablesReference': 1,
                            'expensive': False,
                        }],
                    })

                elif command == 'variables' and launched:
                    variables = []
                    if self._current_frame:
                        for k, v in sorted(self._current_frame.items()):
                            if not k.startswith('_') and not callable(v):
                                vtype = type(v).__name__
                                vstr = repr(v)
                                if len(vstr) > 200:
                                    vstr = vstr[:197] + '...'
                                variables.append({
                                    'name': k,
                                    'value': vstr,
                                    'type': vtype,
                                    'variablesReference': 0,
                                })
                    self._dap.send_response(seq, 'variables', True, {
                        'variables': variables,
                    })

                elif command == 'continue' and launched:
                    self._dap.send_response(seq, 'continue', True, {'allThreadsContinued': True})
                    self._dap_continue()

                elif command == 'next' and launched:
                    self._dap.send_response(seq, 'next', True)
                    self._dap_step()

                elif command == 'stepIn' and launched:
                    self._dap.send_response(seq, 'stepIn', True)
                    self._dap_step()

                elif command == 'stepOut' and launched:
                    self._dap.send_response(seq, 'stepOut', True)
                    self._dap_continue()

                elif command == 'evaluate' and launched:
                    expr = args.get('expression', '')
                    frame_id = args.get('frameId', 0)
                    try:
                        result = eval(expr, {}, self._current_frame or {})
                        self._dap.send_response(seq, 'evaluate', True, {
                            'result': repr(result),
                            'variablesReference': 0,
                        })
                    except Exception as e:
                        self._dap.send_response(seq, 'evaluate', False, message=str(e))

                elif command == 'setVariable' and launched:
                    var_name = args.get('name', '')
                    var_value_str = args.get('value', '')
                    try:
                        val = eval(var_value_str, {}, self._current_frame or {})
                        if self._current_frame is not None:
                            self._current_frame[var_name] = val
                        self._dap.send_response(seq, 'setVariable', True, {
                            'value': repr(val),
                        })
                    except Exception as e:
                        self._dap.send_response(seq, 'setVariable', False, message=str(e))

                elif command == 'disconnect':
                    self._dap.send_response(seq, 'disconnect', True)
                    break

                elif command == 'pause':
                    self._step_mode = True
                    self._dap.send_response(seq, 'pause', True)

                else:
                    self._dap.send_response(seq, command, False, message=f'unknown command: {command}')

    def inject_hooks(self, py_code: str, filename: str, smap=None,
                     tri_source: str = '') -> str:

        self._filename = filename
        self._source_lines = (tri_source or py_code).split('\n')
        lines = py_code.split('\n')
        if smap is None or not getattr(smap, '_entries', None):
            return py_code
        first_py: dict[int, int] = {}
        for pl in sorted(smap._entries):
            tl = smap._entries[pl].line
            if tl > 0 and tl not in first_py:
                first_py[tl] = pl
        _NO_HOOK_BEFORE = ('else', 'elif', 'except', 'finally', 'case ')
        for tl, pl in sorted(first_py.items(), key=lambda kv: -kv[1]):
            idx = pl - 1
            if idx < 0 or idx >= len(lines):
                continue
            line = lines[idx]
            stripped = line.lstrip()
            if not stripped or stripped.startswith(_NO_HOOK_BEFORE):
                continue
            indent = ' ' * (len(line) - len(stripped))
            lines.insert(idx, f'{indent}_triad_dbg._bp_hook({tl}, dict(locals()))')
        return '\n'.join(lines)

    def run_source(self, source: str, filename: str = '<triad>'):
        from frontend.parser_universal import parse
        from runtime.compiler_runtime import TriadCompiler
        mod = parse(source, filename)
        compiler = TriadCompiler()
        code = compiler.compile_to_source(mod)
        code = self.inject_hooks(code, filename, smap=compiler._sourcemap,
                                 tri_source=source)
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
    import argparse as _ap
    p = _ap.ArgumentParser(prog='triad debug')
    p.add_argument('file', help='.tri file to debug')
    p.add_argument('--break', '-b', dest='breaks', default='', help='comma-separated line numbers')
    p.add_argument('--solver', action='store_true', help='Phase 3.3: enable solver-native debug mode (break on convergence/metrics)')
    p.add_argument('--dap', action='store_true', help='run as DAP server (stdio transport)')
    parsed = p.parse_args(args)

    if parsed.dap:
        dbg = TriadDebugger()
        dbg.run_dap()
        return 0

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

    if parsed.solver:
        from runtime.core.multi_runtime import MultiRuntime

        _orig_run = MultiRuntime.run
        def _run_with_solver_dbg(self, verbose=False, **kw):
            from runtime.observers import ConvergenceObserver
            def _on_converge(history, last_metrics):

                sub = max(self.subs, key=lambda s: s.psi.real.max() if s.psi is not None else 0, default=None)
                if sub is not None:
                    dbg._solver_bp_hook(sub.name, self.global_t, sub.psi, sub.y, last_metrics)
            obs = ConvergenceObserver(on_converge=_on_converge)
            return _orig_run(self, verbose=verbose, **kw)
        MultiRuntime.run = _run_with_solver_dbg
    try:
        dbg.run_file(path)
    except SystemExit:
        pass
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning('debugger run failed: %s', exc)
    return 0
