from __future__ import annotations
import sys
from frontend.parser_universal import parse, ParseError
from frontend.lexer_universal import LexError
from runtime.interpreter import Interpreter, Environment, TriadError

def run_repl():
    print('TriadLang REPL v1.0')
    print('Type "exit" or Ctrl+D to quit.\n')
    interp = Interpreter()
    env = Environment(interp.globals)
    while True:
        try:
            line = input('>>> ')
        except (EOFError, KeyboardInterrupt):
            print()
            break
        line = line.strip()
        if not line:
            continue
        if line in ('exit', 'quit'):
            break
        if line.endswith('{'):
            depth = line.count('{') - line.count('}')
            while depth > 0:
                try:
                    cont = input('... ')
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                line += '\n' + cont
                depth += cont.count('{') - cont.count('}')
        try:
            mod = parse(line, '<repl>')
            result = interp.run_repl_line(mod, env)
            if result is not None:
                if isinstance(result, bool):
                    print('true' if result else 'false')
                elif result is None:
                    pass
                else:
                    print(result)
        except (LexError, ParseError) as e:
            print(f'  {e}', file=sys.stderr)
        except TriadError as e:
            print(f'  {e}', file=sys.stderr)
        except Exception as e:
            print(f'  error: {e}', file=sys.stderr)