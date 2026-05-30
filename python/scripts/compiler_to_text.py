import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.compiler import TriadCompiler

def _f(x: float) -> str:
    return repr(float(x))

def _tuple(xs) -> str:
    parts = [_f(v) for v in xs]
    if len(parts) == 1:
        return '(' + parts[0] + ',)'
    return '(' + ', '.join(parts) + ')'

def dump_program(name: str, prog, lines: list) -> None:
    lines.append(f'fixture {name}')
    lines.append(f'T {_f(prog.T)} dt {_f(prog.dt)} backend {prog.backend} n_subs {len(prog.substrates)} n_couplings {len(prog.couplings)}')
    if prog.metadata:
        meta_parts = []
        for k in sorted(prog.metadata):
            v = prog.metadata[k]
            meta_parts.append(f'{k}={v!r}' if isinstance(v, tuple) else f'{k}={v}')
        lines.append('meta ' + ' '.join(meta_parts))
    else:
        lines.append('meta')
    r = prog.readout
    fields = '(' + ','.join(r.fields) + ')'
    lines.append(f'readout what={r.what} fields={fields}')
    for i, s in enumerate(prog.substrates):
        lines.append(f'sub {i} name={s.name} D=1 N={s.N} L={_f(s.L)} Lambda={_f(s.Lambda)} alpha={_f(s.alpha)} sigma={_f(s.sigma)} Gamma={_f(s.Gamma)} f_FDT={_f(s.f_FDT)} omega={_f(s.omega)}')
        v_ext = 'None' if s.V_ext is None else s.V_ext
        lines.append(f'sub {i} nu={_tuple(s.nu)} lam={_tuple(s.lam)} V_ext={v_ext} regime={s.regime}')
    for j, c in enumerate(prog.couplings):
        lines.append(f'couple {j} src={c.src} dst={c.dst} kappa={_f(c.kappa)} mode={c.mode}')

def run() -> str:
    comp = TriadCompiler()
    lines: list[str] = []
    dump_program('classify_default', comp.compile('classify'), lines)
    dump_program('classify_5_classes_depth_6', comp.compile('classify', n_classes=5, n_features=64, depth=6), lines)
    dump_program('classify_2_classes', comp.compile('classify', n_classes=2, n_features=32, depth=3), lines)
    dump_program('generate_crystal', comp.compile('generate', pattern_type='crystal', N=128), lines)
    dump_program('generate_filament', comp.compile('generate', pattern_type='filament', N=64), lines)
    dump_program('generate_lattice', comp.compile('generate', pattern_type='lattice', N=64), lines)
    dump_program('remember_default', comp.compile('remember'), lines)
    dump_program('remember_4ts', comp.compile('remember', timescales=(0.5, 2.0, 8.0, 32.0), N=64), lines)
    dump_program('couple_ring', comp.compile('couple', n_substrates=4, topology='ring'), lines)
    dump_program('couple_full', comp.compile('couple', n_substrates=3, topology='full'), lines)
    dump_program('couple_star', comp.compile('couple', n_substrates=4, topology='star'), lines)
    dump_program('sat_default', comp.compile('sat'), lines)
    dump_program('sat_5vars', comp.compile('sat', n_vars=5, n_clauses=20), lines)
    return '\n'.join(lines) + '\n'
if __name__ == '__main__':
    sys.stdout.write(run())