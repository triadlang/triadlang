from __future__ import annotations

import ctypes

from frontend.ast_nodes import (
    AssignStmt,
    BinOp,
    BoolLit,
    BreakStmt,
    CallExpr,
    ContinueStmt,
    FloatLit,
    ForStmt,
    Ident,
    IfStmt,
    IndexExpr,
    IntLit,
    LetStmt,
    Param,
    ReturnStmt,
    UnaryOp,
    WhileStmt,
)

_ALLOWED_BIN = {'+', '-', '*', '/', '//', '%',
                '==', '!=', '<', '<=', '>', '>=', 'and', 'or'}
_ALLOWED_CALLS = {'abs': 1, 'min': 2, 'max': 2,
                  'sqrt': 1, 'sin': 1, 'cos': 1, 'exp': 1, 'log': 1}
_MATH_CALLS = {'sqrt', 'sin', 'cos', 'exp', 'log'}

class _Reject(Exception):
    pass

def _expr_spec(e):
    if isinstance(e, IntLit):
        return {'k': 'int', 'v': int(e.value)}
    if isinstance(e, FloatLit):
        return {'k': 'float', 'v': float(e.value)}
    if isinstance(e, BoolLit):
        return {'k': 'int', 'v': 1 if e.value else 0}
    if isinstance(e, Ident):
        return {'k': 'name', 'id': e.name}
    if isinstance(e, UnaryOp):
        if e.op == '-':
            return {'k': 'un', 'op': '-', 'e': _expr_spec(e.operand)}
        if e.op == 'not':
            return {'k': 'un', 'op': 'not', 'e': _expr_spec(e.operand)}
        raise _Reject()
    if isinstance(e, BinOp):
        if e.op not in _ALLOWED_BIN:
            raise _Reject()
        return {'k': 'bin', 'op': e.op,
                'l': _expr_spec(e.left), 'r': _expr_spec(e.right)}
    if isinstance(e, IndexExpr):
        if not isinstance(e.obj, Ident):
            raise _Reject()
        return {'k': 'index', 'base': e.obj.name, 'i': _expr_spec(e.index)}
    if isinstance(e, CallExpr):
        if not isinstance(e.func, Ident):
            raise _Reject()
        fn = e.func.name
        if fn == 'len':
            if e.kwargs or len(e.args) != 1 or not isinstance(e.args[0], Ident):
                raise _Reject()
            return {'k': 'len', 'base': e.args[0].name}
        if fn not in _ALLOWED_CALLS or e.kwargs:
            raise _Reject()
        if len(e.args) != _ALLOWED_CALLS[fn]:
            raise _Reject()
        return {'k': 'call', 'fn': fn, 'args': [_expr_spec(a) for a in e.args]}
    raise _Reject()

def _range_spec(it):
    if not (isinstance(it, CallExpr) and isinstance(it.func, Ident)
            and it.func.name == 'range' and not it.kwargs
            and 1 <= len(it.args) <= 3):
        raise _Reject()
    return [_expr_spec(a) for a in it.args]

def _stmt_spec(s):
    if isinstance(s, LetStmt):
        if s.value is None:
            raise _Reject()
        return {'k': 'assign', 't': s.name, 'e': _expr_spec(s.value)}
    if isinstance(s, AssignStmt):
        if isinstance(s.target, IndexExpr) and isinstance(s.target.obj, Ident):
            return {'k': 'setindex', 't': s.target.obj.name,
                    'i': _expr_spec(s.target.index), 'e': _expr_spec(s.value)}
        if not isinstance(s.target, Ident):
            raise _Reject()
        return {'k': 'assign', 't': s.target.name, 'e': _expr_spec(s.value)}
    if isinstance(s, IfStmt):
        return {'k': 'if', 'c': _expr_spec(s.condition),
                'then': [_stmt_spec(x) for x in s.then_body],
                'elifs': [[_expr_spec(c), [_stmt_spec(x) for x in b]]
                          for c, b in (s.elif_clauses or [])],
                'else': ([_stmt_spec(x) for x in s.else_body]
                         if s.else_body else None)}
    if isinstance(s, ForStmt):
        if isinstance(s.iter, Ident):
            return {'k': 'for_list', 'var': s.var, 'list': s.iter.name,
                    'body': [_stmt_spec(x) for x in s.body]}
        return {'k': 'for', 'var': s.var, 'range': _range_spec(s.iter),
                'body': [_stmt_spec(x) for x in s.body]}
    if isinstance(s, WhileStmt):
        return {'k': 'while', 'c': _expr_spec(s.condition),
                'body': [_stmt_spec(x) for x in s.body]}
    if isinstance(s, BreakStmt):
        return {'k': 'break'}
    if isinstance(s, ContinueStmt):
        return {'k': 'continue'}
    if isinstance(s, ReturnStmt):
        return {'k': 'return',
                'e': _expr_spec(s.value) if s.value is not None else None}
    raise _Reject()

def _walk_vars(spec, declared, reads, writes, lists, lwrites):
    k = spec['k']
    if k == 'assign':
        _expr_vars(spec['e'], declared, reads, lists)
        writes.add(spec['t'])
        declared.add(spec['t'])
    elif k == 'setindex':
        _expr_vars(spec['i'], declared, reads, lists)
        _expr_vars(spec['e'], declared, reads, lists)
        lists.add(spec['t'])
        lwrites.add(spec['t'])
    elif k == 'return':
        if spec['e'] is not None:
            _expr_vars(spec['e'], declared, reads, lists)
    elif k == 'if':
        _expr_vars(spec['c'], declared, reads, lists)
        for branch in ([spec['then']] + [b for _, b in spec['elifs']]
                       + ([spec['else']] if spec['else'] else [])):
            d2 = set(declared)
            for st in branch:
                _walk_vars(st, d2, reads, writes, lists, lwrites)
    elif k == 'for':
        for r in spec['range']:
            _expr_vars(r, declared, reads, lists)
        d2 = set(declared); d2.add(spec['var'])
        writes.add(spec['var'])
        for st in spec['body']:
            _walk_vars(st, d2, reads, writes, lists, lwrites)
    elif k == 'for_list':
        lists.add(spec['list'])
        d2 = set(declared); d2.add(spec['var'])
        writes.add(spec['var'])
        for st in spec['body']:
            _walk_vars(st, d2, reads, writes, lists, lwrites)
    elif k == 'while':
        _expr_vars(spec['c'], declared, reads, lists)
        d2 = set(declared)
        for st in spec['body']:
            _walk_vars(st, d2, reads, writes, lists, lwrites)
    elif k == 'fn':
        d2 = set(declared)
        for st in spec['body']:
            _walk_vars(st, d2, reads, writes, lists, lwrites)

def _expr_vars(e, declared, reads, lists):
    k = e['k']
    if k == 'name':
        if e['id'] not in declared:
            reads.add(e['id'])
    elif k == 'index':
        lists.add(e['base'])
        _expr_vars(e['i'], declared, reads, lists)
    elif k == 'len':
        lists.add(e['base'])
    elif k == 'bin':
        _expr_vars(e['l'], declared, reads, lists)
        _expr_vars(e['r'], declared, reads, lists)
    elif k == 'un':
        _expr_vars(e['e'], declared, reads, lists)
    elif k == 'call':
        for a in e['args']:
            _expr_vars(a, declared, reads, lists)

def _finish_spec(spec, params=()):
    reads: set = set()
    writes: set = set()
    lists: set = set()
    lwrites: set = set()
    declared = set(params)
    _walk_vars(spec, declared, reads, writes, lists, lwrites)
    if lists & (reads | writes):
        return None
    spec['reads'] = sorted(reads - lists)
    spec['writes'] = sorted(writes)
    spec['lists'] = sorted(lists)
    spec['lwrites'] = sorted(lwrites)
    return spec

def build_spec(s: ForStmt) -> dict | None:
    try:
        spec = _stmt_spec(s)
    except _Reject:
        return None
    if spec['k'] not in ('for', 'for_list'):
        return None
    if _contains_return(spec):
        return None
    return _finish_spec(spec)

def build_spec_while(s: WhileStmt) -> dict | None:
    try:
        spec = {'k': 'while', 'c': _expr_spec(s.condition),
                'body': [_stmt_spec(x) for x in s.body]}
    except _Reject:
        return None
    if _contains_return(spec):
        return None
    return _finish_spec(spec)

def build_spec_fn(s) -> dict | None:
    try:
        params = []
        for p in s.params:
            name = p.name if isinstance(p, Param) else p
            params.append(name)
        if getattr(s, 'is_async', False) or s.decorators:
            return None
        body = [_stmt_spec(x) for x in s.body]
    except _Reject:
        return None
    spec = {'k': 'fn', 'name': s.name, 'params': params, 'body': body}
    out = _finish_spec(spec, params=params)
    if out is None:
        return None

    plists = [p for p in params if p in out['lists']]
    pscalars = [p for p in params if p not in out['lists']]
    out['reads'] = sorted(set(out['reads']) | set(pscalars))
    out['param_lists'] = plists
    return out

def _contains_return(spec) -> bool:
    k = spec['k']
    if k == 'return':
        return True
    for key in ('body', 'then'):
        for st in spec.get(key) or []:
            if _contains_return(st):
                return True
    for _, b in spec.get('elifs') or []:
        for st in b:
            if _contains_return(st):
                return True
    for st in spec.get('else') or []:
        if _contains_return(st):
            return True
    return False

def _infer_types(spec, entry_types: dict, ltypes: dict | None = None) -> dict:
    types = dict(entry_types)
    ltypes = ltypes or {}
    for w in spec['writes']:
        types.setdefault(w, 'i')
    if spec['k'] == 'for':
        types.setdefault(spec['var'], 'i')
    elif spec['k'] == 'for_list':
        types.setdefault(spec['var'], ltypes.get(spec['list'], 'd'))

    def ety(e) -> str:
        k = e['k']
        if k == 'int':
            return 'i'
        if k == 'float':
            return 'd'
        if k == 'name':
            return types.get(e['id'], 'i')
        if k == 'index':
            return ltypes.get(e['base'], 'd')
        if k == 'len':
            return 'i'
        if k == 'un':
            return 'i' if e['op'] == 'not' else ety(e['e'])
        if k == 'call':
            if e['fn'] in _MATH_CALLS:
                return 'd'
            return 'd' if any(ety(a) == 'd' for a in e['args']) else 'i'
        if k == 'bin':
            op = e['op']
            if op == '/':
                return 'd'
            if op in ('==', '!=', '<', '<=', '>', '>=', 'and', 'or'):
                return 'i'
            return 'd' if (ety(e['l']) == 'd' or ety(e['r']) == 'd') else 'i'
        raise _Reject()

    def walk(st):
        k = st['k']
        if k == 'assign':
            t = ety(st['e'])
            if t == 'd' and types.get(st['t']) == 'i':
                types[st['t']] = 'd'
                return True
            types.setdefault(st['t'], t)
            return False
        if k in ('setindex', 'return', 'break', 'continue'):
            return False
        changed = False
        if k == 'if':
            for b in ([st['then']] + [bb for _, bb in st['elifs']]
                      + ([st['else']] if st['else'] else [])):
                for x in b:
                    changed |= walk(x)
        elif k == 'for_list':
            if st['var'] not in types:
                types[st['var']] = ltypes.get(st['list'], 'd')
                changed = True
            for x in st['body']:
                changed |= walk(x)
        elif k in ('for', 'while'):
            if k == 'for' and st['var'] not in types:
                types[st['var']] = 'i'
                changed = True
            for x in st['body']:
                changed |= walk(x)
        return changed

    for _ in range(8):
        if not any(walk(x) for x in spec['body']):
            break
    return types

_C_PRELUDE = r'''
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

static int _bail = 0;
#define BAIL_OVF  do { _bail = 1; } while (0)
#define BAIL_ZDIV do { _bail = 2; } while (0)

static int64_t add_i(int64_t a, int64_t b){ int64_t r; if(__builtin_add_overflow(a,b,&r)) BAIL_OVF; return r; }
static int64_t sub_i(int64_t a, int64_t b){ int64_t r; if(__builtin_sub_overflow(a,b,&r)) BAIL_OVF; return r; }
static int64_t mul_i(int64_t a, int64_t b){ int64_t r; if(__builtin_mul_overflow(a,b,&r)) BAIL_OVF; return r; }
static int64_t fdiv_i(int64_t a, int64_t b){
    if(b==0){ BAIL_ZDIV; return 0; }
    int64_t q = a/b, r = a%b;
    return (r != 0 && ((r < 0) != (b < 0))) ? q-1 : q;
}
static int64_t mod_i(int64_t a, int64_t b){
    if(b==0){ BAIL_ZDIV; return 0; }
    int64_t r = a%b;
    return (r != 0 && ((r < 0) != (b < 0))) ? r+b : r;
}
static double div_d(double a, double b){ if(b==0.0){ BAIL_ZDIV; return 0.0; } return a/b; }
static double fdiv_d(double a, double b){ if(b==0.0){ BAIL_ZDIV; return 0.0; } return floor(a/b); }
static double mod_d(double a, double b){
    if(b==0.0){ BAIL_ZDIV; return 0.0; }
    double r = fmod(a,b);
    return (r != 0.0 && ((r < 0.0) != (b < 0.0))) ? r+b : r;
}
static int64_t abs_i(int64_t a){ if(a==INT64_MIN) BAIL_OVF; return a<0?-a:a; }
static int64_t idx_n(int64_t i, int64_t n){
    if(i < 0) i += n;
    if(i < 0 || i >= n){ _bail = 4; return 0; }
    return i;
}
'''

class _CEmit:
    def __init__(self, types: dict, ltypes: dict | None = None):
        self.types = types
        self.ltypes = ltypes or {}
        self.lines: list[str] = []
        self.ind = 1

    def w(self, line: str):
        self.lines.append('    ' * self.ind + line)

    def ety(self, e) -> str:
        k = e['k']
        if k == 'int':
            return 'i'
        if k == 'float':
            return 'd'
        if k == 'name':
            return self.types[e['id']]
        if k == 'index':
            return self.ltypes[e['base']]
        if k == 'len':
            return 'i'
        if k == 'un':
            return 'i' if e['op'] == 'not' else self.ety(e['e'])
        if k == 'call':
            if e['fn'] in _MATH_CALLS:
                return 'd'
            return 'd' if any(self.ety(a) == 'd' for a in e['args']) else 'i'
        op = e['op']
        if op == '/':
            return 'd'
        if op in ('==', '!=', '<', '<=', '>', '>=', 'and', 'or'):
            return 'i'
        return 'd' if (self.ety(e['l']) == 'd' or self.ety(e['r']) == 'd') else 'i'

    def ex(self, e, want: str | None = None) -> str:
        k = e['k']
        if k == 'int':
            c = f'INT64_C({e["v"]})'
            t = 'i'
        elif k == 'float':
            c = repr(float(e['v']))
            t = 'd'
        elif k == 'name':
            c = f'v_{e["id"]}'
            t = self.types[e['id']]
        elif k == 'index':
            b = e['base']
            c = f'L_{b}[idx_n({self.ex(e["i"], "i")}, n_{b})]'
            t = self.ltypes[b]
        elif k == 'len':
            c = f'n_{e["base"]}'
            t = 'i'
        elif k == 'un':
            if e['op'] == 'not':
                c = f'(!({self.ex(e["e"])}))'
                t = 'i'
            else:
                inner_t = self.ety(e['e'])
                if inner_t == 'i':
                    c = f'sub_i(0, {self.ex(e["e"])})'
                else:
                    c = f'(-({self.ex(e["e"])}))'
                t = inner_t
        elif k == 'call':
            fn = e['fn']
            if fn in _MATH_CALLS:
                c = f'{fn}((double)({self.ex(e["args"][0], "d")}))'
                t = 'd'
            elif fn == 'abs':
                at = self.ety(e['args'][0])
                c = (f'abs_i({self.ex(e["args"][0])})' if at == 'i'
                     else f'fabs({self.ex(e["args"][0])})')
                t = at
            else:
                a, b = e['args']
                t = 'd' if (self.ety(a) == 'd' or self.ety(b) == 'd') else 'i'
                ca, cb = self.ex(a, t), self.ex(b, t)
                op = '<' if fn == 'min' else '>'
                c = f'(({ca}) {op} ({cb}) ? ({ca}) : ({cb}))'
        else:
            op = e['op']
            if op in ('and', 'or'):
                cop = '&&' if op == 'and' else '||'
                c = f'(({self.ex(e["l"])}) {cop} ({self.ex(e["r"])}))'
                t = 'i'
            elif op in ('==', '!=', '<', '<=', '>', '>='):
                ct = 'd' if (self.ety(e['l']) == 'd' or self.ety(e['r']) == 'd') else 'i'
                c = f'(({self.ex(e["l"], ct)}) {op} ({self.ex(e["r"], ct)}))'
                t = 'i'
            else:
                lt, rt = self.ety(e['l']), self.ety(e['r'])
                ct = 'd' if (op == '/' or lt == 'd' or rt == 'd') else 'i'
                la, ra = self.ex(e['l'], ct), self.ex(e['r'], ct)
                if ct == 'i':
                    fn = {'+': 'add_i', '-': 'sub_i', '*': 'mul_i',
                          '//': 'fdiv_i', '%': 'mod_i'}[op]
                    c = f'{fn}({la}, {ra})'
                else:
                    fn = {'/': 'div_d', '//': 'fdiv_d', '%': 'mod_d'}.get(op)
                    c = f'{fn}({la}, {ra})' if fn else f'(({la}) {op} ({ra}))'
                t = ct
        if want == 'd' and t == 'i':
            return f'((double)({c}))'
        return c

    def stmt(self, st):
        k = st['k']
        if k == 'assign':
            self.w(f'v_{st["t"]} = {self.ex(st["e"], self.types[st["t"]])};')
            self.w('if (_bail) return _bail;')
        elif k == 'setindex':
            b = st['t']
            et = self.ltypes[b]
            self.w(f'L_{b}[idx_n({self.ex(st["i"], "i")}, n_{b})] = {self.ex(st["e"], et)};')
            self.w('if (_bail) return _bail;')
        elif k == 'for_list':
            b = st['list']
            v = f'v_{st["var"]}'
            uid = abs(id(st)) % 100000
            self.w(f'for (int64_t _k{uid} = 0; _k{uid} < n_{b}; _k{uid}++) {{')
            self.ind += 1
            self.w(f'{v} = L_{b}[_k{uid}];')
            if st.get('_top'):
                self.w('_ran = 1;')
            for x in st['body']:
                self.stmt(x)
            self.ind -= 1
            self.w('}')
        elif k == 'return':
            if st['e'] is None:
                self.w('R[2] = 0;')
            else:
                rt = self.ety(st['e'])
                if rt == 'i':
                    self.w(f'R[0] = {self.ex(st["e"], "i")};')
                    self.w('R[2] = 1;')
                else:
                    self.w(f'RD[0] = {self.ex(st["e"], "d")};')
                    self.w('R[2] = 2;')
            self.w('if (_bail) return _bail;')
            self.w('return 0;')
        elif k == 'if':
            self.w(f'if ({self.ex(st["c"])}) {{')
            self.ind += 1
            for x in st['then']:
                self.stmt(x)
            self.ind -= 1
            for c, b in st['elifs']:
                self.w(f'}} else if ({self.ex(c)}) {{')
                self.ind += 1
                for x in b:
                    self.stmt(x)
                self.ind -= 1
            if st['else']:
                self.w('} else {')
                self.ind += 1
                for x in st['else']:
                    self.stmt(x)
                self.ind -= 1
            self.w('}')
        elif k == 'for':
            v = f'v_{st["var"]}'
            r = st['range']
            if len(r) == 1:
                start, stop, step = 'INT64_C(0)', self.ex(r[0]), 'INT64_C(1)'
            elif len(r) == 2:
                start, stop, step = self.ex(r[0]), self.ex(r[1]), 'INT64_C(1)'
            else:
                start, stop, step = self.ex(r[0]), self.ex(r[1]), self.ex(r[2])
            uid = id(st)
            self.w(f'int64_t _start{uid} = {start}, _stop{uid} = {stop}, _step{uid} = {step};')
            self.w(f'if (_step{uid} == 0) return 3;')
            self.w(f'for (int64_t _it{uid} = _start{uid}; '
                   f'(_step{uid} > 0) ? (_it{uid} < _stop{uid}) : (_it{uid} > _stop{uid}); '
                   f'_it{uid} += _step{uid}) {{')
            self.ind += 1
            self.w(f'{v} = _it{uid};')
            if st is not None and st.get('_top'):
                self.w('_ran = 1;')
            for x in st['body']:
                self.stmt(x)
            self.ind -= 1
            self.w('}')
        elif k == 'while':
            self.w(f'while ({self.ex(st["c"])}) {{')
            self.ind += 1
            for x in st['body']:
                self.stmt(x)
            self.w('if (_bail) return _bail;')
            self.ind -= 1
            self.w('}')
        elif k == 'break':
            self.w('break;')
        elif k == 'continue':
            self.w('continue;')

def emit_c(spec: dict, types: dict, order: list, ltypes: dict | None = None) -> str:
    ltypes = ltypes or {}
    em = _CEmit(types, ltypes)
    is_fn = spec['k'] == 'fn'
    top = dict(spec)
    top['_top'] = True
    if is_fn:
        for st in spec['body']:
            em.stmt(st)
        em.w('R[2] = 0;')
    else:
        em.stmt(top)
    decls, loads, stores = [], [], []
    for idx, name in enumerate(order):
        t = types[name]
        ctype = 'int64_t' if t == 'i' else 'double'
        arr = 'I' if t == 'i' else 'D'
        decls.append(f'    {ctype} v_{name};')
        loads.append(f'    v_{name} = {arr}[{idx}];')
        if not is_fn:
            stores.append(f'    {arr}[{idx}] = v_{name};')
    lparams = ''
    for lname in spec.get('lists', []):
        ctype = 'int64_t' if ltypes[lname] == 'i' else 'double'
        lparams += f', {ctype} *L_{lname}, int64_t n_{lname}'
    fnparams = ', int64_t *R, double *RD' if is_fn else ''
    body = '\n'.join(em.lines)
    tail = ('    if (_bail) return _bail;\n'
            + '\n'.join(stores)
            + (f'\n    I[{len(order)}] = _ran;' if not is_fn else ''))
    return (_C_PRELUDE + f'''
int64_t kernel(int64_t *I, double *D{lparams}{fnparams}) {{
    _bail = 0;
    int64_t _ran = 0;
    (void)_ran;
{chr(10).join(decls)}
{chr(10).join(loads)}
{body}
{tail}
    return 0;
}}
''')

_INT_BOUND = 2 ** 63 - 1

_specialized: dict = {}

def _signature(spec: dict, env: dict):
    sig = []
    for name in spec['reads']:
        v = env.get(name, _MISSING)
        if v is _MISSING:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        if isinstance(v, int):
            if abs(v) > _INT_BOUND:
                return None
            sig.append((name, 'i'))
        else:
            sig.append((name, 'd'))
    for name in spec.get('lists', []):
        v = env.get(name, _MISSING)
        if not isinstance(v, list) or len(v) == 0:
            return None
        if all(isinstance(x, int) and not isinstance(x, bool) for x in v):
            if any(abs(x) > _INT_BOUND for x in v):
                return None
            sig.append((name, 'Li'))
        elif all(isinstance(x, float) for x in v):
            sig.append((name, 'Ld'))
        else:
            return None

    lw = set(spec.get('lwrites', []))
    if lw:
        ids = {}
        for name in spec.get('lists', []):
            ids.setdefault(id(env[name]), []).append(name)
        for names in ids.values():
            if len(names) > 1 and any(n in lw for n in names):
                return None
    return tuple(sig)

class _Missing:
    pass

_MISSING = _Missing()

def jit_run(loop_id: str, spec: dict, env: dict):
    sig = _signature(spec, env)
    if sig is None:
        return None

    key = (loop_id, sig)
    entry = _specialized.get(key)
    if entry is None:
        entry = _crystallize(loop_id, spec, sig)
        _specialized[key] = entry
    if entry is False:
        return None
    lib, order, types, ltypes = entry

    n = len(order)
    I = (ctypes.c_int64 * (n + 1))()
    D = (ctypes.c_double * (n + 1))()
    for idx, name in enumerate(order):
        v = env.get(name, 0)
        if types[name] == 'i':
            if isinstance(v, float):
                return None
            I[idx] = int(v)
        else:
            D[idx] = float(v)
    args = [I, D]
    lbufs = {}
    for lname in spec.get('lists', []):
        data = env[lname]
        if ltypes[lname] == 'i':
            buf = (ctypes.c_int64 * len(data))(*data)
        else:
            buf = (ctypes.c_double * len(data))(*data)
        lbufs[lname] = buf
        args.extend([buf, ctypes.c_int64(len(data))])
    rc = lib.kernel(*args)
    if rc != 0:
        return None

    for lname in spec.get('lwrites', []):
        data = env[lname]
        buf = lbufs[lname]
        caster = int if ltypes[lname] == 'i' else float
        for k in range(len(data)):
            data[k] = caster(buf[k])
    ran = bool(I[n])
    out = {}
    var = spec.get('var')
    for idx, name in enumerate(order):
        if name == var and not ran:
            continue
        if name in spec['writes'] or name == var:
            out[name] = int(I[idx]) if types[name] == 'i' else float(D[idx])
    return out

def _crystallize(loop_id: str, spec: dict, sig):
    try:
        entry_types = {n: t for n, t in sig if not t.startswith('L')}
        ltypes = {n: t[1] for n, t in sig if t.startswith('L')}
        types = _infer_types(spec, entry_types, ltypes)
        names = set(spec['reads']) | set(spec['writes'])
        if spec.get('var'):
            names.add(spec['var'])
        order = sorted(names)
        src = emit_c(spec, types, order, ltypes)
        from runtime.jit_tiered import get_jit
        lib = get_jit().compile_function(
            f'{loop_id}_{abs(hash(sig)) % 99991}', src)
        if lib is None:
            return False
        argt = [ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_double)]
        for lname in spec.get('lists', []):
            argt.append(ctypes.POINTER(ctypes.c_int64) if ltypes[lname] == 'i'
                        else ctypes.POINTER(ctypes.c_double))
            argt.append(ctypes.c_int64)
        if spec['k'] == 'fn':
            argt.extend([ctypes.POINTER(ctypes.c_int64),
                         ctypes.POINTER(ctypes.c_double)])
        lib.kernel.restype = ctypes.c_int64
        lib.kernel.argtypes = argt
        return (lib, order, types, ltypes)
    except (OSError, AttributeError, ValueError):
        return False

def jit_call_fn(fn_id: str, spec: dict, env: dict):
    sig = _signature(spec, env)
    if sig is None:
        return (False, None)
    key = (fn_id, sig)
    entry = _specialized.get(key)
    if entry is None:
        entry = _crystallize(fn_id, spec, sig)
        _specialized[key] = entry
    if entry is False:
        return (False, None)
    lib, order, types, ltypes = entry

    n = len(order)
    I = (ctypes.c_int64 * (n + 1))()
    D = (ctypes.c_double * (n + 1))()
    for idx, name in enumerate(order):
        v = env.get(name, 0)
        if types[name] == 'i':
            if isinstance(v, float):
                return (False, None)
            I[idx] = int(v)
        else:
            D[idx] = float(v)
    args = [I, D]
    lbufs = {}
    for lname in spec.get('lists', []):
        data = env[lname]
        if ltypes[lname] == 'i':
            buf = (ctypes.c_int64 * len(data))(*data)
        else:
            buf = (ctypes.c_double * len(data))(*data)
        lbufs[lname] = buf
        args.extend([buf, ctypes.c_int64(len(data))])
    R = (ctypes.c_int64 * 3)()
    RD = (ctypes.c_double * 1)()
    args.extend([R, RD])
    rc = lib.kernel(*args)
    if rc != 0:
        return (False, None)
    for lname in spec.get('lwrites', []):
        data = env[lname]
        buf = lbufs[lname]
        caster = int if ltypes[lname] == 'i' else float
        for k in range(len(data)):
            data[k] = caster(buf[k])
    kind = int(R[2])
    if kind == 0:
        return (True, None)
    if kind == 1:
        return (True, int(R[0]))
    return (True, float(RD[0]))

def jit_stats() -> dict:
    return {'specialized': len(_specialized)}

