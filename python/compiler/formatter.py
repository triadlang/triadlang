from __future__ import annotations
from frontend.parser import Program, RegDecl, Op, LoopBlock, SegmentBlock, IfBlock, OutStmt, HaltStmt, Annotation, EvolveStmt, CoupleStmt as DCoupleStmt, PairStmt as DPairStmt, RingStmt as DRingStmt, SequenceStmt, ObserveStmt as DObserveStmt, AssertStmt, SubstrateDecl, CheckpointStmt, IdentRef, NumLit, BoolLit, StrLit

def _val(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, tuple):
        return '(' + ', '.join((_val(x) for x in v)) + ')'
    return str(v)

def _expr(e):
    if e is None:
        return ''
    if isinstance(e, NumLit):
        return _val(int(e.value) if e.is_int else e.value)
    if isinstance(e, BoolLit):
        return 'true' if e.value else 'false'
    if isinstance(e, StrLit):
        if e.value.startswith('__from_file__:'):
            return f'''from_file("{e.value[len('__from_file__:'):]}")'''
        if e.value.startswith('__from_checkpoint__:'):
            return f'''from_checkpoint("{e.value[len('__from_checkpoint__:'):]}")'''
        return f'"{e.value}"'
    if isinstance(e, IdentRef):
        return e.name
    return str(e)

def _fmt_stmt(stmt, indent: int=0) -> str:
    pad = '    ' * indent
    if isinstance(stmt, Annotation):
        return f'{pad}@{stmt.key}({stmt.raw_args})'
    if isinstance(stmt, RegDecl):
        parts = [f'{pad}reg {stmt.name}']
        if stmt.bit_width != 1:
            parts.append(f'[{stmt.bit_width}]')
        if stmt.regime_name:
            parts.append(f' : {stmt.regime_name}')
        if stmt.regime_overrides:
            inner = '; '.join((f'{k}: {_val(v)}' for k, v in stmt.regime_overrides.items()))
            parts.append(f' {{ {inner}; }}')
        if stmt.initial is not None:
            parts.append(f' = {_expr(stmt.initial)}')
        parts.append(';')
        return ''.join(parts)
    if isinstance(stmt, EvolveStmt):
        return f'{pad}evolve {stmt.target.name} for {stmt.duration};'
    if isinstance(stmt, DCoupleStmt):
        return f'{pad}couple {stmt.src.name} -> {stmt.dst.name} kappa={stmt.kappa} for T={stmt.duration};'
    if isinstance(stmt, DPairStmt):
        return f'{pad}pair({stmt.a.name}, {stmt.b.name}) kappa={stmt.kappa} for T={stmt.duration};'
    if isinstance(stmt, DRingStmt):
        members = ', '.join((m.name for m in stmt.members))
        return f'{pad}ring({members}) kappa={stmt.kappa} for T={stmt.duration};'
    if isinstance(stmt, SequenceStmt):
        inputs = ', '.join((i.name for i in stmt.inputs))
        return f'{pad}sequence {stmt.target.name} via ({inputs}) each_for={stmt.each_for};'
    if isinstance(stmt, DObserveStmt):
        s = f"{pad}OBSERVE {stmt.target.name} {', '.join(stmt.metrics)}"
        if stmt.over_seeds > 1:
            s += f', over_seeds={stmt.over_seeds}'
        if getattr(stmt, 'stream_to', ''):
            s += f' stream_to "{stmt.stream_to}"'
        return s + ';'
    if isinstance(stmt, AssertStmt):
        return f'{pad}assert {stmt.predicate}({stmt.target.name});'
    if isinstance(stmt, CheckpointStmt):
        return f'{pad}CHECKPOINT {stmt.target.name} to "{stmt.path}";'
    if isinstance(stmt, SubstrateDecl):
        members = ', '.join((m.name for m in stmt.composed_of))
        out = [f'{pad}substrate {stmt.name} composed_of ({members})']
        if stmt.properties:
            out.append(' {')
            for k, v in stmt.properties.items():
                out.append(f'    {pad}{k}: {_val(v)};')
            out.append(f'{pad}}}')
        out.append(';')
        return '\n'.join(out)
    if isinstance(stmt, LoopBlock):
        target = _expr(stmt.target)
        body = '\n'.join((_fmt_stmt(s, indent + 1) for s in stmt.body.body))
        return f'{pad}loop {target} {{\n{body}\n{pad}}}'
    if isinstance(stmt, SegmentBlock):
        body = '\n'.join((_fmt_stmt(s, indent + 1) for s in stmt.body.body))
        head = f"segment(t={getattr(stmt, '_duration', stmt.segment_id)})" if getattr(stmt, '_duration', -1) > 0 else f'segment({stmt.segment_id})'
        return f'{pad}{head} {{\n{body}\n{pad}}}'
    if isinstance(stmt, IfBlock):
        then_body = '\n'.join((_fmt_stmt(s, indent + 1) for s in stmt.then_body.body))
        out = [f'{pad}if {stmt.cond.name} {{', then_body, f'{pad}}}']
        if stmt.else_body:
            else_body = '\n'.join((_fmt_stmt(s, indent + 1) for s in stmt.else_body.body))
            out += [f'{pad}else {{', else_body, f'{pad}}}']
        return '\n'.join(out)
    if isinstance(stmt, Op):
        args = ', '.join((_expr(a) for a in stmt.args))
        return f'{pad}{stmt.opcode} {args};'
    if isinstance(stmt, OutStmt):
        return f"{pad}OUT {', '.join((_expr(a) for a in stmt.args))};"
    if isinstance(stmt, HaltStmt):
        return f'{pad}HALT;'
    return f'{pad}{stmt!r}'

def format_program(prog: Program) -> str:
    return '\n'.join((_fmt_stmt(s) for s in prog.body)) + '\n'
from frontend.ast_nodes import *

def _fmt_u_expr(e, indent: int=0) -> str:
    pad = '    ' * indent
    if e is None:
        return 'none'
    if isinstance(e, IntLit):
        return str(e.value)
    if isinstance(e, FloatLit):
        return repr(e.value)
    if isinstance(e, BoolLit):
        return 'true' if e.value else 'false'
    if isinstance(e, StringLit):
        return repr(e.value)
    if isinstance(e, NoneLit):
        return 'none'
    if isinstance(e, Ident):
        return e.name
    if isinstance(e, BinOp):
        l = _fmt_u_expr(e.left)
        r = _fmt_u_expr(e.right)
        if e.op in ('and', 'or'):
            return f'{l} {e.op} {r}'
        return f'{l} {e.op} {r}'
    if isinstance(e, UnaryOp):
        val = _fmt_u_expr(e.operand)
        if e.op == 'not':
            return f'not {val}'
        return f'{e.op}{val}'
    if isinstance(e, CallExpr):
        func = _fmt_u_expr(e.func)
        args = ', '.join((_fmt_u_expr(a) for a in e.args))
        kwargs = ', '.join((f'{k}={_fmt_u_expr(v)}' for k, v in e.kwargs.items()))
        all_args = ', '.join(filter(None, [args, kwargs]))
        return f'{func}({all_args})'
    if isinstance(e, MethodCallExpr):
        obj = _fmt_u_expr(e.obj)
        method = 'push' if e.method == 'append' else e.method
        args = ', '.join((_fmt_u_expr(a) for a in e.args))
        kwargs = ', '.join((f'{k}={_fmt_u_expr(v)}' for k, v in e.kwargs.items()))
        all_args = ', '.join(filter(None, [args, kwargs]))
        return f'{obj}.{method}({all_args})'
    if isinstance(e, IndexExpr):
        obj = _fmt_u_expr(e.obj)
        if isinstance(e.index, SliceExpr):
            s = e.index
            start = _fmt_u_expr(s.start) if s.start else ''
            end = _fmt_u_expr(s.end) if s.end else ''
            step = f':{_fmt_u_expr(s.step)}' if s.step else ''
            return f'{obj}[{start}:{end}{step}]'
        return f'{obj}[{_fmt_u_expr(e.index)}]'
    if isinstance(e, FieldExpr):
        return f'{_fmt_u_expr(e.obj)}.{e.field}'
    if isinstance(e, ListExpr):
        elems = ', '.join((_fmt_u_expr(el) for el in e.elements))
        return f'[{elems}]'
    if isinstance(e, TupleExpr):
        if not e.elements:
            return '()'
        elems = ', '.join((_fmt_u_expr(el) for el in e.elements))
        if len(e.elements) == 1:
            return f'({elems},)'
        return f'({elems})'
    if isinstance(e, ListCompExpr):
        expr = _fmt_u_expr(e.expr)
        var = e.var
        iter_expr = _fmt_u_expr(e.iter)
        if e.condition:
            cond = _fmt_u_expr(e.condition)
            return f'[{expr} for {var} in {iter_expr} if {cond}]'
        return f'[{expr} for {var} in {iter_expr}]'
    if isinstance(e, MapExpr):
        pairs = ', '.join((f'{_fmt_u_expr(k)}: {_fmt_u_expr(v)}' for k, v in e.pairs))
        return f'{{{pairs}}}'
    if isinstance(e, FStringExpr):
        parts = []
        for ptype, pval in e.parts:
            if ptype == 'str':
                parts.append(pval.replace('{', '{{').replace('}', '}}'))
            else:
                parts.append('{' + _fmt_u_expr(pval) + '}')
        return f'''f"{''.join(parts)}"'''
    if isinstance(e, LambdaExpr):
        params = ', '.join((_fmt_u_param(p) for p in e.params))
        if e.body:
            body = '; '.join((_fmt_u_stmt(s) for s in e.body))
            return f'fn({params}) {{ {body} }}'
        return f'fn({params}) {{ }}'
    if isinstance(e, AssignExpr):
        return f'({_fmt_u_expr(e.target)} := {_fmt_u_expr(e.value)})'
    if isinstance(e, YieldExpr):
        if e.value:
            return f'(yield {_fmt_u_expr(e.value)})'
        return '(yield)'
    if isinstance(e, AwaitExpr):
        return f'(await {_fmt_u_expr(e.value)})'
    return f'<expr:{type(e).__name__}>'

def _fmt_u_param(p) -> str:
    if isinstance(p, Param):
        prefix = '**' if p.is_kwargs else '*' if p.is_args else ''
        name = f'{prefix}{p.name}'
        if p.type_ann:
            name += f': {p.type_ann}'
        if p.default is not None:
            name += f'={_fmt_u_expr(p.default)}'
        return name
    return str(p)

def _fmt_u_stmt(s, indent: int=0) -> str:
    pad = '    ' * indent
    if isinstance(s, LetStmt):
        if s.value is not None:
            type_part = f': {s.type_ann}' if s.type_ann else ''
            return f'{pad}let {s.name}{type_part} = {_fmt_u_expr(s.value)};'
        return f'{pad}let {s.name};'
    if isinstance(s, ConstStmt):
        return f'{pad}const {s.name} = {_fmt_u_expr(s.value)};'
    if isinstance(s, DestructLetStmt):
        names = ', '.join(s.names)
        return f'{pad}let ({names}) = {_fmt_u_expr(s.value)};'
    if isinstance(s, MapDestructStmt):
        names = ', '.join(s.names)
        return f'{pad}let {{{names}}} = {_fmt_u_expr(s.value)};'
    if isinstance(s, AssignStmt):
        return f'{pad}{_fmt_u_expr(s.target)} = {_fmt_u_expr(s.value)};'
    if isinstance(s, ExprStmt):
        return f'{pad}{_fmt_u_expr(s.expr)};'
    if isinstance(s, ReturnStmt):
        if s.value:
            return f'{pad}return {_fmt_u_expr(s.value)};'
        return f'{pad}return;'
    if isinstance(s, BreakStmt):
        return f'{pad}break;'
    if isinstance(s, ContinueStmt):
        return f'{pad}continue;'
    if isinstance(s, IfStmt):
        lines = [f'{pad}if {_fmt_u_expr(s.condition)} {{']
        for st in s.then_body:
            lines.append(_fmt_u_stmt(st, indent + 1))
        for cond, body in s.elif_clauses:
            lines.append(f'{pad}}} elif {_fmt_u_expr(cond)} {{')
            for st in body:
                lines.append(_fmt_u_stmt(st, indent + 1))
        if s.else_body:
            lines.append(f'{pad}}} else {{')
            for st in s.else_body:
                lines.append(_fmt_u_stmt(st, indent + 1))
        lines.append(f'{pad}}}')
        return '\n'.join(lines)
    if isinstance(s, ForStmt):
        lines = [f'{pad}for {s.var} in {_fmt_u_expr(s.iter)} {{']
        for st in s.body:
            lines.append(_fmt_u_stmt(st, indent + 1))
        lines.append(f'{pad}}}')
        return '\n'.join(lines)
    if isinstance(s, WhileStmt):
        lines = [f'{pad}while {_fmt_u_expr(s.condition)} {{']
        for st in s.body:
            lines.append(_fmt_u_stmt(st, indent + 1))
        lines.append(f'{pad}}}')
        return '\n'.join(lines)
    if isinstance(s, FnDecl):
        params = ', '.join((_fmt_u_param(p) for p in s.params))
        prefix = 'async ' if s.is_async else ''
        lines = [f'{pad}{prefix}fn {s.name}({params}) {{']
        for st in s.body:
            lines.append(_fmt_u_stmt(st, indent + 1))
        lines.append(f'{pad}}}')
        return '\n'.join(lines)
    if isinstance(s, TypeDecl):
        lines = [f'{pad}type {s.name} {{']
        for f in s.fields:
            default = f' = {_fmt_u_expr(f.default)}' if f.default is not None else ''
            type_part = f': {f.type_ann}' if f.type_ann else ''
            lines.append(f'{pad}    {f.name}{type_part}{default}')
        for m in s.methods:
            params = ', '.join((_fmt_u_param(p) for p in m.params))
            lines.append(f'{pad}    fn {m.name}({params}) {{')
            for st in m.body:
                lines.append(_fmt_u_stmt(st, indent + 2))
            lines.append(f'{pad}    }}')
        lines.append(f'{pad}}}')
        return '\n'.join(lines)
    if isinstance(s, ImportStmt):
        path = '.'.join(s.path)
        alias = f' as {s.alias}' if s.alias else ''
        return f'{pad}import {path}{alias};'
    if isinstance(s, FromImportStmt):
        path = '.'.join(s.path)
        names = ', '.join(s.names)
        return f'{pad}from {path} import {names};'
    if isinstance(s, TryCatchStmt):
        lines = [f'{pad}try {{']
        for st in s.body:
            lines.append(_fmt_u_stmt(st, indent + 1))
        if s.catch_body:
            var = f' {s.catch_var}' if s.catch_var else ''
            lines.append(f'{pad}}} catch{var} {{')
            for st in s.catch_body:
                lines.append(_fmt_u_stmt(st, indent + 1))
        if s.finally_body:
            lines.append(f'{pad}}} finally {{')
            for st in s.finally_body:
                lines.append(_fmt_u_stmt(st, indent + 1))
        lines.append(f'{pad}}}')
        return '\n'.join(lines)
    if isinstance(s, ThrowStmt):
        if s.value:
            return f'{pad}throw {_fmt_u_expr(s.value)};'
        return f'{pad}throw;'
    if isinstance(s, YieldStmt):
        if s.value:
            return f'{pad}yield {_fmt_u_expr(s.value)};'
        return f'{pad}yield;'
    if isinstance(s, RegStmt):
        regime = f' : {s.regime}' if s.regime else ''
        val = f' = {_fmt_u_expr(s.value)}' if s.value else ''
        return f'{pad}reg {s.name}{regime}{val};'
    if isinstance(s, EntityDecl):
        fields = ', '.join((f'{k}={_fmt_u_expr(v)}' for k, v in s.fields.items()))
        base = f'({s.base})' if s.base else ''
        return f'{pad}entity {s.name}{base} {{ {fields} }}'
    if isinstance(s, WorldDecl):
        fields = ', '.join((f'{k}={_fmt_u_expr(v)}' for k, v in s.fields.items()))
        return f'{pad}world {s.name} {{ {fields} }}'
    if isinstance(s, CoupleStmt):
        kappa = f' kappa={_fmt_u_expr(s.kappa)}' if s.kappa else ''
        dur = f' for T={_fmt_u_expr(s.duration)}' if s.duration else ''
        return f'{pad}couple({s.src}, {s.dst}){kappa}{dur};'
    if isinstance(s, PairStmt):
        kappa = f' kappa={_fmt_u_expr(s.kappa)}' if s.kappa else ''
        dur = f' for T={_fmt_u_expr(s.duration)}' if s.duration else ''
        return f'{pad}pair({s.a}, {s.b}){kappa}{dur};'
    if isinstance(s, RingStmt):
        members = ', '.join(s.members)
        kappa = f' kappa={_fmt_u_expr(s.kappa)}' if s.kappa else ''
        dur = f' for T={_fmt_u_expr(s.duration)}' if s.duration else ''
        return f'{pad}ring({members}){kappa}{dur};'
    if isinstance(s, ObserveStmt):
        metrics = ', '.join(s.metrics)
        return f'{pad}OBSERVE {s.target} {metrics};'
    if isinstance(s, RunStmt):
        dur = f' {_fmt_u_expr(s.duration)}' if s.duration else ''
        return f'{pad}run{dur};'
    if isinstance(s, AnnotationStmt):
        return f'{pad}@{s.key}({s.args})'
    if isinstance(s, ClassDecl):
        parent = f' : {s.parent}' if s.parent else ''
        lines = [f'{pad}class {s.name}{parent} {{']
        for f in s.fields:
            default = f' = {_fmt_u_expr(f.default)}' if f.default is not None else ''
            type_part = f': {f.type_ann}' if f.type_ann else ''
            lines.append(f'{pad}    {f.name}{type_part}{default}')
        for m in s.methods:
            params = ', '.join((_fmt_u_param(p) for p in m.params))
            lines.append(f'{pad}    fn {m.name}({params}) {{')
            for st in m.body:
                lines.append(_fmt_u_stmt(st, indent + 2))
            lines.append(f'{pad}    }}')
        lines.append(f'{pad}}}')
        return '\n'.join(lines)
    if isinstance(s, MatchStmt):
        lines = [f'{pad}match {_fmt_u_expr(s.subject)} {{']
        for c in s.cases:
            guard = f' if {_fmt_u_expr(c.guard)}' if c.guard else ''
            lines.append(f'{pad}    case {_fmt_u_expr(c.pattern)}{guard} => {{')
            for st in c.body:
                lines.append(_fmt_u_stmt(st, indent + 2))
            lines.append(f'{pad}    }}')
        if s.else_body:
            lines.append(f'{pad}    else => {{')
            for st in s.else_body:
                lines.append(_fmt_u_stmt(st, indent + 2))
            lines.append(f'{pad}    }}')
        lines.append(f'{pad}}}')
        return '\n'.join(lines)
    return f'{pad}/* unhandled: {type(s).__name__} */'

def format_universal(mod: Module) -> str:
    lines = []
    for s in mod.body:
        lines.append(_fmt_u_stmt(s))
    return '\n'.join(lines) + '\n'