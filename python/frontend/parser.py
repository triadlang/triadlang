from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend.lexer import Token, tokenize

class ParseError(Exception):
    pass

@dataclass
class NumLit:
    value: float
    is_int: bool = True

@dataclass
class BoolLit:
    value: bool

@dataclass
class StrLit:
    value: str

@dataclass
class IdentRef:
    name: str
Expr = Union[NumLit, BoolLit, StrLit, IdentRef]

@dataclass
class RegDecl:
    name: str
    bit_width: int = 1
    initial: Optional[Expr] = None
    line: int = 0
    regime_name: Optional[str] = None
    regime_overrides: Optional[dict] = None

@dataclass
class Op:
    opcode: str
    args: list[Expr]
    line: int = 0

@dataclass
class LoopBlock:
    target: Expr
    body: 'Program'
    line: int = 0

@dataclass
class SegmentBlock:
    segment_id: int
    body: 'Program'
    line: int = 0

@dataclass
class IfBlock:
    cond: IdentRef
    then_body: 'Program'
    else_body: Optional['Program'] = None
    line: int = 0

@dataclass
class OutStmt:
    args: list[Expr]
    line: int = 0

@dataclass
class HaltStmt:
    line: int = 0

@dataclass
class Annotation:
    key: str
    raw_args: str
    line: int = 0

@dataclass
class EvolveStmt:
    target: 'IdentRef'
    duration: float
    line: int = 0

@dataclass
class CoupleStmt:
    src: 'IdentRef'
    dst: 'IdentRef'
    kappa: float
    duration: float
    line: int = 0

@dataclass
class PairStmt:
    a: 'IdentRef'
    b: 'IdentRef'
    kappa: float
    duration: float
    line: int = 0

@dataclass
class RingStmt:
    members: list
    kappa: float
    duration: float
    line: int = 0

@dataclass
class SequenceStmt:
    target: 'IdentRef'
    inputs: list
    each_for: float
    line: int = 0

@dataclass
class ObserveStmt:
    target: 'IdentRef'
    metrics: list
    over_seeds: int = 1
    stream_to: str = ''
    line: int = 0

@dataclass
class AssertStmt:
    predicate: str
    target: 'IdentRef'
    line: int = 0

@dataclass
class CheckpointStmt:
    target: 'IdentRef'
    path: str
    line: int = 0

@dataclass
class SubstrateDecl:
    name: str
    composed_of: list
    properties: dict
    line: int = 0
Stmt = Union[RegDecl, Op, LoopBlock, SegmentBlock, IfBlock, OutStmt, HaltStmt, Annotation, EvolveStmt, CoupleStmt, PairStmt, RingStmt, SequenceStmt, ObserveStmt, AssertStmt, SubstrateDecl, CheckpointStmt]

@dataclass
class Program:
    body: list[Stmt] = field(default_factory=list)

class Parser:

    def __init__(self, tokens: list[Token]):
        self.toks = tokens
        self.i = 0

    def _peek(self, off=0):
        return self.toks[self.i + off]

    def _advance(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def _check(self, kind, value=None):
        t = self._peek()
        if t.kind != kind:
            return False
        if value is not None and t.value != value:
            return False
        return True

    def _expect_kind(self, kind):
        t = self._peek()
        if t.kind != kind:
            raise ParseError(f'expected {kind} but got {t.kind} {t.value!r} at L{t.line}:C{t.col}')
        return self._advance()

    def _expect_sym(self, sym):
        t = self._peek()
        if t.kind != 'SYMBOL' or t.value != sym:
            raise ParseError(f'expected {sym!r} but got {t.kind} {t.value!r} at L{t.line}:C{t.col}')
        return self._advance()

    def _expect_kw(self, kw):
        t = self._peek()
        if t.kind != 'KEYWORD' or t.value != kw:
            raise ParseError(f'expected keyword {kw!r} but got {t.kind} {t.value!r} at L{t.line}:C{t.col}')
        return self._advance()

    def parse(self) -> Program:
        prog = Program()
        while not self._check('EOF'):
            stmt = self._parse_stmt()
            if stmt is not None:
                prog.body.append(stmt)
        return prog

    def _parse_stmt(self) -> Optional[Stmt]:
        t = self._peek()
        if t.kind == 'ANNOT':
            return self._parse_annotation()
        if t.kind == 'KEYWORD':
            if t.value == 'reg':
                return self._parse_reg_decl()
            if t.value == 'loop':
                return self._parse_loop_block()
            if t.value == 'segment':
                return self._parse_segment_block()
            if t.value == 'if':
                return self._parse_if_block()
            if t.value == 'OUT':
                return self._parse_out_stmt()
            if t.value == 'HALT':
                self._advance()
                self._expect_sym(';')
                return HaltStmt(line=t.line)
            if t.value == 'evolve':
                return self._parse_evolve_stmt()
            if t.value == 'couple':
                return self._parse_couple_stmt()
            if t.value == 'pair':
                return self._parse_pair_stmt()
            if t.value == 'ring':
                return self._parse_ring_stmt()
            if t.value == 'sequence':
                return self._parse_sequence_stmt()
            if t.value == 'OBSERVE':
                return self._parse_observe_stmt()
            if t.value == 'assert':
                return self._parse_assert_stmt()
            if t.value == 'substrate':
                return self._parse_substrate_decl()
        if t.kind == 'OPCODE':
            if t.value == 'CHECKPOINT':
                return self._parse_checkpoint_stmt()
            return self._parse_op_stmt()
        raise ParseError(f'unexpected token {t.kind} {t.value!r} at L{t.line}:C{t.col}')

    def _parse_annotation(self) -> Annotation:
        t = self._advance()
        val = t.value
        lp = val.find('(')
        if lp == -1:
            key, args = (val, '')
        else:
            key = val[:lp]
            args = val[lp + 1:].rstrip(')')
        return Annotation(key=key, raw_args=args, line=t.line)

    def _parse_reg_decl(self) -> RegDecl:
        t_kw = self._expect_kw('reg')
        name_tok = self._expect_kind('IDENT')
        bit_width = 1
        regime_name: Optional[str] = None
        regime_overrides: Optional[dict] = None
        if self._check('SYMBOL', '['):
            self._advance()
            n_tok = self._expect_kind('NUMBER')
            bit_width = int(float(n_tok.value))
            self._expect_sym(']')
        if self._check('SYMBOL', ':'):
            self._advance()
            regime_tok = self._expect_kind('IDENT')
            regime_name = regime_tok.value
        if self._check('SYMBOL', '{'):
            self._advance()
            regime_overrides = {}
            while not self._check('SYMBOL', '}'):
                key_tok = self._peek()
                if key_tok.kind not in ('IDENT', 'KEYWORD'):
                    raise ParseError(f'expected property name, got {key_tok.kind} {key_tok.value!r} at L{key_tok.line}:C{key_tok.col}')
                self._advance()
                self._expect_sym(':')
                val = self._parse_atom_or_tuple()
                self._expect_sym(';')
                regime_overrides[key_tok.value] = val
            self._expect_sym('}')
        initial: Optional[Expr] = None
        if self._check('SYMBOL', '='):
            self._advance()
            initial = self._parse_atom_expr()
        self._expect_sym(';')
        return RegDecl(name=name_tok.value, bit_width=bit_width, initial=initial, line=t_kw.line, regime_name=regime_name, regime_overrides=regime_overrides)

    def _parse_atom_or_tuple(self):
        if self._check('SYMBOL', '('):
            self._advance()
            items = []
            if not self._check('SYMBOL', ')'):
                items.append(self._parse_atom_expr())
                while self._check('SYMBOL', ','):
                    self._advance()
                    items.append(self._parse_atom_expr())
            self._expect_sym(')')
            return tuple((it.value if isinstance(it, NumLit) else it for it in items))
        atom = self._parse_atom_expr()
        if isinstance(atom, NumLit):
            return atom.value
        if isinstance(atom, BoolLit):
            return atom.value
        if isinstance(atom, StrLit):
            return atom.value
        if isinstance(atom, IdentRef):
            return atom.name
        return atom

    def _parse_op_stmt(self) -> Op:
        opcode_tok = self._advance()
        args: list[Expr] = []
        if not self._check('SYMBOL', ';'):
            args.append(self._parse_atom_expr())
            while self._check('SYMBOL', ','):
                self._advance()
                args.append(self._parse_atom_expr())
        self._expect_sym(';')
        return Op(opcode=opcode_tok.value, args=args, line=opcode_tok.line)

    def _parse_loop_block(self) -> LoopBlock:
        t_kw = self._expect_kw('loop')
        target_tok = self._peek()
        if target_tok.kind == 'NUMBER':
            target_atom = self._parse_atom_expr()
        elif target_tok.kind == 'IDENT':
            target_atom = self._parse_atom_expr()
        else:
            raise ParseError(f'loop target must be IDENT or NUMBER, got {target_tok.kind} {target_tok.value!r} at L{target_tok.line}:C{target_tok.col}')
        self._expect_sym('{')
        body = self._parse_block_body()
        self._expect_sym('}')
        return LoopBlock(target=target_atom, body=body, line=t_kw.line)

    def _parse_segment_block(self) -> SegmentBlock:
        t_kw = self._expect_kw('segment')
        self._expect_sym('(')
        next_tok = self._peek()
        if next_tok.kind == 'IDENT' and next_tok.value == 't':
            self._advance()
            self._expect_sym('=')
            dur_tok = self._expect_kind('NUMBER')
            duration = float(dur_tok.value)
            seg_id = -1
        else:
            n_tok = self._expect_kind('NUMBER')
            seg_id = int(float(n_tok.value))
            duration = -1.0
        self._expect_sym(')')
        self._expect_sym('{')
        body = self._parse_block_body()
        self._expect_sym('}')
        blk = SegmentBlock(segment_id=seg_id, body=body, line=t_kw.line)
        blk._duration = duration
        return blk

    def _parse_if_block(self) -> IfBlock:
        t_kw = self._expect_kw('if')
        cond_tok = self._expect_kind('IDENT')
        cond = IdentRef(name=cond_tok.value)
        self._expect_sym('{')
        then_body = self._parse_block_body()
        self._expect_sym('}')
        else_body = None
        if self._check('KEYWORD', 'else'):
            self._advance()
            self._expect_sym('{')
            else_body = self._parse_block_body()
            self._expect_sym('}')
        return IfBlock(cond=cond, then_body=then_body, else_body=else_body, line=t_kw.line)

    def _parse_out_stmt(self) -> OutStmt:
        t_kw = self._expect_kw('OUT')
        args = [self._parse_atom_expr()]
        while self._check('SYMBOL', ','):
            self._advance()
            args.append(self._parse_atom_expr())
        self._expect_sym(';')
        return OutStmt(args=args, line=t_kw.line)

    def _parse_block_body(self) -> Program:
        body = Program()
        while not self._check('SYMBOL', '}') and (not self._check('EOF')):
            stmt = self._parse_stmt()
            if stmt is not None:
                body.body.append(stmt)
        return body

    def _parse_evolve_stmt(self) -> EvolveStmt:
        t_kw = self._expect_kw('evolve')
        name_tok = self._expect_kind('IDENT')
        self._expect_kw('for')
        dur_tok = self._expect_kind('NUMBER')
        self._expect_sym(';')
        return EvolveStmt(target=IdentRef(name=name_tok.value), duration=float(dur_tok.value), line=t_kw.line)

    def _consume_kappa_for_duration(self) -> tuple[float, float]:
        self._expect_kw('kappa')
        self._expect_sym('=')
        k_tok = self._advance()
        if k_tok.kind != 'NUMBER':
            raise ParseError(f'kappa expects NUMBER, got {k_tok}')
        self._expect_kw('for')
        nxt = self._peek()
        if nxt.kind == 'IDENT' and nxt.value == 'T':
            self._advance()
            self._expect_sym('=')
        dur_tok = self._expect_kind('NUMBER')
        return (float(k_tok.value), float(dur_tok.value))

    def _parse_couple_stmt(self) -> CoupleStmt:
        t_kw = self._expect_kw('couple')
        src_tok = self._expect_kind('IDENT')
        nxt = self._peek()
        if nxt.kind == 'SYMBOL' and nxt.value == '->':
            self._advance()
        else:
            self._expect_sym('-')
            self._expect_sym('>')
        dst_tok = self._expect_kind('IDENT')
        kappa, dur = self._consume_kappa_for_duration()
        self._expect_sym(';')
        return CoupleStmt(src=IdentRef(name=src_tok.value), dst=IdentRef(name=dst_tok.value), kappa=kappa, duration=dur, line=t_kw.line)

    def _parse_pair_stmt(self) -> PairStmt:
        t_kw = self._expect_kw('pair')
        self._expect_sym('(')
        a = self._expect_kind('IDENT').value
        self._expect_sym(',')
        b = self._expect_kind('IDENT').value
        self._expect_sym(')')
        kappa, dur = self._consume_kappa_for_duration()
        self._expect_sym(';')
        return PairStmt(a=IdentRef(name=a), b=IdentRef(name=b), kappa=kappa, duration=dur, line=t_kw.line)

    def _parse_ring_stmt(self) -> RingStmt:
        t_kw = self._expect_kw('ring')
        self._expect_sym('(')
        members = [self._expect_kind('IDENT').value]
        while self._check('SYMBOL', ','):
            self._advance()
            members.append(self._expect_kind('IDENT').value)
        self._expect_sym(')')
        kappa, dur = self._consume_kappa_for_duration()
        self._expect_sym(';')
        return RingStmt(members=[IdentRef(name=m) for m in members], kappa=kappa, duration=dur, line=t_kw.line)

    def _parse_sequence_stmt(self) -> SequenceStmt:
        t_kw = self._expect_kw('sequence')
        target = self._expect_kind('IDENT').value
        self._expect_kw('via')
        self._expect_sym('(')
        inputs = [self._expect_kind('IDENT').value]
        while self._check('SYMBOL', ','):
            self._advance()
            inputs.append(self._expect_kind('IDENT').value)
        self._expect_sym(')')
        self._expect_kw('each_for')
        self._expect_sym('=')
        dur_tok = self._expect_kind('NUMBER')
        self._expect_sym(';')
        return SequenceStmt(target=IdentRef(name=target), inputs=[IdentRef(name=i) for i in inputs], each_for=float(dur_tok.value), line=t_kw.line)

    def _parse_observe_stmt(self) -> ObserveStmt:
        t_kw = self._expect_kw('OBSERVE')
        target = self._expect_kind('IDENT').value
        metrics = [self._expect_kind('IDENT').value]
        while self._check('SYMBOL', ','):
            self._advance()
            nxt = self._peek()
            if nxt.kind == 'KEYWORD' and nxt.value == 'over_seeds':
                break
            metrics.append(self._expect_kind('IDENT').value)
        over_seeds = 1
        stream_to = ''
        while True:
            nxt = self._peek()
            if nxt.kind == 'KEYWORD' and nxt.value == 'over_seeds':
                self._advance()
                self._expect_sym('=')
                n_tok = self._expect_kind('NUMBER')
                over_seeds = int(float(n_tok.value))
                continue
            if nxt.kind == 'IDENT' and nxt.value == 'stream_to':
                self._advance()
                path_tok = self._expect_kind('STRING')
                stream_to = path_tok.value
                continue
            break
        self._expect_sym(';')
        return ObserveStmt(target=IdentRef(name=target), metrics=metrics, over_seeds=over_seeds, stream_to=stream_to, line=t_kw.line)

    def _parse_assert_stmt(self) -> AssertStmt:
        t_kw = self._expect_kw('assert')
        pred_tok = self._advance()
        if pred_tok.kind != 'KEYWORD':
            raise ParseError(f'assert expects predicate KEYWORD, got {pred_tok}')
        self._expect_sym('(')
        target = self._expect_kind('IDENT').value
        self._expect_sym(')')
        self._expect_sym(';')
        return AssertStmt(predicate=pred_tok.value, target=IdentRef(name=target), line=t_kw.line)

    def _parse_checkpoint_stmt(self) -> CheckpointStmt:
        t_kw = self._advance()
        name_tok = self._expect_kind('IDENT')
        nxt = self._peek()
        if not (nxt.kind == 'KEYWORD' and nxt.value == 'to'):
            raise ParseError(f"CHECKPOINT expects 'to', got {nxt.kind} {nxt.value!r} at L{nxt.line}:C{nxt.col}")
        self._advance()
        path_tok = self._expect_kind('STRING')
        self._expect_sym(';')
        return CheckpointStmt(target=IdentRef(name=name_tok.value), path=path_tok.value, line=t_kw.line)

    def _parse_substrate_decl(self) -> SubstrateDecl:
        t_kw = self._expect_kw('substrate')
        name_tok = self._expect_kind('IDENT')
        self._expect_kw('composed_of')
        self._expect_sym('(')
        members = [self._expect_kind('IDENT').value]
        while self._check('SYMBOL', ','):
            self._advance()
            members.append(self._expect_kind('IDENT').value)
        self._expect_sym(')')
        props: dict = {}
        if self._check('SYMBOL', '{'):
            self._advance()
            while not self._check('SYMBOL', '}'):
                key_tok = self._peek()
                if key_tok.kind not in ('IDENT', 'KEYWORD'):
                    raise ParseError(f'expected property name, got {key_tok.kind} {key_tok.value!r} at L{key_tok.line}:C{key_tok.col}')
                self._advance()
                key_name = key_tok.value
                self._expect_sym(':')
                nxt = self._peek()
                if nxt.kind == 'KEYWORD':
                    self._advance()
                    props[key_name] = nxt.value
                else:
                    val_atom = self._parse_atom_expr()
                    if isinstance(val_atom, NumLit):
                        props[key_name] = val_atom.value
                    elif isinstance(val_atom, StrLit):
                        props[key_name] = val_atom.value
                    elif isinstance(val_atom, IdentRef):
                        props[key_name] = val_atom.name
                    else:
                        props[key_name] = val_atom
                self._expect_sym(';')
            self._expect_sym('}')
        self._expect_sym(';')
        return SubstrateDecl(name=name_tok.value, composed_of=[IdentRef(name=m) for m in members], properties=props, line=t_kw.line)

    def _parse_atom_expr(self) -> Expr:
        t = self._peek()
        if t.kind == 'NUMBER':
            self._advance()
            v = t.value
            try:
                if '.' in v or 'e' in v or 'E' in v:
                    return NumLit(value=float(v), is_int=False)
                return NumLit(value=float(v), is_int=True)
            except ValueError:
                raise ParseError(f'bad number {v!r} at L{t.line}:C{t.col}')
        if t.kind == 'KEYWORD' and t.value in ('true', 'false'):
            self._advance()
            return BoolLit(value=t.value == 'true')
        if t.kind == 'STRING':
            self._advance()
            return StrLit(value=t.value)
        if t.kind == 'IDENT':
            self._advance()
            if self._check('SYMBOL', '('):
                self._advance()
                args = []
                if not self._check('SYMBOL', ')'):
                    args.append(self._parse_atom_expr())
                    while self._check('SYMBOL', ','):
                        self._advance()
                        args.append(self._parse_atom_expr())
                self._expect_sym(')')
                if t.value == 'from_file' and len(args) == 1 and isinstance(args[0], StrLit):
                    return StrLit(value='__from_file__:' + args[0].value)
                if t.value == 'from_checkpoint' and len(args) == 1 and isinstance(args[0], StrLit):
                    return StrLit(value='__from_checkpoint__:' + args[0].value)
                return IdentRef(name=t.value)
            return IdentRef(name=t.value)
        raise ParseError(f'expected atom, got {t.kind} {t.value!r} at L{t.line}:C{t.col}')

def parse(source: str) -> Program:
    return Parser(tokenize(source)).parse()
if __name__ == '__main__':
    import sys, pprint
    src = open(sys.argv[1]).read()
    pprint.pprint(parse(src), depth=8)