from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


@dataclass
class Pos:
    line: int = 0
    col: int = 0
    file: str = ''

@dataclass
class IntLit:
    value: int
    pos: Pos = field(default_factory=Pos)

@dataclass
class FloatLit:
    value: float
    pos: Pos = field(default_factory=Pos)

@dataclass
class ComplexLit:
    value: complex
    pos: Pos = field(default_factory=Pos)

@dataclass
class BoolLit:
    value: bool
    pos: Pos = field(default_factory=Pos)

@dataclass
class StringLit:
    value: str
    pos: Pos = field(default_factory=Pos)

@dataclass
class BytesLit:
    value: str
    pos: Pos = field(default_factory=Pos)

@dataclass
class NoneLit:
    pos: Pos = field(default_factory=Pos)

@dataclass
class Ident:
    name: str
    pos: Pos = field(default_factory=Pos)

@dataclass
class BinOp:
    op: str
    left: Expr
    right: Expr
    pos: Pos = field(default_factory=Pos)

@dataclass
class UnaryOp:
    op: str
    operand: Expr
    pos: Pos = field(default_factory=Pos)

@dataclass
class CallExpr:
    func: Expr
    args: list[Expr]
    kwargs: dict[str, Expr] = field(default_factory=dict)
    pos: Pos = field(default_factory=Pos)

@dataclass
class IndexExpr:
    obj: Expr
    index: Expr
    pos: Pos = field(default_factory=Pos)

@dataclass
class SliceExpr:
    start: Expr | None = None
    end: Expr | None = None
    step: Expr | None = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class FieldExpr:
    obj: Expr
    field: str
    pos: Pos = field(default_factory=Pos)

@dataclass
class ListExpr:
    elements: list[Expr]
    pos: Pos = field(default_factory=Pos)

@dataclass
class ListCompExpr:
    expr: Expr
    var: str
    iter: Expr
    condition: Expr | None = None
    clauses: list[CompClause] | None = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class CompClause:
    var: str
    iter: Expr
    conditions: list[Expr] = field(default_factory=list)

@dataclass
class MapExpr:
    pairs: list[tuple[Expr, Expr]]
    pos: Pos = field(default_factory=Pos)

@dataclass
class TupleExpr:
    elements: list[Expr]
    pos: Pos = field(default_factory=Pos)

@dataclass
class FStringExpr:
    parts: list
    pos: Pos = field(default_factory=Pos)

@dataclass
class LambdaExpr:
    params: list[Param]
    body: list[Stmt]
    pos: Pos = field(default_factory=Pos)

@dataclass
class MethodCallExpr:
    obj: Expr
    method: str
    args: list[Expr]
    kwargs: dict[str, Expr] = field(default_factory=dict)
    pos: Pos = field(default_factory=Pos)

@dataclass
class AssignExpr:
    target: Expr
    value: Expr
    pos: Pos = field(default_factory=Pos)

@dataclass
class Param:
    name: str
    type_ann: str | None = None
    default: Expr | None = None
    is_args: bool = False
    is_kwargs: bool = False
    pos: Pos = field(default_factory=Pos)

@dataclass
class LetStmt:
    name: str
    type_ann: str | GenericType | UnionType | OptionalType | None = None
    value: Expr | None = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class DestructLetStmt:
    names: list[str]
    value: Expr
    star_idx: int = -1
    pos: Pos = field(default_factory=Pos)

@dataclass
class MapDestructStmt:
    names: list[str]
    value: Expr
    pos: Pos = field(default_factory=Pos)

@dataclass
class ConstStmt:
    name: str
    value: Expr
    pos: Pos = field(default_factory=Pos)

@dataclass
class AssignStmt:
    target: Expr
    value: Expr
    pos: Pos = field(default_factory=Pos)

@dataclass
class ExprStmt:
    expr: Expr
    pos: Pos = field(default_factory=Pos)

@dataclass
class ReturnStmt:
    value: Expr | None = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class BreakStmt:
    pos: Pos = field(default_factory=Pos)

@dataclass
class ContinueStmt:
    pos: Pos = field(default_factory=Pos)

@dataclass
class IfStmt:
    condition: Expr
    then_body: list[Stmt]
    elif_clauses: list[tuple[Expr, list[Stmt]]] = field(default_factory=list)
    else_body: list[Stmt] | None = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class ForStmt:
    var: str
    iter: Expr
    body: list[Stmt]
    pos: Pos = field(default_factory=Pos)

@dataclass
class WhileStmt:
    condition: Expr
    body: list[Stmt]
    pos: Pos = field(default_factory=Pos)

@dataclass
class FnDecl:
    name: str
    params: list[Param]
    return_type: str | None = None
    body: list[Stmt] = field(default_factory=list)
    is_async: bool = False
    decorators: list[Expr] | None = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class TryCatchStmt:
    body: list[Stmt] = field(default_factory=list)
    catch_var: str | None = None
    catch_body: list[Stmt] = field(default_factory=list)
    finally_body: list[Stmt] = field(default_factory=list)
    catches: list[CatchBlock] | None = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class CatchBlock:
    exceptions: list[str] = field(default_factory=list)
    var: str | None = None
    body: list[Stmt] = field(default_factory=list)

@dataclass
class ThrowStmt:
    value: Expr | None = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class WithStmt:
    expr: Expr | None = None
    var: str | None = None
    body: list[Stmt] = field(default_factory=list)
    pos: Pos = field(default_factory=Pos)

@dataclass
class TypeField:
    name: str
    type_ann: str
    default: Expr | None = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class TypeDecl:
    name: str
    fields: list[TypeField] = field(default_factory=list)
    methods: list[FnDecl] = field(default_factory=list)
    pos: Pos = field(default_factory=Pos)

@dataclass
class ImportStmt:
    path: list[str]
    alias: str | None = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class FromImportStmt:
    path: list[str]
    names: list[str]
    aliases: list[str | None] | None = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class RegStmt:
    name: str
    regime: str | None = None
    value: Expr | None = None
    overrides: dict | None = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class EntityDecl:
    name: str
    base: str | None = None
    fields: dict[str, Expr] = field(default_factory=dict)
    methods: list[FnDecl] = field(default_factory=list)
    pos: Pos = field(default_factory=Pos)

@dataclass
class WorldDecl:
    name: str
    fields: dict[str, Expr] = field(default_factory=dict)
    entities: list[EntityDecl] = field(default_factory=list)
    body: list[Stmt] = field(default_factory=list)
    pos: Pos = field(default_factory=Pos)

@dataclass
class CoupleStmt:
    src: str
    dst: str
    kappa: Expr | None = None
    duration: Expr | None = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class PairStmt:
    a: str
    b: str
    kappa: Expr | None = None
    duration: Expr | None = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class RingStmt:
    members: list[str] = field(default_factory=list)
    kappa: Expr | None = None
    duration: Expr | None = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class ObserveStmt:
    target: str
    metrics: list[str] = field(default_factory=list)
    over_seeds: int = 1
    pos: Pos = field(default_factory=Pos)

@dataclass
class RunStmt:
    duration: Expr | None = None
    target: str | None = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class SubstrateDecl:
    name: str
    regime: str | None = None
    overrides: dict | None = None
    members: list[str] = field(default_factory=list)
    properties: dict = field(default_factory=dict)
    pos: Pos = field(default_factory=Pos)

@dataclass
class SequenceStmt:
    inputs: Expr
    target: str
    each_for: Expr | None = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class AnnotationStmt:
    key: str
    args: str = ''
    pos: Pos = field(default_factory=Pos)

@dataclass
class ClassDecl:
    name: str
    parent: str | None = None
    parents: list[str] | None = None
    fields: list[TypeField] = field(default_factory=list)
    methods: list[FnDecl] = field(default_factory=list)
    pos: Pos = field(default_factory=Pos)

@dataclass
class MatchCase:
    pattern: Expr
    guard: Expr | None = None
    body: list[Stmt] = field(default_factory=list)

@dataclass
class MatchStmt:
    subject: Expr
    cases: list[MatchCase] = field(default_factory=list)
    else_body: list[Stmt] | None = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class YieldStmt:
    value: Expr | None = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class YieldExpr:
    value: Expr | None = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class AwaitExpr:
    value: Expr
    pos: Pos = field(default_factory=Pos)

@dataclass
class TernaryExpr:
    cond: Expr
    then_val: Expr
    else_val: Expr
    pos: Pos = field(default_factory=Pos)

@dataclass
class SuperExpr:
    args: list[Expr] = field(default_factory=list)
    pos: Pos = field(default_factory=Pos)

@dataclass
class ChainCmpExpr:
    operands: list[Expr]
    ops: list[str]
    pos: Pos = field(default_factory=Pos)

@dataclass
class SetExpr:
    elements: list[Expr]
    pos: Pos = field(default_factory=Pos)

@dataclass
class DictCompExpr:
    key: Expr
    value: Expr
    clauses: list[CompClause] = field(default_factory=list)
    pos: Pos = field(default_factory=Pos)

@dataclass
class SetCompExpr:
    expr: Expr
    clauses: list[CompClause] = field(default_factory=list)
    pos: Pos = field(default_factory=Pos)

@dataclass
class GenCompExpr:
    expr: Expr
    clauses: list[CompClause] = field(default_factory=list)
    pos: Pos = field(default_factory=Pos)

@dataclass
class AssertStmt:
    condition: Expr
    message: Expr | None = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class PassStmt:
    pos: Pos = field(default_factory=Pos)

@dataclass
class DelStmt:
    target: Expr
    pos: Pos = field(default_factory=Pos)

@dataclass
class AsyncForStmt:
    var: str
    iter: Expr
    body: list[Stmt] = field(default_factory=list)
    pos: Pos = field(default_factory=Pos)

@dataclass
class AsyncWithStmt:
    expr: Expr | None = None
    var: str | None = None
    body: list[Stmt] = field(default_factory=list)
    pos: Pos = field(default_factory=Pos)

@dataclass
class GenericType:
    name: str
    args: list[TypeExpr] = field(default_factory=list)
    pos: Pos = field(default_factory=Pos)

@dataclass
class UnionType:
    types: list[TypeExpr] = field(default_factory=list)
    pos: Pos = field(default_factory=Pos)

@dataclass
class OptionalType:
    inner: TypeExpr | None = None
    pos: Pos = field(default_factory=Pos)

TypeExpr = Union[str, GenericType, UnionType, OptionalType]

@dataclass
class ListPattern:
    elements: list[Pattern]
    pos: Pos = field(default_factory=Pos)

@dataclass
class DictPattern:
    pairs: list[tuple[str, Pattern]]
    rest: bool = False
    pos: Pos = field(default_factory=Pos)

@dataclass
class ClassPattern:
    cls_name: str
    fields: list[tuple[str, Pattern]]
    pos: Pos = field(default_factory=Pos)

@dataclass
class OrPattern:
    patterns: list[Pattern]
    pos: Pos = field(default_factory=Pos)

@dataclass
class NullishCoalesceExpr:
    left: Expr
    right: Expr
    pos: Pos = field(default_factory=Pos)

@dataclass
class OptChainExpr:
    obj: Expr
    attr: str | None = None
    method: str | None = None
    args: list[Expr] | None = None
    kwargs: dict[str, Expr] | None = None
    index: Expr | None = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class SpreadExpr:
    value: Expr
    pos: Pos = field(default_factory=Pos)

@dataclass
class PipelineExpr:
    left: Expr
    right: Expr
    pos: Pos = field(default_factory=Pos)

@dataclass
class RangeExpr:
    start: Expr
    end: Expr | None = None
    inclusive: bool = False
    pos: Pos = field(default_factory=Pos)

@dataclass
class RegexLit:
    pattern: str
    flags: str = ''
    pos: Pos = field(default_factory=Pos)

@dataclass
class ElvisExpr:
    cond: Expr
    else_val: Expr
    pos: Pos = field(default_factory=Pos)

@dataclass
class CompoundAssignExpr:
    op: str
    target: Expr
    value: Expr
    pos: Pos = field(default_factory=Pos)

Pattern = Union[Ident, IntLit, FloatLit, BoolLit, StringLit, NoneLit, ListPattern, DictPattern, ClassPattern, OrPattern]

Stmt = Union[LetStmt, DestructLetStmt, MapDestructStmt, ConstStmt, AssignStmt, ExprStmt, ReturnStmt, BreakStmt, ContinueStmt, IfStmt, ForStmt, WhileStmt, FnDecl, TypeDecl, ClassDecl, ImportStmt, FromImportStmt, TryCatchStmt, ThrowStmt, MatchStmt, YieldStmt, RegStmt, EntityDecl, WorldDecl, CoupleStmt, PairStmt, RingStmt, ObserveStmt, RunStmt, SubstrateDecl, SequenceStmt, AnnotationStmt, WithStmt, AssertStmt, PassStmt, DelStmt, AsyncForStmt, AsyncWithStmt]
Expr = Union[IntLit, FloatLit, ComplexLit, BoolLit, StringLit, BytesLit, NoneLit, Ident, BinOp, UnaryOp, CallExpr, IndexExpr, SliceExpr, FieldExpr, ListExpr, ListCompExpr, MapExpr, TupleExpr, FStringExpr, LambdaExpr, MethodCallExpr, AssignExpr, YieldExpr, AwaitExpr, TernaryExpr, SuperExpr, ChainCmpExpr, SetExpr, DictCompExpr, SetCompExpr, GenCompExpr, NullishCoalesceExpr, OptChainExpr, SpreadExpr, PipelineExpr, RangeExpr, RegexLit, ElvisExpr, CompoundAssignExpr]

@dataclass
class Module:
    name: str = ''
    body: list[Stmt] = field(default_factory=list)
    file: str = ''

