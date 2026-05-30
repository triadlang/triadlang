from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union

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
class BoolLit:
    value: bool
    pos: Pos = field(default_factory=Pos)

@dataclass
class StringLit:
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
    start: Optional[Expr] = None
    end: Optional[Expr] = None
    step: Optional[Expr] = None
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
    condition: Optional[Expr] = None
    pos: Pos = field(default_factory=Pos)

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
    type_ann: Optional[str] = None
    default: Optional[Expr] = None
    is_args: bool = False
    is_kwargs: bool = False
    pos: Pos = field(default_factory=Pos)

@dataclass
class LetStmt:
    name: str
    type_ann: Optional[str] = None
    value: Optional[Expr] = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class DestructLetStmt:
    names: list[str]
    value: Expr
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
    value: Optional[Expr] = None
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
    else_body: Optional[list[Stmt]] = None
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
    return_type: Optional[str] = None
    body: list[Stmt] = field(default_factory=list)
    is_async: bool = False
    pos: Pos = field(default_factory=Pos)

@dataclass
class TryCatchStmt:
    body: list['Stmt'] = field(default_factory=list)
    catch_var: Optional[str] = None
    catch_body: list['Stmt'] = field(default_factory=list)
    finally_body: list['Stmt'] = field(default_factory=list)
    pos: Pos = field(default_factory=Pos)

@dataclass
class ThrowStmt:
    value: Optional[Expr] = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class TypeField:
    name: str
    type_ann: str
    default: Optional[Expr] = None
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
    alias: Optional[str] = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class FromImportStmt:
    path: list[str]
    names: list[str]
    pos: Pos = field(default_factory=Pos)

@dataclass
class RegStmt:
    name: str
    regime: Optional[str] = None
    value: Optional[Expr] = None
    overrides: Optional[dict] = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class EntityDecl:
    name: str
    base: Optional[str] = None
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
    kappa: Optional[Expr] = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class PairStmt:
    a: str
    b: str
    kappa: Optional[Expr] = None
    duration: Optional[Expr] = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class RingStmt:
    members: list[str] = field(default_factory=list)
    kappa: Optional[Expr] = None
    duration: Optional[Expr] = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class ObserveStmt:
    target: str
    metrics: list[str] = field(default_factory=list)
    over_seeds: int = 1
    pos: Pos = field(default_factory=Pos)

@dataclass
class RunStmt:
    duration: Optional[Expr] = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class SequenceStmt:
    inputs: Expr
    target: str
    each_for: Optional[Expr] = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class AnnotationStmt:
    key: str
    args: str = ''
    pos: Pos = field(default_factory=Pos)

@dataclass
class ClassDecl:
    name: str
    parent: Optional[str] = None
    fields: list[TypeField] = field(default_factory=list)
    methods: list[FnDecl] = field(default_factory=list)
    pos: Pos = field(default_factory=Pos)

@dataclass
class MatchCase:
    pattern: Expr
    guard: Optional[Expr] = None
    body: list[Stmt] = field(default_factory=list)

@dataclass
class MatchStmt:
    subject: Expr
    cases: list[MatchCase] = field(default_factory=list)
    else_body: Optional[list[Stmt]] = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class YieldStmt:
    value: Optional[Expr] = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class YieldExpr:
    value: Optional[Expr] = None
    pos: Pos = field(default_factory=Pos)

@dataclass
class AwaitExpr:
    value: Expr
    pos: Pos = field(default_factory=Pos)
Stmt = Union[LetStmt, DestructLetStmt, MapDestructStmt, ConstStmt, AssignStmt, ExprStmt, ReturnStmt, BreakStmt, ContinueStmt, IfStmt, ForStmt, WhileStmt, FnDecl, TypeDecl, ClassDecl, ImportStmt, FromImportStmt, TryCatchStmt, ThrowStmt, MatchStmt, YieldStmt, RegStmt, EntityDecl, WorldDecl, CoupleStmt, PairStmt, RingStmt, ObserveStmt, RunStmt, AnnotationStmt]
Expr = Union[IntLit, FloatLit, BoolLit, StringLit, NoneLit, Ident, BinOp, UnaryOp, CallExpr, IndexExpr, FieldExpr, ListExpr, MapExpr, TupleExpr, LambdaExpr, MethodCallExpr, AssignExpr, YieldExpr, AwaitExpr]

@dataclass
class Module:
    name: str = ''
    body: list[Stmt] = field(default_factory=list)
    file: str = ''