from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union

@dataclass
class IRPos:
    line: int = 0
    col: int = 0
    file: str = ''

@dataclass
class IRModule:
    name: str = ''
    imports: list[IRImport] = field(default_factory=list)
    body: list[IRNode] = field(default_factory=list)

@dataclass
class IRImport:
    path: list[str] = field(default_factory=list)
    alias: Optional[str] = None

@dataclass
class IRLet:
    name: str = ''
    type_ann: Optional[str] = None
    value: Optional[IRNode] = None

@dataclass
class IRConst:
    name: str = ''
    value: IRNode = None

@dataclass
class IRAssign:
    target: IRNode = None
    value: IRNode = None

@dataclass
class IRBinOp:
    op: str = ''
    left: IRNode = None
    right: IRNode = None

@dataclass
class IRUnaryOp:
    op: str = ''
    operand: IRNode = None

@dataclass
class IRCall:
    func: IRNode = None
    args: list[IRNode] = field(default_factory=list)
    kwargs: dict[str, IRNode] = field(default_factory=dict)

@dataclass
class IRInt:
    value: int = 0

@dataclass
class IRFloat:
    value: float = 0.0

@dataclass
class IRBool:
    value: bool = False

@dataclass
class IRString:
    value: str = ''

@dataclass
class IRNone:
    pass

@dataclass
class IRIdent:
    name: str = ''

@dataclass
class IRIndex:
    obj: IRNode = None
    index: IRNode = None

@dataclass
class IRField:
    obj: IRNode = None
    field: str = ''

@dataclass
class IRList:
    elements: list[IRNode] = field(default_factory=list)

@dataclass
class IRMap:
    pairs: list[tuple[IRNode, IRNode]] = field(default_factory=list)

@dataclass
class IRFunction:
    name: str = ''
    params: list[str] = field(default_factory=list)
    body: list[IRNode] = field(default_factory=list)

@dataclass
class IRReturn:
    value: Optional[IRNode] = None

@dataclass
class IRIf:
    condition: IRNode = None
    then_body: list[IRNode] = field(default_factory=list)
    elif_clauses: list[tuple[IRNode, list[IRNode]]] = field(default_factory=list)
    else_body: Optional[list[IRNode]] = None

@dataclass
class IRFor:
    var: str = ''
    iter: IRNode = None
    body: list[IRNode] = field(default_factory=list)

@dataclass
class IRWhile:
    condition: IRNode = None
    body: list[IRNode] = field(default_factory=list)

@dataclass
class IRBreak:
    pass

@dataclass
class IRContinue:
    pass

@dataclass
class IRExprStmt:
    expr: IRNode = None

@dataclass
class IRTypeDecl:
    name: str = ''
    fields: list[tuple[str, str]] = field(default_factory=list)

@dataclass
class IREntityDecl:
    name: str = ''
    base: Optional[str] = None
    fields: dict[str, IRNode] = field(default_factory=dict)

@dataclass
class IRWorldDecl:
    name: str = ''
    fields: dict[str, IRNode] = field(default_factory=dict)
    entities: list[IREntityDecl] = field(default_factory=list)

@dataclass
class IRRegDecl:
    name: str = ''
    regime: Optional[str] = None
    value: Optional[IRNode] = None

@dataclass
class IRObserve:
    target: str = ''
    metrics: list[str] = field(default_factory=list)

@dataclass
class IRRun:
    duration: Optional[IRNode] = None

@dataclass
class IRMethodCall:
    obj: IRNode = None
    method: str = ''
    args: list[IRNode] = field(default_factory=list)
    kwargs: dict[str, IRNode] = field(default_factory=dict)

@dataclass
class IRSlice:
    obj: IRNode = None
    start: Optional[IRNode] = None
    end: Optional[IRNode] = None
    step: Optional[IRNode] = None

@dataclass
class IRFString:
    parts: list[tuple[str, Optional[IRNode]]] = field(default_factory=list)

@dataclass
class IRListComp:
    expr: IRNode = None
    var: str = ''
    iter: IRNode = None
    condition: Optional[IRNode] = None

@dataclass
class IRTryCatch:
    body: list[IRNode] = field(default_factory=list)
    catch_var: Optional[str] = None
    catch_body: Optional[list[IRNode]] = None
    finally_body: Optional[list[IRNode]] = None

@dataclass
class IRThrow:
    value: Optional[IRNode] = None

@dataclass
class IRDestructLet:
    names: list[str] = field(default_factory=list)
    value: IRNode = None

@dataclass
class IRAssignExpr:
    target: IRNode = None
    value: IRNode = None

@dataclass
class IRClassDecl:
    name: str = ''
    parent: Optional[str] = None
    fields: list[str] = field(default_factory=list)
    methods: list[IRFunction] = field(default_factory=list)

@dataclass
class IRYield:
    value: Optional[IRNode] = None
IRNode = Union[IRModule, IRImport, IRLet, IRConst, IRAssign, IRBinOp, IRUnaryOp, IRCall, IRInt, IRFloat, IRBool, IRString, IRNone, IRIdent, IRIndex, IRField, IRList, IRMap, IRFunction, IRReturn, IRIf, IRFor, IRWhile, IRBreak, IRContinue, IRExprStmt, IRTypeDecl, IREntityDecl, IRWorldDecl, IRRegDecl, IRObserve, IRRun, IRMethodCall, IRSlice, IRFString, IRListComp, IRTryCatch, IRThrow, IRDestructLet, IRAssignExpr, IRClassDecl, IRYield]