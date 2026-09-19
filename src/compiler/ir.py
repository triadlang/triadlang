from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


@dataclass
class IRModule:
    name: str = ''
    imports: list[IRImport] = field(default_factory=list)
    body: list[IRNode] = field(default_factory=list)

@dataclass
class IRImport:
    path: list[str] = field(default_factory=list)
    alias: str | None = None
    names: list[str] = field(default_factory=list)
    aliases: list = field(default_factory=list)

@dataclass
class IRLet:
    name: str = ''
    type_ann: str | None = None
    value: IRNode | None = None

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
class IRLambda:
    params: list[str] = field(default_factory=list)
    body: list[IRNode] = field(default_factory=list)

@dataclass
class IRFunction:
    name: str = ''
    params: list[str] = field(default_factory=list)
    body: list[IRNode] = field(default_factory=list)
    star_idx: int = -1
    kw_idx: int = -1

@dataclass
class IRReturn:
    value: IRNode | None = None

@dataclass
class IRIf:
    condition: IRNode = None
    then_body: list[IRNode] = field(default_factory=list)
    elif_clauses: list[tuple[IRNode, list[IRNode]]] = field(default_factory=list)
    else_body: list[IRNode] | None = None

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
    base: str | None = None
    fields: dict[str, IRNode] = field(default_factory=dict)

@dataclass
class IRWorldDecl:
    name: str = ''
    fields: dict[str, IRNode] = field(default_factory=dict)
    entities: list[IREntityDecl] = field(default_factory=list)

@dataclass
class IRRegDecl:
    name: str = ''
    regime: str | None = None
    value: IRNode | None = None

@dataclass
class IRSubstrateDecl:
    name: str = ''
    regime: str | None = None
    members: list[str] = field(default_factory=list)
    properties: dict[str, IRNode] = field(default_factory=dict)
    overrides: dict[str, IRNode] = field(default_factory=dict)
    is_composed: bool = False

@dataclass
class IRObserve:
    target: str = ''
    metrics: list[str] = field(default_factory=list)

@dataclass
class IRRun:
    duration: IRNode | None = None
    target: str | None = None

@dataclass
class IRCouple:
    src: str = ''
    dst: str = ''
    kappa: IRNode | None = None
    duration: IRNode | None = None

@dataclass
class IRPair:
    a: str = ''
    b: str = ''
    kappa: IRNode | None = None
    duration: IRNode | None = None

@dataclass
class IRRing:
    members: list[str] = field(default_factory=list)
    kappa: IRNode | None = None
    duration: IRNode | None = None

@dataclass
class IRAnnotation:
    key: str = ''
    args: str = ''

@dataclass
class IRSequence:
    inputs: IRNode | None = None
    target: str = ''
    each_for: IRNode | None = None

@dataclass
class IRWith:
    expr: IRNode | None = None
    var: str | None = None
    body: list[IRNode] = field(default_factory=list)

@dataclass
class IRAssert:
    condition: IRNode = None
    message: IRNode | None = None

@dataclass
class IRPass:
    pass

@dataclass
class IRDel:
    target: IRNode = None

@dataclass
class IRAsyncFor:
    var: str = ''
    iter: IRNode = None
    body: list[IRNode] = field(default_factory=list)

@dataclass
class IRAsyncWith:
    expr: IRNode | None = None
    var: str | None = None
    body: list[IRNode] = field(default_factory=list)

@dataclass
class IRMethodCall:
    obj: IRNode = None
    method: str = ''
    args: list[IRNode] = field(default_factory=list)
    kwargs: dict[str, IRNode] = field(default_factory=dict)

@dataclass
class IRSlice:
    obj: IRNode = None
    start: IRNode | None = None
    end: IRNode | None = None
    step: IRNode | None = None

@dataclass
class IRFString:
    parts: list[tuple[str, IRNode | None]] = field(default_factory=list)

@dataclass
class IRListComp:
    expr: IRNode = None
    var: str = ''
    iter: IRNode = None
    condition: IRNode | None = None
    clauses: list[IRCompClause] = field(default_factory=list)

@dataclass
class IRCatchBlock:
    exceptions: list[str] = field(default_factory=list)
    var: str | None = None
    body: list[IRNode] = field(default_factory=list)

@dataclass
class IRTryCatch:
    body: list[IRNode] = field(default_factory=list)
    catches: list[IRCatchBlock] = field(default_factory=list)
    catch_var: str | None = None
    catch_body: list[IRNode] | None = None
    finally_body: list[IRNode] | None = None

@dataclass
class IRThrow:
    value: IRNode | None = None

@dataclass
class IRDestructLet:
    names: list[str] = field(default_factory=list)
    value: IRNode = None

@dataclass
class IRMapDestruct:
    names: list[str] = field(default_factory=list)
    value: IRNode = None

@dataclass
class IRAssignExpr:
    target: IRNode = None
    value: IRNode = None

@dataclass
class IRMatchCase:
    pattern: IRNode = None
    body: list[IRNode] = field(default_factory=list)

@dataclass
class IRMatch:
    subject: IRNode = None
    cases: list[IRMatchCase] = field(default_factory=list)
    else_body: list[IRNode] | None = None

@dataclass
class IRClassDecl:
    name: str = ''
    parent: str | None = None
    fields: list[str] = field(default_factory=list)
    methods: list[IRFunction] = field(default_factory=list)

@dataclass
class IRYield:
    value: IRNode | None = None

@dataclass
class IRComplex:
    real: float = 0.0
    imag: float = 0.0

@dataclass
class IRBytes:
    value: str = ''

@dataclass
class IRTuple:
    elements: list[IRNode] = field(default_factory=list)

@dataclass
class IRSet:
    elements: list[IRNode] = field(default_factory=list)

@dataclass
class IRTernary:
    condition: IRNode = None
    then_val: IRNode = None
    else_val: IRNode = None

@dataclass
class IRChainCmp:
    operands: list[IRNode] = field(default_factory=list)
    ops: list[str] = field(default_factory=list)

@dataclass
class IRSuper:
    args: list[IRNode] = field(default_factory=list)

@dataclass
class IRCompClause:
    var: str = ''
    iter: IRNode = None
    conditions: list[IRNode] = field(default_factory=list)

@dataclass
class IRDictComp:
    key_expr: IRNode = None
    value_expr: IRNode = None
    clauses: list[IRCompClause] = field(default_factory=list)

@dataclass
class IRSetComp:
    expr: IRNode = None
    clauses: list[IRCompClause] = field(default_factory=list)

@dataclass
class IRGenComp:
    expr: IRNode = None
    clauses: list[IRCompClause] = field(default_factory=list)

@dataclass
class IRYieldExpr:
    value: IRNode | None = None

@dataclass
class IRAwait:
    value: IRNode = None

@dataclass
class IRNullish:
    left: IRNode = None
    right: IRNode = None

@dataclass
class IRElvis:
    cond: IRNode = None
    else_val: IRNode = None

@dataclass
class IROptChain:
    obj: IRNode = None
    kind: str = 'attr'
    name: str | None = None
    args: list[IRNode] = field(default_factory=list)
    kwargs: dict[str, IRNode] = field(default_factory=dict)
    index: IRNode | None = None

@dataclass
class IRRegex:
    pattern: str = ''
    flags: str = ''

@dataclass
class IRCompoundAssign:
    target: IRNode = None
    op: str = ''
    value: IRNode = None

IRNode = Union[IRModule, IRImport, IRLet, IRConst, IRAssign, IRBinOp, IRUnaryOp, IRCall, IRInt, IRFloat, IRBool, IRString, IRNone, IRIdent, IRIndex, IRField, IRList, IRMap, IRLambda, IRFunction, IRReturn, IRIf, IRFor, IRWhile, IRBreak, IRContinue, IRExprStmt, IRTypeDecl, IREntityDecl, IRWorldDecl, IRRegDecl, IRSubstrateDecl, IRObserve, IRRun, IRCouple, IRPair, IRRing, IRAnnotation, IRSequence, IRWith, IRAssert, IRPass, IRDel, IRAsyncFor, IRAsyncWith, IRMethodCall, IRSlice, IRFString, IRListComp, IRCatchBlock, IRTryCatch, IRThrow, IRDestructLet, IRMapDestruct, IRAssignExpr, IRMatchCase, IRMatch, IRClassDecl, IRYield, IRComplex, IRBytes, IRTuple, IRSet, IRTernary, IRChainCmp, IRSuper, IRDictComp, IRSetComp, IRGenComp, IRYieldExpr, IRAwait, IRCompClause, IRNullish, IRElvis, IROptChain, IRRegex, IRCompoundAssign]
