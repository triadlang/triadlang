from __future__ import annotations
from frontend.ast_nodes import *
from compiler.ir import *

def lower_module(mod: Module) -> IRModule:
    return IRModule(name=mod.name, body=[lower_stmt(s) for s in mod.body])

def lower_stmt(s: Stmt) -> IRNode:
    if isinstance(s, LetStmt):
        return IRLet(s.name, s.type_ann, lower_expr(s.value) if s.value else None)
    if isinstance(s, ConstStmt):
        return IRConst(s.name, lower_expr(s.value))
    if isinstance(s, AssignStmt):
        return IRAssign(lower_expr(s.target), lower_expr(s.value))
    if isinstance(s, ExprStmt):
        return IRExprStmt(lower_expr(s.expr))
    if isinstance(s, ReturnStmt):
        return IRReturn(lower_expr(s.value) if s.value else None)
    if isinstance(s, BreakStmt):
        return IRBreak()
    if isinstance(s, ContinueStmt):
        return IRContinue()
    if isinstance(s, IfStmt):
        return IRIf(lower_expr(s.condition), [lower_stmt(st) for st in s.then_body], [(lower_expr(c), [lower_stmt(st) for st in b]) for c, b in s.elif_clauses], [lower_stmt(st) for st in s.else_body] if s.else_body else None)
    if isinstance(s, ForStmt):
        return IRFor(s.var, lower_expr(s.iter), [lower_stmt(st) for st in s.body])
    if isinstance(s, WhileStmt):
        return IRWhile(lower_expr(s.condition), [lower_stmt(st) for st in s.body])
    if isinstance(s, FnDecl):
        return IRFunction(s.name, [p.name for p in s.params], [lower_stmt(st) for st in s.body])
    if isinstance(s, TypeDecl):
        return IRTypeDecl(s.name, [(f.name, f.type_ann) for f in s.fields])
    if isinstance(s, ImportStmt):
        return IRImport(s.path, s.alias)
    if isinstance(s, FromImportStmt):
        return IRImport(s.path)
    if isinstance(s, RegStmt):
        return IRRegDecl(s.name, s.regime, lower_expr(s.value) if s.value else None)
    if isinstance(s, EntityDecl):
        return IREntityDecl(s.name, s.base, {k: lower_expr(v) for k, v in s.fields.items()})
    if isinstance(s, WorldDecl):
        return IRWorldDecl(s.name, {k: lower_expr(v) for k, v in s.fields.items()}, [IREntityDecl(e.name, e.base, {k: lower_expr(v) for k, v in e.fields.items()}) for e in s.entities])
    if isinstance(s, ObserveStmt):
        return IRObserve(s.target, s.metrics)
    if isinstance(s, RunStmt):
        return IRRun(lower_expr(s.duration) if s.duration else None)
    if isinstance(s, DestructLetStmt):
        return IRDestructLet(s.names, lower_expr(s.value))
    if isinstance(s, MapDestructStmt):
        return IRDestructLet(s.names, lower_expr(s.value))
    if isinstance(s, TryCatchStmt):
        return IRTryCatch([lower_stmt(st) for st in s.body], s.catch_var, [lower_stmt(st) for st in s.catch_body] if s.catch_body else None, [lower_stmt(st) for st in s.finally_body] if s.finally_body else None)
    if isinstance(s, ThrowStmt):
        return IRThrow(lower_expr(s.value) if s.value else None)
    if isinstance(s, MatchStmt):
        return _lower_match(s)
    if isinstance(s, ClassDecl):
        return _lower_class(s)
    if isinstance(s, YieldStmt):
        return IRYield(lower_expr(s.value) if s.value else None)
    return IRExprStmt(IRNone())

def _lower_class(s: ClassDecl) -> IRClassDecl:
    methods = []
    for m in s.methods:
        body = [lower_stmt(st) for st in m.body]
        methods.append(IRFunction(m.name, [p.name for p in m.params], body))
    return IRClassDecl(name=s.name, parent=s.parent, fields=[f.name for f in s.fields], methods=methods)

def _lower_match(s: MatchStmt) -> IRIf:
    subject = lower_expr(s.subject)
    elif_clauses = []
    else_body = None
    for i, case in enumerate(s.cases):
        pat = case.pattern
        cond = _match_condition(subject, pat)
        body = [lower_stmt(st) for st in case.body]
        if i == 0:
            first_cond = cond
            first_body = body
        else:
            elif_clauses.append((cond, body))
    if s.else_body:
        else_body = [lower_stmt(st) for st in s.else_body]
    elif elif_clauses:
        last_cond, last_body = elif_clauses.pop()
        else_body = last_body
    else:
        else_body = None
    return IRIf(first_cond, first_body, elif_clauses, else_body)

def _match_condition(subject: IRNode, pattern) -> IRNode:
    if isinstance(pattern, IntLit):
        return IRBinOp('==', subject, IRInt(pattern.value))
    if isinstance(pattern, FloatLit):
        return IRBinOp('==', subject, IRFloat(pattern.value))
    if isinstance(pattern, StringLit):
        return IRBinOp('==', subject, IRString(pattern.value))
    if isinstance(pattern, BoolLit):
        return IRBinOp('==', subject, IRBool(pattern.value))
    if isinstance(pattern, NoneLit):
        return IRBinOp('==', subject, IRNone())
    if isinstance(pattern, Ident):
        if pattern.name == '_':
            return IRBool(True)
        return IRBinOp('==', subject, IRIdent(pattern.name))
    if isinstance(pattern, ListExpr):
        return IRBinOp('==', subject, IRList([lower_expr(el) for el in pattern.elements]))
    return IRBool(True)

def lower_expr(e: Expr) -> IRNode:
    if isinstance(e, IntLit):
        return IRInt(e.value)
    if isinstance(e, FloatLit):
        return IRFloat(e.value)
    if isinstance(e, BoolLit):
        return IRBool(e.value)
    if isinstance(e, StringLit):
        return IRString(e.value)
    if isinstance(e, NoneLit):
        return IRNone()
    if isinstance(e, Ident):
        return IRIdent(e.name)
    if isinstance(e, BinOp):
        return IRBinOp(e.op, lower_expr(e.left), lower_expr(e.right))
    if isinstance(e, UnaryOp):
        return IRUnaryOp(e.op, lower_expr(e.operand))
    if isinstance(e, CallExpr):
        return IRCall(lower_expr(e.func), [lower_expr(a) for a in e.args], {k: lower_expr(v) for k, v in e.kwargs.items()})
    if isinstance(e, IndexExpr):
        if isinstance(e.index, SliceExpr):
            return IRSlice(lower_expr(e.obj), lower_expr(e.index.start) if e.index.start else None, lower_expr(e.index.end) if e.index.end else None, lower_expr(e.index.step) if e.index.step else None)
        return IRIndex(lower_expr(e.obj), lower_expr(e.index))
    if isinstance(e, FieldExpr):
        return IRField(lower_expr(e.obj), e.field)
    if isinstance(e, ListExpr):
        return IRList([lower_expr(el) for el in e.elements])
    if isinstance(e, MapExpr):
        return IRMap([(lower_expr(k), lower_expr(v)) for k, v in e.pairs])
    if isinstance(e, LambdaExpr):
        return IRFunction('', [p.name for p in e.params], [lower_stmt(s) for s in e.body])
    if isinstance(e, MethodCallExpr):
        return IRMethodCall(lower_expr(e.obj), e.method, [lower_expr(a) for a in e.args], {k: lower_expr(v) for k, v in e.kwargs.items()})
    if isinstance(e, AssignExpr):
        return IRAssignExpr(lower_expr(e.target), lower_expr(e.value))
    if isinstance(e, FStringExpr):
        parts = []
        for ptype, pval in e.parts:
            if ptype == 'str':
                parts.append((pval, None))
            else:
                parts.append(('', lower_expr(pval)))
        return IRFString(parts)
    if isinstance(e, ListCompExpr):
        return IRListComp(lower_expr(e.expr), e.var, lower_expr(e.iter), lower_expr(e.condition) if e.condition else None)
    return IRNone()