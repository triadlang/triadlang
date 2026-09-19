from __future__ import annotations

from compiler.ir import *
from frontend.ast_nodes import *


def _combine_conditions(conditions: list[IRNode]) -> IRNode | None:
    if not conditions:
        return None
    cur = conditions[0]
    for cond in conditions[1:]:
        cur = IRBinOp('and', cur, cond)
    return cur

def _lower_comp_clauses(clauses: list[CompClause]) -> list[IRCompClause]:
    result = []
    for clause in clauses:
        conditions = [lower_expr(c) for c in clause.conditions]
        result.append(IRCompClause(
            var=clause.var,
            iter=lower_expr(clause.iter),
            conditions=conditions,
        ))
    return result

def _lower_single_comp(expr: Expr, clauses: list[CompClause] | None) -> IRListComp:
    clauses = clauses or []
    if len(clauses) == 1:
        clause = clauses[0]
        return IRListComp(
            lower_expr(expr),
            clause.var,
            lower_expr(clause.iter),
            _combine_conditions([lower_expr(c) for c in clause.conditions]),
            clauses=_lower_comp_clauses(clauses),
        )
    return IRListComp(
        lower_expr(expr),
        clauses[0].var,
        lower_expr(clauses[0].iter),
        _combine_conditions([lower_expr(c) for c in clauses[0].conditions]),
        clauses=_lower_comp_clauses(clauses),
    )

def lower_module(mod: Module, optimize: bool = True, opt_level: int = 2) -> IRModule:
    ir = IRModule(name=mod.name, body=[lower_stmt(s) for s in mod.body])
    if optimize:
        from compiler.ir_passes import optimize as _optimize_ir
        ir = _optimize_ir(ir, level=opt_level)
    return ir

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
        return IRFunction(s.name, [p.name for p in s.params], [lower_stmt(st) for st in s.body],
                          star_idx=next((i for i, p in enumerate(s.params) if getattr(p, 'is_args', False)), -1),
                          kw_idx=next((i for i, p in enumerate(s.params) if getattr(p, 'is_kwargs', False)), -1))
    if isinstance(s, TypeDecl):
        return IRTypeDecl(s.name, [(f.name, f.type_ann) for f in s.fields])
    if isinstance(s, ImportStmt):
        return IRImport(s.path, s.alias)
    if isinstance(s, FromImportStmt):
        return IRImport(s.path, names=s.names, aliases=list(getattr(s, 'aliases', None) or []))
    if isinstance(s, RegStmt):
        return IRRegDecl(s.name, s.regime, lower_expr(s.value) if s.value else None)
    if isinstance(s, SubstrateDecl):
        members = list(getattr(s, 'members', None) or [])
        return IRSubstrateDecl(
            name=s.name,
            regime=getattr(s, 'regime', None),
            members=members,
            properties={k: lower_expr(v) for k, v in (getattr(s, 'properties', None) or {}).items()},
            overrides={k: lower_expr(v) for k, v in (getattr(s, 'overrides', None) or {}).items()},
            is_composed=bool(members),
        )
    if isinstance(s, EntityDecl):
        return IREntityDecl(s.name, s.base, {k: lower_expr(v) for k, v in s.fields.items()})
    if isinstance(s, WorldDecl):
        return IRWorldDecl(s.name, {k: lower_expr(v) for k, v in s.fields.items()}, [IREntityDecl(e.name, e.base, {k: lower_expr(v) for k, v in e.fields.items()}) for e in s.entities])
    if isinstance(s, ObserveStmt):
        return IRObserve(s.target, s.metrics)
    if isinstance(s, RunStmt):
        return IRRun(lower_expr(s.duration) if s.duration else None, target=s.target)
    if isinstance(s, DestructLetStmt):
        return IRDestructLet(s.names, lower_expr(s.value))
    if isinstance(s, MapDestructStmt):
        return IRMapDestruct(s.names, lower_expr(s.value))
    if isinstance(s, TryCatchStmt):
        catches = []
        if hasattr(s, 'catches') and s.catches:
            for cb in s.catches:
                exc_names = list(getattr(cb, 'exceptions', None) or [])
                catch_var = getattr(cb, 'var', None)
                catch_body = [lower_stmt(st) for st in cb.body] if cb.body else []
                catches.append(IRCatchBlock(exceptions=exc_names, var=catch_var, body=catch_body))
        return IRTryCatch(
            body=[lower_stmt(st) for st in s.body],
            catches=catches,
            catch_var=s.catch_var,
            catch_body=[lower_stmt(st) for st in s.catch_body] if s.catch_body else None,
            finally_body=[lower_stmt(st) for st in s.finally_body] if s.finally_body else None,
        )
    if isinstance(s, ThrowStmt):
        return IRThrow(lower_expr(s.value) if s.value else None)
    if isinstance(s, MatchStmt):
        return _lower_match(s)
    if isinstance(s, ClassDecl):
        return _lower_class(s)
    if isinstance(s, YieldStmt):
        return IRYield(lower_expr(s.value) if s.value else None)
    if isinstance(s, CoupleStmt):
        return IRCouple(s.src, s.dst,
                        lower_expr(s.kappa) if s.kappa else None,
                        lower_expr(s.duration) if s.duration else None)
    if isinstance(s, PairStmt):
        return IRPair(s.a, s.b,
                      lower_expr(s.kappa) if s.kappa else None,
                      lower_expr(s.duration) if s.duration else None)
    if isinstance(s, RingStmt):
        return IRRing(list(s.members),
                      lower_expr(s.kappa) if s.kappa else None,
                      lower_expr(s.duration) if s.duration else None)
    if isinstance(s, AnnotationStmt):
        return IRAnnotation(s.key, s.args)
    if isinstance(s, SequenceStmt):
        return IRSequence(lower_expr(s.inputs), s.target,
                          lower_expr(s.each_for) if s.each_for else None)
    if isinstance(s, WithStmt):
        return IRWith(lower_expr(s.expr) if s.expr else None, s.var,
                      [lower_stmt(st) for st in s.body])
    if isinstance(s, AssertStmt):
        return IRAssert(lower_expr(s.condition),
                        lower_expr(s.message) if s.message else None)
    if isinstance(s, PassStmt):
        return IRPass()
    if isinstance(s, DelStmt):
        return IRDel(lower_expr(s.target))
    if isinstance(s, AsyncForStmt):
        return IRAsyncFor(s.var, lower_expr(s.iter),
                          [lower_stmt(st) for st in s.body])
    if isinstance(s, AsyncWithStmt):
        return IRAsyncWith(lower_expr(s.expr) if s.expr else None, s.var,
                           [lower_stmt(st) for st in s.body])
    if isinstance(s, CompoundAssignExpr):
        return _lower_compound_assign(s)
    raise NotImplementedError(f"lower_stmt: unhandled statement type {type(s).__name__}")

def _lower_class(s: ClassDecl) -> IRClassDecl:
    methods = []
    for m in s.methods:
        body = [lower_stmt(st) for st in m.body]
        methods.append(IRFunction(m.name, [p.name for p in m.params], body))
    return IRClassDecl(name=s.name, parent=s.parent, fields=[f.name for f in s.fields], methods=methods)

def _lower_match(s: MatchStmt) -> IRMatch:
    subject = lower_expr(s.subject)
    cases = []
    for case in s.cases:
        pat_ir = _lower_pattern(case.pattern)
        body = [lower_stmt(st) for st in case.body]
        cases.append(IRMatchCase(pattern=pat_ir, body=body))
    else_body = [lower_stmt(st) for st in s.else_body] if s.else_body else None
    return IRMatch(subject=subject, cases=cases, else_body=else_body)

def _lower_pattern(pattern) -> IRNode:
    if isinstance(pattern, IntLit):
        return IRInt(pattern.value)
    if isinstance(pattern, FloatLit):
        return IRFloat(pattern.value)
    if isinstance(pattern, ComplexLit):
        return IRComplex(real=pattern.value.real, imag=pattern.value.imag)
    if isinstance(pattern, BoolLit):
        return IRBool(pattern.value)
    if isinstance(pattern, StringLit):
        return IRString(pattern.value)
    if isinstance(pattern, BytesLit):
        return IRBytes(value=pattern.value)
    if isinstance(pattern, NoneLit):
        return IRNone()
    if isinstance(pattern, Ident):
        return IRIdent(pattern.name)
    if isinstance(pattern, ListExpr):
        return IRList([_lower_pattern(el) for el in pattern.elements])
    if isinstance(pattern, TupleExpr):
        return IRTuple(elements=[_lower_pattern(el) for el in pattern.elements])
    if isinstance(pattern, SetExpr):
        return IRSet(elements=[_lower_pattern(el) for el in pattern.elements])
    if isinstance(pattern, MapExpr):
        return IRMap([(_lower_pattern(k), _lower_pattern(v)) for k, v in pattern.pairs])
    if isinstance(pattern, ListPattern):
        return IRList([_lower_pattern(el) for el in pattern.elements])
    if isinstance(pattern, DictPattern):
        pairs = [(_lower_pattern(k), _lower_pattern(v)) for k, v in pattern.entries]
        return IRMap(pairs)
    if isinstance(pattern, ClassPattern):
        fields = {k: _lower_pattern(v) for k, v in pattern.fields.items()}
        return IRMap([(IRString(k), v) for k, v in fields.items()])
    if isinstance(pattern, OrPattern):
        return IRList([_lower_pattern(alt) for alt in pattern.alternatives])
    if isinstance(pattern, GenericType):
        return IRIdent(pattern.name)
    if isinstance(pattern, UnionType):
        return IRList([_lower_pattern(t) for t in pattern.types])
    if isinstance(pattern, OptionalType):
        return IRList([_lower_pattern(pattern.inner), IRNone()])
    raise NotImplementedError(f"_lower_pattern: unhandled pattern type {type(pattern).__name__}")

def _match_condition(subject: IRNode, pattern) -> IRNode:
    if isinstance(pattern, IntLit):
        return IRBinOp('==', subject, IRInt(pattern.value))
    if isinstance(pattern, FloatLit):
        return IRBinOp('==', subject, IRFloat(pattern.value))
    if isinstance(pattern, ComplexLit):
        return IRBinOp('==', subject, IRComplex(real=pattern.value.real, imag=pattern.value.imag))
    if isinstance(pattern, StringLit):
        return IRBinOp('==', subject, IRString(pattern.value))
    if isinstance(pattern, BytesLit):
        return IRBinOp('==', subject, IRBytes(value=pattern.value))
    if isinstance(pattern, BoolLit):
        return IRBinOp('==', subject, IRBool(pattern.value))
    if isinstance(pattern, NoneLit):
        return IRBinOp('==', subject, IRNone())
    if isinstance(pattern, Ident):
        if pattern.name == '_':
            return IRBool(True)
        return IRBool(True)
    if isinstance(pattern, ListExpr):
        elem_checks = []
        for i, el in enumerate(pattern.elements):
            elem = IRIndex(subject, IRInt(i))
            elem_checks.append(_match_condition(elem, el))
        len_check = IRBinOp('==', IRCall(IRIdent('len'), [subject]), IRInt(len(pattern.elements)))
        return _combine_conditions([len_check] + elem_checks) or IRBool(True)
    if isinstance(pattern, TupleExpr):
        elem_checks = []
        for i, el in enumerate(pattern.elements):
            elem = IRIndex(subject, IRInt(i))
            elem_checks.append(_match_condition(elem, el))
        len_check = IRBinOp('==', IRCall(IRIdent('len'), [subject]), IRInt(len(pattern.elements)))
        return _combine_conditions([len_check] + elem_checks) or IRBool(True)
    if isinstance(pattern, SetExpr):
        checks = [IRBinOp('in', el_ir, subject) for el_ir in [lower_expr(el) for el in pattern.elements]]
        return _combine_conditions(checks) or IRBool(True)
    if isinstance(pattern, MapExpr):
        checks = []
        for k, v in pattern.pairs:
            has_key = IRBinOp('in', lower_expr(k), subject)
            val_check = _match_condition(IRIndex(subject, lower_expr(k)), v)
            checks.append(IRBinOp('and', has_key, val_check))
        return _combine_conditions(checks) or IRBool(True)
    if isinstance(pattern, ListPattern):
        elem_checks = []
        for i, el in enumerate(pattern.elements):
            elem = IRIndex(subject, IRInt(i))
            elem_checks.append(_match_condition(elem, el))
        len_check = IRBinOp('==', IRCall(IRIdent('len'), [subject]), IRInt(len(pattern.elements)))
        return _combine_conditions([len_check] + elem_checks) or IRBool(True)
    if isinstance(pattern, DictPattern):
        checks = []
        for k, v in pattern.pairs:
            k_ir = lower_expr(k)
            has_key = IRBinOp('in', k_ir, subject)
            val_check = _match_condition(IRIndex(subject, k_ir), v)
            checks.append(IRBinOp('and', has_key, val_check))
        return _combine_conditions(checks) or IRBool(True)
    if isinstance(pattern, ClassPattern):
        checks = []
        for field_name, field_pat in pattern.fields.items():
            field_val = IRField(subject, field_name)
            checks.append(_match_condition(field_val, field_pat))
        return _combine_conditions(checks) or IRBool(True)
    if isinstance(pattern, OrPattern):
        alt_checks = [_match_condition(subject, alt) for alt in pattern.alternatives]
        if alt_checks:
            cur = alt_checks[0]
            for c in alt_checks[1:]:
                cur = IRBinOp('or', cur, c)
            return cur
        return IRBool(True)
    if isinstance(pattern, GenericType):
        return IRBinOp('==', IRCall(IRIdent('type'), [subject]), IRIdent(pattern.name))
    if isinstance(pattern, UnionType):
        alt_checks = [_match_condition(subject, t) for t in pattern.types]
        if alt_checks:
            cur = alt_checks[0]
            for c in alt_checks[1:]:
                cur = IRBinOp('or', cur, c)
            return cur
        return IRBool(True)
    if isinstance(pattern, OptionalType):
        return IRBinOp('or',
                       IRBinOp('==', subject, IRNone()),
                       _match_condition(subject, pattern.inner))
    raise NotImplementedError(f"_match_condition: unhandled pattern type {type(pattern).__name__}")

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
        return IRCall(lower_expr(e.func), [_lower_call_arg(a) for a in e.args], {k: lower_expr(v) for k, v in e.kwargs.items()})
    if isinstance(e, IndexExpr):
        if isinstance(e.index, SliceExpr):
            return IRSlice(lower_expr(e.obj), lower_expr(e.index.start) if e.index.start else None, lower_expr(e.index.end) if e.index.end else None, lower_expr(e.index.step) if e.index.step else None)
        return IRIndex(lower_expr(e.obj), lower_expr(e.index))
    if isinstance(e, FieldExpr):
        return IRField(lower_expr(e.obj), e.field)
    if isinstance(e, ListExpr):
        return IRList([lower_expr(el) for el in e.elements])
    if isinstance(e, SetExpr):
        return IRSet(elements=[lower_expr(el) for el in e.elements])
    if isinstance(e, MapExpr):
        return IRMap([(lower_expr(k), lower_expr(v)) for k, v in e.pairs])
    if isinstance(e, LambdaExpr):
        return IRLambda(params=[p.name for p in e.params], body=[lower_stmt(s) for s in e.body])
    if isinstance(e, MethodCallExpr):
        return IRMethodCall(lower_expr(e.obj), e.method, [_lower_call_arg(a) for a in e.args], {k: lower_expr(v) for k, v in e.kwargs.items()})
    if isinstance(e, AssignExpr):
        return IRAssignExpr(lower_expr(e.target), lower_expr(e.value))
    if isinstance(e, FStringExpr):
        parts = []
        for entry in e.parts:
            ptype = entry[0]
            pval = entry[1]
            spec = entry[2] if len(entry) > 2 else None
            if ptype == 'str':
                parts.append((pval, None, None))
            else:
                parts.append(('', lower_expr(pval), spec))
        return IRFString(parts)
    if isinstance(e, ListCompExpr):
        if e.clauses:
            return _lower_single_comp(e.expr, e.clauses)
        return IRListComp(lower_expr(e.expr), e.var, lower_expr(e.iter), lower_expr(e.condition) if e.condition else None)
    if isinstance(e, SetCompExpr):
        return IRSetComp(expr=lower_expr(e.expr), clauses=_lower_comp_clauses(e.clauses))
    if isinstance(e, GenCompExpr):
        return IRGenComp(expr=lower_expr(e.expr), clauses=_lower_comp_clauses(e.clauses))
    if isinstance(e, ComplexLit):
        return IRComplex(real=e.value.real, imag=e.value.imag)
    if isinstance(e, BytesLit):
        return IRBytes(value=e.value)
    if isinstance(e, TupleExpr):
        return IRTuple(elements=[lower_expr(el) for el in e.elements])
    if isinstance(e, TernaryExpr):
        return IRTernary(condition=lower_expr(e.cond), then_val=lower_expr(e.then_val), else_val=lower_expr(e.else_val))
    if isinstance(e, SuperExpr):
        return IRSuper(args=[lower_expr(a) for a in e.args])
    if isinstance(e, ChainCmpExpr):
        return IRChainCmp(operands=[lower_expr(op) for op in e.operands], ops=list(e.ops))
    if isinstance(e, DictCompExpr):
        return IRDictComp(key_expr=lower_expr(e.key), value_expr=lower_expr(e.value), clauses=_lower_comp_clauses(e.clauses))
    if isinstance(e, YieldExpr):
        return IRYieldExpr(value=lower_expr(e.value) if e.value else None)
    if isinstance(e, AwaitExpr):
        return IRAwait(value=lower_expr(e.value))
    if isinstance(e, NullishCoalesceExpr):
        return IRNullish(lower_expr(e.left), lower_expr(e.right))
    if isinstance(e, ElvisExpr):
        return IRElvis(lower_expr(e.cond), lower_expr(e.else_val))
    if isinstance(e, OptChainExpr):
        if e.attr is not None:
            return IROptChain(lower_expr(e.obj), 'attr', e.attr)
        if e.method is not None:
            return IROptChain(lower_expr(e.obj), 'method', e.method,
                              [lower_expr(a) for a in e.args or []],
                              {k: lower_expr(v) for k, v in (e.kwargs or {}).items()})
        if e.index is not None:
            return IROptChain(lower_expr(e.obj), 'index', None, [], {}, lower_expr(e.index))
        return IRNone()
    if isinstance(e, PipelineExpr):
        fn = e.right
        arg = lower_expr(e.left)
        if isinstance(fn, CallExpr):
            return IRCall(lower_expr(fn.func), [arg] + [lower_expr(a) for a in fn.args],
                          {k: lower_expr(v) for k, v in fn.kwargs.items()})
        return IRCall(lower_expr(fn), [arg])
    if isinstance(e, RangeExpr):
        if e.end is None:
            return IRCall(IRIdent('range'), [lower_expr(e.start)])
        end = lower_expr(e.end)
        if e.inclusive:
            end = IRBinOp('+', end, IRInt(1))
        return IRCall(IRIdent('range'), [lower_expr(e.start), end])
    if isinstance(e, RegexLit):
        return IRRegex(e.pattern, e.flags)
    if isinstance(e, CompoundAssignExpr):
        return _lower_compound_assign_expr(e)
    if isinstance(e, SpreadExpr):
        return IRUnaryOp('*', lower_expr(e.value))
    raise NotImplementedError(f"lower_expr: unhandled expression type {type(e).__name__}")


def _lower_call_arg(a: Expr) -> IRNode:
    if isinstance(a, SpreadExpr):
        return IRUnaryOp('*', lower_expr(a.value))
    return lower_expr(a)


def _lower_compound_assign(e: CompoundAssignExpr) -> IRNode:
    if e.op not in ('??=', '||=', '&&='):
        raise NotImplementedError(f"lower: unhandled compound assign op {e.op!r}")
    if isinstance(e.target, Ident):
        t = lower_expr(e.target)
        v = lower_expr(e.value)
        if e.op == '??=':
            return IRAssign(t, IRNullish(t, v))
        if e.op == '||=':
            return IRAssign(t, IRBinOp('or', t, v))
        return IRAssign(t, IRBinOp('and', t, v))
    return IRExprStmt(IRCompoundAssign(lower_expr(e.target), e.op, lower_expr(e.value)))


def _lower_compound_assign_expr(e: CompoundAssignExpr) -> IRNode:
    if e.op not in ('??=', '||=', '&&='):
        raise NotImplementedError(f"lower: unhandled compound assign op {e.op!r}")
    if isinstance(e.target, Ident):
        t = lower_expr(e.target)
        v = lower_expr(e.value)
        if e.op == '??=':
            return IRAssignExpr(t, IRNullish(t, v))
        if e.op == '||=':
            return IRAssignExpr(t, IRBinOp('or', t, v))
        return IRAssignExpr(t, IRBinOp('and', t, v))
    return IRCompoundAssign(lower_expr(e.target), e.op, lower_expr(e.value))

