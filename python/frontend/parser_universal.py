from __future__ import annotations
from frontend.lexer_universal import Token, tokenize, LexError
from frontend.ast_nodes import *

class ParseError(Exception):

    def __init__(self, msg, line=0, col=0, file=''):
        self.msg = msg
        self.line = line
        self.col = col
        self.file = file
        super().__init__(self._format())

    def _format(self):
        parts = [f'error[PARSE]: {self.msg}']
        if self.file:
            parts.append(f'\n  file: {self.file}')
        if self.line:
            parts.append(f'\n  line: {self.line}')
        if self.col:
            parts.append(f'\n  col: {self.col}')
        return ''.join(parts)

class Parser:

    def __init__(self, tokens: list[Token], file: str=''):
        self.toks = tokens
        self.i = 0
        self.file = file

    def _pos(self) -> Pos:
        t = self._peek()
        return Pos(t.line, t.col, self.file)

    def _peek(self, off=0) -> Token:
        idx = self.i + off
        if idx < len(self.toks):
            return self.toks[idx]
        return self.toks[-1]

    def _at_ident(self, value) -> bool:
        t = self._peek()
        return t.kind == 'IDENT' and t.value == value

    def _at(self, kind, value=None) -> bool:
        t = self._peek()
        if t.kind != kind:
            return False
        return value is None or t.value == value

    def _eat(self) -> Token:
        t = self.toks[self.i]
        self.i += 1
        return t

    def _expect(self, kind, value=None) -> Token:
        t = self._peek()
        if t.kind != kind or (value is not None and t.value != value):
            exp = f'{kind} {value!r}' if value else kind
            raise ParseError(f'expected {exp}, got {t.kind} {t.value!r}', t.line, t.col, self.file)
        return self._eat()

    def _expect_sym(self, v):
        return self._expect('SYMBOL', v)

    def _expect_kw(self, v):
        return self._expect('KEYWORD', v)

    def _skip_semis(self):
        while self._at('SYMBOL', ';'):
            self._eat()

    def parse_module(self) -> Module:
        body = []
        self._skip_semis()
        while not self._at('EOF'):
            body.append(self._parse_stmt())
            self._skip_semis()
        return Module(body=body, file=self.file)

    def _parse_stmt(self) -> Stmt:
        t = self._peek()
        if t.kind == 'KEYWORD':
            if t.value == 'let':
                return self._parse_let()
            if t.value == 'const':
                return self._parse_const()
            if t.value == 'fn':
                return self._parse_fn()
            if t.value == 'async':
                next_t = self._peek(1)
                if next_t.kind == 'KEYWORD' and next_t.value == 'fn':
                    return self._parse_fn(async_=True)
            if t.value == 'if':
                return self._parse_if()
            if t.value == 'for':
                return self._parse_for()
            if t.value == 'while':
                return self._parse_while()
            if t.value == 'return':
                return self._parse_return()
            if t.value == 'yield':
                return self._parse_yield()
            if t.value == 'break':
                self._eat()
                self._eat_semi()
                return BreakStmt(pos=Pos(t.line, t.col, self.file))
            if t.value == 'continue':
                self._eat()
                self._eat_semi()
                return ContinueStmt(pos=Pos(t.line, t.col, self.file))
            if t.value == 'type':
                return self._parse_type()
            if t.value == 'class':
                return self._parse_class()
            if t.value == 'match':
                return self._parse_match()
            if t.value == 'import':
                return self._parse_import()
            if t.value == 'from':
                return self._parse_from_import()
            if t.value == 'try':
                return self._parse_try()
            if t.value == 'throw':
                return self._parse_throw()
            if t.value == 'reg':
                return self._parse_reg()
            if t.value == 'entity':
                return self._parse_entity()
            if t.value == 'world':
                return self._parse_world()
            if t.value == 'couple':
                return self._parse_couple()
            if t.value == 'pair':
                return self._parse_pair()
            if t.value == 'ring':
                return self._parse_ring()
            if t.value in ('observe', 'OBSERVE'):
                return self._parse_observe()
            if t.value == 'run':
                return self._parse_run()
            if t.value == 'evolve':
                return self._parse_evolve()
            if t.value == 'sequence':
                return self._parse_sequence()
        if t.kind == 'SYMBOL' and t.value == '@':
            return self._parse_annotation()
        expr = self._parse_expr()
        if self._at('SYMBOL', '='):
            self._eat()
            val = self._parse_expr()
            self._eat_semi()
            return AssignStmt(target=expr, value=val, pos=expr.pos if hasattr(expr, 'pos') else self._pos())
        self._eat_semi()
        return ExprStmt(expr=expr, pos=expr.pos if hasattr(expr, 'pos') else self._pos())

    def _eat_semi(self):
        if self._at('SYMBOL', ';'):
            self._eat()

    def _expect_ident_or_kw(self) -> Token:
        t = self._peek()
        if t.kind in ('IDENT', 'KEYWORD'):
            return self._eat()
        raise ParseError(f'expected identifier, got {t.kind} {t.value!r}', t.line, t.col, self.file)

    def _parse_let(self):
        from frontend.ast_nodes import DestructLetStmt, MapDestructStmt
        p = self._pos()
        self._eat()
        if self._at('SYMBOL', '('):
            self._eat()
            names = [self._expect('IDENT').value]
            while self._at('SYMBOL', ','):
                self._eat()
                names.append(self._expect('IDENT').value)
            self._expect_sym(')')
            self._expect_sym('=')
            value = self._parse_expr()
            self._eat_semi()
            return DestructLetStmt(names=names, value=value, pos=p)
        if self._at('SYMBOL', '{'):
            self._eat()
            names = [self._expect('IDENT').value]
            while self._at('SYMBOL', ','):
                self._eat()
                if self._at('SYMBOL', '}'):
                    break
                names.append(self._expect('IDENT').value)
            self._expect_sym('}')
            self._expect_sym('=')
            value = self._parse_expr()
            self._eat_semi()
            return MapDestructStmt(names=names, value=value, pos=p)
        name = self._expect_ident_or_kw().value
        type_ann = None
        if self._at('SYMBOL', ':'):
            self._eat()
            type_ann = self._expect('IDENT').value
        value = None
        if self._at('SYMBOL', '='):
            self._eat()
            value = self._parse_expr()
        self._eat_semi()
        return LetStmt(name=name, type_ann=type_ann, value=value, pos=p)

    def _parse_const(self) -> ConstStmt:
        p = self._pos()
        self._eat()
        name = self._expect_ident_or_kw().value
        self._expect_sym('=')
        value = self._parse_expr()
        self._eat_semi()
        return ConstStmt(name=name, value=value, pos=p)

    def _parse_fn(self, async_=False) -> FnDecl:
        p = self._pos()
        if async_:
            self._eat()
        self._eat()
        name = self._expect('IDENT').value
        self._expect_sym('(')
        params = self._parse_params()
        self._expect_sym(')')
        ret = None
        if self._at('SYMBOL', '->'):
            self._eat()
            ret = self._expect('IDENT').value
        body = self._parse_block()
        return FnDecl(name=name, params=params, return_type=ret, body=body, is_async=async_, pos=p)

    def _parse_params(self) -> list[Param]:
        params = []
        while not self._at('SYMBOL', ')') and (not self._at('EOF')):
            p = self._pos()
            is_args = False
            is_kwargs = False
            if self._at('SYMBOL', '**'):
                self._eat()
                is_kwargs = True
            elif self._at('SYMBOL', '*'):
                self._eat()
                is_args = True
            name = self._expect_ident_or_kw().value
            type_ann = None
            if self._at('SYMBOL', ':'):
                self._eat()
                type_ann = self._expect('IDENT').value
            default = None
            if self._at('SYMBOL', '='):
                self._eat()
                default = self._parse_expr()
            params.append(Param(name=name, type_ann=type_ann, default=default, is_args=is_args, is_kwargs=is_kwargs, pos=p))
            if not self._at('SYMBOL', ')'):
                self._expect_sym(',')
        return params

    def _parse_block(self) -> list[Stmt]:
        self._expect_sym('{')
        stmts = []
        self._skip_semis()
        while not self._at('SYMBOL', '}') and (not self._at('EOF')):
            stmts.append(self._parse_stmt())
            self._skip_semis()
        self._expect_sym('}')
        return stmts

    def _parse_if(self) -> IfStmt:
        p = self._pos()
        self._eat()
        cond = self._parse_expr()
        then = self._parse_block()
        elifs = []
        else_body = None
        while self._at('KEYWORD', 'elif'):
            self._eat()
            ec = self._parse_expr()
            eb = self._parse_block()
            elifs.append((ec, eb))
        if self._at('KEYWORD', 'else'):
            self._eat()
            else_body = self._parse_block()
        return IfStmt(condition=cond, then_body=then, elif_clauses=elifs, else_body=else_body, pos=p)

    def _parse_for(self) -> ForStmt:
        p = self._pos()
        self._eat()
        var = self._expect('IDENT').value
        self._expect_kw('in')
        iter_expr = self._parse_expr()
        body = self._parse_block()
        return ForStmt(var=var, iter=iter_expr, body=body, pos=p)

    def _parse_while(self) -> WhileStmt:
        p = self._pos()
        self._eat()
        cond = self._parse_expr()
        body = self._parse_block()
        return WhileStmt(condition=cond, body=body, pos=p)

    def _parse_return(self) -> ReturnStmt:
        p = self._pos()
        self._eat()
        value = None
        if not self._at('SYMBOL', ';') and (not self._at('SYMBOL', '}')) and (not self._at('EOF')):
            value = self._parse_expr()
        self._eat_semi()
        return ReturnStmt(value=value, pos=p)

    def _parse_yield(self):
        p = self._pos()
        self._eat()
        value = None
        if not self._at('SYMBOL', ';') and (not self._at('SYMBOL', '}')) and (not self._at('EOF')):
            value = self._parse_expr()
        self._eat_semi()
        return YieldStmt(value=value, pos=p)

    def _parse_type(self) -> TypeDecl:
        p = self._pos()
        self._eat()
        name = self._expect('IDENT').value
        self._expect_sym('{')
        fields = []
        methods = []
        self._skip_semis()
        while not self._at('SYMBOL', '}') and (not self._at('EOF')):
            if self._at('KEYWORD', 'fn'):
                methods.append(self._parse_fn())
            else:
                fp = self._pos()
                fname = self._expect('IDENT').value
                self._expect_sym(':')
                ftype = self._expect('IDENT').value
                default = None
                if self._at('SYMBOL', '='):
                    self._eat()
                    default = self._parse_expr()
                self._eat_semi()
                fields.append(TypeField(name=fname, type_ann=ftype, default=default, pos=fp))
            self._skip_semis()
        self._expect_sym('}')
        return TypeDecl(name=name, fields=fields, methods=methods, pos=p)

    def _parse_class(self) -> ClassDecl:
        p = self._pos()
        self._eat()
        name = self._expect('IDENT').value
        parent = None
        if self._at('SYMBOL', ':'):
            self._eat()
            parent = self._expect('IDENT').value
        self._expect_sym('{')
        fields = []
        methods = []
        self._skip_semis()
        while not self._at('SYMBOL', '}') and (not self._at('EOF')):
            if self._at('KEYWORD', 'fn'):
                methods.append(self._parse_fn())
            else:
                fp = self._pos()
                fname = self._expect('IDENT').value
                ftype = 'any'
                if self._at('SYMBOL', ':'):
                    self._eat()
                    ftype = self._expect('IDENT').value
                default = None
                if self._at('SYMBOL', '='):
                    self._eat()
                    default = self._parse_expr()
                self._eat_semi()
                fields.append(TypeField(name=fname, type_ann=ftype, default=default, pos=fp))
            self._skip_semis()
        self._expect_sym('}')
        return ClassDecl(name=name, parent=parent, fields=fields, methods=methods, pos=p)

    def _parse_match(self) -> MatchStmt:
        p = self._pos()
        self._eat()
        subject = self._parse_expr()
        self._expect_sym('{')
        self._skip_semis()
        cases = []
        else_body = None
        while not self._at('SYMBOL', '}') and (not self._at('EOF')):
            if self._at('KEYWORD', 'else'):
                self._eat()
                if self._at('SYMBOL', '=>'):
                    self._eat()
                else_body = self._parse_block()
                break
            self._expect_kw('case')
            pattern = self._parse_expr()
            guard = None
            if self._at('KEYWORD', 'if'):
                self._eat()
                guard = self._parse_expr()
            if self._at('SYMBOL', '=>'):
                self._eat()
            body = self._parse_block()
            cases.append(MatchCase(pattern=pattern, guard=guard, body=body))
            self._skip_semis()
        self._expect_sym('}')
        return MatchStmt(subject=subject, cases=cases, else_body=else_body, pos=p)

    def _parse_import(self) -> ImportStmt:
        p = self._pos()
        self._eat()
        path = [self._expect_ident_or_kw().value]
        while self._at('SYMBOL', '.'):
            self._eat()
            path.append(self._expect_ident_or_kw().value)
        alias = None
        if self._at('KEYWORD', 'as'):
            self._eat()
            alias = self._expect('IDENT').value
        self._eat_semi()
        return ImportStmt(path=path, alias=alias, pos=p)

    def _parse_from_import(self) -> FromImportStmt:
        p = self._pos()
        self._eat()
        path = [self._expect('IDENT').value]
        while self._at('SYMBOL', '.'):
            self._eat()
            path.append(self._expect('IDENT').value)
        self._expect_kw('import')
        names = [self._expect('IDENT').value]
        while self._at('SYMBOL', ','):
            self._eat()
            names.append(self._expect('IDENT').value)
        self._eat_semi()
        return FromImportStmt(path=path, names=names, pos=p)

    def _parse_reg(self) -> RegStmt:
        p = self._pos()
        self._eat()
        name = self._expect('IDENT').value
        regime = None
        if self._at('SYMBOL', ':'):
            self._eat()
            regime = self._expect('IDENT').value
        value = None
        if self._at('SYMBOL', '='):
            self._eat()
            value = self._parse_expr()
        overrides = None
        if self._at('SYMBOL', '{'):
            self._eat()
            overrides = {}
            while not self._at('SYMBOL', '}') and (not self._at('EOF')):
                k = self._eat().value
                self._expect_sym(':')
                v = self._parse_expr()
                self._eat_semi()
                overrides[k] = v
            self._expect_sym('}')
        self._eat_semi()
        return RegStmt(name=name, regime=regime, value=value, overrides=overrides, pos=p)

    def _parse_entity(self) -> EntityDecl:
        p = self._pos()
        self._eat()
        name = self._expect('IDENT').value
        base = None
        if self._at('SYMBOL', ':'):
            self._eat()
            base = self._expect('IDENT').value
        self._expect_sym('{')
        fields = {}
        methods = []
        self._skip_semis()
        while not self._at('SYMBOL', '}') and (not self._at('EOF')):
            if self._at('KEYWORD', 'fn'):
                methods.append(self._parse_fn())
            elif self._at('IDENT') and self._peek().value in ('memory', 'phase'):
                k = self._eat().value
                if self._at('SYMBOL', '='):
                    self._eat()
                    fields[k] = self._parse_expr()
                else:
                    while not self._at('SYMBOL', ';') and (not self._at('SYMBOL', '}')) and (not self._at('EOF')):
                        kk = self._eat().value
                        if self._at('SYMBOL', '='):
                            self._eat()
                            fields[f'{k}_{kk}'] = self._parse_expr()
                self._eat_semi()
            else:
                k = self._eat().value
                self._expect_sym('=')
                v = self._parse_expr()
                self._eat_semi()
                fields[k] = v
            self._skip_semis()
        self._expect_sym('}')
        return EntityDecl(name=name, base=base, fields=fields, methods=methods, pos=p)

    def _parse_world(self) -> WorldDecl:
        p = self._pos()
        self._eat()
        name = self._expect('IDENT').value
        self._expect_sym('{')
        fields = {}
        entities = []
        body = []
        self._skip_semis()
        while not self._at('SYMBOL', '}') and (not self._at('EOF')):
            if self._at('KEYWORD', 'entity'):
                entities.append(self._parse_entity())
            elif self._at('KEYWORD', 'run'):
                body.append(self._parse_run())
            else:
                k = self._eat().value
                self._expect_sym('=')
                v = self._parse_expr()
                self._eat_semi()
                fields[k] = v
            self._skip_semis()
        self._expect_sym('}')
        return WorldDecl(name=name, fields=fields, entities=entities, body=body, pos=p)

    def _parse_couple(self) -> CoupleStmt:
        p = self._pos()
        self._eat()
        src = self._expect('IDENT').value
        self._expect_sym('->')
        dst = self._expect('IDENT').value
        kappa = None
        duration = None
        if self._at_ident('kappa'):
            self._eat()
            self._expect_sym('=')
            kappa = self._parse_expr()
        if self._at('KEYWORD', 'for'):
            self._eat()
            if self._at('IDENT') and self._peek().value == 'T':
                self._eat()
                self._expect_sym('=')
            duration = self._parse_expr()
        self._eat_semi()
        return CoupleStmt(src=src, dst=dst, kappa=kappa, duration=duration, pos=p)

    def _parse_pair(self) -> PairStmt:
        p = self._pos()
        self._eat()
        self._expect_sym('(')
        a = self._expect('IDENT').value
        self._expect_sym(',')
        b = self._expect('IDENT').value
        self._expect_sym(')')
        kappa = None
        duration = None
        if self._at_ident('kappa'):
            self._eat()
            self._expect_sym('=')
            kappa = self._parse_expr()
        if self._at('KEYWORD', 'for'):
            self._eat()
            if self._at('IDENT') and self._peek().value == 'T':
                self._eat()
                self._expect_sym('=')
            duration = self._parse_expr()
        self._eat_semi()
        return PairStmt(a=a, b=b, kappa=kappa, duration=duration, pos=p)

    def _parse_ring(self) -> RingStmt:
        p = self._pos()
        self._eat()
        self._expect_sym('(')
        members = [self._expect('IDENT').value]
        while self._at('SYMBOL', ','):
            self._eat()
            members.append(self._expect('IDENT').value)
        self._expect_sym(')')
        kappa = None
        duration = None
        if self._at_ident('kappa'):
            self._eat()
            self._expect_sym('=')
            kappa = self._parse_expr()
        if self._at('KEYWORD', 'for'):
            self._eat()
            if self._at('IDENT') and self._peek().value == 'T':
                self._eat()
                self._expect_sym('=')
            duration = self._parse_expr()
        self._eat_semi()
        return RingStmt(members=members, kappa=kappa, duration=duration, pos=p)

    def _parse_observe(self) -> ObserveStmt:
        p = self._pos()
        self._eat()
        target = self._expect('IDENT').value
        metrics = []
        if not self._at('SYMBOL', ';') and (not self._at('EOF')):
            metrics.append(self._expect('IDENT').value)
            while self._at('SYMBOL', ','):
                self._eat()
                if self._at('KEYWORD', 'over_seeds'):
                    break
                metrics.append(self._expect('IDENT').value)
        over_seeds = 1
        if self._at('KEYWORD', 'over_seeds'):
            self._eat()
            self._expect_sym('=')
            over_seeds = int(self._expect('NUMBER').value)
        self._eat_semi()
        return ObserveStmt(target=target, metrics=metrics, over_seeds=over_seeds, pos=p)

    def _parse_run(self) -> RunStmt:
        p = self._pos()
        self._eat()
        dur = None
        if not self._at('SYMBOL', ';') and (not self._at('SYMBOL', '}')) and (not self._at('EOF')):
            if self._at('IDENT') and self._peek().value == 'T':
                self._eat()
                self._expect_sym('=')
            dur = self._parse_expr()
        self._eat_semi()
        return RunStmt(duration=dur, pos=p)

    def _parse_evolve(self) -> RunStmt:
        p = self._pos()
        self._eat()
        if self._at('IDENT'):
            self._eat()
        dur = None
        if self._at('KEYWORD', 'for'):
            self._eat()
        if not self._at('SYMBOL', ';') and (not self._at('SYMBOL', '}')) and (not self._at('EOF')):
            if self._at('IDENT') and self._peek().value == 'T':
                self._eat()
                self._expect_sym('=')
            dur = self._parse_expr()
        self._eat_semi()
        return RunStmt(duration=dur, pos=p)

    def _parse_sequence(self) -> 'SequenceStmt':
        from frontend.ast_nodes import SequenceStmt
        p = self._pos()
        self._eat()
        inputs = self._parse_expr()
        self._expect_kw('via')
        target = self._expect('IDENT').value
        each_for = None
        if self._at('KEYWORD', 'each_for'):
            self._eat()
            if self._at('IDENT') and self._peek().value == 'T':
                self._eat()
                self._expect_sym('=')
            each_for = self._parse_expr()
        self._eat_semi()
        return SequenceStmt(inputs=inputs, target=target, each_for=each_for, pos=p)

    def _parse_try(self) -> 'TryCatchStmt':
        from frontend.ast_nodes import TryCatchStmt
        p = self._pos()
        self._eat()
        body = self._parse_block()
        catch_var = None
        catch_body = []
        finally_body = []
        if self._at('KEYWORD', 'catch'):
            self._eat()
            if self._at('IDENT'):
                catch_var = self._peek().value
                self._eat()
            catch_body = self._parse_block()
        if self._at('KEYWORD', 'finally'):
            self._eat()
            finally_body = self._parse_block()
        return TryCatchStmt(body=body, catch_var=catch_var, catch_body=catch_body, finally_body=finally_body, pos=p)

    def _parse_throw(self) -> 'ThrowStmt':
        from frontend.ast_nodes import ThrowStmt
        p = self._pos()
        self._eat()
        value = None
        if not self._at('SYMBOL', ';') and (not self._at('SYMBOL', '}')) and (not self._at('EOF')):
            value = self._parse_expr()
        self._eat_semi()
        return ThrowStmt(value=value, pos=p)

    def _parse_annotation(self) -> 'AnnotationStmt':
        from frontend.ast_nodes import AnnotationStmt
        p = self._pos()
        self._eat()
        name = self._peek().value
        self._eat()
        args = ''
        if self._at('SYMBOL', '('):
            self._eat()
            parts = []
            while not self._at('SYMBOL', ')') and (not self._at('EOF')):
                parts.append(self._peek().value)
                self._eat()
            self._expect_sym(')')
            args = ''.join(parts)
        self._eat_semi()
        return AnnotationStmt(key=name, args=args, pos=p)

    def _parse_expr(self) -> Expr:
        return self._parse_or()

    def _parse_or(self) -> Expr:
        left = self._parse_and()
        while self._at('KEYWORD', 'or'):
            self._eat()
            right = self._parse_and()
            left = BinOp('or', left, right, pos=left.pos if hasattr(left, 'pos') else self._pos())
        return left

    def _parse_and(self) -> Expr:
        left = self._parse_not()
        while self._at('KEYWORD', 'and'):
            self._eat()
            right = self._parse_not()
            left = BinOp('and', left, right, pos=left.pos if hasattr(left, 'pos') else self._pos())
        return left

    def _parse_not(self) -> Expr:
        if self._at('KEYWORD', 'not'):
            p = self._pos()
            self._eat()
            return UnaryOp('not', self._parse_not(), pos=p)
        return self._parse_comparison()

    def _parse_comparison(self) -> Expr:
        left = self._parse_add()
        while self._at('SYMBOL') and self._peek().value in ('==', '!=', '<', '<=', '>', '>='):
            op = self._eat().value
            right = self._parse_add()
            left = BinOp(op, left, right, pos=left.pos if hasattr(left, 'pos') else self._pos())
        return left

    def _parse_add(self) -> Expr:
        left = self._parse_mul()
        while self._at('SYMBOL') and self._peek().value in ('+', '-'):
            op = self._eat().value
            right = self._parse_mul()
            left = BinOp(op, left, right, pos=left.pos if hasattr(left, 'pos') else self._pos())
        return left

    def _parse_mul(self) -> Expr:
        left = self._parse_power()
        while self._at('SYMBOL') and self._peek().value in ('*', '/', '%', '@'):
            op = self._eat().value
            right = self._parse_power()
            left = BinOp(op, left, right, pos=left.pos if hasattr(left, 'pos') else self._pos())
        return left

    def _parse_power(self) -> Expr:
        left = self._parse_unary()
        if self._at('SYMBOL', '**'):
            self._eat()
            right = self._parse_power()
            left = BinOp('**', left, right, pos=left.pos if hasattr(left, 'pos') else self._pos())
        return left

    def _parse_unary(self) -> Expr:
        if self._at('KEYWORD', 'await'):
            p = self._pos()
            self._eat()
            return AwaitExpr(value=self._parse_unary(), pos=p)
        if self._at('SYMBOL', '-'):
            p = self._pos()
            self._eat()
            return UnaryOp('-', self._parse_unary(), pos=p)
        if self._at('SYMBOL', '!'):
            p = self._pos()
            self._eat()
            return UnaryOp('not', self._parse_unary(), pos=p)
        return self._parse_postfix()

    def _parse_postfix(self) -> Expr:
        expr = self._parse_primary()
        while True:
            if self._at('SYMBOL', '('):
                self._eat()
                args, kwargs = self._parse_call_args()
                self._expect_sym(')')
                expr = CallExpr(func=expr, args=args, kwargs=kwargs, pos=expr.pos if hasattr(expr, 'pos') else self._pos())
            elif self._at('SYMBOL', '['):
                self._eat()
                p = self._pos()
                if self._at('SYMBOL', ':'):
                    self._eat()
                    end = None if self._at('SYMBOL', ']') or self._at('SYMBOL', ':') else self._parse_expr()
                    step = None
                    if self._at('SYMBOL', ':'):
                        self._eat()
                        step = None if self._at('SYMBOL', ']') else self._parse_expr()
                    self._expect_sym(']')
                    idx = SliceExpr(start=None, end=end, step=step, pos=p)
                    expr = IndexExpr(obj=expr, index=idx, pos=expr.pos if hasattr(expr, 'pos') else p)
                else:
                    first = self._parse_expr()
                    if self._at('SYMBOL', ':'):
                        self._eat()
                        end = None if self._at('SYMBOL', ']') or self._at('SYMBOL', ':') else self._parse_expr()
                        step = None
                        if self._at('SYMBOL', ':'):
                            self._eat()
                            step = None if self._at('SYMBOL', ']') else self._parse_expr()
                        self._expect_sym(']')
                        idx = SliceExpr(start=first, end=end, step=step, pos=p)
                        expr = IndexExpr(obj=expr, index=idx, pos=expr.pos if hasattr(expr, 'pos') else p)
                    else:
                        self._expect_sym(']')
                        expr = IndexExpr(obj=expr, index=first, pos=expr.pos if hasattr(expr, 'pos') else p)
            elif self._at('SYMBOL', '.'):
                self._eat()
                field_name = self._expect('IDENT').value
                if self._at('SYMBOL', '('):
                    self._eat()
                    args, kwargs = self._parse_call_args()
                    self._expect_sym(')')
                    expr = MethodCallExpr(obj=expr, method=field_name, args=args, kwargs=kwargs, pos=expr.pos if hasattr(expr, 'pos') else self._pos())
                else:
                    expr = FieldExpr(obj=expr, field=field_name, pos=expr.pos if hasattr(expr, 'pos') else self._pos())
            else:
                break
        return expr

    def _parse_call_args(self) -> tuple[list[Expr], dict[str, Expr]]:
        args = []
        kwargs = {}
        while not self._at('SYMBOL', ')') and (not self._at('EOF')):
            if self._at('SYMBOL', '**'):
                p = self._pos()
                self._eat()
                args.append(UnaryOp('**', self._parse_expr(), p))
            elif self._at('SYMBOL', '*'):
                p = self._pos()
                self._eat()
                args.append(UnaryOp('*', self._parse_expr(), p))
            elif self._at('IDENT') and self._peek(1).kind == 'SYMBOL' and (self._peek(1).value == '='):
                name = self._eat().value
                self._eat()
                kwargs[name] = self._parse_expr()
            else:
                args.append(self._parse_expr())
            if not self._at('SYMBOL', ')'):
                self._expect_sym(',')
        return (args, kwargs)

    def _parse_primary(self) -> Expr:
        t = self._peek()
        p = self._pos()
        if t.kind == 'NUMBER':
            self._eat()
            v = t.value
            if '.' in v or 'e' in v or 'E' in v:
                return FloatLit(float(v), p)
            if v.startswith('0x') or v.startswith('0X'):
                return IntLit(int(v, 16), p)
            if v.startswith('0b') or v.startswith('0B'):
                return IntLit(int(v, 2), p)
            return IntLit(int(v), p)
        if t.kind == 'STRING':
            self._eat()
            return StringLit(t.value, p)
        if t.kind == 'FSTRING':
            from frontend.ast_nodes import FStringExpr
            self._eat()
            parsed_parts = []
            for ptype, pval in t.value:
                if ptype == 'str':
                    parsed_parts.append(('str', pval))
                else:
                    from frontend.lexer_universal import tokenize
                    sub_toks = tokenize(pval, '<fstring>')
                    sub_parser = Parser(sub_toks, '<fstring>')
                    sub_expr = sub_parser._parse_expr()
                    parsed_parts.append(('expr', sub_expr))
            return FStringExpr(parts=parsed_parts, pos=p)
        if t.kind == 'KEYWORD':
            if t.value == 'true':
                self._eat()
                return BoolLit(True, p)
            if t.value == 'false':
                self._eat()
                return BoolLit(False, p)
            if t.value == 'none':
                self._eat()
                return NoneLit(p)
            if t.value == 'fn':
                return self._parse_lambda()
            if t.value == 'self':
                self._eat()
                return Ident('self', p)
            if t.value == 'yield':
                self._eat()
                val = None
                if not self._at('SYMBOL', ';') and (not self._at('SYMBOL', '}')) and (not self._at('SYMBOL', ')')) and (not self._at('EOF')):
                    val = self._parse_expr()
                return YieldExpr(value=val, pos=p)
            if t.value == 'await':
                self._eat()
                return AwaitExpr(value=self._parse_unary(), pos=p)
        if t.kind == 'IDENT':
            self._eat()
            return Ident(t.value, p)
        if t.kind == 'SYMBOL' and t.value == '(':
            self._eat()
            if self._at('SYMBOL', ')'):
                self._eat()
                from frontend.ast_nodes import TupleExpr
                return TupleExpr(elements=[], pos=p)
            first = self._parse_expr()
            if self._at('SYMBOL', ','):
                from frontend.ast_nodes import TupleExpr
                elems = [first]
                while self._at('SYMBOL', ','):
                    self._eat()
                    if self._at('SYMBOL', ')'):
                        break
                    elems.append(self._parse_expr())
                self._expect_sym(')')
                return TupleExpr(elements=elems, pos=p)
            self._expect_sym(')')
            return first
        if t.kind == 'SYMBOL' and t.value == '[':
            self._eat()
            if self._at('SYMBOL', ']'):
                self._eat()
                return ListExpr([], p)
            first = self._parse_expr()
            if self._at('KEYWORD', 'for'):
                from frontend.ast_nodes import ListCompExpr
                self._eat()
                var = self._expect('IDENT').value
                self._expect('KEYWORD', 'in')
                iter_expr = self._parse_expr()
                cond = None
                if self._at('KEYWORD', 'if'):
                    self._eat()
                    cond = self._parse_expr()
                self._expect_sym(']')
                return ListCompExpr(expr=first, var=var, iter=iter_expr, condition=cond, pos=p)
            elems = [first]
            while self._at('SYMBOL', ','):
                self._eat()
                if self._at('SYMBOL', ']'):
                    break
                elems.append(self._parse_expr())
            self._expect_sym(']')
            return ListExpr(elems, p)
        if t.kind == 'SYMBOL' and t.value == '{':
            self._eat()
            pairs = []
            while not self._at('SYMBOL', '}') and (not self._at('EOF')):
                key = self._parse_expr()
                self._expect_sym(':')
                val = self._parse_expr()
                pairs.append((key, val))
                if not self._at('SYMBOL', '}'):
                    self._expect_sym(',')
            self._expect_sym('}')
            return MapExpr(pairs, p)
        raise ParseError(f'unexpected token {t.kind} {t.value!r}', t.line, t.col, self.file)

    def _parse_lambda(self) -> LambdaExpr:
        p = self._pos()
        self._eat()
        self._expect_sym('(')
        params = self._parse_params()
        self._expect_sym(')')
        body = self._parse_block()
        return LambdaExpr(params=params, body=body, pos=p)

def parse(source: str, file: str='') -> Module:
    tokens = tokenize(source, file)
    return Parser(tokens, file).parse_module()