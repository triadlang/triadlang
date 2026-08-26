from __future__ import annotations

__all__ = [
    'parse', 'tokenize', 'Token', 'LexError', 'ParseError',
    'Module', 'Pos',
    'IntLit', 'FloatLit', 'ComplexLit', 'BoolLit', 'StringLit', 'BytesLit', 'NoneLit',
    'Ident', 'BinOp', 'UnaryOp', 'CallExpr', 'IndexExpr', 'SliceExpr',
    'FieldExpr', 'ListExpr', 'ListCompExpr', 'MapExpr', 'TupleExpr',
    'FStringExpr', 'LambdaExpr', 'MethodCallExpr', 'AssignExpr',
    'TernaryExpr', 'SetExpr', 'DictCompExpr', 'SetCompExpr', 'GenCompExpr',
    'SuperExpr', 'ChainCmpExpr', 'YieldExpr', 'AwaitExpr',
    'Param', 'CompClause',
    'LetStmt', 'DestructLetStmt', 'MapDestructStmt', 'ConstStmt',
    'AssignStmt', 'ExprStmt', 'ReturnStmt', 'BreakStmt', 'ContinueStmt',
    'IfStmt', 'ForStmt', 'WhileStmt', 'FnDecl', 'TypeDecl', 'ClassDecl',
    'ImportStmt', 'FromImportStmt', 'TryCatchStmt', 'ThrowStmt',
    'MatchStmt', 'MatchCase', 'YieldStmt', 'PassStmt', 'AssertStmt', 'DelStmt',
    'WithStmt', 'RegStmt', 'EntityDecl', 'WorldDecl', 'SubstrateDecl',
    'CoupleStmt', 'PairStmt', 'RingStmt', 'ObserveStmt', 'RunStmt',
    'SequenceStmt', 'AnnotationStmt',
    'AsyncForStmt', 'AsyncWithStmt',
    'CatchBlock', 'TypeField',
    'GenericType', 'UnionType', 'OptionalType',
    'ListPattern', 'DictPattern', 'ClassPattern', 'OrPattern',
]

def __getattr__(name):
    if name in __all__:
        if name in ('parse', 'ParseError'):
            from frontend.errors import ParseError
            from frontend.parser_universal import parse
            return locals()[name]
        if name in ('tokenize', 'Token', 'LexError'):
            from frontend.errors import LexError
            from frontend.lexer_universal import Token, tokenize
            return locals()[name]
        import frontend.ast_nodes as _ast
        from frontend.ast_nodes import Module, Pos
        return getattr(_ast, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
