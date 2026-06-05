/* triad_ast.c — AST kind names + tiny helpers. */
#include "triad_frontend.h"

const char *triad_ast_kind_name(TriadAstKind k) {
    switch (k) {
        case TRIAD_AST_INT_LIT:        return "IntLit";
        case TRIAD_AST_FLOAT_LIT:      return "FloatLit";
        case TRIAD_AST_BOOL_LIT:       return "BoolLit";
        case TRIAD_AST_STRING_LIT:     return "StringLit";
        case TRIAD_AST_NONE_LIT:       return "NoneLit";
        case TRIAD_AST_IDENT:          return "Ident";
        case TRIAD_AST_BINOP:          return "BinOp";
        case TRIAD_AST_UNARYOP:        return "UnaryOp";
        case TRIAD_AST_CALL:           return "CallExpr";
        case TRIAD_AST_INDEX:          return "IndexExpr";
        case TRIAD_AST_SLICE:          return "SliceExpr";
        case TRIAD_AST_FIELD:          return "FieldExpr";
        case TRIAD_AST_LIST:           return "ListExpr";
        case TRIAD_AST_LIST_COMP:      return "ListCompExpr";
        case TRIAD_AST_MAP:            return "MapExpr";
        case TRIAD_AST_TUPLE:          return "TupleExpr";
        case TRIAD_AST_FSTRING:        return "FStringExpr";
        case TRIAD_AST_LAMBDA:         return "LambdaExpr";
        case TRIAD_AST_METHOD_CALL:    return "MethodCallExpr";
        case TRIAD_AST_ASSIGN_EXPR:    return "AssignExpr";
        case TRIAD_AST_YIELD_EXPR:     return "YieldExpr";
        case TRIAD_AST_AWAIT_EXPR:     return "AwaitExpr";
        case TRIAD_AST_LET:            return "LetStmt";
        case TRIAD_AST_DESTRUCT_LET:   return "DestructLetStmt";
        case TRIAD_AST_MAP_DESTRUCT:   return "MapDestructStmt";
        case TRIAD_AST_CONST:          return "ConstStmt";
        case TRIAD_AST_ASSIGN:         return "AssignStmt";
        case TRIAD_AST_EXPR_STMT:      return "ExprStmt";
        case TRIAD_AST_RETURN:         return "ReturnStmt";
        case TRIAD_AST_BREAK:          return "BreakStmt";
        case TRIAD_AST_CONTINUE:       return "ContinueStmt";
        case TRIAD_AST_IF:             return "IfStmt";
        case TRIAD_AST_FOR:            return "ForStmt";
        case TRIAD_AST_WHILE:          return "WhileStmt";
        case TRIAD_AST_FN_DECL:        return "FnDecl";
        case TRIAD_AST_TRY_CATCH:      return "TryCatchStmt";
        case TRIAD_AST_THROW:          return "ThrowStmt";
        case TRIAD_AST_TYPE_DECL:      return "TypeDecl";
        case TRIAD_AST_CLASS_DECL:     return "ClassDecl";
        case TRIAD_AST_IMPORT:         return "ImportStmt";
        case TRIAD_AST_FROM_IMPORT:    return "FromImportStmt";
        case TRIAD_AST_MATCH:          return "MatchStmt";
        case TRIAD_AST_YIELD_STMT:     return "YieldStmt";
        case TRIAD_AST_REG:            return "RegStmt";
        case TRIAD_AST_ENTITY:         return "EntityDecl";
        case TRIAD_AST_WORLD:          return "WorldDecl";
        case TRIAD_AST_COUPLE:         return "CoupleStmt";
        case TRIAD_AST_PAIR:           return "PairStmt";
        case TRIAD_AST_RING:           return "RingStmt";
        case TRIAD_AST_OBSERVE:        return "ObserveStmt";
        case TRIAD_AST_RUN:            return "RunStmt";
        case TRIAD_AST_ANNOTATION:     return "AnnotationStmt";
        case TRIAD_AST_MODULE:         return "Module";
        default:                       return "Unknown";
    }
}
