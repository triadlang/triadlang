/*
 * triad_ir.h — Native TriadLang IR (mirror of compiler/ir.py) +
 * lowering from the universal AST (mirror of compiler/lower.py).
 *
 * Ownership: every TriadIRNode and inline array/string returned by
 * this API is bump-allocated from the TriadArena passed in. Freeing
 * the arena frees the whole IRModule. Do NOT free individual nodes.
 *
 * Concurrency: not thread-safe. One arena per lowering pass.
 */
#ifndef TRIAD_IR_H
#define TRIAD_IR_H

#include "triad_frontend.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    /* literals & primitives */
    TRIAD_IR_INT,
    TRIAD_IR_FLOAT,
    TRIAD_IR_BOOL,
    TRIAD_IR_STRING,
    TRIAD_IR_NONE,
    TRIAD_IR_IDENT,

    /* ops & access */
    TRIAD_IR_BINOP,
    TRIAD_IR_UNARYOP,
    TRIAD_IR_CALL,
    TRIAD_IR_METHOD_CALL,
    TRIAD_IR_INDEX,
    TRIAD_IR_SLICE,
    TRIAD_IR_FIELD,
    TRIAD_IR_LIST,
    TRIAD_IR_MAP,
    TRIAD_IR_FSTRING,
    TRIAD_IR_LIST_COMP,
    TRIAD_IR_ASSIGN_EXPR,

    /* statements */
    TRIAD_IR_LET,
    TRIAD_IR_CONST,
    TRIAD_IR_ASSIGN,
    TRIAD_IR_EXPR_STMT,
    TRIAD_IR_RETURN,
    TRIAD_IR_BREAK,
    TRIAD_IR_CONTINUE,
    TRIAD_IR_IF,
    TRIAD_IR_FOR,
    TRIAD_IR_WHILE,
    TRIAD_IR_FUNCTION,
    TRIAD_IR_TRY_CATCH,
    TRIAD_IR_THROW,
    TRIAD_IR_TYPE_DECL,
    TRIAD_IR_ENTITY_DECL,
    TRIAD_IR_WORLD_DECL,
    TRIAD_IR_REG_DECL,
    TRIAD_IR_OBSERVE,
    TRIAD_IR_RUN,
    TRIAD_IR_IMPORT,
    TRIAD_IR_DESTRUCT_LET,
    TRIAD_IR_CLASS_DECL,
    TRIAD_IR_YIELD,

    /* top level */
    TRIAD_IR_MODULE,

    TRIAD_IR_KIND_COUNT
} TriadIRKind;

typedef struct TriadIRNode TriadIRNode;

typedef struct {
    const char  *name;
    TriadIRNode *value;
} TriadIRKwArg;

typedef struct {
    TriadIRNode *key;
    TriadIRNode *value;
} TriadIRMapPair;

typedef struct {
    const char  *key;
    TriadIRNode *value;
} TriadIRStrEntry;

typedef struct {
    TriadIRNode  *cond;
    TriadIRNode **body;
    size_t        body_len;
} TriadIRElifClause;

/* IRFString.parts: list[tuple[str, Optional[IRNode]]].
 * If `expr` is non-NULL the part is an interpolation; otherwise `text`
 * is a literal piece. Matches IRFString in compiler/ir.py. */
typedef struct {
    const char  *text;     /* "" when expr != NULL */
    TriadIRNode *expr;     /* NULL when text-only   */
} TriadIRFStringPart;

/* IRTypeDecl.fields: list[tuple[str, str]] = (name, type_ann). */
typedef struct {
    const char *name;
    const char *type_ann;
} TriadIRTypeField;

struct TriadIRNode {
    TriadIRKind kind;
    union {
        long long           int_val;
        double              float_val;
        int                 bool_val;
        const char         *string_val;
        const char         *ident_name;

        struct {
            const char  *op;
            TriadIRNode *left;
            TriadIRNode *right;
        } binop;

        struct {
            const char  *op;
            TriadIRNode *operand;
        } unaryop;

        struct {
            TriadIRNode  *func_or_obj;
            const char   *method;    /* method-call only */
            TriadIRNode **args;
            size_t        args_len;
            TriadIRKwArg *kwargs;
            size_t        kwargs_len;
        } call;

        struct { TriadIRNode *obj; TriadIRNode *index; } index;
        struct {
            TriadIRNode *obj;
            TriadIRNode *start;
            TriadIRNode *end;
            TriadIRNode *step;
        } slice;
        struct { TriadIRNode *obj; const char *field; } field;

        struct { TriadIRNode **elements; size_t elements_len; } list;
        struct { TriadIRMapPair *pairs; size_t pairs_len; } map;

        struct {
            TriadIRFStringPart *parts;
            size_t              parts_len;
        } fstring;

        struct {
            TriadIRNode *expr;
            const char  *var;
            TriadIRNode *iter;
            TriadIRNode *condition;
        } list_comp;

        struct { TriadIRNode *target; TriadIRNode *value; } assign_expr;

        /* statements */
        struct {
            const char  *name;
            const char  *type_ann;
            TriadIRNode *value;
        } let_stmt;
        struct {
            const char  *name;
            TriadIRNode *value;
        } const_stmt;
        struct { TriadIRNode *target; TriadIRNode *value; } assign_stmt;
        struct { TriadIRNode *expr; }                       expr_stmt;
        struct { TriadIRNode *value; }                      ret_or_throw;  /* for IRReturn/IRThrow/IRYield */

        struct {
            TriadIRNode       *condition;
            TriadIRNode      **then_body;
            size_t             then_body_len;
            TriadIRElifClause *elif_clauses;
            size_t             elif_clauses_len;
            TriadIRNode      **else_body;          /* NULL = absent */
            size_t             else_body_len;
            int                has_else;
        } if_stmt;

        struct {
            const char   *var;
            TriadIRNode  *iter;
            TriadIRNode **body;
            size_t        body_len;
        } for_stmt;

        struct {
            TriadIRNode  *condition;
            TriadIRNode **body;
            size_t        body_len;
        } while_stmt;

        struct {
            const char   *name;       /* "" for lambdas */
            const char  **params;
            size_t        params_len;
            TriadIRNode **body;
            size_t        body_len;
        } func;

        struct {
            TriadIRNode **body;
            size_t        body_len;
            const char   *catch_var;
            TriadIRNode **catch_body;          /* NULL = absent */
            size_t        catch_body_len;
            int           has_catch;
            TriadIRNode **finally_body;        /* NULL = absent */
            size_t        finally_body_len;
            int           has_finally;
        } try_catch;

        struct {
            const char        *name;
            TriadIRTypeField  *fields;
            size_t             fields_len;
        } type_decl;

        struct {
            const char       *name;
            const char       *base;            /* nullable */
            TriadIRStrEntry  *fields;
            size_t            fields_len;
        } entity_decl;

        struct {
            const char       *name;
            TriadIRStrEntry  *fields;
            size_t            fields_len;
            TriadIRNode     **entities;        /* each is an IREntityDecl node */
            size_t            entities_len;
        } world_decl;

        struct {
            const char  *name;
            const char  *regime;       /* nullable */
            TriadIRNode *value;        /* nullable */
        } reg_decl;

        struct {
            const char  *target;
            const char **metrics;
            size_t       metrics_len;
        } observe;

        struct {
            TriadIRNode *duration;     /* nullable */
        } run;

        struct {
            const char **path;
            size_t       path_len;
            const char  *alias;        /* nullable */
        } import_stmt;

        struct {
            const char **names;
            size_t       names_len;
            TriadIRNode *value;
        } destruct_let;

        struct {
            const char  *name;
            const char  *parent;       /* nullable */
            const char **fields;       /* list of field names only */
            size_t       fields_len;
            TriadIRNode **methods;     /* each is an IRFunction node */
            size_t       methods_len;
        } class_decl;

        struct {
            const char    *name;
            TriadIRNode  **imports;    /* each is an IRImport node */
            size_t         imports_len;
            TriadIRNode  **body;
            size_t         body_len;
        } module;
    } u;
};

/* ── Lowering ───────────────────────────────────────────────────── */

/* Lower an AST module into an IR module. Returns NULL on error;
 * `diag` is filled if non-NULL. */
TriadIRNode *triad_ir_lower_module(TriadArena *arena,
                                   const TriadAstNode *module,
                                   TriadDiag *diag);

const char *triad_ir_kind_name(TriadIRKind k);

/* ── JSON dump ──────────────────────────────────────────────────── */

/* Schema matches compiler/emit_json.py: every dataclass becomes
 *   {"_type": "<ClassName>", "<field>": ...}
 * Tuples become arrays. Used by scripts/ir_to_json.py + native parity. */
char *triad_ir_dump_json(const TriadIRNode *module, int indent);
int   triad_ir_dump_json_fp(const TriadIRNode *module, FILE *fp, int indent);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TRIAD_IR_H */
