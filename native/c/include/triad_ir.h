#ifndef TRIAD_IR_H
#define TRIAD_IR_H

#include "triad_frontend.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {

    TRIAD_IR_INT,
    TRIAD_IR_FLOAT,
    TRIAD_IR_BOOL,
    TRIAD_IR_STRING,
    TRIAD_IR_NONE,
    TRIAD_IR_IDENT,

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
    TRIAD_IR_WITH,
    TRIAD_IR_ASSERT,
    TRIAD_IR_PASS,
    TRIAD_IR_DEL,
    TRIAD_IR_ASYNC_FOR,
    TRIAD_IR_ASYNC_WITH,
    TRIAD_IR_TYPE_DECL,
    TRIAD_IR_ENTITY_DECL,
    TRIAD_IR_WORLD_DECL,
    TRIAD_IR_SUBSTRATE_DECL,
    TRIAD_IR_REG_DECL,
    TRIAD_IR_OBSERVE,
    TRIAD_IR_RUN,
    TRIAD_IR_IMPORT,
    TRIAD_IR_DESTRUCT_LET,
    TRIAD_IR_CLASS_DECL,
    TRIAD_IR_YIELD,
    TRIAD_IR_COMPLEX,
    TRIAD_IR_BYTES,
    TRIAD_IR_TUPLE,
    TRIAD_IR_SET,
    TRIAD_IR_TERNARY,
    TRIAD_IR_CHAIN_CMP,
    TRIAD_IR_SUPER,
    TRIAD_IR_DICT_COMP,
    TRIAD_IR_SET_COMP,
    TRIAD_IR_GEN_COMP,
    TRIAD_IR_YIELD_EXPR,
    TRIAD_IR_AWAIT,
    TRIAD_IR_COMP_CLAUSE,
    TRIAD_IR_COUPLE,
    TRIAD_IR_PAIR,
    TRIAD_IR_RING,
    TRIAD_IR_SEQUENCE,
    TRIAD_IR_ANNOTATION,

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

typedef struct {
    const char  *text;
    TriadIRNode *expr;
} TriadIRFStringPart;

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
            const char   *method;
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

            struct {
                const char  *var;
                TriadIRNode *iter;
                TriadIRNode **conditions;
                size_t       conditions_len;
            } *clauses;
            size_t clauses_len;
        } list_comp;

        struct {
            double real_val;
            double imag_val;
        } complex_lit;

        struct {
            const char *bytes_val;
            size_t      bytes_len;
        } bytes_lit;

        struct { TriadIRNode **elements; size_t elements_len; } tuple_lit;
        struct { TriadIRNode **elements; size_t elements_len; } set_lit;

        struct {
            TriadIRNode *condition;
            TriadIRNode *then_val;
            TriadIRNode *else_val;
        } ternary;

        struct {
            TriadIRNode **operands;
            size_t       operands_len;
            const char  **ops;
            size_t       ops_len;
        } chain_cmp;

        struct {
            TriadIRNode **args;
            size_t       args_len;
        } super_expr;

        struct {
            TriadIRNode *key_expr;
            TriadIRNode *value_expr;
            struct {
                const char  *var;
                TriadIRNode *iter;
                TriadIRNode **conditions;
                size_t       conditions_len;
            } *clauses;
            size_t clauses_len;
        } dict_comp;

        struct {
            TriadIRNode *expr;
            struct {
                const char  *var;
                TriadIRNode *iter;
                TriadIRNode **conditions;
                size_t       conditions_len;
            } *clauses;
            size_t clauses_len;
        } set_comp;

        struct {
            TriadIRNode *expr;
            struct {
                const char  *var;
                TriadIRNode *iter;
                TriadIRNode **conditions;
                size_t       conditions_len;
            } *clauses;
            size_t clauses_len;
        } gen_comp;

        struct { TriadIRNode *value; } yield_expr;
        struct { TriadIRNode *value; } await_expr;

        struct { TriadIRNode *target; TriadIRNode *value; } assign_expr;

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
        struct { TriadIRNode *value; }                      ret_or_throw;

        struct {
            TriadIRNode       *condition;
            TriadIRNode      **then_body;
            size_t             then_body_len;
            TriadIRElifClause *elif_clauses;
            size_t             elif_clauses_len;
            TriadIRNode      **else_body;
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
            const char   *name;
            const char  **params;
            size_t        params_len;
            TriadIRNode **body;
            size_t        body_len;
        } func;

        struct {
            TriadIRNode **body;
            size_t        body_len;
            const char   *catch_var;
            TriadIRNode **catch_body;
            size_t        catch_body_len;
            int           has_catch;
            TriadIRNode **finally_body;
            size_t        finally_body_len;
            int           has_finally;
        } try_catch;

        struct {
            TriadIRNode  *expr;
            const char   *var;
            TriadIRNode **body;
            size_t        body_len;
        } with_stmt;

        struct {
            TriadIRNode *condition;
            TriadIRNode *message;
        } assert_stmt;

        struct {
            TriadIRNode *target;
        } del_stmt;

        struct {
            const char        *name;
            TriadIRTypeField  *fields;
            size_t             fields_len;
        } type_decl;

        struct {
            const char       *name;
            const char       *base;
            TriadIRStrEntry  *fields;
            size_t            fields_len;
        } entity_decl;

        struct {
            const char       *name;
            TriadIRStrEntry  *fields;
            size_t            fields_len;
            TriadIRNode     **entities;
            size_t            entities_len;
        } world_decl;

        struct {
            const char       *name;
            const char       *regime;
            const char      **members;
            size_t            members_len;
            TriadIRStrEntry  *properties;
            size_t            properties_len;
            TriadIRStrEntry  *overrides;
            size_t            overrides_len;
            int               is_composed;
        } substrate_decl;

        struct {
            const char  *name;
            const char  *regime;
            TriadIRNode *value;
        } reg_decl;

        struct {
            const char  *target;
            const char **metrics;
            size_t       metrics_len;
        } observe;

        struct {
            TriadIRNode *duration;
            const char  *target;
        } run;

        struct {
            const char  *src;
            const char  *dst;
            TriadIRNode *kappa;
            TriadIRNode *duration;
        } couple;

        struct {
            const char  *a;
            const char  *b;
            TriadIRNode *kappa;
            TriadIRNode *duration;
        } pair;

        struct {
            const char **members;
            size_t       members_len;
            TriadIRNode *kappa;
            TriadIRNode *duration;
        } ring;

        struct {
            const char *key;
            const char *args;
        } annotation;

        struct {
            TriadIRNode *inputs;
            const char  *target;
            TriadIRNode *each_for;
        } sequence;

        struct {
            const char **path;
            size_t       path_len;
            const char  *alias;
            const char **names;
            size_t       names_len;
        } import_stmt;

        struct {
            const char **names;
            size_t       names_len;
            TriadIRNode *value;
        } destruct_let;

        struct {
            const char  *name;
            const char  *parent;
            const char **fields;
            size_t       fields_len;
            TriadIRNode **methods;
            size_t       methods_len;
        } class_decl;

        struct {
            const char    *name;
            TriadIRNode  **imports;
            size_t         imports_len;
            TriadIRNode  **body;
            size_t         body_len;
        } module;
    } u;
};

TriadIRNode *triad_ir_lower_module(TriadArena *arena,
                                   const TriadAstNode *module,
                                   TriadDiag *diag);

const char *triad_ir_kind_name(TriadIRKind k);

char *triad_ir_dump_json(const TriadIRNode *module, int indent);
int   triad_ir_dump_json_fp(const TriadIRNode *module, FILE *fp, int indent);

#ifdef __cplusplus
}
#endif

#endif
