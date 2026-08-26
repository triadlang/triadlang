#ifndef TRIAD_FRONTEND_H
#define TRIAD_FRONTEND_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TriadArenaBlock TriadArenaBlock;

typedef struct TriadArena {
    TriadArenaBlock *head;
    size_t          block_size;
    size_t          total_bytes;
} TriadArena;

void   triad_arena_init(TriadArena *a, size_t block_size);
void  *triad_arena_alloc(TriadArena *a, size_t n);
void  *triad_arena_calloc(TriadArena *a, size_t n);
char  *triad_arena_strdup(TriadArena *a, const char *s);
char  *triad_arena_strndup(TriadArena *a, const char *s, size_t n);
void   triad_arena_free(TriadArena *a);

typedef struct {
    int          line;
    int          col;
    const char  *file;
} TriadPos;

typedef struct {
    int          line;
    int          col;
    const char  *file;
    const char  *kind;
    const char  *msg;
} TriadDiag;

typedef enum {
    TRIAD_TOK_EOF = 0,
    TRIAD_TOK_NUMBER,
    TRIAD_TOK_STRING,
    TRIAD_TOK_FSTRING,
    TRIAD_TOK_KEYWORD,
    TRIAD_TOK_IDENT,
    TRIAD_TOK_SYMBOL
} TriadTokenKind;

typedef struct {
    int          is_expr;
    const char  *text;
    const char  *fmt_spec;
} TriadFStringPart;

typedef struct {
    TriadTokenKind     kind;
    const char        *text;

    TriadFStringPart  *parts;
    size_t             parts_len;
    int                line;
    int                col;
} TriadToken;

typedef struct {
    TriadToken *items;
    size_t      len;
} TriadTokenList;

int triad_tokenize(TriadArena    *arena,
                   const char    *src,
                   const char    *file,
                   TriadTokenList *out,
                   TriadDiag      *diag);

typedef enum {

    TRIAD_AST_INT_LIT,
    TRIAD_AST_FLOAT_LIT,
    TRIAD_AST_BOOL_LIT,
    TRIAD_AST_STRING_LIT,
    TRIAD_AST_NONE_LIT,
    TRIAD_AST_IDENT,
    TRIAD_AST_BINOP,
    TRIAD_AST_UNARYOP,
    TRIAD_AST_CALL,
    TRIAD_AST_INDEX,
    TRIAD_AST_SLICE,
    TRIAD_AST_FIELD,
    TRIAD_AST_LIST,
    TRIAD_AST_LIST_COMP,
    TRIAD_AST_MAP,
    TRIAD_AST_TUPLE,
    TRIAD_AST_FSTRING,
    TRIAD_AST_LAMBDA,
    TRIAD_AST_METHOD_CALL,
    TRIAD_AST_ASSIGN_EXPR,
    TRIAD_AST_YIELD_EXPR,
    TRIAD_AST_AWAIT_EXPR,
    TRIAD_AST_COMPLEX_LIT,
    TRIAD_AST_BYTES_LIT,
    TRIAD_AST_SET_LIT,
    TRIAD_AST_TERNARY,
    TRIAD_AST_CHAIN_CMP,
    TRIAD_AST_SUPER,
    TRIAD_AST_DICT_COMP,
    TRIAD_AST_SET_COMP,
    TRIAD_AST_GEN_COMP,

    TRIAD_AST_LET,
    TRIAD_AST_DESTRUCT_LET,
    TRIAD_AST_MAP_DESTRUCT,
    TRIAD_AST_CONST,
    TRIAD_AST_ASSIGN,
    TRIAD_AST_EXPR_STMT,
    TRIAD_AST_RETURN,
    TRIAD_AST_BREAK,
    TRIAD_AST_CONTINUE,
    TRIAD_AST_IF,
    TRIAD_AST_FOR,
    TRIAD_AST_WHILE,
    TRIAD_AST_FN_DECL,
    TRIAD_AST_TRY_CATCH,
    TRIAD_AST_THROW,
    TRIAD_AST_TYPE_DECL,
    TRIAD_AST_CLASS_DECL,
    TRIAD_AST_IMPORT,
    TRIAD_AST_FROM_IMPORT,
    TRIAD_AST_MATCH,
    TRIAD_AST_YIELD_STMT,
    TRIAD_AST_WITH,
    TRIAD_AST_ASSERT,
    TRIAD_AST_PASS,
    TRIAD_AST_DEL,
    TRIAD_AST_ASYNC_FOR,
    TRIAD_AST_ASYNC_WITH,

    TRIAD_AST_REG,
    TRIAD_AST_ENTITY,
    TRIAD_AST_WORLD,
    TRIAD_AST_SUBSTRATE,
    TRIAD_AST_COUPLE,
    TRIAD_AST_PAIR,
    TRIAD_AST_RING,
    TRIAD_AST_OBSERVE,
    TRIAD_AST_RUN,
    TRIAD_AST_SEQUENCE,
    TRIAD_AST_ANNOTATION,

    TRIAD_AST_MODULE,

    TRIAD_AST_KIND_COUNT
} TriadAstKind;

typedef struct {
    const char *name;
    const char *type_ann;
    struct TriadAstNode *default_value;
    int         is_args;
    int         is_kwargs;
    TriadPos    pos;
} TriadParam;

typedef struct {
    const char *name;
    const char *type_ann;
    struct TriadAstNode *default_value;
    TriadPos    pos;
} TriadTypeField;

typedef struct {
    struct TriadAstNode *cond;
    struct TriadAstNode **body;
    size_t                body_len;
} TriadElifClause;

typedef struct {
    struct TriadAstNode  *pattern;
    struct TriadAstNode  *guard;
    struct TriadAstNode **body;
    size_t                body_len;
} TriadMatchCase;

typedef struct {
    const char           *name;
    struct TriadAstNode  *value;
} TriadKwArg;

typedef struct {
    struct TriadAstNode *key;
    struct TriadAstNode *value;
} TriadMapPair;

typedef struct {
    const char           *key;
    struct TriadAstNode  *value;
} TriadStrEntry;

typedef struct {
    int                   is_expr;
    const char           *text;
    struct TriadAstNode  *expr;
    const char           *fmt_spec;
} TriadFStringAstPart;

typedef struct {
    const char          *var;
    struct TriadAstNode *iter;
    struct TriadAstNode **conditions;
    size_t               conditions_len;
} TriadCompClause;

typedef struct TriadAstNode {
    TriadAstKind kind;
    TriadPos     pos;

    union {

        long long   int_val;
        double      float_val;
        int         bool_val;
        const char *string_val;
        const char *ident_name;

        struct {
            const char          *op;
            struct TriadAstNode *left;
            struct TriadAstNode *right;
        } op;

        struct {
            struct TriadAstNode  *func_or_obj;
            const char           *method;
            struct TriadAstNode **args;
            size_t                args_len;
            TriadKwArg           *kwargs;
            size_t                kwargs_len;
        } call;

        struct {
            struct TriadAstNode *obj;
            struct TriadAstNode *index;
        } index;

        struct {
            struct TriadAstNode *start;
            struct TriadAstNode *end;
            struct TriadAstNode *step;
        } slice;

        struct {
            struct TriadAstNode *obj;
            const char          *field;
        } field;

        struct {
            struct TriadAstNode **elements;
            size_t                elements_len;
        } list;

        struct {
            struct TriadAstNode *expr;
            const char          *var;
            struct TriadAstNode *iter;
            struct TriadAstNode *condition;
            TriadCompClause    *clauses;
            size_t              clauses_len;
        } list_comp;

        struct {
            TriadMapPair *pairs;
            size_t        pairs_len;
        } map;

        struct {
            TriadFStringAstPart *parts;
            size_t               parts_len;
        } fstring;

        struct {
            TriadParam           *params;
            size_t                params_len;
            struct TriadAstNode **body;
            size_t                body_len;
        } lambda;

        struct {
            struct TriadAstNode *target;
            struct TriadAstNode *value;
        } assign_expr;

        struct {
            struct TriadAstNode *value;
        } unary_value;

        struct {
            double real_val;
            double imag_val;
        } complex_lit;

        struct {
            const char *bytes_val;
            size_t      bytes_len;
        } bytes_lit;

        struct {
            struct TriadAstNode **elements;
            size_t                elements_len;
        } set_lit;

        struct {
            struct TriadAstNode *condition;
            struct TriadAstNode *then_val;
            struct TriadAstNode *else_val;
        } ternary;

        struct {
            struct TriadAstNode **operands;
            size_t               operands_len;
            const char          **ops;
            size_t               ops_len;
        } chain_cmp;

        struct {
            struct TriadAstNode **args;
            size_t               args_len;
        } super_expr;

        struct {
            struct TriadAstNode  *key_expr;
            struct TriadAstNode  *value_expr;
            TriadCompClause      *clauses;
            size_t                clauses_len;
        } dict_comp;

        struct {
            struct TriadAstNode  *expr;
            TriadCompClause      *clauses;
            size_t                clauses_len;
        } set_comp;

        struct {
            struct TriadAstNode  *expr;
            TriadCompClause      *clauses;
            size_t                clauses_len;
        } gen_comp;

        struct {
            struct TriadAstNode *value;
        } yield_expr;

        struct {
            struct TriadAstNode *value;
        } await_expr;

        struct {
            const char          *name;
            const char          *type_ann;
            struct TriadAstNode *value;
        } let_stmt;

        struct {
            const char         **names;
            size_t               names_len;
            struct TriadAstNode *value;
            int                  star_idx;
        } destruct;

        struct {
            const char          *name;
            struct TriadAstNode *value;
        } const_stmt;

        struct {
            struct TriadAstNode *target;
            struct TriadAstNode *value;
        } assign_stmt;

        struct {
            struct TriadAstNode *expr;
        } expr_stmt;

        struct {
            struct TriadAstNode  *condition;
            struct TriadAstNode **then_body;
            size_t                then_body_len;
            TriadElifClause      *elif_clauses;
            size_t                elif_clauses_len;
            struct TriadAstNode **else_body;
            size_t                else_body_len;
            int                   has_else;
        } if_stmt;

        struct {
            const char           *var;
            struct TriadAstNode  *iter;
            struct TriadAstNode **body;
            size_t                body_len;
        } for_stmt;

        struct {
            struct TriadAstNode  *condition;
            struct TriadAstNode **body;
            size_t                body_len;
        } while_stmt;

        struct {
            const char           *name;
            TriadParam           *params;
            size_t                params_len;
            const char           *return_type;
            struct TriadAstNode **body;
            size_t                body_len;
            int                   is_async;
            struct TriadAstNode **decorators;
            size_t                decorators_len;
        } fn_decl;

        struct {
            struct TriadAstNode **body;
            size_t                body_len;
            const char           *catch_var;
            struct TriadAstNode **catch_body;
            size_t                catch_body_len;
            struct TriadAstNode **finally_body;
            size_t                finally_body_len;
            const char          **exceptions;
            size_t                exceptions_len;
            int                   has_catch;
        } try_catch;

        struct {
            struct TriadAstNode  *expr;
            const char           *var;
            struct TriadAstNode **body;
            size_t                body_len;
        } with_stmt;

        struct {
            struct TriadAstNode *condition;
            struct TriadAstNode *message;
        } assert_stmt;

        struct {
            struct TriadAstNode *target;
        } del_stmt;

        struct {
            const char           *name;
            const char           *parent;
            TriadTypeField       *fields;
            size_t                fields_len;
            struct TriadAstNode **methods;
            size_t                methods_len;
            const char          **parents;
            size_t                parents_len;
        } type_decl;

        struct {
            const char **path;
            size_t       path_len;
            const char  *alias;
        } import_stmt;

        struct {
            const char **path;
            size_t       path_len;
            const char **names;
            size_t       names_len;
            const char **aliases;
        } from_import;

        struct {
            struct TriadAstNode  *subject;
            TriadMatchCase       *cases;
            size_t                cases_len;
            struct TriadAstNode **else_body;
            size_t                else_body_len;
            int                   has_else;
        } match_stmt;

        struct {
            const char          *name;
            const char          *regime;
            struct TriadAstNode *value;
            TriadStrEntry       *overrides;
            size_t               overrides_len;
            int                  has_overrides;
        } reg_stmt;

        struct {
            const char           *name;
            const char           *base;
            TriadStrEntry        *fields;
            size_t                fields_len;
            struct TriadAstNode **methods;
            size_t                methods_len;
        } entity_decl;

        struct {
            const char           *name;
            TriadStrEntry        *fields;
            size_t                fields_len;
            struct TriadAstNode **entities;
            size_t                entities_len;
            struct TriadAstNode **body;
            size_t                body_len;
        } world_decl;

        struct {
            const char     *name;
            const char     *regime;
            const char    **members;
            size_t          members_len;
            TriadStrEntry  *properties;
            size_t          properties_len;
            TriadStrEntry  *overrides;
            size_t          overrides_len;
            int             is_composed;
        } substrate_decl;

        struct {
            const char          *src;
            const char          *dst;
            struct TriadAstNode *kappa;
            struct TriadAstNode *duration;
        } couple_stmt;

        struct {
            const char          *a;
            const char          *b;
            struct TriadAstNode *kappa;
            struct TriadAstNode *duration;
        } pair_stmt;

        struct {
            const char         **members;
            size_t               members_len;
            struct TriadAstNode *kappa;
            struct TriadAstNode *duration;
        } ring_stmt;

        struct {
            const char  *target;
            const char **metrics;
            size_t       metrics_len;
            int          over_seeds;
        } observe_stmt;

        struct {
            struct TriadAstNode *duration;
            const char          *target;
        } run_stmt;

        struct {
            struct TriadAstNode *inputs;
            const char          *target;
            struct TriadAstNode *each_for;
        } sequence_stmt;

        struct {
            const char *key;
            const char *args;
        } annotation_stmt;

        struct {
            const char           *name;
            const char           *file;
            struct TriadAstNode **body;
            size_t                body_len;
        } module;
    } u;
} TriadAstNode;

TriadAstNode *triad_parse_source(TriadArena *arena,
                                 const char *src,
                                 const char *file,
                                 TriadDiag  *diag);

TriadAstNode *triad_parse_tokens(TriadArena            *arena,
                                 const TriadTokenList  *tokens,
                                 const char            *file,
                                 TriadDiag             *diag);

const char *triad_ast_kind_name(TriadAstKind k);

typedef enum {
    TRIAD_LTOK_EOF = 0,
    TRIAD_LTOK_KEYWORD,
    TRIAD_LTOK_OPCODE,
    TRIAD_LTOK_IDENT,
    TRIAD_LTOK_NUMBER,
    TRIAD_LTOK_STRING,
    TRIAD_LTOK_SYMBOL,
    TRIAD_LTOK_ANNOT
} TriadLegacyTokKind;

typedef struct {
    TriadLegacyTokKind  kind;
    const char         *text;
    int                 line;
    int                 col;
} TriadLegacyToken;

typedef struct {
    TriadLegacyToken *items;
    size_t            len;
} TriadLegacyTokenList;

char *triad_ast_dump_json(const TriadAstNode *module, int indent);

int   triad_ast_dump_json_fp(const TriadAstNode *module, FILE *fp, int indent);

#ifdef __cplusplus
}
#endif

#endif
