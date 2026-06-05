/*
 * triad_frontend.h — Native TriadLang frontend (lexer, AST, parser).
 *
 * Mirrors frontend/lexer_universal.py + frontend/parser_universal.py +
 * frontend/ast_nodes.py. The native AST is bump-allocated from a
 * triad_arena_t supplied by the caller; freeing the arena frees the
 * whole module + every token string.
 *
 * Ownership rules:
 *   - All TriadToken / TriadAstNode / inline string and array storage
 *     returned by this API is owned by the arena passed in.
 *   - triad_tokenize, triad_parse_source, triad_parse_module return
 *     pointers into that arena. Do NOT free individual nodes.
 *   - Error messages on the TriadDiag struct are arena-owned too.
 *
 * Concurrency: not thread-safe. One arena per parse.
 */
#ifndef TRIAD_FRONTEND_H
#define TRIAD_FRONTEND_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Arena ────────────────────────────────────────────────────────── */

typedef struct TriadArenaBlock TriadArenaBlock;

typedef struct TriadArena {
    TriadArenaBlock *head;   /* singly-linked list of blocks */
    size_t          block_size;
    size_t          total_bytes;
} TriadArena;

void   triad_arena_init(TriadArena *a, size_t block_size);
void  *triad_arena_alloc(TriadArena *a, size_t n);
void  *triad_arena_calloc(TriadArena *a, size_t n);
char  *triad_arena_strdup(TriadArena *a, const char *s);
char  *triad_arena_strndup(TriadArena *a, const char *s, size_t n);
void   triad_arena_free(TriadArena *a);

/* ── Position / Diagnostics ──────────────────────────────────────── */

typedef struct {
    int          line;
    int          col;
    const char  *file;   /* arena-owned; may be "" */
} TriadPos;

typedef struct {
    int          line;
    int          col;
    const char  *file;
    const char  *kind;   /* "LEX" or "PARSE" */
    const char  *msg;    /* arena-owned */
} TriadDiag;

/* ── Tokens ──────────────────────────────────────────────────────── */

typedef enum {
    TRIAD_TOK_EOF = 0,
    TRIAD_TOK_NUMBER,
    TRIAD_TOK_STRING,
    TRIAD_TOK_FSTRING,
    TRIAD_TOK_KEYWORD,
    TRIAD_TOK_IDENT,
    TRIAD_TOK_SYMBOL
} TriadTokenKind;

/* FString token payload: an array of FString parts. Each part is
 * either a literal piece ("str") or an unparsed expression source
 * ("expr"). The parser later re-tokenizes the "expr" pieces. */
typedef struct {
    int          is_expr;     /* 0 = literal text, 1 = expression src */
    const char  *text;        /* arena-owned, NUL-terminated          */
} TriadFStringPart;

typedef struct {
    TriadTokenKind     kind;
    const char        *text;  /* arena-owned NUL-terminated lexeme    */
    /* For FSTRING tokens: text == NULL, parts/parts_len describe it. */
    TriadFStringPart  *parts;
    size_t             parts_len;
    int                line;
    int                col;
} TriadToken;

typedef struct {
    TriadToken *items;
    size_t      len;
} TriadTokenList;

/* Tokenize. Returns 0 on success; on failure returns -1 and fills *diag.
 * All output lives in `arena`. */
int triad_tokenize(TriadArena    *arena,
                   const char    *src,
                   const char    *file,
                   TriadTokenList *out,
                   TriadDiag      *diag);

/* ── AST kinds ───────────────────────────────────────────────────── */

typedef enum {
    /* Expressions */
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

    /* Statements */
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

    /* Triad-native */
    TRIAD_AST_REG,
    TRIAD_AST_ENTITY,
    TRIAD_AST_WORLD,
    TRIAD_AST_COUPLE,
    TRIAD_AST_PAIR,
    TRIAD_AST_RING,
    TRIAD_AST_OBSERVE,
    TRIAD_AST_RUN,
    TRIAD_AST_ANNOTATION,

    /* Top level */
    TRIAD_AST_MODULE,

    TRIAD_AST_KIND_COUNT
} TriadAstKind;

/* Parameter (used by fn / lambda) */
typedef struct {
    const char *name;
    const char *type_ann;   /* nullable */
    struct TriadAstNode *default_value; /* nullable */
    int         is_args;
    int         is_kwargs;
    TriadPos    pos;
} TriadParam;

/* Type field (for type / class) */
typedef struct {
    const char *name;
    const char *type_ann;
    struct TriadAstNode *default_value;
    TriadPos    pos;
} TriadTypeField;

/* If statement clause */
typedef struct {
    struct TriadAstNode *cond;
    struct TriadAstNode **body;
    size_t                body_len;
} TriadElifClause;

/* Match case */
typedef struct {
    struct TriadAstNode  *pattern;
    struct TriadAstNode  *guard;   /* nullable */
    struct TriadAstNode **body;
    size_t                body_len;
} TriadMatchCase;

/* Keyword argument */
typedef struct {
    const char           *name;
    struct TriadAstNode  *value;
} TriadKwArg;

/* Map pair / overrides entry */
typedef struct {
    struct TriadAstNode *key;
    struct TriadAstNode *value;
} TriadMapPair;

/* (string-key) entry, e.g. RegStmt.overrides, EntityDecl.fields */
typedef struct {
    const char           *key;
    struct TriadAstNode  *value;
} TriadStrEntry;

/* FString part on the AST side: 0 = literal text, 1 = parsed expression */
typedef struct {
    int                   is_expr;
    const char           *text;     /* if !is_expr */
    struct TriadAstNode  *expr;     /* if  is_expr */
} TriadFStringAstPart;

typedef struct TriadAstNode {
    TriadAstKind kind;
    TriadPos     pos;

    union {
        /* literals & ident */
        long long   int_val;
        double      float_val;
        int         bool_val;
        const char *string_val;
        const char *ident_name;

        /* BinOp / UnaryOp */
        struct {
            const char          *op;
            struct TriadAstNode *left;   /* unary: NULL */
            struct TriadAstNode *right;  /* unary: operand */
        } op;

        /* Call / MethodCall */
        struct {
            struct TriadAstNode  *func_or_obj;
            const char           *method;     /* method-call only */
            struct TriadAstNode **args;
            size_t                args_len;
            TriadKwArg           *kwargs;
            size_t                kwargs_len;
        } call;

        /* Index */
        struct {
            struct TriadAstNode *obj;
            struct TriadAstNode *index;
        } index;

        /* Slice (start/end/step all nullable) */
        struct {
            struct TriadAstNode *start;
            struct TriadAstNode *end;
            struct TriadAstNode *step;
        } slice;

        /* Field */
        struct {
            struct TriadAstNode *obj;
            const char          *field;
        } field;

        /* List / Tuple */
        struct {
            struct TriadAstNode **elements;
            size_t                elements_len;
        } list;

        /* ListComp */
        struct {
            struct TriadAstNode *expr;
            const char          *var;
            struct TriadAstNode *iter;
            struct TriadAstNode *condition;   /* nullable */
        } list_comp;

        /* Map */
        struct {
            TriadMapPair *pairs;
            size_t        pairs_len;
        } map;

        /* FString (AST side) */
        struct {
            TriadFStringAstPart *parts;
            size_t               parts_len;
        } fstring;

        /* Lambda */
        struct {
            TriadParam           *params;
            size_t                params_len;
            struct TriadAstNode **body;
            size_t                body_len;
        } lambda;

        /* AssignExpr */
        struct {
            struct TriadAstNode *target;
            struct TriadAstNode *value;
        } assign_expr;

        /* YieldExpr / AwaitExpr / ReturnStmt / YieldStmt / ThrowStmt */
        struct {
            struct TriadAstNode *value;   /* nullable for yield/return/throw */
        } unary_value;

        /* LetStmt */
        struct {
            const char          *name;
            const char          *type_ann;   /* nullable */
            struct TriadAstNode *value;      /* nullable */
        } let_stmt;

        /* DestructLet / MapDestruct */
        struct {
            const char         **names;
            size_t               names_len;
            struct TriadAstNode *value;
        } destruct;

        /* ConstStmt */
        struct {
            const char          *name;
            struct TriadAstNode *value;
        } const_stmt;

        /* AssignStmt */
        struct {
            struct TriadAstNode *target;
            struct TriadAstNode *value;
        } assign_stmt;

        /* ExprStmt */
        struct {
            struct TriadAstNode *expr;
        } expr_stmt;

        /* IfStmt */
        struct {
            struct TriadAstNode  *condition;
            struct TriadAstNode **then_body;
            size_t                then_body_len;
            TriadElifClause      *elif_clauses;
            size_t                elif_clauses_len;
            struct TriadAstNode **else_body;   /* NULL if absent */
            size_t                else_body_len;
            int                   has_else;
        } if_stmt;

        /* ForStmt */
        struct {
            const char           *var;
            struct TriadAstNode  *iter;
            struct TriadAstNode **body;
            size_t                body_len;
        } for_stmt;

        /* WhileStmt */
        struct {
            struct TriadAstNode  *condition;
            struct TriadAstNode **body;
            size_t                body_len;
        } while_stmt;

        /* FnDecl */
        struct {
            const char           *name;
            TriadParam           *params;
            size_t                params_len;
            const char           *return_type;   /* nullable */
            struct TriadAstNode **body;
            size_t                body_len;
            int                   is_async;
        } fn_decl;

        /* TryCatch */
        struct {
            struct TriadAstNode **body;
            size_t                body_len;
            const char           *catch_var;     /* nullable */
            struct TriadAstNode **catch_body;
            size_t                catch_body_len;
            struct TriadAstNode **finally_body;
            size_t                finally_body_len;
        } try_catch;

        /* TypeDecl / ClassDecl */
        struct {
            const char           *name;
            const char           *parent;        /* class only; nullable */
            TriadTypeField       *fields;
            size_t                fields_len;
            struct TriadAstNode **methods;       /* each is FN_DECL */
            size_t                methods_len;
        } type_decl;

        /* ImportStmt */
        struct {
            const char **path;
            size_t       path_len;
            const char  *alias;                  /* nullable */
        } import_stmt;

        /* FromImportStmt */
        struct {
            const char **path;
            size_t       path_len;
            const char **names;
            size_t       names_len;
        } from_import;

        /* MatchStmt */
        struct {
            struct TriadAstNode  *subject;
            TriadMatchCase       *cases;
            size_t                cases_len;
            struct TriadAstNode **else_body;     /* NULL if absent */
            size_t                else_body_len;
            int                   has_else;
        } match_stmt;

        /* RegStmt */
        struct {
            const char          *name;
            const char          *regime;       /* nullable */
            struct TriadAstNode *value;        /* nullable */
            TriadStrEntry       *overrides;    /* nullable */
            size_t               overrides_len;
            int                  has_overrides;
        } reg_stmt;

        /* EntityDecl */
        struct {
            const char           *name;
            const char           *base;          /* nullable */
            TriadStrEntry        *fields;
            size_t                fields_len;
            struct TriadAstNode **methods;
            size_t                methods_len;
        } entity_decl;

        /* WorldDecl */
        struct {
            const char           *name;
            TriadStrEntry        *fields;
            size_t                fields_len;
            struct TriadAstNode **entities;
            size_t                entities_len;
            struct TriadAstNode **body;
            size_t                body_len;
        } world_decl;

        /* CoupleStmt */
        struct {
            const char          *src;
            const char          *dst;
            struct TriadAstNode *kappa;          /* nullable */
        } couple_stmt;

        /* PairStmt */
        struct {
            const char          *a;
            const char          *b;
            struct TriadAstNode *kappa;          /* nullable */
            struct TriadAstNode *duration;       /* nullable */
        } pair_stmt;

        /* RingStmt */
        struct {
            const char         **members;
            size_t               members_len;
            struct TriadAstNode *kappa;
            struct TriadAstNode *duration;
        } ring_stmt;

        /* ObserveStmt */
        struct {
            const char  *target;
            const char **metrics;
            size_t       metrics_len;
            int          over_seeds;
        } observe_stmt;

        /* RunStmt */
        struct {
            struct TriadAstNode *duration;       /* nullable */
        } run_stmt;

        /* AnnotationStmt */
        struct {
            const char *key;
            const char *args;
        } annotation_stmt;

        /* Module */
        struct {
            const char           *name;
            const char           *file;
            struct TriadAstNode **body;
            size_t                body_len;
        } module;
    } u;
} TriadAstNode;

/* ── Parse API ───────────────────────────────────────────────────── */

/* Parse a complete module. Returns NULL on failure (fills *diag).
 * `arena` must already be initialized. */
TriadAstNode *triad_parse_source(TriadArena *arena,
                                 const char *src,
                                 const char *file,
                                 TriadDiag  *diag);

/* Lower-level: parse from a pre-tokenized stream. */
TriadAstNode *triad_parse_tokens(TriadArena            *arena,
                                 const TriadTokenList  *tokens,
                                 const char            *file,
                                 TriadDiag             *diag);

/* ── AST helpers ─────────────────────────────────────────────────── */

const char *triad_ast_kind_name(TriadAstKind k);

/* ════════════════════════════════════════════════════════════════════
 * Legacy frontend (v1 DSL) — port of frontend/lexer.py + parser.py.
 * Used by compiler/triadc.py and compiler/typecheck.py. Shares the same
 * arena. AST nodes are tagged with TriadLegacyKind.
 * ════════════════════════════════════════════════════════════════════ */

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

int triad_legacy_tokenize(TriadArena *arena, const char *src,
                          TriadLegacyTokenList *out, TriadDiag *diag);

typedef enum {
    TRIAD_LAST_NUM_LIT,
    TRIAD_LAST_BOOL_LIT,
    TRIAD_LAST_STR_LIT,
    TRIAD_LAST_IDENT_REF,

    TRIAD_LAST_REG_DECL,
    TRIAD_LAST_OP,
    TRIAD_LAST_LOOP_BLOCK,
    TRIAD_LAST_SEGMENT_BLOCK,
    TRIAD_LAST_IF_BLOCK,
    TRIAD_LAST_OUT_STMT,
    TRIAD_LAST_HALT_STMT,
    TRIAD_LAST_ANNOTATION,
    TRIAD_LAST_EVOLVE_STMT,
    TRIAD_LAST_COUPLE_STMT,
    TRIAD_LAST_PAIR_STMT,
    TRIAD_LAST_RING_STMT,
    TRIAD_LAST_SEQUENCE_STMT,
    TRIAD_LAST_OBSERVE_STMT,
    TRIAD_LAST_ASSERT_STMT,
    TRIAD_LAST_CHECKPOINT_STMT,
    TRIAD_LAST_SUBSTRATE_DECL,

    TRIAD_LAST_PROGRAM,

    TRIAD_LAST_KIND_COUNT
} TriadLegacyKind;

/* Overrides for RegDecl: a dict[str, value] where value is one of
 * (int, float, bool, str, tuple-of-primitives, ident). We represent
 * each value as a small tagged variant. */
typedef enum {
    TRIAD_LVAL_INT,
    TRIAD_LVAL_FLOAT,
    TRIAD_LVAL_BOOL,
    TRIAD_LVAL_STR,
    TRIAD_LVAL_IDENT,
    TRIAD_LVAL_TUPLE
} TriadLegacyValKind;

typedef struct TriadLegacyVal {
    TriadLegacyValKind kind;
    long long          int_val;
    double             float_val;
    int                bool_val;
    const char        *str_val;     /* used by STR and IDENT */
    struct TriadLegacyVal *tuple_items;
    size_t             tuple_len;
} TriadLegacyVal;

typedef struct {
    const char       *key;
    TriadLegacyVal    value;
} TriadLegacyOverride;

typedef struct {
    const char     *key;
    const char     *raw_args;       /* without parens */
    int             line;
} TriadLegacyAnnot;

typedef struct TriadLegacyNode TriadLegacyNode;

struct TriadLegacyNode {
    TriadLegacyKind kind;
    int             line;
    union {
        /* literals */
        struct { double value; int is_int; }  num;
        struct { int value; }                  boolean;
        struct { const char *value; }          str;
        struct { const char *name; }           ident;

        /* RegDecl */
        struct {
            const char            *name;
            int                    bit_width;
            TriadLegacyNode       *initial;        /* nullable */
            const char            *regime_name;    /* nullable */
            TriadLegacyOverride   *overrides;
            size_t                 overrides_len;
            int                    has_overrides;
        } reg_decl;

        /* Op */
        struct {
            const char        *opcode;
            TriadLegacyNode  **args;
            size_t             args_len;
        } op;

        /* LoopBlock */
        struct {
            TriadLegacyNode   *target;
            TriadLegacyNode  **body;
            size_t             body_len;
        } loop_block;

        /* SegmentBlock */
        struct {
            int                segment_id;
            double             duration;
            TriadLegacyNode  **body;
            size_t             body_len;
        } segment_block;

        /* IfBlock */
        struct {
            const char        *cond_name;
            TriadLegacyNode  **then_body;
            size_t             then_body_len;
            TriadLegacyNode  **else_body;          /* NULL when absent */
            size_t             else_body_len;
            int                has_else;
        } if_block;

        /* OutStmt */
        struct {
            TriadLegacyNode  **args;
            size_t             args_len;
        } out_stmt;

        /* HaltStmt */ /* — kind alone is enough */

        /* Annotation */
        TriadLegacyAnnot annotation;

        /* EvolveStmt */
        struct {
            const char *target;
            double      duration;
        } evolve_stmt;

        /* CoupleStmt / PairStmt */
        struct {
            const char *src_or_a;
            const char *dst_or_b;
            double      kappa;
            double      duration;
        } couple_pair;

        /* RingStmt / SequenceStmt */
        struct {
            const char **members;       /* identifiers */
            size_t       members_len;
            const char  *target;         /* sequence only */
            double       kappa;          /* ring only */
            double       duration;       /* ring or sequence each_for */
        } ring_seq;

        /* ObserveStmt */
        struct {
            const char  *target;
            const char **metrics;
            size_t       metrics_len;
            int          over_seeds;
            const char  *stream_to;      /* "" when absent */
        } observe_stmt;

        /* AssertStmt */
        struct {
            const char *predicate;
            const char *target;
        } assert_stmt;

        /* CheckpointStmt */
        struct {
            const char *target;
            const char *path;
        } checkpoint_stmt;

        /* SubstrateDecl */
        struct {
            const char           *name;
            const char          **composed_of;
            size_t                composed_of_len;
            TriadLegacyOverride  *properties;
            size_t                properties_len;
        } substrate_decl;

        /* Program */
        struct {
            TriadLegacyNode **body;
            size_t            body_len;
        } program;
    } u;
};

TriadLegacyNode *triad_legacy_parse_source(TriadArena *a, const char *src,
                                           TriadDiag *diag);
TriadLegacyNode *triad_legacy_parse_tokens(TriadArena *a,
                                           const TriadLegacyTokenList *t,
                                           TriadDiag *diag);

const char *triad_legacy_kind_name(TriadLegacyKind k);

/* JSON dump of the legacy AST. Schema mirrors the Python dataclasses
 * (RegDecl, Op, LoopBlock, ...) with _type/field/value layout, used by
 * parity_ast_legacy. */
char *triad_legacy_dump_json(const TriadLegacyNode *prog, int indent);

/* ── AST → JSON ──────────────────────────────────────────────────── */

/* Dump the module as a JSON string in a malloc'd buffer (caller frees).
 * The schema matches scripts/ast_to_json.py on the Python side. */
char *triad_ast_dump_json(const TriadAstNode *module, int indent);

/* Same but write directly to a FILE *. Returns 0 on success. */
int   triad_ast_dump_json_fp(const TriadAstNode *module, FILE *fp, int indent);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TRIAD_FRONTEND_H */
