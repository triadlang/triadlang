/* triad_parser.c — Port of frontend/parser_universal.py.
 *
 * The Python parser uses a recursive descent + precedence-climbing
 * scheme. We mirror it structure-for-structure.
 */
#include "triad_frontend.h"

#include <ctype.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Parser state ───────────────────────────────────────────────── */

typedef struct {
    TriadArena           *arena;
    const TriadToken     *toks;
    size_t                ntoks;
    size_t                i;
    const char           *file;
    TriadDiag            *diag;
    jmp_buf               err_jmp;
} Pz;

/* growable pointer array for any list of ast nodes / strings, freed
 * after copy into arena. */
typedef struct {
    void  **items;
    size_t  len;
    size_t  cap;
} PVec;

static void pvec_push(PVec *v, void *p) {
    if (v->len == v->cap) {
        size_t nc = v->cap ? v->cap * 2 : 8;
        v->items = (void **)realloc(v->items, nc * sizeof(void *));
        v->cap = nc;
    }
    v->items[v->len++] = p;
}
static void pvec_free(PVec *v) { free(v->items); v->items = NULL; v->len = v->cap = 0; }

static void *pvec_to_arena_ptrs(TriadArena *a, PVec *v, size_t elem_sz) {
    if (v->len == 0) { pvec_free(v); return NULL; }
    void *out = triad_arena_alloc(a, v->len * elem_sz);
    memcpy(out, v->items, v->len * sizeof(void *));
    pvec_free(v);
    return out;
}

/* ── Errors ─────────────────────────────────────────────────────── */

static void NORETURN_parse_error(Pz *P, int line, int col, const char *fmt, ...)
    __attribute__((noreturn));

static void NORETURN_parse_error(Pz *P, int line, int col, const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (P->diag) {
        P->diag->line = line;
        P->diag->col  = col;
        P->diag->file = P->file;
        P->diag->kind = "PARSE";
        P->diag->msg  = triad_arena_strdup(P->arena, buf);
    }
    longjmp(P->err_jmp, 1);
}

/* ── Token helpers ──────────────────────────────────────────────── */

static const TriadToken *p_peek(Pz *P, size_t off) {
    size_t idx = P->i + off;
    if (idx < P->ntoks) return &P->toks[idx];
    return &P->toks[P->ntoks - 1]; /* EOF */
}

static const TriadToken *p_eat(Pz *P) {
    const TriadToken *t = &P->toks[P->i];
    P->i++;
    return t;
}

static int p_at_kind(Pz *P, TriadTokenKind k) {
    return p_peek(P, 0)->kind == k;
}

static int p_at(Pz *P, TriadTokenKind k, const char *value) {
    const TriadToken *t = p_peek(P, 0);
    if (t->kind != k) return 0;
    if (!value) return 1;
    return t->text && strcmp(t->text, value) == 0;
}

static int p_at_ident_val(Pz *P, const char *v) {
    const TriadToken *t = p_peek(P, 0);
    return t->kind == TRIAD_TOK_IDENT && t->text && strcmp(t->text, v) == 0;
}

static const char *kind_name(TriadTokenKind k) {
    switch (k) {
        case TRIAD_TOK_EOF: return "EOF";
        case TRIAD_TOK_NUMBER: return "NUMBER";
        case TRIAD_TOK_STRING: return "STRING";
        case TRIAD_TOK_FSTRING: return "FSTRING";
        case TRIAD_TOK_KEYWORD: return "KEYWORD";
        case TRIAD_TOK_IDENT: return "IDENT";
        case TRIAD_TOK_SYMBOL: return "SYMBOL";
    }
    return "?";
}

static const TriadToken *p_expect(Pz *P, TriadTokenKind k, const char *value) {
    const TriadToken *t = p_peek(P, 0);
    if (t->kind != k || (value && (!t->text || strcmp(t->text, value) != 0))) {
        if (value) {
            NORETURN_parse_error(P, t->line, t->col,
                "expected %s '%s', got %s '%s'",
                kind_name(k), value, kind_name(t->kind), t->text ? t->text : "");
        } else {
            NORETURN_parse_error(P, t->line, t->col,
                "expected %s, got %s '%s'",
                kind_name(k), kind_name(t->kind), t->text ? t->text : "");
        }
    }
    return p_eat(P);
}

static const TriadToken *p_expect_sym(Pz *P, const char *v) { return p_expect(P, TRIAD_TOK_SYMBOL, v); }
static const TriadToken *p_expect_kw (Pz *P, const char *v) { return p_expect(P, TRIAD_TOK_KEYWORD, v); }

static const TriadToken *p_expect_ident_or_kw(Pz *P) {
    const TriadToken *t = p_peek(P, 0);
    if (t->kind == TRIAD_TOK_IDENT || t->kind == TRIAD_TOK_KEYWORD) return p_eat(P);
    NORETURN_parse_error(P, t->line, t->col,
        "expected identifier, got %s '%s'", kind_name(t->kind), t->text ? t->text : "");
}

static void p_skip_semis(Pz *P) {
    while (p_at(P, TRIAD_TOK_SYMBOL, ";")) p_eat(P);
}

static void p_eat_semi(Pz *P) {
    if (p_at(P, TRIAD_TOK_SYMBOL, ";")) p_eat(P);
}

/* ── Position helper ────────────────────────────────────────────── */

static TriadPos cur_pos(Pz *P) {
    const TriadToken *t = p_peek(P, 0);
    TriadPos pos;
    pos.line = t->line;
    pos.col  = t->col;
    pos.file = P->file;
    return pos;
}

/* ── Node allocation ────────────────────────────────────────────── */

static TriadAstNode *new_node(Pz *P, TriadAstKind k, TriadPos pos) {
    TriadAstNode *n = (TriadAstNode *)triad_arena_calloc(P->arena, sizeof(TriadAstNode));
    n->kind = k;
    n->pos  = pos;
    return n;
}

/* ── Forward decls ──────────────────────────────────────────────── */

static TriadAstNode *parse_stmt(Pz *P);
static TriadAstNode *parse_expr(Pz *P);
static TriadAstNode *parse_or(Pz *P);
static TriadAstNode *parse_and(Pz *P);
static TriadAstNode *parse_not(Pz *P);
static TriadAstNode *parse_comparison(Pz *P);
static TriadAstNode *parse_add(Pz *P);
static TriadAstNode *parse_mul(Pz *P);
static TriadAstNode *parse_power(Pz *P);
static TriadAstNode *parse_unary(Pz *P);
static TriadAstNode *parse_postfix(Pz *P);
static TriadAstNode *parse_primary(Pz *P);
static TriadAstNode *parse_lambda(Pz *P);
static void parse_params(Pz *P, TriadParam **out, size_t *out_len);
static void parse_block(Pz *P, TriadAstNode ***body, size_t *body_len);
static void parse_call_args(Pz *P,
                            TriadAstNode ***args, size_t *args_len,
                            TriadKwArg     **kwargs, size_t *kwargs_len);

/* ── Statement parsers ──────────────────────────────────────────── */

static TriadAstNode *parse_let(Pz *P) {
    TriadPos p = cur_pos(P);
    p_eat(P); /* let */
    if (p_at(P, TRIAD_TOK_SYMBOL, "(")) {
        p_eat(P);
        PVec names = {0};
        pvec_push(&names, (void *)p_expect(P, TRIAD_TOK_IDENT, NULL)->text);
        while (p_at(P, TRIAD_TOK_SYMBOL, ",")) {
            p_eat(P);
            pvec_push(&names, (void *)p_expect(P, TRIAD_TOK_IDENT, NULL)->text);
        }
        p_expect_sym(P, ")");
        p_expect_sym(P, "=");
        TriadAstNode *val = parse_expr(P);
        p_eat_semi(P);
        TriadAstNode *n = new_node(P, TRIAD_AST_DESTRUCT_LET, p);
        n->u.destruct.names_len = names.len;
        n->u.destruct.names     = (const char **)pvec_to_arena_ptrs(P->arena, &names, sizeof(const char *));
        n->u.destruct.value     = val;
        return n;
    }
    if (p_at(P, TRIAD_TOK_SYMBOL, "{")) {
        p_eat(P);
        PVec names = {0};
        pvec_push(&names, (void *)p_expect(P, TRIAD_TOK_IDENT, NULL)->text);
        while (p_at(P, TRIAD_TOK_SYMBOL, ",")) {
            p_eat(P);
            if (p_at(P, TRIAD_TOK_SYMBOL, "}")) break;
            pvec_push(&names, (void *)p_expect(P, TRIAD_TOK_IDENT, NULL)->text);
        }
        p_expect_sym(P, "}");
        p_expect_sym(P, "=");
        TriadAstNode *val = parse_expr(P);
        p_eat_semi(P);
        TriadAstNode *n = new_node(P, TRIAD_AST_MAP_DESTRUCT, p);
        n->u.destruct.names_len = names.len;
        n->u.destruct.names     = (const char **)pvec_to_arena_ptrs(P->arena, &names, sizeof(const char *));
        n->u.destruct.value     = val;
        return n;
    }
    const char *name = p_expect_ident_or_kw(P)->text;
    const char *type_ann = NULL;
    if (p_at(P, TRIAD_TOK_SYMBOL, ":")) {
        p_eat(P);
        type_ann = p_expect(P, TRIAD_TOK_IDENT, NULL)->text;
    }
    TriadAstNode *val = NULL;
    if (p_at(P, TRIAD_TOK_SYMBOL, "=")) {
        p_eat(P);
        val = parse_expr(P);
    }
    p_eat_semi(P);
    TriadAstNode *n = new_node(P, TRIAD_AST_LET, p);
    n->u.let_stmt.name     = name;
    n->u.let_stmt.type_ann = type_ann;
    n->u.let_stmt.value    = val;
    return n;
}

static TriadAstNode *parse_const(Pz *P) {
    TriadPos p = cur_pos(P);
    p_eat(P); /* const */
    const char *name = p_expect_ident_or_kw(P)->text;
    p_expect_sym(P, "=");
    TriadAstNode *val = parse_expr(P);
    p_eat_semi(P);
    TriadAstNode *n = new_node(P, TRIAD_AST_CONST, p);
    n->u.const_stmt.name = name;
    n->u.const_stmt.value = val;
    return n;
}

static TriadAstNode *parse_fn(Pz *P, int async_) {
    TriadPos p = cur_pos(P);
    if (async_) p_eat(P); /* async */
    p_eat(P); /* fn */
    const char *name = p_expect(P, TRIAD_TOK_IDENT, NULL)->text;
    p_expect_sym(P, "(");
    TriadParam *params = NULL;
    size_t params_len = 0;
    parse_params(P, &params, &params_len);
    p_expect_sym(P, ")");
    const char *ret = NULL;
    if (p_at(P, TRIAD_TOK_SYMBOL, "->")) {
        p_eat(P);
        ret = p_expect(P, TRIAD_TOK_IDENT, NULL)->text;
    }
    TriadAstNode **body = NULL;
    size_t body_len = 0;
    parse_block(P, &body, &body_len);

    TriadAstNode *n = new_node(P, TRIAD_AST_FN_DECL, p);
    n->u.fn_decl.name        = name;
    n->u.fn_decl.params      = params;
    n->u.fn_decl.params_len  = params_len;
    n->u.fn_decl.return_type = ret;
    n->u.fn_decl.body        = body;
    n->u.fn_decl.body_len    = body_len;
    n->u.fn_decl.is_async    = async_ ? 1 : 0;
    return n;
}

static void parse_params(Pz *P, TriadParam **out, size_t *out_len) {
    PVec tmp = {0};
    while (!p_at(P, TRIAD_TOK_SYMBOL, ")") && !p_at_kind(P, TRIAD_TOK_EOF)) {
        TriadParam *pm = (TriadParam *)triad_arena_calloc(P->arena, sizeof(TriadParam));
        pm->pos = cur_pos(P);
        if (p_at(P, TRIAD_TOK_SYMBOL, "**")) { p_eat(P); pm->is_kwargs = 1; }
        else if (p_at(P, TRIAD_TOK_SYMBOL, "*")) { p_eat(P); pm->is_args = 1; }
        pm->name = p_expect_ident_or_kw(P)->text;
        if (p_at(P, TRIAD_TOK_SYMBOL, ":")) {
            p_eat(P);
            pm->type_ann = p_expect(P, TRIAD_TOK_IDENT, NULL)->text;
        }
        if (p_at(P, TRIAD_TOK_SYMBOL, "=")) {
            p_eat(P);
            pm->default_value = parse_expr(P);
        }
        pvec_push(&tmp, pm);
        if (!p_at(P, TRIAD_TOK_SYMBOL, ")")) p_expect_sym(P, ",");
    }
    if (tmp.len == 0) {
        *out = NULL;
        *out_len = 0;
        pvec_free(&tmp);
        return;
    }
    TriadParam *arr = (TriadParam *)triad_arena_alloc(P->arena, tmp.len * sizeof(TriadParam));
    for (size_t k = 0; k < tmp.len; ++k) arr[k] = *(TriadParam *)tmp.items[k];
    *out = arr;
    *out_len = tmp.len;
    pvec_free(&tmp);
}

static void parse_block(Pz *P, TriadAstNode ***body, size_t *body_len) {
    p_expect_sym(P, "{");
    PVec stmts = {0};
    p_skip_semis(P);
    while (!p_at(P, TRIAD_TOK_SYMBOL, "}") && !p_at_kind(P, TRIAD_TOK_EOF)) {
        pvec_push(&stmts, parse_stmt(P));
        p_skip_semis(P);
    }
    p_expect_sym(P, "}");
    *body_len = stmts.len;
    *body     = (TriadAstNode **)pvec_to_arena_ptrs(P->arena, &stmts, sizeof(TriadAstNode *));
}

static TriadAstNode *parse_if(Pz *P) {
    TriadPos p = cur_pos(P);
    p_eat(P); /* if */
    TriadAstNode *cond = parse_expr(P);
    TriadAstNode **then = NULL; size_t then_len = 0;
    parse_block(P, &then, &then_len);
    PVec elifs = {0};
    int has_else = 0;
    TriadAstNode **else_body = NULL; size_t else_len = 0;
    while (p_at(P, TRIAD_TOK_KEYWORD, "elif")) {
        p_eat(P);
        TriadAstNode *ec = parse_expr(P);
        TriadAstNode **eb = NULL; size_t eb_len = 0;
        parse_block(P, &eb, &eb_len);
        TriadElifClause *cl = (TriadElifClause *)triad_arena_calloc(P->arena, sizeof(TriadElifClause));
        cl->cond = ec;
        cl->body = eb;
        cl->body_len = eb_len;
        pvec_push(&elifs, cl);
    }
    if (p_at(P, TRIAD_TOK_KEYWORD, "else")) {
        p_eat(P);
        parse_block(P, &else_body, &else_len);
        has_else = 1;
    }
    TriadAstNode *n = new_node(P, TRIAD_AST_IF, p);
    n->u.if_stmt.condition  = cond;
    n->u.if_stmt.then_body  = then;
    n->u.if_stmt.then_body_len = then_len;
    if (elifs.len) {
        TriadElifClause *arr = (TriadElifClause *)triad_arena_alloc(P->arena, elifs.len * sizeof(TriadElifClause));
        for (size_t k = 0; k < elifs.len; ++k) arr[k] = *(TriadElifClause *)elifs.items[k];
        n->u.if_stmt.elif_clauses = arr;
        n->u.if_stmt.elif_clauses_len = elifs.len;
    }
    pvec_free(&elifs);
    n->u.if_stmt.has_else      = has_else;
    n->u.if_stmt.else_body     = else_body;
    n->u.if_stmt.else_body_len = else_len;
    return n;
}

static TriadAstNode *parse_for(Pz *P) {
    TriadPos p = cur_pos(P);
    p_eat(P); /* for */
    const char *var = p_expect(P, TRIAD_TOK_IDENT, NULL)->text;
    p_expect_kw(P, "in");
    TriadAstNode *it = parse_expr(P);
    TriadAstNode **body = NULL; size_t body_len = 0;
    parse_block(P, &body, &body_len);
    TriadAstNode *n = new_node(P, TRIAD_AST_FOR, p);
    n->u.for_stmt.var      = var;
    n->u.for_stmt.iter     = it;
    n->u.for_stmt.body     = body;
    n->u.for_stmt.body_len = body_len;
    return n;
}

static TriadAstNode *parse_while(Pz *P) {
    TriadPos p = cur_pos(P);
    p_eat(P);
    TriadAstNode *cond = parse_expr(P);
    TriadAstNode **body = NULL; size_t body_len = 0;
    parse_block(P, &body, &body_len);
    TriadAstNode *n = new_node(P, TRIAD_AST_WHILE, p);
    n->u.while_stmt.condition = cond;
    n->u.while_stmt.body      = body;
    n->u.while_stmt.body_len  = body_len;
    return n;
}

static TriadAstNode *parse_return(Pz *P) {
    TriadPos p = cur_pos(P);
    p_eat(P); /* return */
    TriadAstNode *val = NULL;
    if (!p_at(P, TRIAD_TOK_SYMBOL, ";") && !p_at(P, TRIAD_TOK_SYMBOL, "}") && !p_at_kind(P, TRIAD_TOK_EOF)) {
        val = parse_expr(P);
    }
    p_eat_semi(P);
    TriadAstNode *n = new_node(P, TRIAD_AST_RETURN, p);
    n->u.unary_value.value = val;
    return n;
}

static TriadAstNode *parse_yield_stmt(Pz *P) {
    TriadPos p = cur_pos(P);
    p_eat(P); /* yield */
    TriadAstNode *val = NULL;
    if (!p_at(P, TRIAD_TOK_SYMBOL, ";") && !p_at(P, TRIAD_TOK_SYMBOL, "}") && !p_at_kind(P, TRIAD_TOK_EOF)) {
        val = parse_expr(P);
    }
    p_eat_semi(P);
    TriadAstNode *n = new_node(P, TRIAD_AST_YIELD_STMT, p);
    n->u.unary_value.value = val;
    return n;
}

static TriadAstNode *parse_type_decl(Pz *P) {
    TriadPos p = cur_pos(P);
    p_eat(P); /* type */
    const char *name = p_expect(P, TRIAD_TOK_IDENT, NULL)->text;
    p_expect_sym(P, "{");
    PVec fields = {0};
    PVec methods = {0};
    p_skip_semis(P);
    while (!p_at(P, TRIAD_TOK_SYMBOL, "}") && !p_at_kind(P, TRIAD_TOK_EOF)) {
        if (p_at(P, TRIAD_TOK_KEYWORD, "fn")) {
            pvec_push(&methods, parse_fn(P, 0));
        } else {
            TriadTypeField *tf = (TriadTypeField *)triad_arena_calloc(P->arena, sizeof(TriadTypeField));
            tf->pos = cur_pos(P);
            tf->name = p_expect(P, TRIAD_TOK_IDENT, NULL)->text;
            p_expect_sym(P, ":");
            tf->type_ann = p_expect(P, TRIAD_TOK_IDENT, NULL)->text;
            if (p_at(P, TRIAD_TOK_SYMBOL, "=")) {
                p_eat(P);
                tf->default_value = parse_expr(P);
            }
            p_eat_semi(P);
            pvec_push(&fields, tf);
        }
        p_skip_semis(P);
    }
    p_expect_sym(P, "}");
    TriadAstNode *n = new_node(P, TRIAD_AST_TYPE_DECL, p);
    n->u.type_decl.name = name;
    n->u.type_decl.parent = NULL;
    if (fields.len) {
        TriadTypeField *arr = (TriadTypeField *)triad_arena_alloc(P->arena, fields.len * sizeof(TriadTypeField));
        for (size_t k = 0; k < fields.len; ++k) arr[k] = *(TriadTypeField *)fields.items[k];
        n->u.type_decl.fields = arr;
        n->u.type_decl.fields_len = fields.len;
    }
    pvec_free(&fields);
    n->u.type_decl.methods_len = methods.len;
    n->u.type_decl.methods = (TriadAstNode **)pvec_to_arena_ptrs(P->arena, &methods, sizeof(TriadAstNode *));
    return n;
}

static TriadAstNode *parse_class(Pz *P) {
    TriadPos p = cur_pos(P);
    p_eat(P); /* class */
    const char *name = p_expect(P, TRIAD_TOK_IDENT, NULL)->text;
    const char *parent = NULL;
    if (p_at(P, TRIAD_TOK_SYMBOL, ":")) {
        p_eat(P);
        parent = p_expect(P, TRIAD_TOK_IDENT, NULL)->text;
    }
    p_expect_sym(P, "{");
    PVec fields = {0};
    PVec methods = {0};
    p_skip_semis(P);
    while (!p_at(P, TRIAD_TOK_SYMBOL, "}") && !p_at_kind(P, TRIAD_TOK_EOF)) {
        if (p_at(P, TRIAD_TOK_KEYWORD, "fn")) {
            pvec_push(&methods, parse_fn(P, 0));
        } else {
            TriadTypeField *tf = (TriadTypeField *)triad_arena_calloc(P->arena, sizeof(TriadTypeField));
            tf->pos = cur_pos(P);
            tf->name = p_expect(P, TRIAD_TOK_IDENT, NULL)->text;
            tf->type_ann = "any";
            if (p_at(P, TRIAD_TOK_SYMBOL, ":")) {
                p_eat(P);
                tf->type_ann = p_expect(P, TRIAD_TOK_IDENT, NULL)->text;
            }
            if (p_at(P, TRIAD_TOK_SYMBOL, "=")) {
                p_eat(P);
                tf->default_value = parse_expr(P);
            }
            p_eat_semi(P);
            pvec_push(&fields, tf);
        }
        p_skip_semis(P);
    }
    p_expect_sym(P, "}");
    TriadAstNode *n = new_node(P, TRIAD_AST_CLASS_DECL, p);
    n->u.type_decl.name   = name;
    n->u.type_decl.parent = parent;
    if (fields.len) {
        TriadTypeField *arr = (TriadTypeField *)triad_arena_alloc(P->arena, fields.len * sizeof(TriadTypeField));
        for (size_t k = 0; k < fields.len; ++k) arr[k] = *(TriadTypeField *)fields.items[k];
        n->u.type_decl.fields = arr;
        n->u.type_decl.fields_len = fields.len;
    }
    pvec_free(&fields);
    n->u.type_decl.methods_len = methods.len;
    n->u.type_decl.methods = (TriadAstNode **)pvec_to_arena_ptrs(P->arena, &methods, sizeof(TriadAstNode *));
    return n;
}

static TriadAstNode *parse_match(Pz *P) {
    TriadPos p = cur_pos(P);
    p_eat(P); /* match */
    TriadAstNode *subject = parse_expr(P);
    p_expect_sym(P, "{");
    p_skip_semis(P);
    PVec cases = {0};
    TriadAstNode **else_body = NULL;
    size_t else_len = 0;
    int has_else = 0;
    while (!p_at(P, TRIAD_TOK_SYMBOL, "}") && !p_at_kind(P, TRIAD_TOK_EOF)) {
        if (p_at(P, TRIAD_TOK_KEYWORD, "else")) {
            p_eat(P);
            if (p_at(P, TRIAD_TOK_SYMBOL, "=>")) p_eat(P);
            parse_block(P, &else_body, &else_len);
            has_else = 1;
            break;
        }
        p_expect_kw(P, "case");
        TriadAstNode *pattern = parse_expr(P);
        TriadAstNode *guard = NULL;
        if (p_at(P, TRIAD_TOK_KEYWORD, "if")) {
            p_eat(P);
            guard = parse_expr(P);
        }
        if (p_at(P, TRIAD_TOK_SYMBOL, "=>")) p_eat(P);
        TriadAstNode **body = NULL; size_t blen = 0;
        parse_block(P, &body, &blen);
        TriadMatchCase *mc = (TriadMatchCase *)triad_arena_calloc(P->arena, sizeof(TriadMatchCase));
        mc->pattern = pattern;
        mc->guard   = guard;
        mc->body    = body;
        mc->body_len = blen;
        pvec_push(&cases, mc);
        p_skip_semis(P);
    }
    p_expect_sym(P, "}");
    TriadAstNode *n = new_node(P, TRIAD_AST_MATCH, p);
    n->u.match_stmt.subject = subject;
    if (cases.len) {
        TriadMatchCase *arr = (TriadMatchCase *)triad_arena_alloc(P->arena, cases.len * sizeof(TriadMatchCase));
        for (size_t k = 0; k < cases.len; ++k) arr[k] = *(TriadMatchCase *)cases.items[k];
        n->u.match_stmt.cases = arr;
        n->u.match_stmt.cases_len = cases.len;
    }
    pvec_free(&cases);
    n->u.match_stmt.has_else = has_else;
    n->u.match_stmt.else_body = else_body;
    n->u.match_stmt.else_body_len = else_len;
    return n;
}

static TriadAstNode *parse_import(Pz *P) {
    TriadPos p = cur_pos(P);
    p_eat(P); /* import */
    PVec path = {0};
    pvec_push(&path, (void *)p_expect_ident_or_kw(P)->text);
    while (p_at(P, TRIAD_TOK_SYMBOL, ".")) {
        p_eat(P);
        pvec_push(&path, (void *)p_expect_ident_or_kw(P)->text);
    }
    const char *alias = NULL;
    if (p_at(P, TRIAD_TOK_KEYWORD, "as")) {
        p_eat(P);
        alias = p_expect(P, TRIAD_TOK_IDENT, NULL)->text;
    }
    p_eat_semi(P);
    TriadAstNode *n = new_node(P, TRIAD_AST_IMPORT, p);
    n->u.import_stmt.path_len = path.len;
    n->u.import_stmt.path = (const char **)pvec_to_arena_ptrs(P->arena, &path, sizeof(const char *));
    n->u.import_stmt.alias = alias;
    return n;
}

static TriadAstNode *parse_from_import(Pz *P) {
    TriadPos p = cur_pos(P);
    p_eat(P); /* from */
    PVec path = {0};
    pvec_push(&path, (void *)p_expect(P, TRIAD_TOK_IDENT, NULL)->text);
    while (p_at(P, TRIAD_TOK_SYMBOL, ".")) {
        p_eat(P);
        pvec_push(&path, (void *)p_expect(P, TRIAD_TOK_IDENT, NULL)->text);
    }
    p_expect_kw(P, "import");
    PVec names = {0};
    pvec_push(&names, (void *)p_expect(P, TRIAD_TOK_IDENT, NULL)->text);
    while (p_at(P, TRIAD_TOK_SYMBOL, ",")) {
        p_eat(P);
        pvec_push(&names, (void *)p_expect(P, TRIAD_TOK_IDENT, NULL)->text);
    }
    p_eat_semi(P);
    TriadAstNode *n = new_node(P, TRIAD_AST_FROM_IMPORT, p);
    n->u.from_import.path_len = path.len;
    n->u.from_import.path = (const char **)pvec_to_arena_ptrs(P->arena, &path, sizeof(const char *));
    n->u.from_import.names_len = names.len;
    n->u.from_import.names = (const char **)pvec_to_arena_ptrs(P->arena, &names, sizeof(const char *));
    return n;
}

static TriadAstNode *parse_try(Pz *P) {
    TriadPos p = cur_pos(P);
    p_eat(P); /* try */
    TriadAstNode **body = NULL; size_t blen = 0;
    parse_block(P, &body, &blen);
    const char *catch_var = NULL;
    TriadAstNode **catch_body = NULL; size_t clen = 0;
    TriadAstNode **finally_body = NULL; size_t flen = 0;
    if (p_at(P, TRIAD_TOK_KEYWORD, "catch")) {
        p_eat(P);
        if (p_at_kind(P, TRIAD_TOK_IDENT)) {
            catch_var = p_peek(P, 0)->text;
            p_eat(P);
        }
        parse_block(P, &catch_body, &clen);
    }
    if (p_at(P, TRIAD_TOK_KEYWORD, "finally")) {
        p_eat(P);
        parse_block(P, &finally_body, &flen);
    }
    TriadAstNode *n = new_node(P, TRIAD_AST_TRY_CATCH, p);
    n->u.try_catch.body = body;
    n->u.try_catch.body_len = blen;
    n->u.try_catch.catch_var = catch_var;
    n->u.try_catch.catch_body = catch_body;
    n->u.try_catch.catch_body_len = clen;
    n->u.try_catch.finally_body = finally_body;
    n->u.try_catch.finally_body_len = flen;
    return n;
}

static TriadAstNode *parse_throw(Pz *P) {
    TriadPos p = cur_pos(P);
    p_eat(P); /* throw */
    TriadAstNode *val = NULL;
    if (!p_at(P, TRIAD_TOK_SYMBOL, ";") && !p_at(P, TRIAD_TOK_SYMBOL, "}") && !p_at_kind(P, TRIAD_TOK_EOF)) {
        val = parse_expr(P);
    }
    p_eat_semi(P);
    TriadAstNode *n = new_node(P, TRIAD_AST_THROW, p);
    n->u.unary_value.value = val;
    return n;
}

/* ── Triad-native ──────────────────────────────────────────────── */

static TriadAstNode *parse_reg(Pz *P) {
    TriadPos p = cur_pos(P);
    p_eat(P); /* reg */
    const char *name = p_expect(P, TRIAD_TOK_IDENT, NULL)->text;
    const char *regime = NULL;
    if (p_at(P, TRIAD_TOK_SYMBOL, ":")) {
        p_eat(P);
        regime = p_expect(P, TRIAD_TOK_IDENT, NULL)->text;
    }
    TriadAstNode *val = NULL;
    if (p_at(P, TRIAD_TOK_SYMBOL, "=")) {
        p_eat(P);
        val = parse_expr(P);
    }
    int has_overrides = 0;
    PVec overrides = {0};
    if (p_at(P, TRIAD_TOK_SYMBOL, "{")) {
        p_eat(P);
        has_overrides = 1;
        while (!p_at(P, TRIAD_TOK_SYMBOL, "}") && !p_at_kind(P, TRIAD_TOK_EOF)) {
            const char *k = p_eat(P)->text;
            p_expect_sym(P, ":");
            TriadAstNode *v = parse_expr(P);
            p_eat_semi(P);
            TriadStrEntry *e = (TriadStrEntry *)triad_arena_calloc(P->arena, sizeof(TriadStrEntry));
            e->key = k;
            e->value = v;
            pvec_push(&overrides, e);
        }
        p_expect_sym(P, "}");
    }
    p_eat_semi(P);
    TriadAstNode *n = new_node(P, TRIAD_AST_REG, p);
    n->u.reg_stmt.name = name;
    n->u.reg_stmt.regime = regime;
    n->u.reg_stmt.value = val;
    n->u.reg_stmt.has_overrides = has_overrides;
    if (overrides.len) {
        TriadStrEntry *arr = (TriadStrEntry *)triad_arena_alloc(P->arena, overrides.len * sizeof(TriadStrEntry));
        for (size_t k = 0; k < overrides.len; ++k) arr[k] = *(TriadStrEntry *)overrides.items[k];
        n->u.reg_stmt.overrides = arr;
        n->u.reg_stmt.overrides_len = overrides.len;
    }
    pvec_free(&overrides);
    return n;
}

static TriadAstNode *parse_entity(Pz *P) {
    TriadPos p = cur_pos(P);
    p_eat(P); /* entity */
    const char *name = p_expect(P, TRIAD_TOK_IDENT, NULL)->text;
    const char *base = NULL;
    if (p_at(P, TRIAD_TOK_SYMBOL, ":")) {
        p_eat(P);
        base = p_expect(P, TRIAD_TOK_IDENT, NULL)->text;
    }
    p_expect_sym(P, "{");
    PVec fields = {0};
    PVec methods = {0};
    p_skip_semis(P);
    while (!p_at(P, TRIAD_TOK_SYMBOL, "}") && !p_at_kind(P, TRIAD_TOK_EOF)) {
        if (p_at(P, TRIAD_TOK_KEYWORD, "fn")) {
            pvec_push(&methods, parse_fn(P, 0));
        } else if (p_at_kind(P, TRIAD_TOK_IDENT) &&
                   (strcmp(p_peek(P, 0)->text, "memory") == 0 ||
                    strcmp(p_peek(P, 0)->text, "phase") == 0)) {
            const char *k = p_eat(P)->text;
            if (p_at(P, TRIAD_TOK_SYMBOL, "=")) {
                p_eat(P);
                TriadAstNode *v = parse_expr(P);
                TriadStrEntry *e = (TriadStrEntry *)triad_arena_calloc(P->arena, sizeof(TriadStrEntry));
                e->key = k;
                e->value = v;
                pvec_push(&fields, e);
            } else {
                /* memory depth=4 form */
                while (!p_at(P, TRIAD_TOK_SYMBOL, ";") &&
                       !p_at(P, TRIAD_TOK_SYMBOL, "}") &&
                       !p_at_kind(P, TRIAD_TOK_EOF)) {
                    const char *kk = p_eat(P)->text;
                    if (p_at(P, TRIAD_TOK_SYMBOL, "=")) {
                        p_eat(P);
                        TriadAstNode *vv = parse_expr(P);
                        char *combined = (char *)triad_arena_alloc(P->arena, strlen(k) + strlen(kk) + 2);
                        sprintf(combined, "%s_%s", k, kk);
                        TriadStrEntry *e = (TriadStrEntry *)triad_arena_calloc(P->arena, sizeof(TriadStrEntry));
                        e->key = combined;
                        e->value = vv;
                        pvec_push(&fields, e);
                    }
                }
            }
            p_eat_semi(P);
        } else {
            const char *k = p_eat(P)->text;
            p_expect_sym(P, "=");
            TriadAstNode *v = parse_expr(P);
            p_eat_semi(P);
            TriadStrEntry *e = (TriadStrEntry *)triad_arena_calloc(P->arena, sizeof(TriadStrEntry));
            e->key = k;
            e->value = v;
            pvec_push(&fields, e);
        }
        p_skip_semis(P);
    }
    p_expect_sym(P, "}");
    TriadAstNode *n = new_node(P, TRIAD_AST_ENTITY, p);
    n->u.entity_decl.name = name;
    n->u.entity_decl.base = base;
    if (fields.len) {
        TriadStrEntry *arr = (TriadStrEntry *)triad_arena_alloc(P->arena, fields.len * sizeof(TriadStrEntry));
        for (size_t k = 0; k < fields.len; ++k) arr[k] = *(TriadStrEntry *)fields.items[k];
        n->u.entity_decl.fields = arr;
        n->u.entity_decl.fields_len = fields.len;
    }
    pvec_free(&fields);
    n->u.entity_decl.methods_len = methods.len;
    n->u.entity_decl.methods = (TriadAstNode **)pvec_to_arena_ptrs(P->arena, &methods, sizeof(TriadAstNode *));
    return n;
}

static TriadAstNode *parse_run(Pz *P);
static TriadAstNode *parse_world(Pz *P) {
    TriadPos p = cur_pos(P);
    p_eat(P); /* world */
    const char *name = p_expect(P, TRIAD_TOK_IDENT, NULL)->text;
    p_expect_sym(P, "{");
    PVec fields = {0};
    PVec entities = {0};
    PVec body = {0};
    p_skip_semis(P);
    while (!p_at(P, TRIAD_TOK_SYMBOL, "}") && !p_at_kind(P, TRIAD_TOK_EOF)) {
        if (p_at(P, TRIAD_TOK_KEYWORD, "entity")) {
            pvec_push(&entities, parse_entity(P));
        } else if (p_at(P, TRIAD_TOK_KEYWORD, "run")) {
            pvec_push(&body, parse_run(P));
        } else {
            const char *k = p_eat(P)->text;
            p_expect_sym(P, "=");
            TriadAstNode *v = parse_expr(P);
            p_eat_semi(P);
            TriadStrEntry *e = (TriadStrEntry *)triad_arena_calloc(P->arena, sizeof(TriadStrEntry));
            e->key = k;
            e->value = v;
            pvec_push(&fields, e);
        }
        p_skip_semis(P);
    }
    p_expect_sym(P, "}");
    TriadAstNode *n = new_node(P, TRIAD_AST_WORLD, p);
    n->u.world_decl.name = name;
    if (fields.len) {
        TriadStrEntry *arr = (TriadStrEntry *)triad_arena_alloc(P->arena, fields.len * sizeof(TriadStrEntry));
        for (size_t k = 0; k < fields.len; ++k) arr[k] = *(TriadStrEntry *)fields.items[k];
        n->u.world_decl.fields = arr;
        n->u.world_decl.fields_len = fields.len;
    }
    pvec_free(&fields);
    n->u.world_decl.entities_len = entities.len;
    n->u.world_decl.entities = (TriadAstNode **)pvec_to_arena_ptrs(P->arena, &entities, sizeof(TriadAstNode *));
    n->u.world_decl.body_len = body.len;
    n->u.world_decl.body = (TriadAstNode **)pvec_to_arena_ptrs(P->arena, &body, sizeof(TriadAstNode *));
    return n;
}

static TriadAstNode *parse_couple(Pz *P) {
    TriadPos p = cur_pos(P);
    p_eat(P);
    const char *src = p_expect(P, TRIAD_TOK_IDENT, NULL)->text;
    p_expect_sym(P, "->");
    const char *dst = p_expect(P, TRIAD_TOK_IDENT, NULL)->text;
    TriadAstNode *kappa = NULL;
    if (p_at_ident_val(P, "kappa")) {
        p_eat(P);
        p_expect_sym(P, "=");
        kappa = parse_expr(P);
    }
    p_eat_semi(P);
    TriadAstNode *n = new_node(P, TRIAD_AST_COUPLE, p);
    n->u.couple_stmt.src = src;
    n->u.couple_stmt.dst = dst;
    n->u.couple_stmt.kappa = kappa;
    return n;
}

static TriadAstNode *parse_pair(Pz *P) {
    TriadPos p = cur_pos(P);
    p_eat(P);
    p_expect_sym(P, "(");
    const char *a = p_expect(P, TRIAD_TOK_IDENT, NULL)->text;
    p_expect_sym(P, ",");
    const char *b = p_expect(P, TRIAD_TOK_IDENT, NULL)->text;
    p_expect_sym(P, ")");
    TriadAstNode *kappa = NULL;
    TriadAstNode *dur = NULL;
    if (p_at_ident_val(P, "kappa")) {
        p_eat(P);
        p_expect_sym(P, "=");
        kappa = parse_expr(P);
    }
    if (p_at(P, TRIAD_TOK_KEYWORD, "for")) {
        p_eat(P);
        if (p_at_kind(P, TRIAD_TOK_IDENT) && strcmp(p_peek(P, 0)->text, "T") == 0) {
            p_eat(P);
            p_expect_sym(P, "=");
        }
        dur = parse_expr(P);
    }
    p_eat_semi(P);
    TriadAstNode *n = new_node(P, TRIAD_AST_PAIR, p);
    n->u.pair_stmt.a = a;
    n->u.pair_stmt.b = b;
    n->u.pair_stmt.kappa = kappa;
    n->u.pair_stmt.duration = dur;
    return n;
}

static TriadAstNode *parse_ring(Pz *P) {
    TriadPos p = cur_pos(P);
    p_eat(P);
    p_expect_sym(P, "(");
    PVec members = {0};
    pvec_push(&members, (void *)p_expect(P, TRIAD_TOK_IDENT, NULL)->text);
    while (p_at(P, TRIAD_TOK_SYMBOL, ",")) {
        p_eat(P);
        pvec_push(&members, (void *)p_expect(P, TRIAD_TOK_IDENT, NULL)->text);
    }
    p_expect_sym(P, ")");
    TriadAstNode *kappa = NULL;
    TriadAstNode *dur = NULL;
    if (p_at_ident_val(P, "kappa")) {
        p_eat(P);
        p_expect_sym(P, "=");
        kappa = parse_expr(P);
    }
    if (p_at(P, TRIAD_TOK_KEYWORD, "for")) {
        p_eat(P);
        if (p_at_kind(P, TRIAD_TOK_IDENT) && strcmp(p_peek(P, 0)->text, "T") == 0) {
            p_eat(P);
            p_expect_sym(P, "=");
        }
        dur = parse_expr(P);
    }
    p_eat_semi(P);
    TriadAstNode *n = new_node(P, TRIAD_AST_RING, p);
    n->u.ring_stmt.members_len = members.len;
    n->u.ring_stmt.members = (const char **)pvec_to_arena_ptrs(P->arena, &members, sizeof(const char *));
    n->u.ring_stmt.kappa = kappa;
    n->u.ring_stmt.duration = dur;
    return n;
}

static TriadAstNode *parse_observe(Pz *P) {
    TriadPos p = cur_pos(P);
    p_eat(P);
    const char *target = p_expect(P, TRIAD_TOK_IDENT, NULL)->text;
    PVec metrics = {0};
    if (!p_at(P, TRIAD_TOK_SYMBOL, ";") && !p_at_kind(P, TRIAD_TOK_EOF)) {
        pvec_push(&metrics, (void *)p_expect(P, TRIAD_TOK_IDENT, NULL)->text);
        while (p_at(P, TRIAD_TOK_SYMBOL, ",")) {
            p_eat(P);
            if (p_at(P, TRIAD_TOK_KEYWORD, "over_seeds")) break;
            pvec_push(&metrics, (void *)p_expect(P, TRIAD_TOK_IDENT, NULL)->text);
        }
    }
    int over_seeds = 1;
    if (p_at(P, TRIAD_TOK_KEYWORD, "over_seeds")) {
        p_eat(P);
        p_expect_sym(P, "=");
        const TriadToken *num = p_expect(P, TRIAD_TOK_NUMBER, NULL);
        over_seeds = atoi(num->text);
    }
    p_eat_semi(P);
    TriadAstNode *n = new_node(P, TRIAD_AST_OBSERVE, p);
    n->u.observe_stmt.target = target;
    n->u.observe_stmt.metrics_len = metrics.len;
    n->u.observe_stmt.metrics = (const char **)pvec_to_arena_ptrs(P->arena, &metrics, sizeof(const char *));
    n->u.observe_stmt.over_seeds = over_seeds;
    return n;
}

static TriadAstNode *parse_run(Pz *P) {
    TriadPos p = cur_pos(P);
    p_eat(P); /* run */
    TriadAstNode *dur = NULL;
    if (!p_at(P, TRIAD_TOK_SYMBOL, ";") && !p_at(P, TRIAD_TOK_SYMBOL, "}") && !p_at_kind(P, TRIAD_TOK_EOF)) {
        if (p_at_kind(P, TRIAD_TOK_IDENT) && strcmp(p_peek(P, 0)->text, "T") == 0) {
            p_eat(P);
            p_expect_sym(P, "=");
        }
        dur = parse_expr(P);
    }
    p_eat_semi(P);
    TriadAstNode *n = new_node(P, TRIAD_AST_RUN, p);
    n->u.run_stmt.duration = dur;
    return n;
}

static TriadAstNode *parse_annotation(Pz *P) {
    TriadPos p = cur_pos(P);
    p_eat(P); /* @ */
    const char *name = p_peek(P, 0)->text;
    p_eat(P);
    const char *args = "";
    if (p_at(P, TRIAD_TOK_SYMBOL, "(")) {
        p_eat(P);
        /* concatenate raw token text up to ) */
        Pz tmp = *P;
        (void)tmp;
        size_t total = 0;
        size_t start_i = P->i;
        while (!p_at(P, TRIAD_TOK_SYMBOL, ")") && !p_at_kind(P, TRIAD_TOK_EOF)) {
            const char *tx = p_peek(P, 0)->text;
            total += tx ? strlen(tx) : 0;
            p_eat(P);
        }
        char *buf = (char *)triad_arena_alloc(P->arena, total + 1);
        size_t pos = 0;
        for (size_t k = start_i; k < P->i; ++k) {
            const char *tx = P->toks[k].text;
            if (!tx) continue;
            size_t l = strlen(tx);
            memcpy(buf + pos, tx, l);
            pos += l;
        }
        buf[pos] = '\0';
        args = buf;
        p_expect_sym(P, ")");
    }
    p_eat_semi(P);
    TriadAstNode *n = new_node(P, TRIAD_AST_ANNOTATION, p);
    n->u.annotation_stmt.key = name;
    n->u.annotation_stmt.args = args;
    return n;
}

/* ── Expressions ────────────────────────────────────────────────── */

static TriadAstNode *parse_expr(Pz *P) { return parse_or(P); }

static TriadAstNode *parse_or(Pz *P) {
    TriadAstNode *left = parse_and(P);
    while (p_at(P, TRIAD_TOK_KEYWORD, "or")) {
        p_eat(P);
        TriadAstNode *right = parse_and(P);
        TriadAstNode *n = new_node(P, TRIAD_AST_BINOP, left->pos);
        n->u.op.op = "or";
        n->u.op.left = left;
        n->u.op.right = right;
        left = n;
    }
    return left;
}

static TriadAstNode *parse_and(Pz *P) {
    TriadAstNode *left = parse_not(P);
    while (p_at(P, TRIAD_TOK_KEYWORD, "and")) {
        p_eat(P);
        TriadAstNode *right = parse_not(P);
        TriadAstNode *n = new_node(P, TRIAD_AST_BINOP, left->pos);
        n->u.op.op = "and";
        n->u.op.left = left;
        n->u.op.right = right;
        left = n;
    }
    return left;
}

static TriadAstNode *parse_not(Pz *P) {
    if (p_at(P, TRIAD_TOK_KEYWORD, "not")) {
        TriadPos p = cur_pos(P);
        p_eat(P);
        TriadAstNode *operand = parse_not(P);
        TriadAstNode *n = new_node(P, TRIAD_AST_UNARYOP, p);
        n->u.op.op = "not";
        n->u.op.right = operand;
        return n;
    }
    return parse_comparison(P);
}

static int is_cmp_op(const char *v) {
    return strcmp(v, "==") == 0 || strcmp(v, "!=") == 0 ||
           strcmp(v, "<")  == 0 || strcmp(v, "<=") == 0 ||
           strcmp(v, ">")  == 0 || strcmp(v, ">=") == 0;
}

static TriadAstNode *parse_comparison(Pz *P) {
    TriadAstNode *left = parse_add(P);
    while (p_at_kind(P, TRIAD_TOK_SYMBOL) && is_cmp_op(p_peek(P, 0)->text)) {
        const char *op = p_eat(P)->text;
        TriadAstNode *right = parse_add(P);
        TriadAstNode *n = new_node(P, TRIAD_AST_BINOP, left->pos);
        n->u.op.op = op;
        n->u.op.left = left;
        n->u.op.right = right;
        left = n;
    }
    return left;
}

static TriadAstNode *parse_add(Pz *P) {
    TriadAstNode *left = parse_mul(P);
    while (p_at_kind(P, TRIAD_TOK_SYMBOL) &&
           (strcmp(p_peek(P, 0)->text, "+") == 0 || strcmp(p_peek(P, 0)->text, "-") == 0)) {
        const char *op = p_eat(P)->text;
        TriadAstNode *right = parse_mul(P);
        TriadAstNode *n = new_node(P, TRIAD_AST_BINOP, left->pos);
        n->u.op.op = op;
        n->u.op.left = left;
        n->u.op.right = right;
        left = n;
    }
    return left;
}

static int is_mul_op(const char *v) {
    return strcmp(v, "*") == 0 || strcmp(v, "/") == 0 ||
           strcmp(v, "%") == 0 || strcmp(v, "@") == 0;
}

static TriadAstNode *parse_mul(Pz *P) {
    TriadAstNode *left = parse_power(P);
    while (p_at_kind(P, TRIAD_TOK_SYMBOL) && is_mul_op(p_peek(P, 0)->text)) {
        const char *op = p_eat(P)->text;
        TriadAstNode *right = parse_power(P);
        TriadAstNode *n = new_node(P, TRIAD_AST_BINOP, left->pos);
        n->u.op.op = op;
        n->u.op.left = left;
        n->u.op.right = right;
        left = n;
    }
    return left;
}

static TriadAstNode *parse_power(Pz *P) {
    TriadAstNode *left = parse_unary(P);
    if (p_at(P, TRIAD_TOK_SYMBOL, "**")) {
        p_eat(P);
        TriadAstNode *right = parse_power(P);
        TriadAstNode *n = new_node(P, TRIAD_AST_BINOP, left->pos);
        n->u.op.op = "**";
        n->u.op.left = left;
        n->u.op.right = right;
        return n;
    }
    return left;
}

static TriadAstNode *parse_unary(Pz *P) {
    if (p_at(P, TRIAD_TOK_KEYWORD, "await")) {
        TriadPos p = cur_pos(P);
        p_eat(P);
        TriadAstNode *operand = parse_unary(P);
        TriadAstNode *n = new_node(P, TRIAD_AST_AWAIT_EXPR, p);
        n->u.unary_value.value = operand;
        return n;
    }
    if (p_at(P, TRIAD_TOK_SYMBOL, "-")) {
        TriadPos p = cur_pos(P);
        p_eat(P);
        TriadAstNode *operand = parse_unary(P);
        TriadAstNode *n = new_node(P, TRIAD_AST_UNARYOP, p);
        n->u.op.op = "-";
        n->u.op.right = operand;
        return n;
    }
    if (p_at(P, TRIAD_TOK_SYMBOL, "!")) {
        TriadPos p = cur_pos(P);
        p_eat(P);
        TriadAstNode *operand = parse_unary(P);
        TriadAstNode *n = new_node(P, TRIAD_AST_UNARYOP, p);
        n->u.op.op = "not";
        n->u.op.right = operand;
        return n;
    }
    return parse_postfix(P);
}

static void parse_call_args(Pz *P,
                            TriadAstNode ***args, size_t *args_len,
                            TriadKwArg     **kwargs, size_t *kwargs_len) {
    PVec av = {0};
    PVec kv = {0};
    while (!p_at(P, TRIAD_TOK_SYMBOL, ")") && !p_at_kind(P, TRIAD_TOK_EOF)) {
        if (p_at(P, TRIAD_TOK_SYMBOL, "**")) {
            TriadPos pp = cur_pos(P);
            p_eat(P);
            TriadAstNode *v = parse_expr(P);
            TriadAstNode *n = new_node(P, TRIAD_AST_UNARYOP, pp);
            n->u.op.op = "**";
            n->u.op.right = v;
            pvec_push(&av, n);
        } else if (p_at(P, TRIAD_TOK_SYMBOL, "*")) {
            TriadPos pp = cur_pos(P);
            p_eat(P);
            TriadAstNode *v = parse_expr(P);
            TriadAstNode *n = new_node(P, TRIAD_AST_UNARYOP, pp);
            n->u.op.op = "*";
            n->u.op.right = v;
            pvec_push(&av, n);
        } else if (p_at_kind(P, TRIAD_TOK_IDENT) &&
                   p_peek(P, 1)->kind == TRIAD_TOK_SYMBOL &&
                   strcmp(p_peek(P, 1)->text, "=") == 0) {
            const char *name = p_eat(P)->text;
            p_eat(P); /* = */
            TriadKwArg *kw = (TriadKwArg *)triad_arena_calloc(P->arena, sizeof(TriadKwArg));
            kw->name = name;
            kw->value = parse_expr(P);
            pvec_push(&kv, kw);
        } else {
            pvec_push(&av, parse_expr(P));
        }
        if (!p_at(P, TRIAD_TOK_SYMBOL, ")")) p_expect_sym(P, ",");
    }
    *args_len = av.len;
    *args = (TriadAstNode **)pvec_to_arena_ptrs(P->arena, &av, sizeof(TriadAstNode *));
    *kwargs_len = kv.len;
    if (kv.len) {
        TriadKwArg *arr = (TriadKwArg *)triad_arena_alloc(P->arena, kv.len * sizeof(TriadKwArg));
        for (size_t k = 0; k < kv.len; ++k) arr[k] = *(TriadKwArg *)kv.items[k];
        *kwargs = arr;
    } else *kwargs = NULL;
    pvec_free(&kv);
}

static TriadAstNode *parse_postfix(Pz *P) {
    TriadAstNode *expr = parse_primary(P);
    for (;;) {
        if (p_at(P, TRIAD_TOK_SYMBOL, "(")) {
            p_eat(P);
            TriadAstNode **args = NULL; size_t args_len = 0;
            TriadKwArg *kwargs = NULL; size_t kwargs_len = 0;
            parse_call_args(P, &args, &args_len, &kwargs, &kwargs_len);
            p_expect_sym(P, ")");
            TriadAstNode *n = new_node(P, TRIAD_AST_CALL, expr->pos);
            n->u.call.func_or_obj = expr;
            n->u.call.args = args;
            n->u.call.args_len = args_len;
            n->u.call.kwargs = kwargs;
            n->u.call.kwargs_len = kwargs_len;
            expr = n;
        } else if (p_at(P, TRIAD_TOK_SYMBOL, "[")) {
            p_eat(P);
            TriadPos sp = cur_pos(P);
            TriadAstNode *idx_node = NULL;
            if (p_at(P, TRIAD_TOK_SYMBOL, ":")) {
                p_eat(P);
                TriadAstNode *end = NULL, *step = NULL;
                if (!p_at(P, TRIAD_TOK_SYMBOL, "]") && !p_at(P, TRIAD_TOK_SYMBOL, ":"))
                    end = parse_expr(P);
                if (p_at(P, TRIAD_TOK_SYMBOL, ":")) {
                    p_eat(P);
                    if (!p_at(P, TRIAD_TOK_SYMBOL, "]")) step = parse_expr(P);
                }
                p_expect_sym(P, "]");
                TriadAstNode *sl = new_node(P, TRIAD_AST_SLICE, sp);
                sl->u.slice.start = NULL; sl->u.slice.end = end; sl->u.slice.step = step;
                idx_node = sl;
            } else {
                TriadAstNode *first = parse_expr(P);
                if (p_at(P, TRIAD_TOK_SYMBOL, ":")) {
                    p_eat(P);
                    TriadAstNode *end = NULL, *step = NULL;
                    if (!p_at(P, TRIAD_TOK_SYMBOL, "]") && !p_at(P, TRIAD_TOK_SYMBOL, ":"))
                        end = parse_expr(P);
                    if (p_at(P, TRIAD_TOK_SYMBOL, ":")) {
                        p_eat(P);
                        if (!p_at(P, TRIAD_TOK_SYMBOL, "]")) step = parse_expr(P);
                    }
                    p_expect_sym(P, "]");
                    TriadAstNode *sl = new_node(P, TRIAD_AST_SLICE, sp);
                    sl->u.slice.start = first; sl->u.slice.end = end; sl->u.slice.step = step;
                    idx_node = sl;
                } else {
                    p_expect_sym(P, "]");
                    idx_node = first;
                }
            }
            TriadAstNode *n = new_node(P, TRIAD_AST_INDEX, expr->pos);
            n->u.index.obj = expr;
            n->u.index.index = idx_node;
            expr = n;
        } else if (p_at(P, TRIAD_TOK_SYMBOL, ".")) {
            p_eat(P);
            const char *fname = p_expect(P, TRIAD_TOK_IDENT, NULL)->text;
            if (p_at(P, TRIAD_TOK_SYMBOL, "(")) {
                p_eat(P);
                TriadAstNode **args = NULL; size_t args_len = 0;
                TriadKwArg *kwargs = NULL; size_t kwargs_len = 0;
                parse_call_args(P, &args, &args_len, &kwargs, &kwargs_len);
                p_expect_sym(P, ")");
                TriadAstNode *n = new_node(P, TRIAD_AST_METHOD_CALL, expr->pos);
                n->u.call.func_or_obj = expr;
                n->u.call.method = fname;
                n->u.call.args = args;
                n->u.call.args_len = args_len;
                n->u.call.kwargs = kwargs;
                n->u.call.kwargs_len = kwargs_len;
                expr = n;
            } else {
                TriadAstNode *n = new_node(P, TRIAD_AST_FIELD, expr->pos);
                n->u.field.obj = expr;
                n->u.field.field = fname;
                expr = n;
            }
        } else {
            break;
        }
    }
    return expr;
}

static TriadAstNode *parse_primary(Pz *P) {
    const TriadToken *t = p_peek(P, 0);
    TriadPos p = cur_pos(P);

    if (t->kind == TRIAD_TOK_NUMBER) {
        p_eat(P);
        const char *v = t->text;
        size_t L = strlen(v);
        int has_dot_or_exp = 0;
        for (size_t k = 0; k < L; ++k) {
            if (v[k] == '.' || v[k] == 'e' || v[k] == 'E') { has_dot_or_exp = 1; break; }
        }
        if (has_dot_or_exp) {
            TriadAstNode *n = new_node(P, TRIAD_AST_FLOAT_LIT, p);
            n->u.float_val = strtod(v, NULL);
            return n;
        }
        TriadAstNode *n = new_node(P, TRIAD_AST_INT_LIT, p);
        if (L > 2 && v[0] == '0' && (v[1] == 'x' || v[1] == 'X'))      n->u.int_val = strtoll(v + 2, NULL, 16);
        else if (L > 2 && v[0] == '0' && (v[1] == 'b' || v[1] == 'B')) n->u.int_val = strtoll(v + 2, NULL, 2);
        else                                                            n->u.int_val = strtoll(v, NULL, 10);
        return n;
    }

    if (t->kind == TRIAD_TOK_STRING) {
        p_eat(P);
        TriadAstNode *n = new_node(P, TRIAD_AST_STRING_LIT, p);
        n->u.string_val = t->text;
        return n;
    }

    if (t->kind == TRIAD_TOK_FSTRING) {
        p_eat(P);
        TriadFStringAstPart *aparts = (TriadFStringAstPart *)triad_arena_calloc(
            P->arena, t->parts_len * sizeof(TriadFStringAstPart));
        for (size_t k = 0; k < t->parts_len; ++k) {
            if (!t->parts[k].is_expr) {
                aparts[k].is_expr = 0;
                aparts[k].text    = t->parts[k].text;
            } else {
                /* Re-lex + re-parse the expression source. */
                TriadTokenList sub = {0};
                TriadDiag sub_diag = {0};
                if (triad_tokenize(P->arena, t->parts[k].text, "<fstring>", &sub, &sub_diag) != 0) {
                    NORETURN_parse_error(P, t->line, t->col, "fstring lex: %s", sub_diag.msg ? sub_diag.msg : "?");
                }
                Pz sub_P = {0};
                sub_P.arena = P->arena;
                sub_P.toks  = sub.items;
                sub_P.ntoks = sub.len;
                sub_P.i     = 0;
                sub_P.file  = "<fstring>";
                sub_P.diag  = P->diag;
                if (setjmp(sub_P.err_jmp) != 0) {
                    longjmp(P->err_jmp, 1);
                }
                aparts[k].is_expr = 1;
                aparts[k].expr    = parse_expr(&sub_P);
            }
        }
        TriadAstNode *n = new_node(P, TRIAD_AST_FSTRING, p);
        n->u.fstring.parts = aparts;
        n->u.fstring.parts_len = t->parts_len;
        return n;
    }

    if (t->kind == TRIAD_TOK_KEYWORD) {
        if (strcmp(t->text, "true") == 0) {
            p_eat(P);
            TriadAstNode *n = new_node(P, TRIAD_AST_BOOL_LIT, p);
            n->u.bool_val = 1; return n;
        }
        if (strcmp(t->text, "false") == 0) {
            p_eat(P);
            TriadAstNode *n = new_node(P, TRIAD_AST_BOOL_LIT, p);
            n->u.bool_val = 0; return n;
        }
        if (strcmp(t->text, "none") == 0) {
            p_eat(P);
            return new_node(P, TRIAD_AST_NONE_LIT, p);
        }
        if (strcmp(t->text, "fn") == 0) {
            return parse_lambda(P);
        }
        if (strcmp(t->text, "self") == 0) {
            p_eat(P);
            TriadAstNode *n = new_node(P, TRIAD_AST_IDENT, p);
            n->u.ident_name = "self";
            return n;
        }
        if (strcmp(t->text, "yield") == 0) {
            p_eat(P);
            TriadAstNode *val = NULL;
            if (!p_at(P, TRIAD_TOK_SYMBOL, ";") && !p_at(P, TRIAD_TOK_SYMBOL, "}") &&
                !p_at(P, TRIAD_TOK_SYMBOL, ")") && !p_at_kind(P, TRIAD_TOK_EOF)) {
                val = parse_expr(P);
            }
            TriadAstNode *n = new_node(P, TRIAD_AST_YIELD_EXPR, p);
            n->u.unary_value.value = val;
            return n;
        }
        if (strcmp(t->text, "await") == 0) {
            p_eat(P);
            TriadAstNode *operand = parse_unary(P);
            TriadAstNode *n = new_node(P, TRIAD_AST_AWAIT_EXPR, p);
            n->u.unary_value.value = operand;
            return n;
        }
    }

    if (t->kind == TRIAD_TOK_IDENT) {
        p_eat(P);
        TriadAstNode *n = new_node(P, TRIAD_AST_IDENT, p);
        n->u.ident_name = t->text;
        return n;
    }

    if (t->kind == TRIAD_TOK_SYMBOL && strcmp(t->text, "(") == 0) {
        p_eat(P);
        if (p_at(P, TRIAD_TOK_SYMBOL, ")")) {
            p_eat(P);
            TriadAstNode *n = new_node(P, TRIAD_AST_TUPLE, p);
            n->u.list.elements = NULL;
            n->u.list.elements_len = 0;
            return n;
        }
        TriadAstNode *first = parse_expr(P);
        if (p_at(P, TRIAD_TOK_SYMBOL, ",")) {
            PVec elems = {0};
            pvec_push(&elems, first);
            while (p_at(P, TRIAD_TOK_SYMBOL, ",")) {
                p_eat(P);
                if (p_at(P, TRIAD_TOK_SYMBOL, ")")) break;
                pvec_push(&elems, parse_expr(P));
            }
            p_expect_sym(P, ")");
            TriadAstNode *n = new_node(P, TRIAD_AST_TUPLE, p);
            n->u.list.elements_len = elems.len;
            n->u.list.elements = (TriadAstNode **)pvec_to_arena_ptrs(P->arena, &elems, sizeof(TriadAstNode *));
            return n;
        }
        p_expect_sym(P, ")");
        return first;
    }

    if (t->kind == TRIAD_TOK_SYMBOL && strcmp(t->text, "[") == 0) {
        p_eat(P);
        if (p_at(P, TRIAD_TOK_SYMBOL, "]")) {
            p_eat(P);
            TriadAstNode *n = new_node(P, TRIAD_AST_LIST, p);
            n->u.list.elements = NULL;
            n->u.list.elements_len = 0;
            return n;
        }
        TriadAstNode *first = parse_expr(P);
        if (p_at(P, TRIAD_TOK_KEYWORD, "for")) {
            p_eat(P);
            const char *var = p_expect(P, TRIAD_TOK_IDENT, NULL)->text;
            p_expect_kw(P, "in");
            TriadAstNode *it = parse_expr(P);
            TriadAstNode *cond = NULL;
            if (p_at(P, TRIAD_TOK_KEYWORD, "if")) {
                p_eat(P);
                cond = parse_expr(P);
            }
            p_expect_sym(P, "]");
            TriadAstNode *n = new_node(P, TRIAD_AST_LIST_COMP, p);
            n->u.list_comp.expr = first;
            n->u.list_comp.var  = var;
            n->u.list_comp.iter = it;
            n->u.list_comp.condition = cond;
            return n;
        }
        PVec elems = {0};
        pvec_push(&elems, first);
        while (p_at(P, TRIAD_TOK_SYMBOL, ",")) {
            p_eat(P);
            if (p_at(P, TRIAD_TOK_SYMBOL, "]")) break;
            pvec_push(&elems, parse_expr(P));
        }
        p_expect_sym(P, "]");
        TriadAstNode *n = new_node(P, TRIAD_AST_LIST, p);
        n->u.list.elements_len = elems.len;
        n->u.list.elements = (TriadAstNode **)pvec_to_arena_ptrs(P->arena, &elems, sizeof(TriadAstNode *));
        return n;
    }

    if (t->kind == TRIAD_TOK_SYMBOL && strcmp(t->text, "{") == 0) {
        p_eat(P);
        PVec pairs = {0};
        while (!p_at(P, TRIAD_TOK_SYMBOL, "}") && !p_at_kind(P, TRIAD_TOK_EOF)) {
            TriadMapPair *pp = (TriadMapPair *)triad_arena_calloc(P->arena, sizeof(TriadMapPair));
            pp->key = parse_expr(P);
            p_expect_sym(P, ":");
            pp->value = parse_expr(P);
            pvec_push(&pairs, pp);
            if (!p_at(P, TRIAD_TOK_SYMBOL, "}")) p_expect_sym(P, ",");
        }
        p_expect_sym(P, "}");
        TriadAstNode *n = new_node(P, TRIAD_AST_MAP, p);
        if (pairs.len) {
            TriadMapPair *arr = (TriadMapPair *)triad_arena_alloc(P->arena, pairs.len * sizeof(TriadMapPair));
            for (size_t k = 0; k < pairs.len; ++k) arr[k] = *(TriadMapPair *)pairs.items[k];
            n->u.map.pairs = arr;
            n->u.map.pairs_len = pairs.len;
        }
        pvec_free(&pairs);
        return n;
    }

    NORETURN_parse_error(P, t->line, t->col, "unexpected token %s '%s'",
                         kind_name(t->kind), t->text ? t->text : "");
}

static TriadAstNode *parse_lambda(Pz *P) {
    TriadPos p = cur_pos(P);
    p_eat(P); /* fn */
    p_expect_sym(P, "(");
    TriadParam *params = NULL; size_t params_len = 0;
    parse_params(P, &params, &params_len);
    p_expect_sym(P, ")");
    TriadAstNode **body = NULL; size_t body_len = 0;
    parse_block(P, &body, &body_len);
    TriadAstNode *n = new_node(P, TRIAD_AST_LAMBDA, p);
    n->u.lambda.params = params;
    n->u.lambda.params_len = params_len;
    n->u.lambda.body = body;
    n->u.lambda.body_len = body_len;
    return n;
}

/* ── Top-level dispatch ─────────────────────────────────────────── */

static TriadAstNode *parse_stmt(Pz *P) {
    const TriadToken *t = p_peek(P, 0);

    if (t->kind == TRIAD_TOK_KEYWORD) {
        const char *v = t->text;
        if (strcmp(v, "let") == 0)      return parse_let(P);
        if (strcmp(v, "const") == 0)    return parse_const(P);
        if (strcmp(v, "fn") == 0)       return parse_fn(P, 0);
        if (strcmp(v, "async") == 0) {
            const TriadToken *nx = p_peek(P, 1);
            if (nx->kind == TRIAD_TOK_KEYWORD && strcmp(nx->text, "fn") == 0)
                return parse_fn(P, 1);
        }
        if (strcmp(v, "if") == 0)       return parse_if(P);
        if (strcmp(v, "for") == 0)      return parse_for(P);
        if (strcmp(v, "while") == 0)    return parse_while(P);
        if (strcmp(v, "return") == 0)   return parse_return(P);
        if (strcmp(v, "yield") == 0)    return parse_yield_stmt(P);
        if (strcmp(v, "break") == 0)    { TriadPos p = cur_pos(P); p_eat(P); p_eat_semi(P); return new_node(P, TRIAD_AST_BREAK, p); }
        if (strcmp(v, "continue") == 0) { TriadPos p = cur_pos(P); p_eat(P); p_eat_semi(P); return new_node(P, TRIAD_AST_CONTINUE, p); }
        if (strcmp(v, "type") == 0)     return parse_type_decl(P);
        if (strcmp(v, "class") == 0)    return parse_class(P);
        if (strcmp(v, "match") == 0)    return parse_match(P);
        if (strcmp(v, "import") == 0)   return parse_import(P);
        if (strcmp(v, "from") == 0)     return parse_from_import(P);
        if (strcmp(v, "try") == 0)      return parse_try(P);
        if (strcmp(v, "throw") == 0)    return parse_throw(P);
        if (strcmp(v, "reg") == 0)      return parse_reg(P);
        if (strcmp(v, "entity") == 0)   return parse_entity(P);
        if (strcmp(v, "world") == 0)    return parse_world(P);
        if (strcmp(v, "couple") == 0)   return parse_couple(P);
        if (strcmp(v, "pair") == 0)     return parse_pair(P);
        if (strcmp(v, "ring") == 0)     return parse_ring(P);
        if (strcmp(v, "observe") == 0 || strcmp(v, "OBSERVE") == 0) return parse_observe(P);
        if (strcmp(v, "run") == 0)      return parse_run(P);
    }

    if (t->kind == TRIAD_TOK_SYMBOL && strcmp(t->text, "@") == 0)
        return parse_annotation(P);

    TriadAstNode *expr = parse_expr(P);
    if (p_at(P, TRIAD_TOK_SYMBOL, "=")) {
        p_eat(P);
        TriadAstNode *val = parse_expr(P);
        p_eat_semi(P);
        TriadAstNode *n = new_node(P, TRIAD_AST_ASSIGN, expr->pos);
        n->u.assign_stmt.target = expr;
        n->u.assign_stmt.value  = val;
        return n;
    }
    p_eat_semi(P);
    TriadAstNode *n = new_node(P, TRIAD_AST_EXPR_STMT, expr->pos);
    n->u.expr_stmt.expr = expr;
    return n;
}

/* ── Public entry ───────────────────────────────────────────────── */

TriadAstNode *triad_parse_tokens(TriadArena *arena, const TriadTokenList *tokens,
                                 const char *file, TriadDiag *diag) {
    Pz P = {0};
    P.arena = arena;
    P.toks  = tokens->items;
    P.ntoks = tokens->len;
    P.i     = 0;
    P.file  = file ? triad_arena_strdup(arena, file) : "";
    P.diag  = diag;

    if (setjmp(P.err_jmp) != 0) {
        return NULL;
    }

    PVec body = {0};
    p_skip_semis(&P);
    while (!p_at_kind(&P, TRIAD_TOK_EOF)) {
        pvec_push(&body, parse_stmt(&P));
        p_skip_semis(&P);
    }
    TriadPos zero = {0};
    zero.file = P.file;
    TriadAstNode *mod = new_node(&P, TRIAD_AST_MODULE, zero);
    mod->u.module.name = "";
    mod->u.module.file = P.file;
    mod->u.module.body_len = body.len;
    mod->u.module.body = (TriadAstNode **)pvec_to_arena_ptrs(P.arena, &body, sizeof(TriadAstNode *));
    return mod;
}

TriadAstNode *triad_parse_source(TriadArena *arena, const char *src,
                                 const char *file, TriadDiag *diag) {
    TriadTokenList toks = {0};
    if (triad_tokenize(arena, src, file, &toks, diag) != 0) return NULL;
    return triad_parse_tokens(arena, &toks, file, diag);
}
