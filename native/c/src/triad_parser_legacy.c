/* triad_parser_legacy.c — Port of frontend/parser.py (v1 DSL). */
#include "triad_frontend.h"

#include <setjmp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    TriadArena                 *arena;
    const TriadLegacyToken     *toks;
    size_t                      ntoks;
    size_t                      i;
    TriadDiag                  *diag;
    jmp_buf                     err_jmp;
} LP;

typedef struct {
    void  **items;
    size_t  len;
    size_t  cap;
} V;
static void V_push(V *v, void *p) {
    if (v->len == v->cap) {
        size_t nc = v->cap ? v->cap * 2 : 8;
        v->items = (void **)realloc(v->items, nc * sizeof(void *));
        v->cap = nc;
    }
    v->items[v->len++] = p;
}
static void V_free(V *v) { free(v->items); v->items = NULL; v->len = v->cap = 0; }

static void NORETURN_lperr(LP *P, int line, int col, const char *fmt, ...)
    __attribute__((noreturn));
static void NORETURN_lperr(LP *P, int line, int col, const char *fmt, ...) {
    char buf[256];
    va_list ap; va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap); va_end(ap);
    if (P->diag) {
        P->diag->line = line; P->diag->col = col;
        P->diag->file = "";
        P->diag->kind = "PARSE";
        P->diag->msg  = triad_arena_strdup(P->arena, buf);
    }
    longjmp(P->err_jmp, 1);
}

static const TriadLegacyToken *lp_peek(LP *P, size_t off) {
    size_t idx = P->i + off;
    if (idx < P->ntoks) return &P->toks[idx];
    return &P->toks[P->ntoks - 1];
}
static const TriadLegacyToken *lp_adv(LP *P) {
    const TriadLegacyToken *t = &P->toks[P->i++];
    return t;
}
static int lp_check(LP *P, TriadLegacyTokKind k, const char *value) {
    const TriadLegacyToken *t = lp_peek(P, 0);
    if (t->kind != k) return 0;
    if (!value) return 1;
    return t->text && strcmp(t->text, value) == 0;
}
static const TriadLegacyToken *lp_expect(LP *P, TriadLegacyTokKind k, const char *value) {
    const TriadLegacyToken *t = lp_peek(P, 0);
    if (t->kind != k || (value && (!t->text || strcmp(t->text, value) != 0))) {
        NORETURN_lperr(P, t->line, t->col,
            "expected %s%s%s, got %d '%s'",
            value ? "" : "kind ",
            value ? value : "",
            value ? "" : "",
            (int)k, t->text ? t->text : "");
    }
    return lp_adv(P);
}
static const TriadLegacyToken *lp_expect_sym(LP *P, const char *v) { return lp_expect(P, TRIAD_LTOK_SYMBOL, v); }
static const TriadLegacyToken *lp_expect_kw (LP *P, const char *v) { return lp_expect(P, TRIAD_LTOK_KEYWORD, v); }
static const TriadLegacyToken *lp_expect_kind(LP *P, TriadLegacyTokKind k) {
    return lp_expect(P, k, NULL);
}

static TriadLegacyNode *lnew(LP *P, TriadLegacyKind k, int line) {
    TriadLegacyNode *n = (TriadLegacyNode *)triad_arena_calloc(P->arena, sizeof(TriadLegacyNode));
    n->kind = k;
    n->line = line;
    return n;
}

/* ── Forward ────────────────────────────────────────────────────── */
static TriadLegacyNode *lparse_stmt(LP *P);
static void             lparse_block(LP *P, TriadLegacyNode ***out, size_t *out_len);
static TriadLegacyNode *lparse_atom_expr(LP *P);
static TriadLegacyVal   lparse_atom_or_tuple(LP *P);

/* ── Atoms ─────────────────────────────────────────────────────── */

static TriadLegacyNode *lparse_atom_expr(LP *P) {
    const TriadLegacyToken *t = lp_peek(P, 0);
    if (t->kind == TRIAD_LTOK_NUMBER) {
        lp_adv(P);
        TriadLegacyNode *n = lnew(P, TRIAD_LAST_NUM_LIT, t->line);
        const char *v = t->text;
        int is_int = 1;
        for (const char *p = v; *p; ++p) if (*p == '.' || *p == 'e' || *p == 'E') { is_int = 0; break; }
        n->u.num.is_int = is_int;
        n->u.num.value  = strtod(v, NULL);
        return n;
    }
    if (t->kind == TRIAD_LTOK_KEYWORD && (strcmp(t->text, "true") == 0 || strcmp(t->text, "false") == 0)) {
        lp_adv(P);
        TriadLegacyNode *n = lnew(P, TRIAD_LAST_BOOL_LIT, t->line);
        n->u.boolean.value = (strcmp(t->text, "true") == 0);
        return n;
    }
    if (t->kind == TRIAD_LTOK_STRING) {
        lp_adv(P);
        TriadLegacyNode *n = lnew(P, TRIAD_LAST_STR_LIT, t->line);
        n->u.str.value = t->text;
        return n;
    }
    if (t->kind == TRIAD_LTOK_IDENT) {
        lp_adv(P);
        /* function-call form: from_file(...) / from_checkpoint(...) */
        if (lp_check(P, TRIAD_LTOK_SYMBOL, "(")) {
            lp_adv(P);
            V args = {0};
            if (!lp_check(P, TRIAD_LTOK_SYMBOL, ")")) {
                V_push(&args, lparse_atom_expr(P));
                while (lp_check(P, TRIAD_LTOK_SYMBOL, ",")) {
                    lp_adv(P);
                    V_push(&args, lparse_atom_expr(P));
                }
            }
            lp_expect_sym(P, ")");
            if (strcmp(t->text, "from_file") == 0 && args.len == 1) {
                TriadLegacyNode *a0 = (TriadLegacyNode *)args.items[0];
                if (a0->kind == TRIAD_LAST_STR_LIT) {
                    size_t need = strlen("__from_file__:") + strlen(a0->u.str.value) + 1;
                    char *combined = (char *)triad_arena_alloc(P->arena, need);
                    snprintf(combined, need, "__from_file__:%s", a0->u.str.value);
                    TriadLegacyNode *n = lnew(P, TRIAD_LAST_STR_LIT, t->line);
                    n->u.str.value = combined;
                    V_free(&args);
                    return n;
                }
            }
            if (strcmp(t->text, "from_checkpoint") == 0 && args.len == 1) {
                TriadLegacyNode *a0 = (TriadLegacyNode *)args.items[0];
                if (a0->kind == TRIAD_LAST_STR_LIT) {
                    size_t need = strlen("__from_checkpoint__:") + strlen(a0->u.str.value) + 1;
                    char *combined = (char *)triad_arena_alloc(P->arena, need);
                    snprintf(combined, need, "__from_checkpoint__:%s", a0->u.str.value);
                    TriadLegacyNode *n = lnew(P, TRIAD_LAST_STR_LIT, t->line);
                    n->u.str.value = combined;
                    V_free(&args);
                    return n;
                }
            }
            V_free(&args);
            TriadLegacyNode *n = lnew(P, TRIAD_LAST_IDENT_REF, t->line);
            n->u.ident.name = t->text;
            return n;
        }
        TriadLegacyNode *n = lnew(P, TRIAD_LAST_IDENT_REF, t->line);
        n->u.ident.name = t->text;
        return n;
    }
    NORETURN_lperr(P, t->line, t->col, "expected atom, got '%s'", t->text ? t->text : "");
}

static TriadLegacyVal node_to_val(TriadLegacyNode *n) {
    TriadLegacyVal v = {0};
    if (!n) { v.kind = TRIAD_LVAL_INT; v.int_val = 0; return v; }
    switch (n->kind) {
        case TRIAD_LAST_NUM_LIT:
            if (n->u.num.is_int) { v.kind = TRIAD_LVAL_INT; v.int_val = (long long)n->u.num.value; }
            else                  { v.kind = TRIAD_LVAL_FLOAT; v.float_val = n->u.num.value; }
            break;
        case TRIAD_LAST_BOOL_LIT:
            v.kind = TRIAD_LVAL_BOOL; v.bool_val = n->u.boolean.value;
            break;
        case TRIAD_LAST_STR_LIT:
            v.kind = TRIAD_LVAL_STR; v.str_val = n->u.str.value;
            break;
        case TRIAD_LAST_IDENT_REF:
            v.kind = TRIAD_LVAL_IDENT; v.str_val = n->u.ident.name;
            break;
        default:
            v.kind = TRIAD_LVAL_STR; v.str_val = "";
            break;
    }
    return v;
}

static TriadLegacyVal lparse_atom_or_tuple(LP *P) {
    if (lp_check(P, TRIAD_LTOK_SYMBOL, "(")) {
        int line = lp_peek(P, 0)->line;
        (void)line;
        lp_adv(P);
        V items = {0};
        if (!lp_check(P, TRIAD_LTOK_SYMBOL, ")")) {
            V_push(&items, lparse_atom_expr(P));
            while (lp_check(P, TRIAD_LTOK_SYMBOL, ",")) {
                lp_adv(P);
                V_push(&items, lparse_atom_expr(P));
            }
        }
        lp_expect_sym(P, ")");
        TriadLegacyVal out = {0};
        out.kind = TRIAD_LVAL_TUPLE;
        out.tuple_len = items.len;
        if (items.len) {
            TriadLegacyVal *arr = (TriadLegacyVal *)triad_arena_alloc(P->arena, items.len * sizeof(TriadLegacyVal));
            for (size_t k = 0; k < items.len; ++k) arr[k] = node_to_val((TriadLegacyNode *)items.items[k]);
            out.tuple_items = arr;
        }
        V_free(&items);
        return out;
    }
    return node_to_val(lparse_atom_expr(P));
}

/* ── Statement parsers ─────────────────────────────────────────── */

static TriadLegacyNode *lparse_annotation(LP *P) {
    const TriadLegacyToken *t = lp_adv(P);
    /* t->text is "ident(args)" */
    const char *lp_pos = strchr(t->text, '(');
    const char *key, *args;
    if (!lp_pos) {
        key = t->text;
        args = "";
    } else {
        size_t klen = (size_t)(lp_pos - t->text);
        key = triad_arena_strndup(P->arena, t->text, klen);
        size_t alen = strlen(lp_pos + 1);
        /* trailing ')' */
        if (alen && lp_pos[1 + alen - 1] == ')') alen--;
        args = triad_arena_strndup(P->arena, lp_pos + 1, alen);
    }
    TriadLegacyNode *n = lnew(P, TRIAD_LAST_ANNOTATION, t->line);
    n->u.annotation.key = key;
    n->u.annotation.raw_args = args;
    n->u.annotation.line = t->line;
    return n;
}

static TriadLegacyNode *lparse_reg_decl(LP *P) {
    const TriadLegacyToken *kw = lp_expect_kw(P, "reg");
    const TriadLegacyToken *name = lp_expect_kind(P, TRIAD_LTOK_IDENT);
    int bit_width = 1;
    const char *regime_name = NULL;
    int has_overrides = 0;
    V overrides = {0};

    if (lp_check(P, TRIAD_LTOK_SYMBOL, "[")) {
        lp_adv(P);
        const TriadLegacyToken *nt = lp_expect_kind(P, TRIAD_LTOK_NUMBER);
        bit_width = (int)strtod(nt->text, NULL);
        lp_expect_sym(P, "]");
    }
    if (lp_check(P, TRIAD_LTOK_SYMBOL, ":")) {
        lp_adv(P);
        const TriadLegacyToken *rt = lp_expect_kind(P, TRIAD_LTOK_IDENT);
        regime_name = rt->text;
    }
    if (lp_check(P, TRIAD_LTOK_SYMBOL, "{")) {
        lp_adv(P);
        has_overrides = 1;
        while (!lp_check(P, TRIAD_LTOK_SYMBOL, "}")) {
            const TriadLegacyToken *kt = lp_peek(P, 0);
            if (kt->kind != TRIAD_LTOK_IDENT && kt->kind != TRIAD_LTOK_KEYWORD) {
                NORETURN_lperr(P, kt->line, kt->col, "expected property name");
            }
            lp_adv(P);
            lp_expect_sym(P, ":");
            TriadLegacyVal val = lparse_atom_or_tuple(P);
            lp_expect_sym(P, ";");
            TriadLegacyOverride *o = (TriadLegacyOverride *)triad_arena_calloc(P->arena, sizeof(*o));
            o->key = kt->text;
            o->value = val;
            V_push(&overrides, o);
        }
        lp_expect_sym(P, "}");
    }
    TriadLegacyNode *initial = NULL;
    if (lp_check(P, TRIAD_LTOK_SYMBOL, "=")) {
        lp_adv(P);
        initial = lparse_atom_expr(P);
    }
    lp_expect_sym(P, ";");

    TriadLegacyNode *n = lnew(P, TRIAD_LAST_REG_DECL, kw->line);
    n->u.reg_decl.name = name->text;
    n->u.reg_decl.bit_width = bit_width;
    n->u.reg_decl.initial = initial;
    n->u.reg_decl.regime_name = regime_name;
    n->u.reg_decl.has_overrides = has_overrides;
    n->u.reg_decl.overrides_len = overrides.len;
    if (overrides.len) {
        TriadLegacyOverride *arr = (TriadLegacyOverride *)triad_arena_alloc(P->arena, overrides.len * sizeof(*arr));
        for (size_t k = 0; k < overrides.len; ++k) arr[k] = *(TriadLegacyOverride *)overrides.items[k];
        n->u.reg_decl.overrides = arr;
    }
    V_free(&overrides);
    return n;
}

static TriadLegacyNode *lparse_op_stmt(LP *P) {
    const TriadLegacyToken *op = lp_adv(P);  /* OPCODE */
    V args = {0};
    if (!lp_check(P, TRIAD_LTOK_SYMBOL, ";")) {
        V_push(&args, lparse_atom_expr(P));
        while (lp_check(P, TRIAD_LTOK_SYMBOL, ",")) {
            lp_adv(P);
            V_push(&args, lparse_atom_expr(P));
        }
    }
    lp_expect_sym(P, ";");
    TriadLegacyNode *n = lnew(P, TRIAD_LAST_OP, op->line);
    n->u.op.opcode = op->text;
    n->u.op.args_len = args.len;
    if (args.len) {
        TriadLegacyNode **arr = (TriadLegacyNode **)triad_arena_alloc(P->arena, args.len * sizeof(void *));
        for (size_t k = 0; k < args.len; ++k) arr[k] = (TriadLegacyNode *)args.items[k];
        n->u.op.args = arr;
    }
    V_free(&args);
    return n;
}

static void lparse_block(LP *P, TriadLegacyNode ***out, size_t *out_len) {
    V v = {0};
    while (!lp_check(P, TRIAD_LTOK_SYMBOL, "}") && !lp_check(P, TRIAD_LTOK_EOF, NULL)) {
        TriadLegacyNode *s = lparse_stmt(P);
        if (s) V_push(&v, s);
    }
    *out_len = v.len;
    if (v.len) {
        TriadLegacyNode **arr = (TriadLegacyNode **)triad_arena_alloc(P->arena, v.len * sizeof(void *));
        for (size_t k = 0; k < v.len; ++k) arr[k] = (TriadLegacyNode *)v.items[k];
        *out = arr;
    } else *out = NULL;
    V_free(&v);
}

static TriadLegacyNode *lparse_loop_block(LP *P) {
    const TriadLegacyToken *kw = lp_expect_kw(P, "loop");
    const TriadLegacyToken *tt = lp_peek(P, 0);
    if (tt->kind != TRIAD_LTOK_NUMBER && tt->kind != TRIAD_LTOK_IDENT) {
        NORETURN_lperr(P, tt->line, tt->col, "loop target must be IDENT or NUMBER");
    }
    TriadLegacyNode *target = lparse_atom_expr(P);
    lp_expect_sym(P, "{");
    TriadLegacyNode **body = NULL; size_t blen = 0;
    lparse_block(P, &body, &blen);
    lp_expect_sym(P, "}");
    TriadLegacyNode *n = lnew(P, TRIAD_LAST_LOOP_BLOCK, kw->line);
    n->u.loop_block.target = target;
    n->u.loop_block.body = body;
    n->u.loop_block.body_len = blen;
    return n;
}

static TriadLegacyNode *lparse_segment_block(LP *P) {
    const TriadLegacyToken *kw = lp_expect_kw(P, "segment");
    lp_expect_sym(P, "(");
    int seg_id = -1;
    double duration = -1.0;
    const TriadLegacyToken *nt = lp_peek(P, 0);
    if (nt->kind == TRIAD_LTOK_IDENT && strcmp(nt->text, "t") == 0) {
        lp_adv(P);
        lp_expect_sym(P, "=");
        const TriadLegacyToken *d = lp_expect_kind(P, TRIAD_LTOK_NUMBER);
        duration = strtod(d->text, NULL);
    } else {
        const TriadLegacyToken *n2 = lp_expect_kind(P, TRIAD_LTOK_NUMBER);
        seg_id = (int)strtod(n2->text, NULL);
    }
    lp_expect_sym(P, ")");
    lp_expect_sym(P, "{");
    TriadLegacyNode **body = NULL; size_t blen = 0;
    lparse_block(P, &body, &blen);
    lp_expect_sym(P, "}");
    TriadLegacyNode *n = lnew(P, TRIAD_LAST_SEGMENT_BLOCK, kw->line);
    n->u.segment_block.segment_id = seg_id;
    n->u.segment_block.duration = duration;
    n->u.segment_block.body = body;
    n->u.segment_block.body_len = blen;
    return n;
}

static TriadLegacyNode *lparse_if_block(LP *P) {
    const TriadLegacyToken *kw = lp_expect_kw(P, "if");
    const TriadLegacyToken *cond = lp_expect_kind(P, TRIAD_LTOK_IDENT);
    lp_expect_sym(P, "{");
    TriadLegacyNode **then_body = NULL; size_t tlen = 0;
    lparse_block(P, &then_body, &tlen);
    lp_expect_sym(P, "}");
    TriadLegacyNode **else_body = NULL; size_t elen = 0;
    int has_else = 0;
    if (lp_check(P, TRIAD_LTOK_KEYWORD, "else")) {
        lp_adv(P);
        lp_expect_sym(P, "{");
        lparse_block(P, &else_body, &elen);
        lp_expect_sym(P, "}");
        has_else = 1;
    }
    TriadLegacyNode *n = lnew(P, TRIAD_LAST_IF_BLOCK, kw->line);
    n->u.if_block.cond_name = cond->text;
    n->u.if_block.then_body = then_body;
    n->u.if_block.then_body_len = tlen;
    n->u.if_block.else_body = else_body;
    n->u.if_block.else_body_len = elen;
    n->u.if_block.has_else = has_else;
    return n;
}

static TriadLegacyNode *lparse_out_stmt(LP *P) {
    const TriadLegacyToken *kw = lp_expect_kw(P, "OUT");
    V args = {0};
    V_push(&args, lparse_atom_expr(P));
    while (lp_check(P, TRIAD_LTOK_SYMBOL, ",")) {
        lp_adv(P);
        V_push(&args, lparse_atom_expr(P));
    }
    lp_expect_sym(P, ";");
    TriadLegacyNode *n = lnew(P, TRIAD_LAST_OUT_STMT, kw->line);
    n->u.out_stmt.args_len = args.len;
    if (args.len) {
        TriadLegacyNode **arr = (TriadLegacyNode **)triad_arena_alloc(P->arena, args.len * sizeof(void *));
        for (size_t k = 0; k < args.len; ++k) arr[k] = (TriadLegacyNode *)args.items[k];
        n->u.out_stmt.args = arr;
    }
    V_free(&args);
    return n;
}

static TriadLegacyNode *lparse_evolve_stmt(LP *P) {
    const TriadLegacyToken *kw = lp_expect_kw(P, "evolve");
    const TriadLegacyToken *nt = lp_expect_kind(P, TRIAD_LTOK_IDENT);
    lp_expect_kw(P, "for");
    const TriadLegacyToken *dt = lp_expect_kind(P, TRIAD_LTOK_NUMBER);
    lp_expect_sym(P, ";");
    TriadLegacyNode *n = lnew(P, TRIAD_LAST_EVOLVE_STMT, kw->line);
    n->u.evolve_stmt.target = nt->text;
    n->u.evolve_stmt.duration = strtod(dt->text, NULL);
    return n;
}

static void lp_consume_kappa_for_dur(LP *P, double *out_k, double *out_d) {
    lp_expect_kw(P, "kappa");
    lp_expect_sym(P, "=");
    const TriadLegacyToken *kt = lp_expect_kind(P, TRIAD_LTOK_NUMBER);
    lp_expect_kw(P, "for");
    const TriadLegacyToken *nx = lp_peek(P, 0);
    if (nx->kind == TRIAD_LTOK_IDENT && strcmp(nx->text, "T") == 0) {
        lp_adv(P);
        lp_expect_sym(P, "=");
    }
    const TriadLegacyToken *dt = lp_expect_kind(P, TRIAD_LTOK_NUMBER);
    *out_k = strtod(kt->text, NULL);
    *out_d = strtod(dt->text, NULL);
}

static TriadLegacyNode *lparse_couple_stmt(LP *P) {
    const TriadLegacyToken *kw = lp_expect_kw(P, "couple");
    const TriadLegacyToken *src = lp_expect_kind(P, TRIAD_LTOK_IDENT);
    const TriadLegacyToken *nx = lp_peek(P, 0);
    if (nx->kind == TRIAD_LTOK_SYMBOL && strcmp(nx->text, "->") == 0) {
        lp_adv(P);
    } else {
        lp_expect_sym(P, "-");
        lp_expect_sym(P, ">");
    }
    const TriadLegacyToken *dst = lp_expect_kind(P, TRIAD_LTOK_IDENT);
    double kappa, dur;
    lp_consume_kappa_for_dur(P, &kappa, &dur);
    lp_expect_sym(P, ";");
    TriadLegacyNode *n = lnew(P, TRIAD_LAST_COUPLE_STMT, kw->line);
    n->u.couple_pair.src_or_a = src->text;
    n->u.couple_pair.dst_or_b = dst->text;
    n->u.couple_pair.kappa = kappa;
    n->u.couple_pair.duration = dur;
    return n;
}

static TriadLegacyNode *lparse_pair_stmt(LP *P) {
    const TriadLegacyToken *kw = lp_expect_kw(P, "pair");
    lp_expect_sym(P, "(");
    const TriadLegacyToken *a = lp_expect_kind(P, TRIAD_LTOK_IDENT);
    lp_expect_sym(P, ",");
    const TriadLegacyToken *b = lp_expect_kind(P, TRIAD_LTOK_IDENT);
    lp_expect_sym(P, ")");
    double kappa, dur;
    lp_consume_kappa_for_dur(P, &kappa, &dur);
    lp_expect_sym(P, ";");
    TriadLegacyNode *n = lnew(P, TRIAD_LAST_PAIR_STMT, kw->line);
    n->u.couple_pair.src_or_a = a->text;
    n->u.couple_pair.dst_or_b = b->text;
    n->u.couple_pair.kappa = kappa;
    n->u.couple_pair.duration = dur;
    return n;
}

static TriadLegacyNode *lparse_ring_stmt(LP *P) {
    const TriadLegacyToken *kw = lp_expect_kw(P, "ring");
    lp_expect_sym(P, "(");
    V mem = {0};
    V_push(&mem, (void *)lp_expect_kind(P, TRIAD_LTOK_IDENT)->text);
    while (lp_check(P, TRIAD_LTOK_SYMBOL, ",")) {
        lp_adv(P);
        V_push(&mem, (void *)lp_expect_kind(P, TRIAD_LTOK_IDENT)->text);
    }
    lp_expect_sym(P, ")");
    double kappa, dur;
    lp_consume_kappa_for_dur(P, &kappa, &dur);
    lp_expect_sym(P, ";");
    TriadLegacyNode *n = lnew(P, TRIAD_LAST_RING_STMT, kw->line);
    n->u.ring_seq.members_len = mem.len;
    if (mem.len) {
        const char **arr = (const char **)triad_arena_alloc(P->arena, mem.len * sizeof(char *));
        for (size_t k = 0; k < mem.len; ++k) arr[k] = (const char *)mem.items[k];
        n->u.ring_seq.members = arr;
    }
    n->u.ring_seq.kappa = kappa;
    n->u.ring_seq.duration = dur;
    V_free(&mem);
    return n;
}

static TriadLegacyNode *lparse_sequence_stmt(LP *P) {
    const TriadLegacyToken *kw = lp_expect_kw(P, "sequence");
    const TriadLegacyToken *tgt = lp_expect_kind(P, TRIAD_LTOK_IDENT);
    lp_expect_kw(P, "via");
    lp_expect_sym(P, "(");
    V mem = {0};
    V_push(&mem, (void *)lp_expect_kind(P, TRIAD_LTOK_IDENT)->text);
    while (lp_check(P, TRIAD_LTOK_SYMBOL, ",")) {
        lp_adv(P);
        V_push(&mem, (void *)lp_expect_kind(P, TRIAD_LTOK_IDENT)->text);
    }
    lp_expect_sym(P, ")");
    lp_expect_kw(P, "each_for");
    lp_expect_sym(P, "=");
    const TriadLegacyToken *dt = lp_expect_kind(P, TRIAD_LTOK_NUMBER);
    lp_expect_sym(P, ";");
    TriadLegacyNode *n = lnew(P, TRIAD_LAST_SEQUENCE_STMT, kw->line);
    n->u.ring_seq.target = tgt->text;
    n->u.ring_seq.members_len = mem.len;
    if (mem.len) {
        const char **arr = (const char **)triad_arena_alloc(P->arena, mem.len * sizeof(char *));
        for (size_t k = 0; k < mem.len; ++k) arr[k] = (const char *)mem.items[k];
        n->u.ring_seq.members = arr;
    }
    n->u.ring_seq.duration = strtod(dt->text, NULL);
    V_free(&mem);
    return n;
}

static TriadLegacyNode *lparse_observe_stmt(LP *P) {
    const TriadLegacyToken *kw = lp_expect_kw(P, "OBSERVE");
    const TriadLegacyToken *tgt = lp_expect_kind(P, TRIAD_LTOK_IDENT);
    V metrics = {0};
    V_push(&metrics, (void *)lp_expect_kind(P, TRIAD_LTOK_IDENT)->text);
    while (lp_check(P, TRIAD_LTOK_SYMBOL, ",")) {
        lp_adv(P);
        const TriadLegacyToken *nx = lp_peek(P, 0);
        if (nx->kind == TRIAD_LTOK_KEYWORD && strcmp(nx->text, "over_seeds") == 0) break;
        V_push(&metrics, (void *)lp_expect_kind(P, TRIAD_LTOK_IDENT)->text);
    }
    int over_seeds = 1;
    const char *stream_to = "";
    for (;;) {
        const TriadLegacyToken *nx = lp_peek(P, 0);
        if (nx->kind == TRIAD_LTOK_KEYWORD && strcmp(nx->text, "over_seeds") == 0) {
            lp_adv(P);
            lp_expect_sym(P, "=");
            const TriadLegacyToken *nt = lp_expect_kind(P, TRIAD_LTOK_NUMBER);
            over_seeds = (int)strtod(nt->text, NULL);
            continue;
        }
        if (nx->kind == TRIAD_LTOK_IDENT && strcmp(nx->text, "stream_to") == 0) {
            lp_adv(P);
            const TriadLegacyToken *st = lp_expect_kind(P, TRIAD_LTOK_STRING);
            stream_to = st->text;
            continue;
        }
        break;
    }
    lp_expect_sym(P, ";");
    TriadLegacyNode *n = lnew(P, TRIAD_LAST_OBSERVE_STMT, kw->line);
    n->u.observe_stmt.target = tgt->text;
    n->u.observe_stmt.metrics_len = metrics.len;
    if (metrics.len) {
        const char **arr = (const char **)triad_arena_alloc(P->arena, metrics.len * sizeof(char *));
        for (size_t k = 0; k < metrics.len; ++k) arr[k] = (const char *)metrics.items[k];
        n->u.observe_stmt.metrics = arr;
    }
    n->u.observe_stmt.over_seeds = over_seeds;
    n->u.observe_stmt.stream_to = stream_to;
    V_free(&metrics);
    return n;
}

static TriadLegacyNode *lparse_assert_stmt(LP *P) {
    const TriadLegacyToken *kw = lp_expect_kw(P, "assert");
    const TriadLegacyToken *pred = lp_adv(P);
    if (pred->kind != TRIAD_LTOK_KEYWORD) {
        NORETURN_lperr(P, pred->line, pred->col, "assert expects predicate KEYWORD");
    }
    lp_expect_sym(P, "(");
    const TriadLegacyToken *tgt = lp_expect_kind(P, TRIAD_LTOK_IDENT);
    lp_expect_sym(P, ")");
    lp_expect_sym(P, ";");
    TriadLegacyNode *n = lnew(P, TRIAD_LAST_ASSERT_STMT, kw->line);
    n->u.assert_stmt.predicate = pred->text;
    n->u.assert_stmt.target = tgt->text;
    return n;
}

static TriadLegacyNode *lparse_checkpoint_stmt(LP *P) {
    const TriadLegacyToken *kw = lp_adv(P);  /* OPCODE CHECKPOINT */
    const TriadLegacyToken *nt = lp_expect_kind(P, TRIAD_LTOK_IDENT);
    const TriadLegacyToken *nx = lp_peek(P, 0);
    if (!(nx->kind == TRIAD_LTOK_KEYWORD && strcmp(nx->text, "to") == 0)) {
        NORETURN_lperr(P, nx->line, nx->col, "CHECKPOINT expects 'to'");
    }
    lp_adv(P);
    const TriadLegacyToken *st = lp_expect_kind(P, TRIAD_LTOK_STRING);
    lp_expect_sym(P, ";");
    TriadLegacyNode *n = lnew(P, TRIAD_LAST_CHECKPOINT_STMT, kw->line);
    n->u.checkpoint_stmt.target = nt->text;
    n->u.checkpoint_stmt.path = st->text;
    return n;
}

static TriadLegacyNode *lparse_substrate_decl(LP *P) {
    const TriadLegacyToken *kw = lp_expect_kw(P, "substrate");
    const TriadLegacyToken *name = lp_expect_kind(P, TRIAD_LTOK_IDENT);
    lp_expect_kw(P, "composed_of");
    lp_expect_sym(P, "(");
    V mem = {0};
    V_push(&mem, (void *)lp_expect_kind(P, TRIAD_LTOK_IDENT)->text);
    while (lp_check(P, TRIAD_LTOK_SYMBOL, ",")) {
        lp_adv(P);
        V_push(&mem, (void *)lp_expect_kind(P, TRIAD_LTOK_IDENT)->text);
    }
    lp_expect_sym(P, ")");
    V props = {0};
    if (lp_check(P, TRIAD_LTOK_SYMBOL, "{")) {
        lp_adv(P);
        while (!lp_check(P, TRIAD_LTOK_SYMBOL, "}")) {
            const TriadLegacyToken *kt = lp_peek(P, 0);
            if (kt->kind != TRIAD_LTOK_IDENT && kt->kind != TRIAD_LTOK_KEYWORD) {
                NORETURN_lperr(P, kt->line, kt->col, "expected property name");
            }
            lp_adv(P);
            lp_expect_sym(P, ":");
            TriadLegacyVal val = {0};
            const TriadLegacyToken *nx = lp_peek(P, 0);
            if (nx->kind == TRIAD_LTOK_KEYWORD) {
                lp_adv(P);
                val.kind = TRIAD_LVAL_STR;
                val.str_val = nx->text;
            } else {
                val = node_to_val(lparse_atom_expr(P));
            }
            lp_expect_sym(P, ";");
            TriadLegacyOverride *o = (TriadLegacyOverride *)triad_arena_calloc(P->arena, sizeof(*o));
            o->key = kt->text;
            o->value = val;
            V_push(&props, o);
        }
        lp_expect_sym(P, "}");
    }
    lp_expect_sym(P, ";");
    TriadLegacyNode *n = lnew(P, TRIAD_LAST_SUBSTRATE_DECL, kw->line);
    n->u.substrate_decl.name = name->text;
    n->u.substrate_decl.composed_of_len = mem.len;
    if (mem.len) {
        const char **arr = (const char **)triad_arena_alloc(P->arena, mem.len * sizeof(char *));
        for (size_t k = 0; k < mem.len; ++k) arr[k] = (const char *)mem.items[k];
        n->u.substrate_decl.composed_of = arr;
    }
    n->u.substrate_decl.properties_len = props.len;
    if (props.len) {
        TriadLegacyOverride *arr = (TriadLegacyOverride *)triad_arena_alloc(P->arena, props.len * sizeof(*arr));
        for (size_t k = 0; k < props.len; ++k) arr[k] = *(TriadLegacyOverride *)props.items[k];
        n->u.substrate_decl.properties = arr;
    }
    V_free(&mem);
    V_free(&props);
    return n;
}

/* ── Top-level dispatch ─────────────────────────────────────────── */

static TriadLegacyNode *lparse_stmt(LP *P) {
    const TriadLegacyToken *t = lp_peek(P, 0);

    if (t->kind == TRIAD_LTOK_ANNOT)   return lparse_annotation(P);

    if (t->kind == TRIAD_LTOK_KEYWORD) {
        const char *v = t->text;
        if (strcmp(v, "reg") == 0)      return lparse_reg_decl(P);
        if (strcmp(v, "loop") == 0)     return lparse_loop_block(P);
        if (strcmp(v, "segment") == 0)  return lparse_segment_block(P);
        if (strcmp(v, "if") == 0)       return lparse_if_block(P);
        if (strcmp(v, "OUT") == 0)      return lparse_out_stmt(P);
        if (strcmp(v, "HALT") == 0) {
            lp_adv(P);
            lp_expect_sym(P, ";");
            return lnew(P, TRIAD_LAST_HALT_STMT, t->line);
        }
        if (strcmp(v, "evolve") == 0)   return lparse_evolve_stmt(P);
        if (strcmp(v, "couple") == 0)   return lparse_couple_stmt(P);
        if (strcmp(v, "pair") == 0)     return lparse_pair_stmt(P);
        if (strcmp(v, "ring") == 0)     return lparse_ring_stmt(P);
        if (strcmp(v, "sequence") == 0) return lparse_sequence_stmt(P);
        if (strcmp(v, "OBSERVE") == 0)  return lparse_observe_stmt(P);
        if (strcmp(v, "assert") == 0)   return lparse_assert_stmt(P);
        if (strcmp(v, "substrate") == 0)return lparse_substrate_decl(P);
    }

    if (t->kind == TRIAD_LTOK_OPCODE) {
        if (strcmp(t->text, "CHECKPOINT") == 0) return lparse_checkpoint_stmt(P);
        return lparse_op_stmt(P);
    }

    NORETURN_lperr(P, t->line, t->col, "unexpected token '%s'", t->text ? t->text : "");
}

const char *triad_legacy_kind_name(TriadLegacyKind k) {
    switch (k) {
        case TRIAD_LAST_NUM_LIT:         return "NumLit";
        case TRIAD_LAST_BOOL_LIT:        return "BoolLit";
        case TRIAD_LAST_STR_LIT:         return "StrLit";
        case TRIAD_LAST_IDENT_REF:       return "IdentRef";
        case TRIAD_LAST_REG_DECL:        return "RegDecl";
        case TRIAD_LAST_OP:              return "Op";
        case TRIAD_LAST_LOOP_BLOCK:      return "LoopBlock";
        case TRIAD_LAST_SEGMENT_BLOCK:   return "SegmentBlock";
        case TRIAD_LAST_IF_BLOCK:        return "IfBlock";
        case TRIAD_LAST_OUT_STMT:        return "OutStmt";
        case TRIAD_LAST_HALT_STMT:       return "HaltStmt";
        case TRIAD_LAST_ANNOTATION:      return "Annotation";
        case TRIAD_LAST_EVOLVE_STMT:     return "EvolveStmt";
        case TRIAD_LAST_COUPLE_STMT:     return "CoupleStmt";
        case TRIAD_LAST_PAIR_STMT:       return "PairStmt";
        case TRIAD_LAST_RING_STMT:       return "RingStmt";
        case TRIAD_LAST_SEQUENCE_STMT:   return "SequenceStmt";
        case TRIAD_LAST_OBSERVE_STMT:    return "ObserveStmt";
        case TRIAD_LAST_ASSERT_STMT:     return "AssertStmt";
        case TRIAD_LAST_CHECKPOINT_STMT: return "CheckpointStmt";
        case TRIAD_LAST_SUBSTRATE_DECL:  return "SubstrateDecl";
        case TRIAD_LAST_PROGRAM:         return "Program";
        default:                         return "Unknown";
    }
}

TriadLegacyNode *triad_legacy_parse_tokens(TriadArena *a, const TriadLegacyTokenList *t, TriadDiag *diag) {
    LP P = {0};
    P.arena = a;
    P.toks  = t->items;
    P.ntoks = t->len;
    P.i     = 0;
    P.diag  = diag;
    if (setjmp(P.err_jmp) != 0) return NULL;
    V body = {0};
    while (!lp_check(&P, TRIAD_LTOK_EOF, NULL)) {
        TriadLegacyNode *s = lparse_stmt(&P);
        if (s) V_push(&body, s);
    }
    TriadLegacyNode *prog = lnew(&P, TRIAD_LAST_PROGRAM, 0);
    prog->u.program.body_len = body.len;
    if (body.len) {
        TriadLegacyNode **arr = (TriadLegacyNode **)triad_arena_alloc(a, body.len * sizeof(void *));
        for (size_t k = 0; k < body.len; ++k) arr[k] = (TriadLegacyNode *)body.items[k];
        prog->u.program.body = arr;
    }
    V_free(&body);
    return prog;
}

TriadLegacyNode *triad_legacy_parse_source(TriadArena *a, const char *src, TriadDiag *diag) {
    TriadLegacyTokenList toks = {0};
    if (triad_legacy_tokenize(a, src, &toks, diag) != 0) return NULL;
    return triad_legacy_parse_tokens(a, &toks, diag);
}
