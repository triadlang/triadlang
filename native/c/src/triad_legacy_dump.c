/* triad_legacy_dump.c — JSON dumper for the legacy AST.
 *
 * Schema matches scripts/ast_to_json_legacy.py, which serializes the
 * frontend.parser dataclasses with _serialize semantics from
 * compiler/emit_json.py:
 *   - dataclass -> {"_type": "<ClassName>", <field>: ...}
 *   - tuples    -> arrays
 *   - dicts     -> objects
 *   - None      -> null
 */
#include "triad_frontend.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char  *data;
    size_t len;
    size_t cap;
} S;

static void s_reserve(S *s, size_t extra) {
    if (s->len + extra + 1 > s->cap) {
        size_t nc = s->cap ? s->cap * 2 : 256;
        while (s->len + extra + 1 > nc) nc *= 2;
        s->data = (char *)realloc(s->data, nc);
        s->cap = nc;
    }
}
static void s_putc(S *s, char c) { s_reserve(s, 1); s->data[s->len++] = c; s->data[s->len] = 0; }
static void s_puts(S *s, const char *t) { size_t n = strlen(t); s_reserve(s, n); memcpy(s->data + s->len, t, n); s->len += n; s->data[s->len] = 0; }

static void s_printf(S *s, const char *fmt, ...) {
    char buf[128];
    va_list ap; va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap); va_end(ap);
    if (n < 0) return;
    if ((size_t)n < sizeof(buf)) { s_puts(s, buf); return; }
    char *big = (char *)malloc((size_t)n + 1);
    va_start(ap, fmt);
    vsnprintf(big, n + 1, fmt, ap);
    va_end(ap);
    s_puts(s, big);
    free(big);
}

static void emit_string(S *s, const char *t) {
    s_putc(s, '"');
    if (!t) { s_putc(s, '"'); return; }
    for (const unsigned char *p = (const unsigned char *)t; *p; ++p) {
        unsigned char c = *p;
        switch (c) {
            case '\\': s_puts(s, "\\\\"); break;
            case '"':  s_puts(s, "\\\""); break;
            case '\n': s_puts(s, "\\n"); break;
            case '\r': s_puts(s, "\\r"); break;
            case '\t': s_puts(s, "\\t"); break;
            case '\b': s_puts(s, "\\b"); break;
            case '\f': s_puts(s, "\\f"); break;
            default:
                if (c < 0x20) s_printf(s, "\\u%04x", c);
                else          s_putc(s, (char)c);
        }
    }
    s_putc(s, '"');
}

typedef struct { int indent; int depth; } Ctx;
static void emit_nl_indent(S *s, Ctx *c) {
    if (c->indent <= 0) return;
    s_putc(s, '\n');
    for (int i = 0; i < c->indent * c->depth; ++i) s_putc(s, ' ');
}

/* Match Python repr for double — same routine as triad_ast_dump.c. */
static void emit_float(S *s, double v) {
    if (isnan(v)) { s_puts(s, "NaN"); return; }
    if (isinf(v)) { s_puts(s, v > 0 ? "Infinity" : "-Infinity"); return; }
    char buf[64];
    int prec = 17;
    for (int p = 1; p <= 17; ++p) {
        snprintf(buf, sizeof(buf), "%.*g", p, v);
        if (strtod(buf, NULL) == v) { prec = p; break; }
    }
    char ebuf[64];
    snprintf(ebuf, sizeof(ebuf), "%.*e", prec - 1, v);
    const char *epos = strchr(ebuf, 'e');
    int decexp = epos ? atoi(epos + 1) : 0;
    int use_fixed = (decexp >= -4 && decexp < 16);
    if (use_fixed) {
        int after;
        if (decexp >= 0) after = prec - 1 - decexp;
        else             after = prec - 1 + (-decexp);
        if (after < 0) after = 0;
        snprintf(buf, sizeof(buf), "%.*f", after, v);
        char *dot = strchr(buf, '.');
        if (dot) {
            char *end = buf + strlen(buf) - 1;
            while (end > dot + 1 && *end == '0') { *end = '\0'; --end; }
        } else {
            size_t L = strlen(buf);
            if (L + 2 < sizeof(buf)) { buf[L] = '.'; buf[L+1] = '0'; buf[L+2] = '\0'; }
        }
        if (strtod(buf, NULL) != v) use_fixed = 0;
    }
    if (!use_fixed) {
        char *e2 = strchr(ebuf, 'e');
        if (e2) {
            char *dot = strchr(ebuf, '.');
            if (dot && dot < e2) {
                char *q = e2 - 1;
                while (q > dot && *q == '0') { memmove(q, q + 1, strlen(q)); --q; e2--; }
                if (q == dot) { memmove(dot, dot + 1, strlen(dot)); e2--; }
            }
            char *ep = e2 + 1;
            char sign = '+';
            if (*ep == '+' || *ep == '-') { sign = *ep; ep++; }
            while (*ep == '0' && *(ep + 1)) ep++;
            int expv = atoi(ep);
            if (sign == '-') expv = -expv;
            snprintf(buf, sizeof(buf), "%.*se%+03d",
                     (int)(e2 - ebuf), ebuf, expv);
        } else {
            snprintf(buf, sizeof(buf), "%s", ebuf);
        }
    }
    s_puts(s, buf);
}

typedef struct { int first; } FS;
static void f_sep(S *s, Ctx *c, FS *fs) { if (!fs->first) s_putc(s, ','); fs->first = 0; emit_nl_indent(s, c); }
static void f_key(S *s, const char *k) { s_putc(s, '"'); s_puts(s, k); s_puts(s, "\": "); }
static void f_start(S *s, Ctx *c, FS *fs, const char *tn) {
    s_putc(s, '{'); c->depth++; fs->first = 1;
    f_sep(s, c, fs); f_key(s, "_type"); emit_string(s, tn);
}
static void f_end(S *s, Ctx *c) { c->depth--; emit_nl_indent(s, c); s_putc(s, '}'); }

static void emit_null(S *s) { s_puts(s, "null"); }
static void emit_int(S *s, long long v) { s_printf(s, "%lld", v); }
static void emit_bool(S *s, int v) { s_puts(s, v ? "true" : "false"); }
static void emit_str_or_null(S *s, const char *t) { if (!t) emit_null(s); else emit_string(s, t); }

static void emit_node(S *s, Ctx *c, const TriadLegacyNode *n);

static void emit_node_array(S *s, Ctx *c, TriadLegacyNode *const *items, size_t n) {
    if (n == 0) { s_puts(s, "[]"); return; }
    s_putc(s, '['); c->depth++;
    for (size_t k = 0; k < n; ++k) {
        if (k) s_putc(s, ',');
        emit_nl_indent(s, c);
        emit_node(s, c, items[k]);
    }
    c->depth--; emit_nl_indent(s, c); s_putc(s, ']');
}

static void emit_string_array(S *s, Ctx *c, const char *const *items, size_t n) {
    if (n == 0) { s_puts(s, "[]"); return; }
    s_putc(s, '['); c->depth++;
    for (size_t k = 0; k < n; ++k) {
        if (k) s_putc(s, ',');
        emit_nl_indent(s, c);
        emit_string(s, items[k]);
    }
    c->depth--; emit_nl_indent(s, c); s_putc(s, ']');
}

/* For RingStmt/SequenceStmt: members are wrapped as IdentRef objects in
 * the Python AST. Same for OutStmt/Op args (already nodes). */
static void emit_ident_array(S *s, Ctx *c, const char *const *names, size_t n) {
    if (n == 0) { s_puts(s, "[]"); return; }
    s_putc(s, '['); c->depth++;
    for (size_t k = 0; k < n; ++k) {
        if (k) s_putc(s, ',');
        emit_nl_indent(s, c);
        FS fs = {1};
        f_start(s, c, &fs, "IdentRef");
        f_sep(s, c, &fs); f_key(s, "name"); emit_string(s, names[k]);
        f_end(s, c);
    }
    c->depth--; emit_nl_indent(s, c); s_putc(s, ']');
}

/* IdentRef inline (for ObserveStmt.target, AssertStmt.target, etc.) */
static void emit_identref(S *s, Ctx *c, const char *name) {
    FS fs = {1};
    f_start(s, c, &fs, "IdentRef");
    f_sep(s, c, &fs); f_key(s, "name"); emit_string(s, name);
    f_end(s, c);
}

static void emit_val(S *s, Ctx *c, const TriadLegacyVal *v) {
    switch (v->kind) {
        case TRIAD_LVAL_INT:   emit_int(s, v->int_val); break;
        case TRIAD_LVAL_FLOAT: emit_float(s, v->float_val); break;
        case TRIAD_LVAL_BOOL:  emit_bool(s, v->bool_val); break;
        case TRIAD_LVAL_STR:   emit_string(s, v->str_val); break;
        case TRIAD_LVAL_IDENT: emit_string(s, v->str_val); break;
        case TRIAD_LVAL_TUPLE: {
            if (v->tuple_len == 0) { s_puts(s, "[]"); return; }
            s_putc(s, '['); c->depth++;
            for (size_t k = 0; k < v->tuple_len; ++k) {
                if (k) s_putc(s, ',');
                emit_nl_indent(s, c);
                emit_val(s, c, &v->tuple_items[k]);
            }
            c->depth--; emit_nl_indent(s, c); s_putc(s, ']');
            break;
        }
    }
}

static void emit_override_dict(S *s, Ctx *c, const TriadLegacyOverride *o, size_t n) {
    s_putc(s, '{'); c->depth++;
    for (size_t k = 0; k < n; ++k) {
        if (k) s_putc(s, ',');
        emit_nl_indent(s, c);
        emit_string(s, o[k].key);
        s_puts(s, ": ");
        emit_val(s, c, &o[k].value);
    }
    c->depth--;
    if (n) emit_nl_indent(s, c);
    s_putc(s, '}');
}

static void emit_program_body(S *s, Ctx *c, TriadLegacyNode *const *items, size_t n) {
    /* Python: dataclass Program with field `body: list[Stmt]`.
     * Serialized as {"_type": "Program", "body": [...]} */
    FS fs = {1};
    f_start(s, c, &fs, "Program");
    f_sep(s, c, &fs); f_key(s, "body");
    emit_node_array(s, c, items, n);
    f_end(s, c);
}

static void emit_node(S *s, Ctx *c, const TriadLegacyNode *n) {
    if (!n) { emit_null(s); return; }
    FS fs = {1};
    const char *tname = triad_legacy_kind_name(n->kind);
    if (n->kind == TRIAD_LAST_PROGRAM) {
        emit_program_body(s, c, n->u.program.body, n->u.program.body_len);
        return;
    }
    f_start(s, c, &fs, tname);
    switch (n->kind) {
        case TRIAD_LAST_NUM_LIT:
            f_sep(s, c, &fs); f_key(s, "value");
            if (n->u.num.is_int) emit_float(s, n->u.num.value); /* Python stores as float either way */
            else                 emit_float(s, n->u.num.value);
            f_sep(s, c, &fs); f_key(s, "is_int"); emit_bool(s, n->u.num.is_int);
            break;
        case TRIAD_LAST_BOOL_LIT:
            f_sep(s, c, &fs); f_key(s, "value"); emit_bool(s, n->u.boolean.value);
            break;
        case TRIAD_LAST_STR_LIT:
            f_sep(s, c, &fs); f_key(s, "value"); emit_string(s, n->u.str.value);
            break;
        case TRIAD_LAST_IDENT_REF:
            f_sep(s, c, &fs); f_key(s, "name"); emit_string(s, n->u.ident.name);
            break;

        case TRIAD_LAST_REG_DECL:
            f_sep(s, c, &fs); f_key(s, "name"); emit_string(s, n->u.reg_decl.name);
            f_sep(s, c, &fs); f_key(s, "bit_width"); emit_int(s, n->u.reg_decl.bit_width);
            f_sep(s, c, &fs); f_key(s, "initial");
            if (n->u.reg_decl.initial) emit_node(s, c, n->u.reg_decl.initial); else emit_null(s);
            f_sep(s, c, &fs); f_key(s, "line"); emit_int(s, n->line);
            f_sep(s, c, &fs); f_key(s, "regime_name"); emit_str_or_null(s, n->u.reg_decl.regime_name);
            f_sep(s, c, &fs); f_key(s, "regime_overrides");
            if (n->u.reg_decl.has_overrides)
                emit_override_dict(s, c, n->u.reg_decl.overrides, n->u.reg_decl.overrides_len);
            else emit_null(s);
            break;

        case TRIAD_LAST_OP:
            f_sep(s, c, &fs); f_key(s, "opcode"); emit_string(s, n->u.op.opcode);
            f_sep(s, c, &fs); f_key(s, "args"); emit_node_array(s, c, n->u.op.args, n->u.op.args_len);
            f_sep(s, c, &fs); f_key(s, "line"); emit_int(s, n->line);
            break;

        case TRIAD_LAST_LOOP_BLOCK:
            f_sep(s, c, &fs); f_key(s, "target"); emit_node(s, c, n->u.loop_block.target);
            f_sep(s, c, &fs); f_key(s, "body");
            emit_program_body(s, c, n->u.loop_block.body, n->u.loop_block.body_len);
            f_sep(s, c, &fs); f_key(s, "line"); emit_int(s, n->line);
            break;

        case TRIAD_LAST_SEGMENT_BLOCK:
            f_sep(s, c, &fs); f_key(s, "segment_id"); emit_int(s, n->u.segment_block.segment_id);
            f_sep(s, c, &fs); f_key(s, "body");
            emit_program_body(s, c, n->u.segment_block.body, n->u.segment_block.body_len);
            f_sep(s, c, &fs); f_key(s, "line"); emit_int(s, n->line);
            break;

        case TRIAD_LAST_IF_BLOCK:
            f_sep(s, c, &fs); f_key(s, "cond");
            emit_identref(s, c, n->u.if_block.cond_name);
            f_sep(s, c, &fs); f_key(s, "then_body");
            emit_program_body(s, c, n->u.if_block.then_body, n->u.if_block.then_body_len);
            f_sep(s, c, &fs); f_key(s, "else_body");
            if (n->u.if_block.has_else)
                emit_program_body(s, c, n->u.if_block.else_body, n->u.if_block.else_body_len);
            else emit_null(s);
            f_sep(s, c, &fs); f_key(s, "line"); emit_int(s, n->line);
            break;

        case TRIAD_LAST_OUT_STMT:
            f_sep(s, c, &fs); f_key(s, "args"); emit_node_array(s, c, n->u.out_stmt.args, n->u.out_stmt.args_len);
            f_sep(s, c, &fs); f_key(s, "line"); emit_int(s, n->line);
            break;

        case TRIAD_LAST_HALT_STMT:
            f_sep(s, c, &fs); f_key(s, "line"); emit_int(s, n->line);
            break;

        case TRIAD_LAST_ANNOTATION:
            f_sep(s, c, &fs); f_key(s, "key"); emit_string(s, n->u.annotation.key);
            f_sep(s, c, &fs); f_key(s, "raw_args"); emit_string(s, n->u.annotation.raw_args);
            f_sep(s, c, &fs); f_key(s, "line"); emit_int(s, n->line);
            break;

        case TRIAD_LAST_EVOLVE_STMT:
            f_sep(s, c, &fs); f_key(s, "target"); emit_identref(s, c, n->u.evolve_stmt.target);
            f_sep(s, c, &fs); f_key(s, "duration"); emit_float(s, n->u.evolve_stmt.duration);
            f_sep(s, c, &fs); f_key(s, "line"); emit_int(s, n->line);
            break;

        case TRIAD_LAST_COUPLE_STMT:
            f_sep(s, c, &fs); f_key(s, "src"); emit_identref(s, c, n->u.couple_pair.src_or_a);
            f_sep(s, c, &fs); f_key(s, "dst"); emit_identref(s, c, n->u.couple_pair.dst_or_b);
            f_sep(s, c, &fs); f_key(s, "kappa"); emit_float(s, n->u.couple_pair.kappa);
            f_sep(s, c, &fs); f_key(s, "duration"); emit_float(s, n->u.couple_pair.duration);
            f_sep(s, c, &fs); f_key(s, "line"); emit_int(s, n->line);
            break;

        case TRIAD_LAST_PAIR_STMT:
            f_sep(s, c, &fs); f_key(s, "a"); emit_identref(s, c, n->u.couple_pair.src_or_a);
            f_sep(s, c, &fs); f_key(s, "b"); emit_identref(s, c, n->u.couple_pair.dst_or_b);
            f_sep(s, c, &fs); f_key(s, "kappa"); emit_float(s, n->u.couple_pair.kappa);
            f_sep(s, c, &fs); f_key(s, "duration"); emit_float(s, n->u.couple_pair.duration);
            f_sep(s, c, &fs); f_key(s, "line"); emit_int(s, n->line);
            break;

        case TRIAD_LAST_RING_STMT:
            f_sep(s, c, &fs); f_key(s, "members");
            emit_ident_array(s, c, n->u.ring_seq.members, n->u.ring_seq.members_len);
            f_sep(s, c, &fs); f_key(s, "kappa"); emit_float(s, n->u.ring_seq.kappa);
            f_sep(s, c, &fs); f_key(s, "duration"); emit_float(s, n->u.ring_seq.duration);
            f_sep(s, c, &fs); f_key(s, "line"); emit_int(s, n->line);
            break;

        case TRIAD_LAST_SEQUENCE_STMT:
            f_sep(s, c, &fs); f_key(s, "target"); emit_identref(s, c, n->u.ring_seq.target);
            f_sep(s, c, &fs); f_key(s, "inputs");
            emit_ident_array(s, c, n->u.ring_seq.members, n->u.ring_seq.members_len);
            f_sep(s, c, &fs); f_key(s, "each_for"); emit_float(s, n->u.ring_seq.duration);
            f_sep(s, c, &fs); f_key(s, "line"); emit_int(s, n->line);
            break;

        case TRIAD_LAST_OBSERVE_STMT:
            f_sep(s, c, &fs); f_key(s, "target"); emit_identref(s, c, n->u.observe_stmt.target);
            f_sep(s, c, &fs); f_key(s, "metrics");
            emit_string_array(s, c, n->u.observe_stmt.metrics, n->u.observe_stmt.metrics_len);
            f_sep(s, c, &fs); f_key(s, "over_seeds"); emit_int(s, n->u.observe_stmt.over_seeds);
            f_sep(s, c, &fs); f_key(s, "stream_to"); emit_string(s, n->u.observe_stmt.stream_to);
            f_sep(s, c, &fs); f_key(s, "line"); emit_int(s, n->line);
            break;

        case TRIAD_LAST_ASSERT_STMT:
            f_sep(s, c, &fs); f_key(s, "predicate"); emit_string(s, n->u.assert_stmt.predicate);
            f_sep(s, c, &fs); f_key(s, "target"); emit_identref(s, c, n->u.assert_stmt.target);
            f_sep(s, c, &fs); f_key(s, "line"); emit_int(s, n->line);
            break;

        case TRIAD_LAST_CHECKPOINT_STMT:
            f_sep(s, c, &fs); f_key(s, "target"); emit_identref(s, c, n->u.checkpoint_stmt.target);
            f_sep(s, c, &fs); f_key(s, "path"); emit_string(s, n->u.checkpoint_stmt.path);
            f_sep(s, c, &fs); f_key(s, "line"); emit_int(s, n->line);
            break;

        case TRIAD_LAST_SUBSTRATE_DECL:
            f_sep(s, c, &fs); f_key(s, "name"); emit_string(s, n->u.substrate_decl.name);
            f_sep(s, c, &fs); f_key(s, "composed_of");
            emit_ident_array(s, c, n->u.substrate_decl.composed_of, n->u.substrate_decl.composed_of_len);
            f_sep(s, c, &fs); f_key(s, "properties");
            emit_override_dict(s, c, n->u.substrate_decl.properties, n->u.substrate_decl.properties_len);
            f_sep(s, c, &fs); f_key(s, "line"); emit_int(s, n->line);
            break;

        default:
            break;
    }
    f_end(s, c);
}

char *triad_legacy_dump_json(const TriadLegacyNode *prog, int indent) {
    S s = {0};
    Ctx c = { .indent = indent, .depth = 0 };
    emit_node(&s, &c, prog);
    if (s.data) return s.data;
    char *fb = (char *)malloc(5);
    if (fb) memcpy(fb, "null", 5);
    return fb;
}
