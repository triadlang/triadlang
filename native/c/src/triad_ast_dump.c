/* triad_ast_dump.c — JSON dumper for the native AST.
 *
 * The output schema matches what scripts/ast_to_json.py emits on the
 * Python side: every dataclass becomes {"_type": "<ClassName>", ...
 * field: value}. Positions are objects with line/col/file. None values
 * are emitted as JSON null. Tuples become arrays.
 */
#include "triad_frontend.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* growable string buffer */
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
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if ((size_t)n < sizeof(buf)) { s_puts(s, buf); return; }
    char *big = (char *)malloc((size_t)n + 1);
    va_start(ap, fmt);
    vsnprintf(big, n + 1, fmt, ap);
    va_end(ap);
    s_puts(s, big);
    free(big);
}

static void emit_json_string(S *s, const char *t) {
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

typedef struct {
    int indent;       /* spaces per level; 0 = compact */
    int depth;
} Ctx;

static void emit_nl_indent(S *s, Ctx *c) {
    if (c->indent <= 0) return;
    s_putc(s, '\n');
    for (int i = 0; i < c->indent * c->depth; ++i) s_putc(s, ' ');
}

static void emit_node(S *s, Ctx *c, const TriadAstNode *n);

static void emit_pos(S *s, Ctx *c, TriadPos p) {
    s_putc(s, '{');
    c->depth++;
    emit_nl_indent(s, c);
    s_puts(s, "\"_type\": \"Pos\",");
    emit_nl_indent(s, c);
    s_printf(s, "\"line\": %d,", p.line);
    emit_nl_indent(s, c);
    s_printf(s, "\"col\": %d,", p.col);
    emit_nl_indent(s, c);
    s_puts(s, "\"file\": ");
    emit_json_string(s, p.file ? p.file : "");
    c->depth--;
    emit_nl_indent(s, c);
    s_putc(s, '}');
}

static void emit_null(S *s) { s_puts(s, "null"); }

static void emit_int(S *s, long long v)  { s_printf(s, "%lld", v); }
static void emit_bool(S *s, int v)       { s_puts(s, v ? "true" : "false"); }

static void emit_float(S *s, double v) {
    if (isnan(v)) { s_puts(s, "NaN"); return; }
    if (isinf(v)) { s_puts(s, v > 0 ? "Infinity" : "-Infinity"); return; }
    /* Match Python float.__repr__: shortest decimal that round-trips,
     * with Python's preference for fixed-point over exponential when
     * the exponent is in [-4, 16] (the same window used by repr).
     *
     * Strategy: find the minimum precision p in 1..17 such that
     *   strtod(snprintf("%.*g", p, v)) == v.
     * Then format with both "%.*e" (which gives the canonical mantissa
     * at p-1 significant digits) and a fixed-point form, and pick the
     * one Python would pick: fixed when the decimal exponent is in
     * [-4, 16], otherwise exponential. */
    char buf[64];
    int prec = 17;
    for (int p = 1; p <= 17; ++p) {
        snprintf(buf, sizeof(buf), "%.*g", p, v);
        if (strtod(buf, NULL) == v) { prec = p; break; }
    }

    /* Extract decimal exponent of v using %.*e at `prec`. */
    char ebuf[64];
    snprintf(ebuf, sizeof(ebuf), "%.*e", prec - 1, v);
    /* Parse exponent. */
    const char *epos = strchr(ebuf, 'e');
    int decexp = 0;
    if (epos) decexp = atoi(epos + 1);

    int use_fixed = (decexp >= -4 && decexp < 16);

    if (use_fixed) {
        /* Compute number of digits after the decimal point for %.*f.
         * Total significant digits is `prec`; the integer part has
         * (decexp + 1) digits when decexp >= 0, else 0 (with leading
         * zeros after the point). */
        int after;
        if (decexp >= 0) after = prec - 1 - decexp;
        else             after = prec - 1 + (-decexp);
        if (after < 0) after = 0;
        snprintf(buf, sizeof(buf), "%.*f", after, v);
        /* Trim trailing zeros after the decimal point, but keep at
         * least one digit (Python keeps "10.0", not "10"). */
        char *dot = strchr(buf, '.');
        if (dot) {
            char *end = buf + strlen(buf) - 1;
            while (end > dot + 1 && *end == '0') { *end = '\0'; --end; }
        } else {
            /* No decimal point yet — append .0 so JSON stays a float. */
            size_t L = strlen(buf);
            if (L + 2 < sizeof(buf)) { buf[L] = '.'; buf[L+1] = '0'; buf[L+2] = '\0'; }
        }
        /* Sanity: round-trip; if it fails (shouldn't for prec sigfigs),
         * fall through to exponential form. */
        if (strtod(buf, NULL) != v) use_fixed = 0;
    }
    if (!use_fixed) {
        /* Build "<mantissa>e<sign><exp>" matching Python: mantissa
         * shortened by trimming trailing zeros, exponent without
         * leading zeros (but at least two digits is *not* required
         * in Python; e.g. repr(1e20) == '1e+20'). */
        /* ebuf currently is "<sign>d.dddde<sign>NN". Trim mantissa zeros. */
        char *e2 = strchr(ebuf, 'e');
        if (e2) {
            char *dot = strchr(ebuf, '.');
            if (dot && dot < e2) {
                char *q = e2 - 1;
                while (q > dot && *q == '0') { memmove(q, q + 1, strlen(q)); --q; e2--; }
                if (q == dot) { memmove(dot, dot + 1, strlen(dot)); e2--; }
            }
            /* Normalise exponent: keep sign, strip leading zeros, but
             * Python emits at least two digits ("1e+20", "1e-05"). */
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

static void emit_str_or_null(S *s, const char *t) {
    if (!t) emit_null(s);
    else    emit_json_string(s, t);
}

/* ── Field helpers ──────────────────────────────────────────────── */

typedef struct {
    int  first;
} FieldState;

static void emit_field_sep(S *s, Ctx *c, FieldState *fs) {
    if (!fs->first) s_putc(s, ',');
    fs->first = 0;
    emit_nl_indent(s, c);
}

static void emit_field_key(S *s, const char *key) {
    s_putc(s, '"'); s_puts(s, key); s_puts(s, "\": ");
}

static void start_obj(S *s, Ctx *c, FieldState *fs, const char *type_name) {
    s_putc(s, '{');
    c->depth++;
    fs->first = 1;
    emit_field_sep(s, c, fs);
    emit_field_key(s, "_type");
    emit_json_string(s, type_name);
}

static void end_obj(S *s, Ctx *c) {
    c->depth--;
    emit_nl_indent(s, c);
    s_putc(s, '}');
}

/* Emit a pos field on every dataclass that has one. */
static void emit_pos_field(S *s, Ctx *c, FieldState *fs, TriadPos p) {
    emit_field_sep(s, c, fs);
    emit_field_key(s, "pos");
    emit_pos(s, c, p);
}

/* Emit an array of statement/expression nodes. */
static void emit_node_array(S *s, Ctx *c, TriadAstNode *const *items, size_t n) {
    if (n == 0) { s_puts(s, "[]"); return; }
    s_putc(s, '[');
    c->depth++;
    for (size_t k = 0; k < n; ++k) {
        if (k) s_putc(s, ',');
        emit_nl_indent(s, c);
        emit_node(s, c, items[k]);
    }
    c->depth--;
    emit_nl_indent(s, c);
    s_putc(s, ']');
}

static void emit_string_array(S *s, Ctx *c, const char *const *items, size_t n) {
    if (n == 0) { s_puts(s, "[]"); return; }
    s_putc(s, '[');
    c->depth++;
    for (size_t k = 0; k < n; ++k) {
        if (k) s_putc(s, ',');
        emit_nl_indent(s, c);
        emit_json_string(s, items[k] ? items[k] : "");
    }
    c->depth--;
    emit_nl_indent(s, c);
    s_putc(s, ']');
}

/* Map (string -> node) → JSON object. */
static void emit_str_node_map(S *s, Ctx *c, const TriadStrEntry *e, size_t n) {
    s_putc(s, '{');
    c->depth++;
    for (size_t k = 0; k < n; ++k) {
        if (k) s_putc(s, ',');
        emit_nl_indent(s, c);
        emit_json_string(s, e[k].key);
        s_puts(s, ": ");
        emit_node(s, c, e[k].value);
    }
    c->depth--;
    if (n) emit_nl_indent(s, c);
    s_putc(s, '}');
}

/* Map (string -> node) using Python {key: expr}-like JSON. Used for
 * RegStmt.overrides (Python: dict). */

/* Param object */
static void emit_param(S *s, Ctx *c, const TriadParam *p) {
    FieldState fs = {1};
    start_obj(s, c, &fs, "Param");
    emit_field_sep(s, c, &fs); emit_field_key(s, "name"); emit_str_or_null(s, p->name);
    emit_field_sep(s, c, &fs); emit_field_key(s, "type_ann"); emit_str_or_null(s, p->type_ann);
    emit_field_sep(s, c, &fs); emit_field_key(s, "default");
    if (p->default_value) emit_node(s, c, p->default_value); else emit_null(s);
    emit_field_sep(s, c, &fs); emit_field_key(s, "is_args"); emit_bool(s, p->is_args);
    emit_field_sep(s, c, &fs); emit_field_key(s, "is_kwargs"); emit_bool(s, p->is_kwargs);
    emit_pos_field(s, c, &fs, p->pos);
    end_obj(s, c);
}

static void emit_param_array(S *s, Ctx *c, const TriadParam *items, size_t n) {
    if (n == 0) { s_puts(s, "[]"); return; }
    s_putc(s, '[');
    c->depth++;
    for (size_t k = 0; k < n; ++k) {
        if (k) s_putc(s, ',');
        emit_nl_indent(s, c);
        emit_param(s, c, &items[k]);
    }
    c->depth--;
    emit_nl_indent(s, c);
    s_putc(s, ']');
}

static void emit_type_field(S *s, Ctx *c, const TriadTypeField *f) {
    FieldState fs = {1};
    start_obj(s, c, &fs, "TypeField");
    emit_field_sep(s, c, &fs); emit_field_key(s, "name"); emit_str_or_null(s, f->name);
    emit_field_sep(s, c, &fs); emit_field_key(s, "type_ann"); emit_str_or_null(s, f->type_ann);
    emit_field_sep(s, c, &fs); emit_field_key(s, "default");
    if (f->default_value) emit_node(s, c, f->default_value); else emit_null(s);
    emit_pos_field(s, c, &fs, f->pos);
    end_obj(s, c);
}

static void emit_typefield_array(S *s, Ctx *c, const TriadTypeField *items, size_t n) {
    if (n == 0) { s_puts(s, "[]"); return; }
    s_putc(s, '[');
    c->depth++;
    for (size_t k = 0; k < n; ++k) {
        if (k) s_putc(s, ',');
        emit_nl_indent(s, c);
        emit_type_field(s, c, &items[k]);
    }
    c->depth--;
    emit_nl_indent(s, c);
    s_putc(s, ']');
}

/* IfStmt.elif_clauses : list[tuple[Expr, list[Stmt]]]. _serialize maps
 * tuples to JSON arrays, so each clause becomes [cond, [body...]]. */
static void emit_elif_clauses(S *s, Ctx *c, const TriadElifClause *items, size_t n) {
    if (n == 0) { s_puts(s, "[]"); return; }
    s_putc(s, '[');
    c->depth++;
    for (size_t k = 0; k < n; ++k) {
        if (k) s_putc(s, ',');
        emit_nl_indent(s, c);
        s_putc(s, '[');
        c->depth++;
        emit_nl_indent(s, c);
        emit_node(s, c, items[k].cond);
        s_putc(s, ',');
        emit_nl_indent(s, c);
        emit_node_array(s, c, items[k].body, items[k].body_len);
        c->depth--;
        emit_nl_indent(s, c);
        s_putc(s, ']');
    }
    c->depth--;
    emit_nl_indent(s, c);
    s_putc(s, ']');
}

/* MatchCase: not a dataclass with default field("Pos"), but plain dataclass. */
static void emit_match_case(S *s, Ctx *c, const TriadMatchCase *mc) {
    FieldState fs = {1};
    start_obj(s, c, &fs, "MatchCase");
    emit_field_sep(s, c, &fs); emit_field_key(s, "pattern"); emit_node(s, c, mc->pattern);
    emit_field_sep(s, c, &fs); emit_field_key(s, "guard");
    if (mc->guard) emit_node(s, c, mc->guard); else emit_null(s);
    emit_field_sep(s, c, &fs); emit_field_key(s, "body");
    emit_node_array(s, c, mc->body, mc->body_len);
    end_obj(s, c);
}

static void emit_match_case_array(S *s, Ctx *c, const TriadMatchCase *items, size_t n) {
    if (n == 0) { s_puts(s, "[]"); return; }
    s_putc(s, '[');
    c->depth++;
    for (size_t k = 0; k < n; ++k) {
        if (k) s_putc(s, ',');
        emit_nl_indent(s, c);
        emit_match_case(s, c, &items[k]);
    }
    c->depth--;
    emit_nl_indent(s, c);
    s_putc(s, ']');
}

/* Map literal pairs: list[tuple[Expr, Expr]] -> list of 2-element arrays. */
static void emit_map_pairs(S *s, Ctx *c, const TriadMapPair *items, size_t n) {
    if (n == 0) { s_puts(s, "[]"); return; }
    s_putc(s, '[');
    c->depth++;
    for (size_t k = 0; k < n; ++k) {
        if (k) s_putc(s, ',');
        emit_nl_indent(s, c);
        s_putc(s, '[');
        c->depth++;
        emit_nl_indent(s, c);
        emit_node(s, c, items[k].key);
        s_putc(s, ',');
        emit_nl_indent(s, c);
        emit_node(s, c, items[k].value);
        c->depth--;
        emit_nl_indent(s, c);
        s_putc(s, ']');
    }
    c->depth--;
    emit_nl_indent(s, c);
    s_putc(s, ']');
}

/* CallExpr/MethodCallExpr kwargs: dict[str, Expr] */
static void emit_kwargs(S *s, Ctx *c, const TriadKwArg *items, size_t n) {
    s_putc(s, '{');
    c->depth++;
    for (size_t k = 0; k < n; ++k) {
        if (k) s_putc(s, ',');
        emit_nl_indent(s, c);
        emit_json_string(s, items[k].name);
        s_puts(s, ": ");
        emit_node(s, c, items[k].value);
    }
    c->depth--;
    if (n) emit_nl_indent(s, c);
    s_putc(s, '}');
}

/* FStringExpr.parts: list of (type, value). Python type is "str" or
 * "expr". _serialize maps tuples to arrays. */
static void emit_fstring_parts(S *s, Ctx *c, const TriadFStringAstPart *items, size_t n) {
    if (n == 0) { s_puts(s, "[]"); return; }
    s_putc(s, '[');
    c->depth++;
    for (size_t k = 0; k < n; ++k) {
        if (k) s_putc(s, ',');
        emit_nl_indent(s, c);
        s_putc(s, '[');
        c->depth++;
        emit_nl_indent(s, c);
        emit_json_string(s, items[k].is_expr ? "expr" : "str");
        s_putc(s, ',');
        emit_nl_indent(s, c);
        if (items[k].is_expr) emit_node(s, c, items[k].expr);
        else                  emit_json_string(s, items[k].text ? items[k].text : "");
        c->depth--;
        emit_nl_indent(s, c);
        s_putc(s, ']');
    }
    c->depth--;
    emit_nl_indent(s, c);
    s_putc(s, ']');
}

/* ── Main node dispatcher ───────────────────────────────────────── */

static void emit_node(S *s, Ctx *c, const TriadAstNode *n) {
    if (!n) { emit_null(s); return; }

    FieldState fs = {1};
    const char *tname = triad_ast_kind_name(n->kind);
    start_obj(s, c, &fs, tname);

    switch (n->kind) {
        case TRIAD_AST_INT_LIT:
            emit_field_sep(s, c, &fs); emit_field_key(s, "value"); emit_int(s, n->u.int_val);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_FLOAT_LIT:
            emit_field_sep(s, c, &fs); emit_field_key(s, "value"); emit_float(s, n->u.float_val);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_BOOL_LIT:
            emit_field_sep(s, c, &fs); emit_field_key(s, "value"); emit_bool(s, n->u.bool_val);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_STRING_LIT:
            emit_field_sep(s, c, &fs); emit_field_key(s, "value"); emit_str_or_null(s, n->u.string_val);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_NONE_LIT:
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_IDENT:
            emit_field_sep(s, c, &fs); emit_field_key(s, "name"); emit_str_or_null(s, n->u.ident_name);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_BINOP:
            emit_field_sep(s, c, &fs); emit_field_key(s, "op"); emit_str_or_null(s, n->u.op.op);
            emit_field_sep(s, c, &fs); emit_field_key(s, "left"); emit_node(s, c, n->u.op.left);
            emit_field_sep(s, c, &fs); emit_field_key(s, "right"); emit_node(s, c, n->u.op.right);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_UNARYOP:
            emit_field_sep(s, c, &fs); emit_field_key(s, "op"); emit_str_or_null(s, n->u.op.op);
            emit_field_sep(s, c, &fs); emit_field_key(s, "operand"); emit_node(s, c, n->u.op.right);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_CALL:
            emit_field_sep(s, c, &fs); emit_field_key(s, "func"); emit_node(s, c, n->u.call.func_or_obj);
            emit_field_sep(s, c, &fs); emit_field_key(s, "args"); emit_node_array(s, c, n->u.call.args, n->u.call.args_len);
            emit_field_sep(s, c, &fs); emit_field_key(s, "kwargs"); emit_kwargs(s, c, n->u.call.kwargs, n->u.call.kwargs_len);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_INDEX:
            emit_field_sep(s, c, &fs); emit_field_key(s, "obj"); emit_node(s, c, n->u.index.obj);
            emit_field_sep(s, c, &fs); emit_field_key(s, "index"); emit_node(s, c, n->u.index.index);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_SLICE:
            emit_field_sep(s, c, &fs); emit_field_key(s, "start");
            if (n->u.slice.start) emit_node(s, c, n->u.slice.start); else emit_null(s);
            emit_field_sep(s, c, &fs); emit_field_key(s, "end");
            if (n->u.slice.end) emit_node(s, c, n->u.slice.end); else emit_null(s);
            emit_field_sep(s, c, &fs); emit_field_key(s, "step");
            if (n->u.slice.step) emit_node(s, c, n->u.slice.step); else emit_null(s);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_FIELD:
            emit_field_sep(s, c, &fs); emit_field_key(s, "obj"); emit_node(s, c, n->u.field.obj);
            emit_field_sep(s, c, &fs); emit_field_key(s, "field"); emit_str_or_null(s, n->u.field.field);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_LIST:
            emit_field_sep(s, c, &fs); emit_field_key(s, "elements");
            emit_node_array(s, c, n->u.list.elements, n->u.list.elements_len);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_LIST_COMP:
            emit_field_sep(s, c, &fs); emit_field_key(s, "expr"); emit_node(s, c, n->u.list_comp.expr);
            emit_field_sep(s, c, &fs); emit_field_key(s, "var"); emit_str_or_null(s, n->u.list_comp.var);
            emit_field_sep(s, c, &fs); emit_field_key(s, "iter"); emit_node(s, c, n->u.list_comp.iter);
            emit_field_sep(s, c, &fs); emit_field_key(s, "condition");
            if (n->u.list_comp.condition) emit_node(s, c, n->u.list_comp.condition); else emit_null(s);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_MAP:
            emit_field_sep(s, c, &fs); emit_field_key(s, "pairs");
            emit_map_pairs(s, c, n->u.map.pairs, n->u.map.pairs_len);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_TUPLE:
            emit_field_sep(s, c, &fs); emit_field_key(s, "elements");
            emit_node_array(s, c, n->u.list.elements, n->u.list.elements_len);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_FSTRING:
            emit_field_sep(s, c, &fs); emit_field_key(s, "parts");
            emit_fstring_parts(s, c, n->u.fstring.parts, n->u.fstring.parts_len);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_LAMBDA:
            emit_field_sep(s, c, &fs); emit_field_key(s, "params");
            emit_param_array(s, c, n->u.lambda.params, n->u.lambda.params_len);
            emit_field_sep(s, c, &fs); emit_field_key(s, "body");
            emit_node_array(s, c, n->u.lambda.body, n->u.lambda.body_len);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_METHOD_CALL:
            emit_field_sep(s, c, &fs); emit_field_key(s, "obj"); emit_node(s, c, n->u.call.func_or_obj);
            emit_field_sep(s, c, &fs); emit_field_key(s, "method"); emit_str_or_null(s, n->u.call.method);
            emit_field_sep(s, c, &fs); emit_field_key(s, "args"); emit_node_array(s, c, n->u.call.args, n->u.call.args_len);
            emit_field_sep(s, c, &fs); emit_field_key(s, "kwargs"); emit_kwargs(s, c, n->u.call.kwargs, n->u.call.kwargs_len);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_ASSIGN_EXPR:
            emit_field_sep(s, c, &fs); emit_field_key(s, "target"); emit_node(s, c, n->u.assign_expr.target);
            emit_field_sep(s, c, &fs); emit_field_key(s, "value"); emit_node(s, c, n->u.assign_expr.value);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_YIELD_EXPR:
        case TRIAD_AST_AWAIT_EXPR:
            emit_field_sep(s, c, &fs); emit_field_key(s, "value");
            if (n->u.unary_value.value) emit_node(s, c, n->u.unary_value.value); else emit_null(s);
            emit_pos_field(s, c, &fs, n->pos);
            break;

        case TRIAD_AST_LET:
            emit_field_sep(s, c, &fs); emit_field_key(s, "name"); emit_str_or_null(s, n->u.let_stmt.name);
            emit_field_sep(s, c, &fs); emit_field_key(s, "type_ann"); emit_str_or_null(s, n->u.let_stmt.type_ann);
            emit_field_sep(s, c, &fs); emit_field_key(s, "value");
            if (n->u.let_stmt.value) emit_node(s, c, n->u.let_stmt.value); else emit_null(s);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_DESTRUCT_LET:
        case TRIAD_AST_MAP_DESTRUCT:
            emit_field_sep(s, c, &fs); emit_field_key(s, "names");
            emit_string_array(s, c, n->u.destruct.names, n->u.destruct.names_len);
            emit_field_sep(s, c, &fs); emit_field_key(s, "value"); emit_node(s, c, n->u.destruct.value);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_CONST:
            emit_field_sep(s, c, &fs); emit_field_key(s, "name"); emit_str_or_null(s, n->u.const_stmt.name);
            emit_field_sep(s, c, &fs); emit_field_key(s, "value"); emit_node(s, c, n->u.const_stmt.value);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_ASSIGN:
            emit_field_sep(s, c, &fs); emit_field_key(s, "target"); emit_node(s, c, n->u.assign_stmt.target);
            emit_field_sep(s, c, &fs); emit_field_key(s, "value"); emit_node(s, c, n->u.assign_stmt.value);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_EXPR_STMT:
            emit_field_sep(s, c, &fs); emit_field_key(s, "expr"); emit_node(s, c, n->u.expr_stmt.expr);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_RETURN:
        case TRIAD_AST_THROW:
        case TRIAD_AST_YIELD_STMT:
            emit_field_sep(s, c, &fs); emit_field_key(s, "value");
            if (n->u.unary_value.value) emit_node(s, c, n->u.unary_value.value); else emit_null(s);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_BREAK:
        case TRIAD_AST_CONTINUE:
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_IF:
            emit_field_sep(s, c, &fs); emit_field_key(s, "condition"); emit_node(s, c, n->u.if_stmt.condition);
            emit_field_sep(s, c, &fs); emit_field_key(s, "then_body");
            emit_node_array(s, c, n->u.if_stmt.then_body, n->u.if_stmt.then_body_len);
            emit_field_sep(s, c, &fs); emit_field_key(s, "elif_clauses");
            emit_elif_clauses(s, c, n->u.if_stmt.elif_clauses, n->u.if_stmt.elif_clauses_len);
            emit_field_sep(s, c, &fs); emit_field_key(s, "else_body");
            if (n->u.if_stmt.has_else)
                emit_node_array(s, c, n->u.if_stmt.else_body, n->u.if_stmt.else_body_len);
            else
                emit_null(s);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_FOR:
            emit_field_sep(s, c, &fs); emit_field_key(s, "var"); emit_str_or_null(s, n->u.for_stmt.var);
            emit_field_sep(s, c, &fs); emit_field_key(s, "iter"); emit_node(s, c, n->u.for_stmt.iter);
            emit_field_sep(s, c, &fs); emit_field_key(s, "body");
            emit_node_array(s, c, n->u.for_stmt.body, n->u.for_stmt.body_len);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_WHILE:
            emit_field_sep(s, c, &fs); emit_field_key(s, "condition"); emit_node(s, c, n->u.while_stmt.condition);
            emit_field_sep(s, c, &fs); emit_field_key(s, "body");
            emit_node_array(s, c, n->u.while_stmt.body, n->u.while_stmt.body_len);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_FN_DECL:
            emit_field_sep(s, c, &fs); emit_field_key(s, "name"); emit_str_or_null(s, n->u.fn_decl.name);
            emit_field_sep(s, c, &fs); emit_field_key(s, "params");
            emit_param_array(s, c, n->u.fn_decl.params, n->u.fn_decl.params_len);
            emit_field_sep(s, c, &fs); emit_field_key(s, "return_type"); emit_str_or_null(s, n->u.fn_decl.return_type);
            emit_field_sep(s, c, &fs); emit_field_key(s, "body");
            emit_node_array(s, c, n->u.fn_decl.body, n->u.fn_decl.body_len);
            emit_field_sep(s, c, &fs); emit_field_key(s, "is_async"); emit_bool(s, n->u.fn_decl.is_async);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_TRY_CATCH:
            emit_field_sep(s, c, &fs); emit_field_key(s, "body");
            emit_node_array(s, c, n->u.try_catch.body, n->u.try_catch.body_len);
            emit_field_sep(s, c, &fs); emit_field_key(s, "catch_var"); emit_str_or_null(s, n->u.try_catch.catch_var);
            emit_field_sep(s, c, &fs); emit_field_key(s, "catch_body");
            emit_node_array(s, c, n->u.try_catch.catch_body, n->u.try_catch.catch_body_len);
            emit_field_sep(s, c, &fs); emit_field_key(s, "finally_body");
            emit_node_array(s, c, n->u.try_catch.finally_body, n->u.try_catch.finally_body_len);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_TYPE_DECL:
            emit_field_sep(s, c, &fs); emit_field_key(s, "name"); emit_str_or_null(s, n->u.type_decl.name);
            emit_field_sep(s, c, &fs); emit_field_key(s, "fields");
            emit_typefield_array(s, c, n->u.type_decl.fields, n->u.type_decl.fields_len);
            emit_field_sep(s, c, &fs); emit_field_key(s, "methods");
            emit_node_array(s, c, n->u.type_decl.methods, n->u.type_decl.methods_len);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_CLASS_DECL:
            emit_field_sep(s, c, &fs); emit_field_key(s, "name"); emit_str_or_null(s, n->u.type_decl.name);
            emit_field_sep(s, c, &fs); emit_field_key(s, "parent"); emit_str_or_null(s, n->u.type_decl.parent);
            emit_field_sep(s, c, &fs); emit_field_key(s, "fields");
            emit_typefield_array(s, c, n->u.type_decl.fields, n->u.type_decl.fields_len);
            emit_field_sep(s, c, &fs); emit_field_key(s, "methods");
            emit_node_array(s, c, n->u.type_decl.methods, n->u.type_decl.methods_len);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_IMPORT:
            emit_field_sep(s, c, &fs); emit_field_key(s, "path");
            emit_string_array(s, c, n->u.import_stmt.path, n->u.import_stmt.path_len);
            emit_field_sep(s, c, &fs); emit_field_key(s, "alias"); emit_str_or_null(s, n->u.import_stmt.alias);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_FROM_IMPORT:
            emit_field_sep(s, c, &fs); emit_field_key(s, "path");
            emit_string_array(s, c, n->u.from_import.path, n->u.from_import.path_len);
            emit_field_sep(s, c, &fs); emit_field_key(s, "names");
            emit_string_array(s, c, n->u.from_import.names, n->u.from_import.names_len);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_MATCH:
            emit_field_sep(s, c, &fs); emit_field_key(s, "subject"); emit_node(s, c, n->u.match_stmt.subject);
            emit_field_sep(s, c, &fs); emit_field_key(s, "cases");
            emit_match_case_array(s, c, n->u.match_stmt.cases, n->u.match_stmt.cases_len);
            emit_field_sep(s, c, &fs); emit_field_key(s, "else_body");
            if (n->u.match_stmt.has_else)
                emit_node_array(s, c, n->u.match_stmt.else_body, n->u.match_stmt.else_body_len);
            else
                emit_null(s);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_REG:
            emit_field_sep(s, c, &fs); emit_field_key(s, "name"); emit_str_or_null(s, n->u.reg_stmt.name);
            emit_field_sep(s, c, &fs); emit_field_key(s, "regime"); emit_str_or_null(s, n->u.reg_stmt.regime);
            emit_field_sep(s, c, &fs); emit_field_key(s, "value");
            if (n->u.reg_stmt.value) emit_node(s, c, n->u.reg_stmt.value); else emit_null(s);
            emit_field_sep(s, c, &fs); emit_field_key(s, "overrides");
            if (n->u.reg_stmt.has_overrides)
                emit_str_node_map(s, c, n->u.reg_stmt.overrides, n->u.reg_stmt.overrides_len);
            else
                emit_null(s);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_ENTITY:
            emit_field_sep(s, c, &fs); emit_field_key(s, "name"); emit_str_or_null(s, n->u.entity_decl.name);
            emit_field_sep(s, c, &fs); emit_field_key(s, "base"); emit_str_or_null(s, n->u.entity_decl.base);
            emit_field_sep(s, c, &fs); emit_field_key(s, "fields");
            emit_str_node_map(s, c, n->u.entity_decl.fields, n->u.entity_decl.fields_len);
            emit_field_sep(s, c, &fs); emit_field_key(s, "methods");
            emit_node_array(s, c, n->u.entity_decl.methods, n->u.entity_decl.methods_len);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_WORLD:
            emit_field_sep(s, c, &fs); emit_field_key(s, "name"); emit_str_or_null(s, n->u.world_decl.name);
            emit_field_sep(s, c, &fs); emit_field_key(s, "fields");
            emit_str_node_map(s, c, n->u.world_decl.fields, n->u.world_decl.fields_len);
            emit_field_sep(s, c, &fs); emit_field_key(s, "entities");
            emit_node_array(s, c, n->u.world_decl.entities, n->u.world_decl.entities_len);
            emit_field_sep(s, c, &fs); emit_field_key(s, "body");
            emit_node_array(s, c, n->u.world_decl.body, n->u.world_decl.body_len);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_COUPLE:
            emit_field_sep(s, c, &fs); emit_field_key(s, "src"); emit_str_or_null(s, n->u.couple_stmt.src);
            emit_field_sep(s, c, &fs); emit_field_key(s, "dst"); emit_str_or_null(s, n->u.couple_stmt.dst);
            emit_field_sep(s, c, &fs); emit_field_key(s, "kappa");
            if (n->u.couple_stmt.kappa) emit_node(s, c, n->u.couple_stmt.kappa); else emit_null(s);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_PAIR:
            emit_field_sep(s, c, &fs); emit_field_key(s, "a"); emit_str_or_null(s, n->u.pair_stmt.a);
            emit_field_sep(s, c, &fs); emit_field_key(s, "b"); emit_str_or_null(s, n->u.pair_stmt.b);
            emit_field_sep(s, c, &fs); emit_field_key(s, "kappa");
            if (n->u.pair_stmt.kappa) emit_node(s, c, n->u.pair_stmt.kappa); else emit_null(s);
            emit_field_sep(s, c, &fs); emit_field_key(s, "duration");
            if (n->u.pair_stmt.duration) emit_node(s, c, n->u.pair_stmt.duration); else emit_null(s);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_RING:
            emit_field_sep(s, c, &fs); emit_field_key(s, "members");
            emit_string_array(s, c, n->u.ring_stmt.members, n->u.ring_stmt.members_len);
            emit_field_sep(s, c, &fs); emit_field_key(s, "kappa");
            if (n->u.ring_stmt.kappa) emit_node(s, c, n->u.ring_stmt.kappa); else emit_null(s);
            emit_field_sep(s, c, &fs); emit_field_key(s, "duration");
            if (n->u.ring_stmt.duration) emit_node(s, c, n->u.ring_stmt.duration); else emit_null(s);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_OBSERVE:
            emit_field_sep(s, c, &fs); emit_field_key(s, "target"); emit_str_or_null(s, n->u.observe_stmt.target);
            emit_field_sep(s, c, &fs); emit_field_key(s, "metrics");
            emit_string_array(s, c, n->u.observe_stmt.metrics, n->u.observe_stmt.metrics_len);
            emit_field_sep(s, c, &fs); emit_field_key(s, "over_seeds"); emit_int(s, n->u.observe_stmt.over_seeds);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_RUN:
            emit_field_sep(s, c, &fs); emit_field_key(s, "duration");
            if (n->u.run_stmt.duration) emit_node(s, c, n->u.run_stmt.duration); else emit_null(s);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_ANNOTATION:
            emit_field_sep(s, c, &fs); emit_field_key(s, "key"); emit_str_or_null(s, n->u.annotation_stmt.key);
            emit_field_sep(s, c, &fs); emit_field_key(s, "args"); emit_str_or_null(s, n->u.annotation_stmt.args);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_MODULE:
            emit_field_sep(s, c, &fs); emit_field_key(s, "name"); emit_str_or_null(s, n->u.module.name);
            emit_field_sep(s, c, &fs); emit_field_key(s, "body");
            emit_node_array(s, c, n->u.module.body, n->u.module.body_len);
            emit_field_sep(s, c, &fs); emit_field_key(s, "file"); emit_str_or_null(s, n->u.module.file);
            break;
        default:
            break;
    }
    end_obj(s, c);
}

/* ── Public API ─────────────────────────────────────────────────── */

char *triad_ast_dump_json(const TriadAstNode *module, int indent) {
    S s = {0};
    Ctx c = { .indent = indent, .depth = 0 };
    emit_node(&s, &c, module);
    if (s.data) return s.data;
    char *fallback = (char *)malloc(5);
    if (fallback) memcpy(fallback, "null", 5);
    return fallback;
}

int triad_ast_dump_json_fp(const TriadAstNode *module, FILE *fp, int indent) {
    char *t = triad_ast_dump_json(module, indent);
    if (!t) return -1;
    fputs(t, fp);
    free(t);
    return 0;
}
