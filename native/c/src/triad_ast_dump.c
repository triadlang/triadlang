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
                if (c < 0x20) {
                    s_printf(s, "\\u%04x", c);
                } else if (c < 0x80) {
                    s_putc(s, (char)c);
                } else {

                    unsigned cp = 0;
                    int extra = 0;
                    if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; extra = 1; }
                    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; extra = 2; }
                    else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; extra = 3; }
                    else { s_putc(s, (char)c); continue; }
                    int ok = 1;
                    for (int k = 1; k <= extra; ++k) {
                        if ((p[k] & 0xC0) != 0x80) { ok = 0; break; }
                        cp = (cp << 6) | (p[k] & 0x3F);
                    }
                    if (!ok) { s_putc(s, (char)c); continue; }
                    p += extra;
                    if (cp > 0xFFFF) {
                        cp -= 0x10000;
                        s_printf(s, "\\u%04x", 0xD800 + (cp >> 10));
                        s_printf(s, "\\u%04x", 0xDC00 + (cp & 0x3FF));
                    } else {
                        s_printf(s, "\\u%04x", cp);
                    }
                }
        }
    }
    s_putc(s, '"');
}

typedef struct {
    int indent;
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

    char buf[64];
    int prec = 17;
    for (int p = 1; p <= 17; ++p) {
        snprintf(buf, sizeof(buf), "%.*g", p, v);
        if (strtod(buf, NULL) == v) { prec = p; break; }
    }

    char ebuf[64];
    snprintf(ebuf, sizeof(ebuf), "%.*e", prec - 1, v);

    const char *epos = strchr(ebuf, 'e');
    int decexp = 0;
    if (epos) decexp = atoi(epos + 1);

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

static void emit_str_or_null(S *s, const char *t) {
    if (!t) emit_null(s);
    else    emit_json_string(s, t);
}

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

static void emit_pos_field(S *s, Ctx *c, FieldState *fs, TriadPos p) {
    emit_field_sep(s, c, fs);
    emit_field_key(s, "pos");
    emit_pos(s, c, p);
}

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

static void emit_str_or_null_array(S *s, Ctx *c, const char *const *items, size_t n) {
    if (n == 0) { s_puts(s, "[]"); return; }
    s_putc(s, '[');
    c->depth++;
    for (size_t k = 0; k < n; ++k) {
        if (k) s_putc(s, ',');
        emit_nl_indent(s, c);
        if (items && items[k]) emit_json_string(s, items[k]);
        else emit_null(s);
    }
    c->depth--;
    emit_nl_indent(s, c);
    s_putc(s, ']');
}

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
        if (items[k].is_expr) {
            emit_node(s, c, items[k].expr);
            s_putc(s, ',');
            emit_nl_indent(s, c);
            emit_str_or_null(s, items[k].fmt_spec);
        } else {
            emit_json_string(s, items[k].text ? items[k].text : "");
        }
        c->depth--;
        emit_nl_indent(s, c);
        s_putc(s, ']');
    }
    c->depth--;
    emit_nl_indent(s, c);
    s_putc(s, ']');
}

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
            emit_null(s);
            emit_field_sep(s, c, &fs); emit_field_key(s, "clauses");
            s_putc(s, '[');
            c->depth++;
            emit_nl_indent(s, c);
            {
                FieldState cfs = {1};
                start_obj(s, c, &cfs, "CompClause");
                emit_field_sep(s, c, &cfs); emit_field_key(s, "var"); emit_str_or_null(s, n->u.list_comp.var);
                emit_field_sep(s, c, &cfs); emit_field_key(s, "iter"); emit_node(s, c, n->u.list_comp.iter);
                emit_field_sep(s, c, &cfs); emit_field_key(s, "conditions");
                if (n->u.list_comp.condition) {
                    TriadAstNode *conds1[1];
                    conds1[0] = n->u.list_comp.condition;
                    emit_node_array(s, c, conds1, 1);
                } else {
                    s_puts(s, "[]");
                }
                end_obj(s, c);
            }
            c->depth--;
            emit_nl_indent(s, c);
            s_putc(s, ']');
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
        case TRIAD_AST_COMPLEX_LIT:
            emit_field_sep(s, c, &fs); emit_field_key(s, "real"); emit_float(s, n->u.complex_lit.real_val);
            emit_field_sep(s, c, &fs); emit_field_key(s, "imag"); emit_float(s, n->u.complex_lit.imag_val);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_BYTES_LIT:
            emit_field_sep(s, c, &fs); emit_field_key(s, "value"); emit_str_or_null(s, n->u.bytes_lit.bytes_val);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_SET_LIT:
            emit_field_sep(s, c, &fs); emit_field_key(s, "elements");
            emit_node_array(s, c, n->u.set_lit.elements, n->u.set_lit.elements_len);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_TERNARY:
            emit_field_sep(s, c, &fs); emit_field_key(s, "condition"); emit_node(s, c, n->u.ternary.condition);
            emit_field_sep(s, c, &fs); emit_field_key(s, "then_val"); emit_node(s, c, n->u.ternary.then_val);
            emit_field_sep(s, c, &fs); emit_field_key(s, "else_val"); emit_node(s, c, n->u.ternary.else_val);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_CHAIN_CMP:
            emit_field_sep(s, c, &fs); emit_field_key(s, "operands");
            emit_node_array(s, c, n->u.chain_cmp.operands, n->u.chain_cmp.operands_len);
            emit_field_sep(s, c, &fs); emit_field_key(s, "ops");
            emit_string_array(s, c, n->u.chain_cmp.ops, n->u.chain_cmp.ops_len);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_SUPER:
            emit_field_sep(s, c, &fs); emit_field_key(s, "args");
            emit_node_array(s, c, n->u.super_expr.args, n->u.super_expr.args_len);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_DICT_COMP:
            emit_field_sep(s, c, &fs); emit_field_key(s, "key_expr"); emit_node(s, c, n->u.dict_comp.key_expr);
            emit_field_sep(s, c, &fs); emit_field_key(s, "value_expr"); emit_node(s, c, n->u.dict_comp.value_expr);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_SET_COMP:
            emit_field_sep(s, c, &fs); emit_field_key(s, "expr"); emit_node(s, c, n->u.set_comp.expr);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_GEN_COMP:
            emit_field_sep(s, c, &fs); emit_field_key(s, "expr"); emit_node(s, c, n->u.gen_comp.expr);
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
            emit_field_sep(s, c, &fs); emit_field_key(s, "names");
            emit_string_array(s, c, n->u.destruct.names, n->u.destruct.names_len);
            emit_field_sep(s, c, &fs); emit_field_key(s, "value"); emit_node(s, c, n->u.destruct.value);
            emit_field_sep(s, c, &fs); emit_field_key(s, "star_idx"); emit_int(s, n->u.destruct.star_idx);
            emit_pos_field(s, c, &fs, n->pos);
            break;
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
        case TRIAD_AST_PASS:
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_ASSERT:
            emit_field_sep(s, c, &fs); emit_field_key(s, "condition");
            emit_node(s, c, n->u.assert_stmt.condition);
            emit_field_sep(s, c, &fs); emit_field_key(s, "message");
            if (n->u.assert_stmt.message) emit_node(s, c, n->u.assert_stmt.message); else emit_null(s);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_DEL:
            emit_field_sep(s, c, &fs); emit_field_key(s, "target");
            emit_node(s, c, n->u.del_stmt.target);
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
        case TRIAD_AST_ASYNC_FOR:
        case TRIAD_AST_FOR:
            emit_field_sep(s, c, &fs); emit_field_key(s, "var"); emit_str_or_null(s, n->u.for_stmt.var);
            emit_field_sep(s, c, &fs); emit_field_key(s, "iter"); emit_node(s, c, n->u.for_stmt.iter);
            emit_field_sep(s, c, &fs); emit_field_key(s, "body");
            emit_node_array(s, c, n->u.for_stmt.body, n->u.for_stmt.body_len);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_ASYNC_WITH:
        case TRIAD_AST_WITH:
            emit_field_sep(s, c, &fs); emit_field_key(s, "expr"); emit_node(s, c, n->u.with_stmt.expr);
            emit_field_sep(s, c, &fs); emit_field_key(s, "var"); emit_str_or_null(s, n->u.with_stmt.var);
            emit_field_sep(s, c, &fs); emit_field_key(s, "body");
            emit_node_array(s, c, n->u.with_stmt.body, n->u.with_stmt.body_len);
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
            emit_field_sep(s, c, &fs); emit_field_key(s, "decorators");
            if (n->u.fn_decl.decorators)
                emit_node_array(s, c, n->u.fn_decl.decorators, n->u.fn_decl.decorators_len);
            else
                emit_null(s);
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
            emit_field_sep(s, c, &fs); emit_field_key(s, "catches");
            if (n->u.try_catch.has_catch) {
                s_putc(s, '[');
                c->depth++;
                emit_nl_indent(s, c);
                {
                    FieldState kfs = {1};
                    start_obj(s, c, &kfs, "CatchBlock");
                    emit_field_sep(s, c, &kfs); emit_field_key(s, "exceptions");
                    emit_string_array(s, c, n->u.try_catch.exceptions, n->u.try_catch.exceptions_len);
                    emit_field_sep(s, c, &kfs); emit_field_key(s, "var");
                    emit_str_or_null(s, n->u.try_catch.catch_var);
                    emit_field_sep(s, c, &kfs); emit_field_key(s, "body");
                    emit_node_array(s, c, n->u.try_catch.catch_body, n->u.try_catch.catch_body_len);
                    end_obj(s, c);
                }
                c->depth--;
                emit_nl_indent(s, c);
                s_putc(s, ']');
            } else {
                emit_null(s);
            }
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
            emit_field_sep(s, c, &fs); emit_field_key(s, "parents");
            emit_string_array(s, c, n->u.type_decl.parents, n->u.type_decl.parents_len);
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
            emit_field_sep(s, c, &fs); emit_field_key(s, "aliases");
            emit_str_or_null_array(s, c, n->u.from_import.aliases, n->u.from_import.names_len);
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
        case TRIAD_AST_SUBSTRATE:
            emit_field_sep(s, c, &fs); emit_field_key(s, "name"); emit_str_or_null(s, n->u.substrate_decl.name);
            emit_field_sep(s, c, &fs); emit_field_key(s, "regime"); emit_str_or_null(s, n->u.substrate_decl.regime);
            emit_field_sep(s, c, &fs); emit_field_key(s, "members");
            emit_string_array(s, c, n->u.substrate_decl.members, n->u.substrate_decl.members_len);
            emit_field_sep(s, c, &fs); emit_field_key(s, "properties");
            emit_str_node_map(s, c, n->u.substrate_decl.properties, n->u.substrate_decl.properties_len);
            emit_field_sep(s, c, &fs); emit_field_key(s, "overrides");
            emit_str_node_map(s, c, n->u.substrate_decl.overrides, n->u.substrate_decl.overrides_len);
            emit_field_sep(s, c, &fs); emit_field_key(s, "is_composed"); emit_bool(s, n->u.substrate_decl.is_composed);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_COUPLE:
            emit_field_sep(s, c, &fs); emit_field_key(s, "src"); emit_str_or_null(s, n->u.couple_stmt.src);
            emit_field_sep(s, c, &fs); emit_field_key(s, "dst"); emit_str_or_null(s, n->u.couple_stmt.dst);
            emit_field_sep(s, c, &fs); emit_field_key(s, "kappa");
            if (n->u.couple_stmt.kappa) emit_node(s, c, n->u.couple_stmt.kappa); else emit_null(s);
            emit_field_sep(s, c, &fs); emit_field_key(s, "duration");
            if (n->u.couple_stmt.duration) emit_node(s, c, n->u.couple_stmt.duration); else emit_null(s);
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
            emit_field_sep(s, c, &fs); emit_field_key(s, "target");
            emit_str_or_null(s, n->u.run_stmt.target);
            emit_pos_field(s, c, &fs, n->pos);
            break;
        case TRIAD_AST_SEQUENCE:
            emit_field_sep(s, c, &fs); emit_field_key(s, "inputs");
            emit_node(s, c, n->u.sequence_stmt.inputs);
            emit_field_sep(s, c, &fs); emit_field_key(s, "target");
            emit_str_or_null(s, n->u.sequence_stmt.target);
            emit_field_sep(s, c, &fs); emit_field_key(s, "each_for");
            if (n->u.sequence_stmt.each_for) emit_node(s, c, n->u.sequence_stmt.each_for); else emit_null(s);
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
