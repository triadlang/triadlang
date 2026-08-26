#include "triad_ir.h"

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
    vsnprintf(big, n + 1, fmt, ap); va_end(ap);
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

typedef struct { int indent; int depth; } Ctx;
static void emit_nl(S *s, Ctx *c) {
    if (c->indent <= 0) return;
    s_putc(s, '\n');
    for (int i = 0; i < c->indent * c->depth; ++i) s_putc(s, ' ');
}

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
static void f_sep(S *s, Ctx *c, FS *fs) { if (!fs->first) s_putc(s, ','); fs->first = 0; emit_nl(s, c); }
static void f_key(S *s, const char *k) { s_putc(s, '"'); s_puts(s, k); s_puts(s, "\": "); }
static void f_start(S *s, Ctx *c, FS *fs, const char *tn) {
    s_putc(s, '{'); c->depth++; fs->first = 1;
    f_sep(s, c, fs); f_key(s, "_type"); emit_string(s, tn);
}
static void f_end(S *s, Ctx *c) { c->depth--; emit_nl(s, c); s_putc(s, '}'); }

static void emit_null(S *s) { s_puts(s, "null"); }
static void emit_int(S *s, long long v) { s_printf(s, "%lld", v); }
static void emit_bool(S *s, int v) { s_puts(s, v ? "true" : "false"); }
static void emit_str_or_null(S *s, const char *t) { if (!t) emit_null(s); else emit_string(s, t); }

static void emit_node(S *s, Ctx *c, const TriadIRNode *n);

static void emit_node_array(S *s, Ctx *c, TriadIRNode *const *items, size_t n) {
    if (n == 0) { s_puts(s, "[]"); return; }
    s_putc(s, '['); c->depth++;
    for (size_t k = 0; k < n; ++k) {
        if (k) s_putc(s, ',');
        emit_nl(s, c);
        emit_node(s, c, items[k]);
    }
    c->depth--; emit_nl(s, c); s_putc(s, ']');
}

static void emit_string_array(S *s, Ctx *c, const char *const *items, size_t n) {
    if (n == 0) { s_puts(s, "[]"); return; }
    s_putc(s, '['); c->depth++;
    for (size_t k = 0; k < n; ++k) {
        if (k) s_putc(s, ',');
        emit_nl(s, c);
        emit_string(s, items[k] ? items[k] : "");
    }
    c->depth--; emit_nl(s, c); s_putc(s, ']');
}

static void emit_string_pair_array(S *s, Ctx *c, const TriadIRTypeField *items, size_t n) {
    if (n == 0) { s_puts(s, "[]"); return; }
    s_putc(s, '['); c->depth++;
    for (size_t k = 0; k < n; ++k) {
        if (k) s_putc(s, ',');
        emit_nl(s, c);
        s_putc(s, '[');
        c->depth++;
        emit_nl(s, c); emit_string(s, items[k].name ? items[k].name : "");
        s_putc(s, ','); emit_nl(s, c); emit_string(s, items[k].type_ann ? items[k].type_ann : "");
        c->depth--; emit_nl(s, c); s_putc(s, ']');
    }
    c->depth--; emit_nl(s, c); s_putc(s, ']');
}

static void emit_map_pairs(S *s, Ctx *c, const TriadIRMapPair *items, size_t n) {
    if (n == 0) { s_puts(s, "[]"); return; }
    s_putc(s, '['); c->depth++;
    for (size_t k = 0; k < n; ++k) {
        if (k) s_putc(s, ',');
        emit_nl(s, c);
        s_putc(s, '[');
        c->depth++;
        emit_nl(s, c); emit_node(s, c, items[k].key);
        s_putc(s, ','); emit_nl(s, c); emit_node(s, c, items[k].value);
        c->depth--; emit_nl(s, c); s_putc(s, ']');
    }
    c->depth--; emit_nl(s, c); s_putc(s, ']');
}

static void emit_fstring_parts(S *s, Ctx *c, const TriadIRFStringPart *items, size_t n) {
    if (n == 0) { s_puts(s, "[]"); return; }
    s_putc(s, '['); c->depth++;
    for (size_t k = 0; k < n; ++k) {
        if (k) s_putc(s, ',');
        emit_nl(s, c);
        s_putc(s, '[');
        c->depth++;
        emit_nl(s, c); emit_string(s, items[k].text ? items[k].text : "");
        s_putc(s, ','); emit_nl(s, c);
        if (items[k].expr) emit_node(s, c, items[k].expr);
        else               emit_null(s);
        c->depth--; emit_nl(s, c); s_putc(s, ']');
    }
    c->depth--; emit_nl(s, c); s_putc(s, ']');
}

static void emit_str_node_map(S *s, Ctx *c, const TriadIRStrEntry *e, size_t n) {
    s_putc(s, '{'); c->depth++;
    for (size_t k = 0; k < n; ++k) {
        if (k) s_putc(s, ',');
        emit_nl(s, c);
        emit_string(s, e[k].key);
        s_puts(s, ": ");
        emit_node(s, c, e[k].value);
    }
    c->depth--;
    if (n) emit_nl(s, c);
    s_putc(s, '}');
}

static void emit_kwargs(S *s, Ctx *c, const TriadIRKwArg *items, size_t n) {
    s_putc(s, '{'); c->depth++;
    for (size_t k = 0; k < n; ++k) {
        if (k) s_putc(s, ',');
        emit_nl(s, c);
        emit_string(s, items[k].name);
        s_puts(s, ": ");
        emit_node(s, c, items[k].value);
    }
    c->depth--;
    if (n) emit_nl(s, c);
    s_putc(s, '}');
}

static void emit_elif_clauses(S *s, Ctx *c, const TriadIRElifClause *items, size_t n) {
    if (n == 0) { s_puts(s, "[]"); return; }
    s_putc(s, '['); c->depth++;
    for (size_t k = 0; k < n; ++k) {
        if (k) s_putc(s, ',');
        emit_nl(s, c);
        s_putc(s, '[');
        c->depth++;
        emit_nl(s, c); emit_node(s, c, items[k].cond);
        s_putc(s, ','); emit_nl(s, c);
        emit_node_array(s, c, items[k].body, items[k].body_len);
        c->depth--; emit_nl(s, c); s_putc(s, ']');
    }
    c->depth--; emit_nl(s, c); s_putc(s, ']');
}

static void emit_node(S *s, Ctx *c, const TriadIRNode *n) {
    if (!n) { emit_null(s); return; }
    FS fs = {1};
    const char *tname = triad_ir_kind_name(n->kind);
    f_start(s, c, &fs, tname);
    switch (n->kind) {
        case TRIAD_IR_INT:
            f_sep(s, c, &fs); f_key(s, "value"); emit_int(s, n->u.int_val);
            break;
        case TRIAD_IR_FLOAT:
            f_sep(s, c, &fs); f_key(s, "value"); emit_float(s, n->u.float_val);
            break;
        case TRIAD_IR_BOOL:
            f_sep(s, c, &fs); f_key(s, "value"); emit_bool(s, n->u.bool_val);
            break;
        case TRIAD_IR_STRING:
            f_sep(s, c, &fs); f_key(s, "value"); emit_string(s, n->u.string_val ? n->u.string_val : "");
            break;
        case TRIAD_IR_NONE:
            break;
        case TRIAD_IR_IDENT:
            f_sep(s, c, &fs); f_key(s, "name"); emit_string(s, n->u.ident_name ? n->u.ident_name : "");
            break;
        case TRIAD_IR_BINOP:
            f_sep(s, c, &fs); f_key(s, "op"); emit_string(s, n->u.binop.op ? n->u.binop.op : "");
            f_sep(s, c, &fs); f_key(s, "left"); emit_node(s, c, n->u.binop.left);
            f_sep(s, c, &fs); f_key(s, "right"); emit_node(s, c, n->u.binop.right);
            break;
        case TRIAD_IR_UNARYOP:
            f_sep(s, c, &fs); f_key(s, "op"); emit_string(s, n->u.unaryop.op ? n->u.unaryop.op : "");
            f_sep(s, c, &fs); f_key(s, "operand"); emit_node(s, c, n->u.unaryop.operand);
            break;
        case TRIAD_IR_CALL:
            f_sep(s, c, &fs); f_key(s, "func"); emit_node(s, c, n->u.call.func_or_obj);
            f_sep(s, c, &fs); f_key(s, "args"); emit_node_array(s, c, n->u.call.args, n->u.call.args_len);
            f_sep(s, c, &fs); f_key(s, "kwargs"); emit_kwargs(s, c, n->u.call.kwargs, n->u.call.kwargs_len);
            break;
        case TRIAD_IR_METHOD_CALL:
            f_sep(s, c, &fs); f_key(s, "obj"); emit_node(s, c, n->u.call.func_or_obj);
            f_sep(s, c, &fs); f_key(s, "method"); emit_string(s, n->u.call.method ? n->u.call.method : "");
            f_sep(s, c, &fs); f_key(s, "args"); emit_node_array(s, c, n->u.call.args, n->u.call.args_len);
            f_sep(s, c, &fs); f_key(s, "kwargs"); emit_kwargs(s, c, n->u.call.kwargs, n->u.call.kwargs_len);
            break;
        case TRIAD_IR_INDEX:
            f_sep(s, c, &fs); f_key(s, "obj"); emit_node(s, c, n->u.index.obj);
            f_sep(s, c, &fs); f_key(s, "index"); emit_node(s, c, n->u.index.index);
            break;
        case TRIAD_IR_SLICE:
            f_sep(s, c, &fs); f_key(s, "obj"); emit_node(s, c, n->u.slice.obj);
            f_sep(s, c, &fs); f_key(s, "start"); if (n->u.slice.start) emit_node(s, c, n->u.slice.start); else emit_null(s);
            f_sep(s, c, &fs); f_key(s, "end");   if (n->u.slice.end)   emit_node(s, c, n->u.slice.end);   else emit_null(s);
            f_sep(s, c, &fs); f_key(s, "step");  if (n->u.slice.step)  emit_node(s, c, n->u.slice.step);  else emit_null(s);
            break;
        case TRIAD_IR_FIELD:
            f_sep(s, c, &fs); f_key(s, "obj"); emit_node(s, c, n->u.field.obj);
            f_sep(s, c, &fs); f_key(s, "field"); emit_string(s, n->u.field.field ? n->u.field.field : "");
            break;
        case TRIAD_IR_LIST:
            f_sep(s, c, &fs); f_key(s, "elements"); emit_node_array(s, c, n->u.list.elements, n->u.list.elements_len);
            break;
        case TRIAD_IR_MAP:
            f_sep(s, c, &fs); f_key(s, "pairs"); emit_map_pairs(s, c, n->u.map.pairs, n->u.map.pairs_len);
            break;
        case TRIAD_IR_FSTRING:
            f_sep(s, c, &fs); f_key(s, "parts"); emit_fstring_parts(s, c, n->u.fstring.parts, n->u.fstring.parts_len);
            break;
        case TRIAD_IR_LIST_COMP:
            f_sep(s, c, &fs); f_key(s, "expr"); emit_node(s, c, n->u.list_comp.expr);
            f_sep(s, c, &fs); f_key(s, "var"); emit_string(s, n->u.list_comp.var ? n->u.list_comp.var : "");
            f_sep(s, c, &fs); f_key(s, "iter"); emit_node(s, c, n->u.list_comp.iter);
            f_sep(s, c, &fs); f_key(s, "condition");
            if (n->u.list_comp.condition) emit_node(s, c, n->u.list_comp.condition); else emit_null(s);
            break;
        case TRIAD_IR_ASSIGN_EXPR:
            f_sep(s, c, &fs); f_key(s, "target"); emit_node(s, c, n->u.assign_expr.target);
            f_sep(s, c, &fs); f_key(s, "value"); emit_node(s, c, n->u.assign_expr.value);
            break;
        case TRIAD_IR_COMPLEX:
            f_sep(s, c, &fs); f_key(s, "real"); emit_float(s, n->u.complex_lit.real_val);
            f_sep(s, c, &fs); f_key(s, "imag"); emit_float(s, n->u.complex_lit.imag_val);
            break;
        case TRIAD_IR_BYTES:
            f_sep(s, c, &fs); f_key(s, "value"); emit_str_or_null(s, n->u.bytes_lit.bytes_val);
            break;
        case TRIAD_IR_TUPLE:
            f_sep(s, c, &fs); f_key(s, "elements");
            emit_node_array(s, c, n->u.tuple_lit.elements, n->u.tuple_lit.elements_len);
            break;
        case TRIAD_IR_SET:
            f_sep(s, c, &fs); f_key(s, "elements");
            emit_node_array(s, c, n->u.set_lit.elements, n->u.set_lit.elements_len);
            break;
        case TRIAD_IR_TERNARY:
            f_sep(s, c, &fs); f_key(s, "condition"); emit_node(s, c, n->u.ternary.condition);
            f_sep(s, c, &fs); f_key(s, "then_val"); emit_node(s, c, n->u.ternary.then_val);
            f_sep(s, c, &fs); f_key(s, "else_val"); emit_node(s, c, n->u.ternary.else_val);
            break;
        case TRIAD_IR_CHAIN_CMP:
            f_sep(s, c, &fs); f_key(s, "operands");
            emit_node_array(s, c, n->u.chain_cmp.operands, n->u.chain_cmp.operands_len);
            f_sep(s, c, &fs); f_key(s, "ops");
            emit_string_array(s, c, n->u.chain_cmp.ops, n->u.chain_cmp.ops_len);
            break;
        case TRIAD_IR_SUPER:
            f_sep(s, c, &fs); f_key(s, "args");
            emit_node_array(s, c, n->u.super_expr.args, n->u.super_expr.args_len);
            break;
        case TRIAD_IR_DICT_COMP:
            f_sep(s, c, &fs); f_key(s, "key_expr"); emit_node(s, c, n->u.dict_comp.key_expr);
            f_sep(s, c, &fs); f_key(s, "value_expr"); emit_node(s, c, n->u.dict_comp.value_expr);
            break;
        case TRIAD_IR_SET_COMP:
            f_sep(s, c, &fs); f_key(s, "expr"); emit_node(s, c, n->u.set_comp.expr);
            break;
        case TRIAD_IR_GEN_COMP:
            f_sep(s, c, &fs); f_key(s, "expr"); emit_node(s, c, n->u.gen_comp.expr);
            break;
        case TRIAD_IR_YIELD_EXPR:
            f_sep(s, c, &fs); f_key(s, "value");
            if (n->u.yield_expr.value) emit_node(s, c, n->u.yield_expr.value); else emit_null(s);
            break;
        case TRIAD_IR_AWAIT:
            f_sep(s, c, &fs); f_key(s, "value"); emit_node(s, c, n->u.await_expr.value);
            break;

        case TRIAD_IR_LET:
            f_sep(s, c, &fs); f_key(s, "name"); emit_string(s, n->u.let_stmt.name ? n->u.let_stmt.name : "");
            f_sep(s, c, &fs); f_key(s, "type_ann"); emit_str_or_null(s, n->u.let_stmt.type_ann);
            f_sep(s, c, &fs); f_key(s, "value");
            if (n->u.let_stmt.value) emit_node(s, c, n->u.let_stmt.value); else emit_null(s);
            break;
        case TRIAD_IR_CONST:
            f_sep(s, c, &fs); f_key(s, "name"); emit_string(s, n->u.const_stmt.name ? n->u.const_stmt.name : "");
            f_sep(s, c, &fs); f_key(s, "value");
            if (n->u.const_stmt.value) emit_node(s, c, n->u.const_stmt.value); else emit_null(s);
            break;
        case TRIAD_IR_ASSIGN:
            f_sep(s, c, &fs); f_key(s, "target"); emit_node(s, c, n->u.assign_stmt.target);
            f_sep(s, c, &fs); f_key(s, "value");  emit_node(s, c, n->u.assign_stmt.value);
            break;
        case TRIAD_IR_EXPR_STMT:
            f_sep(s, c, &fs); f_key(s, "expr"); emit_node(s, c, n->u.expr_stmt.expr);
            break;
        case TRIAD_IR_RETURN:
            f_sep(s, c, &fs); f_key(s, "value");
            if (n->u.ret_or_throw.value) emit_node(s, c, n->u.ret_or_throw.value); else emit_null(s);
            break;
        case TRIAD_IR_PASS:
            break;
        case TRIAD_IR_ASSERT:
            f_sep(s, c, &fs); f_key(s, "condition");
            emit_node(s, c, n->u.assert_stmt.condition);
            f_sep(s, c, &fs); f_key(s, "message");
            if (n->u.assert_stmt.message) emit_node(s, c, n->u.assert_stmt.message); else emit_null(s);
            break;
        case TRIAD_IR_DEL:
            f_sep(s, c, &fs); f_key(s, "target");
            emit_node(s, c, n->u.del_stmt.target);
            break;
        case TRIAD_IR_BREAK:
        case TRIAD_IR_CONTINUE:
            break;
        case TRIAD_IR_IF:
            f_sep(s, c, &fs); f_key(s, "condition"); emit_node(s, c, n->u.if_stmt.condition);
            f_sep(s, c, &fs); f_key(s, "then_body"); emit_node_array(s, c, n->u.if_stmt.then_body, n->u.if_stmt.then_body_len);
            f_sep(s, c, &fs); f_key(s, "elif_clauses"); emit_elif_clauses(s, c, n->u.if_stmt.elif_clauses, n->u.if_stmt.elif_clauses_len);
            f_sep(s, c, &fs); f_key(s, "else_body");
            if (n->u.if_stmt.has_else) emit_node_array(s, c, n->u.if_stmt.else_body, n->u.if_stmt.else_body_len);
            else                       emit_null(s);
            break;
        case TRIAD_IR_ASYNC_FOR:
        case TRIAD_IR_FOR:
            f_sep(s, c, &fs); f_key(s, "var"); emit_string(s, n->u.for_stmt.var ? n->u.for_stmt.var : "");
            f_sep(s, c, &fs); f_key(s, "iter"); emit_node(s, c, n->u.for_stmt.iter);
            f_sep(s, c, &fs); f_key(s, "body"); emit_node_array(s, c, n->u.for_stmt.body, n->u.for_stmt.body_len);
            break;
        case TRIAD_IR_ASYNC_WITH:
        case TRIAD_IR_WITH:
            f_sep(s, c, &fs); f_key(s, "expr"); emit_node(s, c, n->u.with_stmt.expr);
            f_sep(s, c, &fs); f_key(s, "var"); emit_str_or_null(s, n->u.with_stmt.var);
            f_sep(s, c, &fs); f_key(s, "body"); emit_node_array(s, c, n->u.with_stmt.body, n->u.with_stmt.body_len);
            break;
        case TRIAD_IR_WHILE:
            f_sep(s, c, &fs); f_key(s, "condition"); emit_node(s, c, n->u.while_stmt.condition);
            f_sep(s, c, &fs); f_key(s, "body"); emit_node_array(s, c, n->u.while_stmt.body, n->u.while_stmt.body_len);
            break;
        case TRIAD_IR_FUNCTION:
            f_sep(s, c, &fs); f_key(s, "name"); emit_string(s, n->u.func.name ? n->u.func.name : "");
            f_sep(s, c, &fs); f_key(s, "params"); emit_string_array(s, c, n->u.func.params, n->u.func.params_len);
            f_sep(s, c, &fs); f_key(s, "body"); emit_node_array(s, c, n->u.func.body, n->u.func.body_len);
            break;
        case TRIAD_IR_TRY_CATCH:
            f_sep(s, c, &fs); f_key(s, "body"); emit_node_array(s, c, n->u.try_catch.body, n->u.try_catch.body_len);
            f_sep(s, c, &fs); f_key(s, "catch_var"); emit_str_or_null(s, n->u.try_catch.catch_var);
            f_sep(s, c, &fs); f_key(s, "catch_body");
            if (n->u.try_catch.has_catch) emit_node_array(s, c, n->u.try_catch.catch_body, n->u.try_catch.catch_body_len);
            else                          emit_null(s);
            f_sep(s, c, &fs); f_key(s, "finally_body");
            if (n->u.try_catch.has_finally) emit_node_array(s, c, n->u.try_catch.finally_body, n->u.try_catch.finally_body_len);
            else                            emit_null(s);
            break;
        case TRIAD_IR_THROW:
            f_sep(s, c, &fs); f_key(s, "value");
            if (n->u.ret_or_throw.value) emit_node(s, c, n->u.ret_or_throw.value); else emit_null(s);
            break;
        case TRIAD_IR_TYPE_DECL:
            f_sep(s, c, &fs); f_key(s, "name"); emit_string(s, n->u.type_decl.name ? n->u.type_decl.name : "");
            f_sep(s, c, &fs); f_key(s, "fields"); emit_string_pair_array(s, c, n->u.type_decl.fields, n->u.type_decl.fields_len);
            break;
        case TRIAD_IR_ENTITY_DECL:
            f_sep(s, c, &fs); f_key(s, "name"); emit_string(s, n->u.entity_decl.name ? n->u.entity_decl.name : "");
            f_sep(s, c, &fs); f_key(s, "base"); emit_str_or_null(s, n->u.entity_decl.base);
            f_sep(s, c, &fs); f_key(s, "fields"); emit_str_node_map(s, c, n->u.entity_decl.fields, n->u.entity_decl.fields_len);
            break;
        case TRIAD_IR_WORLD_DECL:
            f_sep(s, c, &fs); f_key(s, "name"); emit_string(s, n->u.world_decl.name ? n->u.world_decl.name : "");
            f_sep(s, c, &fs); f_key(s, "fields"); emit_str_node_map(s, c, n->u.world_decl.fields, n->u.world_decl.fields_len);
            f_sep(s, c, &fs); f_key(s, "entities"); emit_node_array(s, c, n->u.world_decl.entities, n->u.world_decl.entities_len);
            break;
        case TRIAD_IR_SUBSTRATE_DECL:
            f_sep(s, c, &fs); f_key(s, "name"); emit_string(s, n->u.substrate_decl.name ? n->u.substrate_decl.name : "");
            f_sep(s, c, &fs); f_key(s, "regime"); emit_str_or_null(s, n->u.substrate_decl.regime);
            f_sep(s, c, &fs); f_key(s, "members"); emit_string_array(s, c, n->u.substrate_decl.members, n->u.substrate_decl.members_len);
            f_sep(s, c, &fs); f_key(s, "properties"); emit_str_node_map(s, c, n->u.substrate_decl.properties, n->u.substrate_decl.properties_len);
            f_sep(s, c, &fs); f_key(s, "overrides"); emit_str_node_map(s, c, n->u.substrate_decl.overrides, n->u.substrate_decl.overrides_len);
            f_sep(s, c, &fs); f_key(s, "is_composed"); emit_bool(s, n->u.substrate_decl.is_composed);
            break;
        case TRIAD_IR_REG_DECL:
            f_sep(s, c, &fs); f_key(s, "name"); emit_string(s, n->u.reg_decl.name ? n->u.reg_decl.name : "");
            f_sep(s, c, &fs); f_key(s, "regime"); emit_str_or_null(s, n->u.reg_decl.regime);
            f_sep(s, c, &fs); f_key(s, "value");
            if (n->u.reg_decl.value) emit_node(s, c, n->u.reg_decl.value); else emit_null(s);
            break;
        case TRIAD_IR_OBSERVE:
            f_sep(s, c, &fs); f_key(s, "target"); emit_string(s, n->u.observe.target ? n->u.observe.target : "");
            f_sep(s, c, &fs); f_key(s, "metrics"); emit_string_array(s, c, n->u.observe.metrics, n->u.observe.metrics_len);
            break;
        case TRIAD_IR_RUN:
            f_sep(s, c, &fs); f_key(s, "duration");
            if (n->u.run.duration) emit_node(s, c, n->u.run.duration); else emit_null(s);
            f_sep(s, c, &fs); f_key(s, "target"); emit_str_or_null(s, n->u.run.target);
            break;
        case TRIAD_IR_SEQUENCE:
            f_sep(s, c, &fs); f_key(s, "inputs");
            emit_node(s, c, n->u.sequence.inputs);
            f_sep(s, c, &fs); f_key(s, "target");
            emit_str_or_null(s, n->u.sequence.target);
            f_sep(s, c, &fs); f_key(s, "each_for");
            if (n->u.sequence.each_for) emit_node(s, c, n->u.sequence.each_for); else emit_null(s);
            break;
        case TRIAD_IR_COUPLE:
            f_sep(s, c, &fs); f_key(s, "src"); emit_str_or_null(s, n->u.couple.src);
            f_sep(s, c, &fs); f_key(s, "dst"); emit_str_or_null(s, n->u.couple.dst);
            f_sep(s, c, &fs); f_key(s, "kappa");
            if (n->u.couple.kappa) emit_node(s, c, n->u.couple.kappa); else emit_null(s);
            f_sep(s, c, &fs); f_key(s, "duration");
            if (n->u.couple.duration) emit_node(s, c, n->u.couple.duration); else emit_null(s);
            break;
        case TRIAD_IR_PAIR:
            f_sep(s, c, &fs); f_key(s, "a"); emit_str_or_null(s, n->u.pair.a);
            f_sep(s, c, &fs); f_key(s, "b"); emit_str_or_null(s, n->u.pair.b);
            f_sep(s, c, &fs); f_key(s, "kappa");
            if (n->u.pair.kappa) emit_node(s, c, n->u.pair.kappa); else emit_null(s);
            f_sep(s, c, &fs); f_key(s, "duration");
            if (n->u.pair.duration) emit_node(s, c, n->u.pair.duration); else emit_null(s);
            break;
        case TRIAD_IR_RING:
            f_sep(s, c, &fs); f_key(s, "members");
            emit_string_array(s, c, n->u.ring.members, n->u.ring.members_len);
            f_sep(s, c, &fs); f_key(s, "kappa");
            if (n->u.ring.kappa) emit_node(s, c, n->u.ring.kappa); else emit_null(s);
            f_sep(s, c, &fs); f_key(s, "duration");
            if (n->u.ring.duration) emit_node(s, c, n->u.ring.duration); else emit_null(s);
            break;
        case TRIAD_IR_ANNOTATION:
            f_sep(s, c, &fs); f_key(s, "key"); emit_str_or_null(s, n->u.annotation.key);
            f_sep(s, c, &fs); f_key(s, "args"); emit_str_or_null(s, n->u.annotation.args);
            break;
        case TRIAD_IR_IMPORT:
            f_sep(s, c, &fs); f_key(s, "path"); emit_string_array(s, c, n->u.import_stmt.path, n->u.import_stmt.path_len);
            f_sep(s, c, &fs); f_key(s, "alias"); emit_str_or_null(s, n->u.import_stmt.alias);
            f_sep(s, c, &fs); f_key(s, "names"); emit_string_array(s, c, n->u.import_stmt.names, n->u.import_stmt.names_len);
            break;
        case TRIAD_IR_DESTRUCT_LET:
            f_sep(s, c, &fs); f_key(s, "names"); emit_string_array(s, c, n->u.destruct_let.names, n->u.destruct_let.names_len);
            f_sep(s, c, &fs); f_key(s, "value"); emit_node(s, c, n->u.destruct_let.value);
            break;
        case TRIAD_IR_CLASS_DECL:
            f_sep(s, c, &fs); f_key(s, "name"); emit_string(s, n->u.class_decl.name ? n->u.class_decl.name : "");
            f_sep(s, c, &fs); f_key(s, "parent"); emit_str_or_null(s, n->u.class_decl.parent);
            f_sep(s, c, &fs); f_key(s, "fields"); emit_string_array(s, c, n->u.class_decl.fields, n->u.class_decl.fields_len);
            f_sep(s, c, &fs); f_key(s, "methods"); emit_node_array(s, c, n->u.class_decl.methods, n->u.class_decl.methods_len);
            break;
        case TRIAD_IR_YIELD:
            f_sep(s, c, &fs); f_key(s, "value");
            if (n->u.ret_or_throw.value) emit_node(s, c, n->u.ret_or_throw.value); else emit_null(s);
            break;
        case TRIAD_IR_MODULE:
            f_sep(s, c, &fs); f_key(s, "name"); emit_string(s, n->u.module.name ? n->u.module.name : "");
            f_sep(s, c, &fs); f_key(s, "imports"); emit_node_array(s, c, n->u.module.imports, n->u.module.imports_len);
            f_sep(s, c, &fs); f_key(s, "body"); emit_node_array(s, c, n->u.module.body, n->u.module.body_len);
            break;
        default:
            break;
    }
    f_end(s, c);
}

char *triad_ir_dump_json(const TriadIRNode *module, int indent) {
    S s = {0};
    Ctx c = { .indent = indent, .depth = 0 };
    emit_node(&s, &c, module);
    if (s.data) return s.data;
    char *fb = (char *)malloc(5);
    if (fb) memcpy(fb, "null", 5);
    return fb;
}

int triad_ir_dump_json_fp(const TriadIRNode *module, FILE *fp, int indent) {
    char *t = triad_ir_dump_json(module, indent);
    if (!t) return -1;
    fputs(t, fp);
    free(t);
    return 0;
}
