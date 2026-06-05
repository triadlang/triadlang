/* triad_format.c — Port of compiler/formatter.py:format_universal.
 *
 * Emits canonically-indented universal-AST source. Output is
 * byte-identical to the Python formatter, validated against every .tri
 * fixture under examples/ via native/c/parity_format.
 *
 * Strategy: a single Buf-based emitter walks the AST, building each
 * statement string then "\n"-joining them. Float and string literals
 * go through repr-mimicking helpers (triad_py_repr_float /
 * triad_py_repr_string) to match CPython byte-for-byte.
 */
#include "triad_format.h"

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ──────────────────────────────────────────────────────────────────
 * Buf: growable byte buffer, freed by caller. Final result is copied
 * into the arena.
 * ────────────────────────────────────────────────────────────────── */

typedef struct {
    char  *data;
    size_t len;
    size_t cap;
} Buf;

static void buf_init(Buf *b) {
    b->data = NULL;
    b->len = b->cap = 0;
}

static void buf_reserve(Buf *b, size_t need) {
    if (b->len + need + 1 > b->cap) {
        size_t nc = b->cap ? b->cap * 2 : 64;
        while (nc < b->len + need + 1) nc *= 2;
        b->data = (char *)realloc(b->data, nc);
        b->cap = nc;
    }
}

static void buf_append(Buf *b, const char *s, size_t n) {
    buf_reserve(b, n);
    memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = '\0';
}

static void buf_puts(Buf *b, const char *s) {
    buf_append(b, s, strlen(s));
}

static void buf_putc(Buf *b, char c) {
    buf_reserve(b, 1);
    b->data[b->len++] = c;
    b->data[b->len] = '\0';
}

static void buf_printf(Buf *b, const char *fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) { va_end(ap2); return; }
    buf_reserve(b, (size_t)n);
    vsnprintf(b->data + b->len, (size_t)n + 1, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)n;
}

static void buf_free(Buf *b) {
    free(b->data);
    b->data = NULL;
    b->len = b->cap = 0;
}

static const char *buf_to_arena(TriadArena *a, Buf *b) {
    char *out = (char *)triad_arena_alloc(a, b->len + 1);
    memcpy(out, b->data, b->len);
    out[b->len] = '\0';
    buf_free(b);
    return out;
}

static const char *pad_str(int indent) {
    static char pads[64][256];
    if (indent < 0) indent = 0;
    if (indent >= 64) indent = 63;
    if (pads[indent][0] == '\0' && indent > 0) {
        for (int i = 0; i < indent * 4; ++i) pads[indent][i] = ' ';
        pads[indent][indent * 4] = '\0';
    }
    return pads[indent];
}

/* ──────────────────────────────────────────────────────────────────
 * Python repr() mimics.
 * ────────────────────────────────────────────────────────────────── */

void triad_py_repr_float(double v, char *out, size_t outlen) {
    if (isnan(v)) { snprintf(out, outlen, "nan"); return; }
    if (isinf(v)) { snprintf(out, outlen, v < 0 ? "-inf" : "inf"); return; }
    if (v == 0.0) {
        snprintf(out, outlen, signbit(v) ? "-0.0" : "0.0");
        return;
    }
    char tmp[64];
    for (int p = 1; p <= 17; ++p) {
        snprintf(tmp, sizeof tmp, "%.*g", p, v);
        if (strtod(tmp, NULL) == v) break;
    }
    char *e = strchr(tmp, 'e');
    char *dot = strchr(tmp, '.');

    double absv = fabs(v);
    int prefer_fixed = (absv >= 1e-4 && absv < 1e16);

    if (e && prefer_fixed) {
        /* Reflow scientific -> fixed using the parsed digits. */
        char sign = (tmp[0] == '-') ? '-' : '+';
        char *p = tmp + (sign == '-' ? 1 : 0);
        char digits[64];
        int  ndig = 0;
        int  dec_pos = 0;
        int  seen_dot = 0;
        while (p < e) {
            if (*p == '.') seen_dot = 1;
            else { digits[ndig++] = *p; if (!seen_dot) dec_pos++; }
            p++;
        }
        int  exp = atoi(e + 1);
        int  new_dec = dec_pos + exp;
        char fixed[96];
        int  fpos = 0;
        if (sign == '-') fixed[fpos++] = '-';
        if (new_dec <= 0) {
            fixed[fpos++] = '0';
            fixed[fpos++] = '.';
            for (int i = 0; i < -new_dec; ++i) fixed[fpos++] = '0';
            for (int i = 0; i < ndig; ++i) fixed[fpos++] = digits[i];
        } else if (new_dec >= ndig) {
            for (int i = 0; i < ndig; ++i) fixed[fpos++] = digits[i];
            for (int i = 0; i < new_dec - ndig; ++i) fixed[fpos++] = '0';
            fixed[fpos++] = '.';
            fixed[fpos++] = '0';
        } else {
            for (int i = 0; i < new_dec; ++i) fixed[fpos++] = digits[i];
            fixed[fpos++] = '.';
            for (int i = new_dec; i < ndig; ++i) fixed[fpos++] = digits[i];
        }
        fixed[fpos] = '\0';
        snprintf(out, outlen, "%s", fixed);
        return;
    }

    if (!e && !dot) {
        snprintf(out, outlen, "%s.0", tmp);
        return;
    }

    if (e) {
        int mlen = (int)(e - tmp);
        char mant[64];
        memcpy(mant, tmp, (size_t)mlen);
        mant[mlen] = '\0';
        int exp = atoi(e + 1);
        char esign = exp < 0 ? '-' : '+';
        int abse = exp < 0 ? -exp : exp;
        if (abse < 10) snprintf(out, outlen, "%se%c0%d", mant, esign, abse);
        else           snprintf(out, outlen, "%se%c%d",  mant, esign, abse);
        return;
    }

    snprintf(out, outlen, "%s", tmp);
}

/* CPython PyUnicode_Repr rules for strings (ASCII subset; non-ASCII
 * bytes are passed through verbatim, matching repr() of an str that
 * contains the same UTF-8 bytes since Python 3 treats them as printable
 * unicode chars in repr by default). */
const char *triad_py_repr_string(TriadArena *arena,
                                 const char *s,
                                 size_t      len) {
    int has_single = 0, has_double = 0;
    for (size_t i = 0; i < len; ++i) {
        if (s[i] == '\'') has_single = 1;
        else if (s[i] == '"') has_double = 1;
    }
    char quote = '\'';
    if (has_single && !has_double) quote = '"';
    /* Python's choice when both are present: use single, escape '. */

    Buf b; buf_init(&b);
    buf_putc(&b, quote);
    for (size_t i = 0; i < len; ++i) {
        unsigned char c = (unsigned char)s[i];
        if (c == '\\') buf_puts(&b, "\\\\");
        else if (c == '\n') buf_puts(&b, "\\n");
        else if (c == '\r') buf_puts(&b, "\\r");
        else if (c == '\t') buf_puts(&b, "\\t");
        else if (c == (unsigned char)quote) {
            buf_putc(&b, '\\');
            buf_putc(&b, (char)c);
        }
        else if (c < 0x20 || c == 0x7f) {
            /* control char: \xNN */
            buf_printf(&b, "\\x%02x", c);
        }
        else {
            buf_putc(&b, (char)c);
        }
    }
    buf_putc(&b, quote);
    return buf_to_arena(arena, &b);
}

/* ──────────────────────────────────────────────────────────────────
 * Forward decls.
 * ────────────────────────────────────────────────────────────────── */

static void emit_expr(Buf *b, TriadArena *a, const TriadAstNode *e);
static void emit_stmt(Buf *b, TriadArena *a, const TriadAstNode *s, int indent);

/* ──────────────────────────────────────────────────────────────────
 * Param.
 * ────────────────────────────────────────────────────────────────── */

static void emit_param(Buf *b, TriadArena *a, const TriadParam *p) {
    if (p->is_kwargs) buf_puts(b, "**");
    else if (p->is_args) buf_putc(b, '*');
    buf_puts(b, p->name);
    if (p->type_ann) {
        buf_puts(b, ": ");
        buf_puts(b, p->type_ann);
    }
    if (p->default_value) {
        buf_putc(b, '=');
        emit_expr(b, a, p->default_value);
    }
}

/* ──────────────────────────────────────────────────────────────────
 * Expressions.
 * ────────────────────────────────────────────────────────────────── */

static void emit_kwargs(Buf *b, TriadArena *a,
                        TriadAstNode *const *args, size_t args_len,
                        const TriadKwArg *kwargs, size_t kwargs_len) {
    int wrote = 0;
    for (size_t i = 0; i < args_len; ++i) {
        if (wrote) buf_puts(b, ", ");
        emit_expr(b, a, args[i]);
        wrote = 1;
    }
    for (size_t i = 0; i < kwargs_len; ++i) {
        if (wrote) buf_puts(b, ", ");
        buf_puts(b, kwargs[i].name);
        buf_putc(b, '=');
        emit_expr(b, a, kwargs[i].value);
        wrote = 1;
    }
}

static void emit_expr(Buf *b, TriadArena *a, const TriadAstNode *e) {
    if (!e) { buf_puts(b, "none"); return; }
    switch (e->kind) {
    case TRIAD_AST_INT_LIT:
        buf_printf(b, "%lld", e->u.int_val);
        return;
    case TRIAD_AST_FLOAT_LIT: {
        char tmp[64];
        triad_py_repr_float(e->u.float_val, tmp, sizeof tmp);
        buf_puts(b, tmp);
        return;
    }
    case TRIAD_AST_BOOL_LIT:
        buf_puts(b, e->u.bool_val ? "true" : "false");
        return;
    case TRIAD_AST_STRING_LIT: {
        const char *s = e->u.string_val ? e->u.string_val : "";
        const char *r = triad_py_repr_string(a, s, strlen(s));
        buf_puts(b, r);
        return;
    }
    case TRIAD_AST_NONE_LIT:
        buf_puts(b, "none");
        return;
    case TRIAD_AST_IDENT:
        buf_puts(b, e->u.ident_name);
        return;
    case TRIAD_AST_BINOP: {
        emit_expr(b, a, e->u.op.left);
        buf_putc(b, ' ');
        buf_puts(b, e->u.op.op);
        buf_putc(b, ' ');
        emit_expr(b, a, e->u.op.right);
        return;
    }
    case TRIAD_AST_UNARYOP: {
        const char *op = e->u.op.op;
        if (strcmp(op, "not") == 0) {
            buf_puts(b, "not ");
            emit_expr(b, a, e->u.op.right);
        } else {
            buf_puts(b, op);
            emit_expr(b, a, e->u.op.right);
        }
        return;
    }
    case TRIAD_AST_CALL: {
        emit_expr(b, a, e->u.call.func_or_obj);
        buf_putc(b, '(');
        emit_kwargs(b, a, e->u.call.args, e->u.call.args_len,
                    e->u.call.kwargs, e->u.call.kwargs_len);
        buf_putc(b, ')');
        return;
    }
    case TRIAD_AST_METHOD_CALL: {
        emit_expr(b, a, e->u.call.func_or_obj);
        buf_putc(b, '.');
        const char *m = e->u.call.method;
        if (m && strcmp(m, "append") == 0) m = "push";
        buf_puts(b, m);
        buf_putc(b, '(');
        emit_kwargs(b, a, e->u.call.args, e->u.call.args_len,
                    e->u.call.kwargs, e->u.call.kwargs_len);
        buf_putc(b, ')');
        return;
    }
    case TRIAD_AST_INDEX: {
        emit_expr(b, a, e->u.index.obj);
        buf_putc(b, '[');
        const TriadAstNode *idx = e->u.index.index;
        if (idx && idx->kind == TRIAD_AST_SLICE) {
            if (idx->u.slice.start) emit_expr(b, a, idx->u.slice.start);
            buf_putc(b, ':');
            if (idx->u.slice.end) emit_expr(b, a, idx->u.slice.end);
            if (idx->u.slice.step) {
                buf_putc(b, ':');
                emit_expr(b, a, idx->u.slice.step);
            }
        } else {
            emit_expr(b, a, idx);
        }
        buf_putc(b, ']');
        return;
    }
    case TRIAD_AST_FIELD: {
        emit_expr(b, a, e->u.field.obj);
        buf_putc(b, '.');
        buf_puts(b, e->u.field.field);
        return;
    }
    case TRIAD_AST_LIST: {
        buf_putc(b, '[');
        for (size_t i = 0; i < e->u.list.elements_len; ++i) {
            if (i) buf_puts(b, ", ");
            emit_expr(b, a, e->u.list.elements[i]);
        }
        buf_putc(b, ']');
        return;
    }
    case TRIAD_AST_TUPLE: {
        size_t n = e->u.list.elements_len;
        if (n == 0) { buf_puts(b, "()"); return; }
        buf_putc(b, '(');
        for (size_t i = 0; i < n; ++i) {
            if (i) buf_puts(b, ", ");
            emit_expr(b, a, e->u.list.elements[i]);
        }
        if (n == 1) buf_putc(b, ',');
        buf_putc(b, ')');
        return;
    }
    case TRIAD_AST_LIST_COMP: {
        buf_putc(b, '[');
        emit_expr(b, a, e->u.list_comp.expr);
        buf_puts(b, " for ");
        buf_puts(b, e->u.list_comp.var);
        buf_puts(b, " in ");
        emit_expr(b, a, e->u.list_comp.iter);
        if (e->u.list_comp.condition) {
            buf_puts(b, " if ");
            emit_expr(b, a, e->u.list_comp.condition);
        }
        buf_putc(b, ']');
        return;
    }
    case TRIAD_AST_MAP: {
        buf_putc(b, '{');
        for (size_t i = 0; i < e->u.map.pairs_len; ++i) {
            if (i) buf_puts(b, ", ");
            emit_expr(b, a, e->u.map.pairs[i].key);
            buf_puts(b, ": ");
            emit_expr(b, a, e->u.map.pairs[i].value);
        }
        buf_putc(b, '}');
        return;
    }
    case TRIAD_AST_FSTRING: {
        /* f"..." with {{ and }} doubling on literal '{'/'}', and
         * {expr} for embedded expressions. */
        buf_puts(b, "f\"");
        for (size_t i = 0; i < e->u.fstring.parts_len; ++i) {
            const TriadFStringAstPart *p = &e->u.fstring.parts[i];
            if (p->is_expr) {
                buf_putc(b, '{');
                emit_expr(b, a, p->expr);
                buf_putc(b, '}');
            } else {
                for (const char *q = p->text; *q; ++q) {
                    if (*q == '{') buf_puts(b, "{{");
                    else if (*q == '}') buf_puts(b, "}}");
                    else buf_putc(b, *q);
                }
            }
        }
        buf_putc(b, '"');
        return;
    }
    case TRIAD_AST_LAMBDA: {
        buf_puts(b, "fn(");
        for (size_t i = 0; i < e->u.lambda.params_len; ++i) {
            if (i) buf_puts(b, ", ");
            emit_param(b, a, &e->u.lambda.params[i]);
        }
        buf_puts(b, ") {");
        if (e->u.lambda.body_len > 0) {
            buf_putc(b, ' ');
            for (size_t i = 0; i < e->u.lambda.body_len; ++i) {
                if (i) buf_puts(b, "; ");
                /* Lambda body is rendered inline (no indent / no newline);
                 * emit_stmt would add a trailing pad. Use a temporary
                 * sub-buffer to drop the leading pad. */
                Buf sub; buf_init(&sub);
                emit_stmt(&sub, a, e->u.lambda.body[i], 0);
                /* Drop the trailing ';' to match Python's "; "-join */
                if (sub.len > 0 && sub.data[sub.len - 1] == ';') sub.len--;
                if (sub.len) buf_append(b, sub.data, sub.len);
                buf_free(&sub);
            }
            buf_putc(b, ' ');
        } else {
            buf_putc(b, ' ');
        }
        buf_putc(b, '}');
        return;
    }
    case TRIAD_AST_ASSIGN_EXPR: {
        buf_putc(b, '(');
        emit_expr(b, a, e->u.assign_expr.target);
        buf_puts(b, " := ");
        emit_expr(b, a, e->u.assign_expr.value);
        buf_putc(b, ')');
        return;
    }
    case TRIAD_AST_YIELD_EXPR: {
        if (e->u.unary_value.value) {
            buf_puts(b, "(yield ");
            emit_expr(b, a, e->u.unary_value.value);
            buf_putc(b, ')');
        } else {
            buf_puts(b, "(yield)");
        }
        return;
    }
    case TRIAD_AST_AWAIT_EXPR: {
        buf_puts(b, "(await ");
        emit_expr(b, a, e->u.unary_value.value);
        buf_putc(b, ')');
        return;
    }
    default:
        buf_printf(b, "<expr:%s>", triad_ast_kind_name(e->kind));
        return;
    }
}

/* ──────────────────────────────────────────────────────────────────
 * Statements.
 * ────────────────────────────────────────────────────────────────── */

static void emit_body(Buf *b, TriadArena *a,
                      TriadAstNode *const *body, size_t n, int indent) {
    for (size_t i = 0; i < n; ++i) {
        buf_putc(b, '\n');
        emit_stmt(b, a, body[i], indent);
    }
}

static void emit_stmt(Buf *b, TriadArena *a,
                      const TriadAstNode *s, int indent) {
    const char *pad = pad_str(indent);
    switch (s->kind) {
    case TRIAD_AST_LET: {
        buf_puts(b, pad);
        buf_puts(b, "let ");
        buf_puts(b, s->u.let_stmt.name);
        if (s->u.let_stmt.type_ann) {
            buf_puts(b, ": ");
            buf_puts(b, s->u.let_stmt.type_ann);
        }
        if (s->u.let_stmt.value) {
            buf_puts(b, " = ");
            emit_expr(b, a, s->u.let_stmt.value);
        }
        buf_putc(b, ';');
        return;
    }
    case TRIAD_AST_CONST: {
        buf_puts(b, pad);
        buf_puts(b, "const ");
        buf_puts(b, s->u.const_stmt.name);
        buf_puts(b, " = ");
        emit_expr(b, a, s->u.const_stmt.value);
        buf_putc(b, ';');
        return;
    }
    case TRIAD_AST_DESTRUCT_LET: {
        buf_puts(b, pad);
        buf_puts(b, "let (");
        for (size_t i = 0; i < s->u.destruct.names_len; ++i) {
            if (i) buf_puts(b, ", ");
            buf_puts(b, s->u.destruct.names[i]);
        }
        buf_puts(b, ") = ");
        emit_expr(b, a, s->u.destruct.value);
        buf_putc(b, ';');
        return;
    }
    case TRIAD_AST_MAP_DESTRUCT: {
        buf_puts(b, pad);
        buf_puts(b, "let {");
        for (size_t i = 0; i < s->u.destruct.names_len; ++i) {
            if (i) buf_puts(b, ", ");
            buf_puts(b, s->u.destruct.names[i]);
        }
        buf_puts(b, "} = ");
        emit_expr(b, a, s->u.destruct.value);
        buf_putc(b, ';');
        return;
    }
    case TRIAD_AST_ASSIGN: {
        buf_puts(b, pad);
        emit_expr(b, a, s->u.assign_stmt.target);
        buf_puts(b, " = ");
        emit_expr(b, a, s->u.assign_stmt.value);
        buf_putc(b, ';');
        return;
    }
    case TRIAD_AST_EXPR_STMT: {
        buf_puts(b, pad);
        emit_expr(b, a, s->u.expr_stmt.expr);
        buf_putc(b, ';');
        return;
    }
    case TRIAD_AST_RETURN: {
        buf_puts(b, pad);
        if (s->u.unary_value.value) {
            buf_puts(b, "return ");
            emit_expr(b, a, s->u.unary_value.value);
        } else {
            buf_puts(b, "return");
        }
        buf_putc(b, ';');
        return;
    }
    case TRIAD_AST_BREAK:    buf_puts(b, pad); buf_puts(b, "break;"); return;
    case TRIAD_AST_CONTINUE: buf_puts(b, pad); buf_puts(b, "continue;"); return;
    case TRIAD_AST_IF: {
        buf_puts(b, pad);
        buf_puts(b, "if ");
        emit_expr(b, a, s->u.if_stmt.condition);
        buf_puts(b, " {");
        emit_body(b, a, s->u.if_stmt.then_body, s->u.if_stmt.then_body_len,
                  indent + 1);
        for (size_t i = 0; i < s->u.if_stmt.elif_clauses_len; ++i) {
            buf_putc(b, '\n');
            buf_puts(b, pad);
            buf_puts(b, "} elif ");
            emit_expr(b, a, s->u.if_stmt.elif_clauses[i].cond);
            buf_puts(b, " {");
            emit_body(b, a, s->u.if_stmt.elif_clauses[i].body,
                      s->u.if_stmt.elif_clauses[i].body_len, indent + 1);
        }
        if (s->u.if_stmt.has_else) {
            buf_putc(b, '\n');
            buf_puts(b, pad);
            buf_puts(b, "} else {");
            emit_body(b, a, s->u.if_stmt.else_body,
                      s->u.if_stmt.else_body_len, indent + 1);
        }
        buf_putc(b, '\n');
        buf_puts(b, pad);
        buf_putc(b, '}');
        return;
    }
    case TRIAD_AST_FOR: {
        buf_puts(b, pad);
        buf_puts(b, "for ");
        buf_puts(b, s->u.for_stmt.var);
        buf_puts(b, " in ");
        emit_expr(b, a, s->u.for_stmt.iter);
        buf_puts(b, " {");
        emit_body(b, a, s->u.for_stmt.body, s->u.for_stmt.body_len, indent + 1);
        buf_putc(b, '\n');
        buf_puts(b, pad);
        buf_putc(b, '}');
        return;
    }
    case TRIAD_AST_WHILE: {
        buf_puts(b, pad);
        buf_puts(b, "while ");
        emit_expr(b, a, s->u.while_stmt.condition);
        buf_puts(b, " {");
        emit_body(b, a, s->u.while_stmt.body, s->u.while_stmt.body_len,
                  indent + 1);
        buf_putc(b, '\n');
        buf_puts(b, pad);
        buf_putc(b, '}');
        return;
    }
    case TRIAD_AST_FN_DECL: {
        buf_puts(b, pad);
        if (s->u.fn_decl.is_async) buf_puts(b, "async ");
        buf_puts(b, "fn ");
        buf_puts(b, s->u.fn_decl.name);
        buf_putc(b, '(');
        for (size_t i = 0; i < s->u.fn_decl.params_len; ++i) {
            if (i) buf_puts(b, ", ");
            emit_param(b, a, &s->u.fn_decl.params[i]);
        }
        buf_puts(b, ") {");
        emit_body(b, a, s->u.fn_decl.body, s->u.fn_decl.body_len, indent + 1);
        buf_putc(b, '\n');
        buf_puts(b, pad);
        buf_putc(b, '}');
        return;
    }
    case TRIAD_AST_TYPE_DECL: {
        buf_puts(b, pad);
        buf_puts(b, "type ");
        buf_puts(b, s->u.type_decl.name);
        buf_puts(b, " {");
        const char *inner = pad_str(indent + 1);
        for (size_t i = 0; i < s->u.type_decl.fields_len; ++i) {
            const TriadTypeField *f = &s->u.type_decl.fields[i];
            buf_putc(b, '\n');
            buf_puts(b, inner);
            buf_puts(b, f->name);
            if (f->type_ann) { buf_puts(b, ": "); buf_puts(b, f->type_ann); }
            if (f->default_value) {
                buf_puts(b, " = ");
                emit_expr(b, a, f->default_value);
            }
        }
        for (size_t i = 0; i < s->u.type_decl.methods_len; ++i) {
            const TriadAstNode *m = s->u.type_decl.methods[i];
            buf_putc(b, '\n');
            buf_puts(b, inner);
            buf_puts(b, "fn ");
            buf_puts(b, m->u.fn_decl.name);
            buf_putc(b, '(');
            for (size_t j = 0; j < m->u.fn_decl.params_len; ++j) {
                if (j) buf_puts(b, ", ");
                emit_param(b, a, &m->u.fn_decl.params[j]);
            }
            buf_puts(b, ") {");
            emit_body(b, a, m->u.fn_decl.body, m->u.fn_decl.body_len,
                      indent + 2);
            buf_putc(b, '\n');
            buf_puts(b, inner);
            buf_putc(b, '}');
        }
        buf_putc(b, '\n');
        buf_puts(b, pad);
        buf_putc(b, '}');
        return;
    }
    case TRIAD_AST_CLASS_DECL: {
        buf_puts(b, pad);
        buf_puts(b, "class ");
        buf_puts(b, s->u.type_decl.name);
        if (s->u.type_decl.parent) {
            buf_puts(b, " : ");
            buf_puts(b, s->u.type_decl.parent);
        }
        buf_puts(b, " {");
        const char *inner = pad_str(indent + 1);
        for (size_t i = 0; i < s->u.type_decl.fields_len; ++i) {
            const TriadTypeField *f = &s->u.type_decl.fields[i];
            buf_putc(b, '\n');
            buf_puts(b, inner);
            buf_puts(b, f->name);
            if (f->type_ann) { buf_puts(b, ": "); buf_puts(b, f->type_ann); }
            if (f->default_value) {
                buf_puts(b, " = ");
                emit_expr(b, a, f->default_value);
            }
        }
        for (size_t i = 0; i < s->u.type_decl.methods_len; ++i) {
            const TriadAstNode *m = s->u.type_decl.methods[i];
            buf_putc(b, '\n');
            buf_puts(b, inner);
            buf_puts(b, "fn ");
            buf_puts(b, m->u.fn_decl.name);
            buf_putc(b, '(');
            for (size_t j = 0; j < m->u.fn_decl.params_len; ++j) {
                if (j) buf_puts(b, ", ");
                emit_param(b, a, &m->u.fn_decl.params[j]);
            }
            buf_puts(b, ") {");
            emit_body(b, a, m->u.fn_decl.body, m->u.fn_decl.body_len,
                      indent + 2);
            buf_putc(b, '\n');
            buf_puts(b, inner);
            buf_putc(b, '}');
        }
        buf_putc(b, '\n');
        buf_puts(b, pad);
        buf_putc(b, '}');
        return;
    }
    case TRIAD_AST_IMPORT: {
        buf_puts(b, pad);
        buf_puts(b, "import ");
        for (size_t i = 0; i < s->u.import_stmt.path_len; ++i) {
            if (i) buf_putc(b, '.');
            buf_puts(b, s->u.import_stmt.path[i]);
        }
        if (s->u.import_stmt.alias) {
            buf_puts(b, " as ");
            buf_puts(b, s->u.import_stmt.alias);
        }
        buf_putc(b, ';');
        return;
    }
    case TRIAD_AST_FROM_IMPORT: {
        buf_puts(b, pad);
        buf_puts(b, "from ");
        for (size_t i = 0; i < s->u.from_import.path_len; ++i) {
            if (i) buf_putc(b, '.');
            buf_puts(b, s->u.from_import.path[i]);
        }
        buf_puts(b, " import ");
        for (size_t i = 0; i < s->u.from_import.names_len; ++i) {
            if (i) buf_puts(b, ", ");
            buf_puts(b, s->u.from_import.names[i]);
        }
        buf_putc(b, ';');
        return;
    }
    case TRIAD_AST_TRY_CATCH: {
        buf_puts(b, pad);
        buf_puts(b, "try {");
        emit_body(b, a, s->u.try_catch.body, s->u.try_catch.body_len,
                  indent + 1);
        if (s->u.try_catch.catch_body_len > 0 || s->u.try_catch.catch_var) {
            buf_putc(b, '\n');
            buf_puts(b, pad);
            buf_puts(b, "} catch");
            if (s->u.try_catch.catch_var) {
                buf_putc(b, ' ');
                buf_puts(b, s->u.try_catch.catch_var);
            }
            buf_puts(b, " {");
            emit_body(b, a, s->u.try_catch.catch_body,
                      s->u.try_catch.catch_body_len, indent + 1);
        }
        if (s->u.try_catch.finally_body_len > 0) {
            buf_putc(b, '\n');
            buf_puts(b, pad);
            buf_puts(b, "} finally {");
            emit_body(b, a, s->u.try_catch.finally_body,
                      s->u.try_catch.finally_body_len, indent + 1);
        }
        buf_putc(b, '\n');
        buf_puts(b, pad);
        buf_putc(b, '}');
        return;
    }
    case TRIAD_AST_THROW: {
        buf_puts(b, pad);
        if (s->u.unary_value.value) {
            buf_puts(b, "throw ");
            emit_expr(b, a, s->u.unary_value.value);
        } else {
            buf_puts(b, "throw");
        }
        buf_putc(b, ';');
        return;
    }
    case TRIAD_AST_YIELD_STMT: {
        buf_puts(b, pad);
        if (s->u.unary_value.value) {
            buf_puts(b, "yield ");
            emit_expr(b, a, s->u.unary_value.value);
        } else {
            buf_puts(b, "yield");
        }
        buf_putc(b, ';');
        return;
    }
    case TRIAD_AST_REG: {
        buf_puts(b, pad);
        buf_puts(b, "reg ");
        buf_puts(b, s->u.reg_stmt.name);
        if (s->u.reg_stmt.regime) {
            buf_puts(b, " : ");
            buf_puts(b, s->u.reg_stmt.regime);
        }
        if (s->u.reg_stmt.value) {
            buf_puts(b, " = ");
            emit_expr(b, a, s->u.reg_stmt.value);
        }
        buf_putc(b, ';');
        return;
    }
    case TRIAD_AST_ENTITY: {
        buf_puts(b, pad);
        buf_puts(b, "entity ");
        buf_puts(b, s->u.entity_decl.name);
        if (s->u.entity_decl.base) {
            buf_putc(b, '(');
            buf_puts(b, s->u.entity_decl.base);
            buf_putc(b, ')');
        }
        buf_puts(b, " { ");
        for (size_t i = 0; i < s->u.entity_decl.fields_len; ++i) {
            if (i) buf_puts(b, ", ");
            buf_puts(b, s->u.entity_decl.fields[i].key);
            buf_putc(b, '=');
            emit_expr(b, a, s->u.entity_decl.fields[i].value);
        }
        buf_puts(b, " }");
        return;
    }
    case TRIAD_AST_WORLD: {
        buf_puts(b, pad);
        buf_puts(b, "world ");
        buf_puts(b, s->u.world_decl.name);
        buf_puts(b, " { ");
        for (size_t i = 0; i < s->u.world_decl.fields_len; ++i) {
            if (i) buf_puts(b, ", ");
            buf_puts(b, s->u.world_decl.fields[i].key);
            buf_putc(b, '=');
            emit_expr(b, a, s->u.world_decl.fields[i].value);
        }
        buf_puts(b, " }");
        return;
    }
    case TRIAD_AST_COUPLE: {
        buf_puts(b, pad);
        buf_puts(b, "couple(");
        buf_puts(b, s->u.couple_stmt.src);
        buf_puts(b, ", ");
        buf_puts(b, s->u.couple_stmt.dst);
        buf_putc(b, ')');
        if (s->u.couple_stmt.kappa) {
            buf_puts(b, " kappa=");
            emit_expr(b, a, s->u.couple_stmt.kappa);
        }
        buf_putc(b, ';');
        return;
    }
    case TRIAD_AST_PAIR: {
        buf_puts(b, pad);
        buf_puts(b, "pair(");
        buf_puts(b, s->u.pair_stmt.a);
        buf_puts(b, ", ");
        buf_puts(b, s->u.pair_stmt.b);
        buf_putc(b, ')');
        if (s->u.pair_stmt.kappa) {
            buf_puts(b, " kappa=");
            emit_expr(b, a, s->u.pair_stmt.kappa);
        }
        if (s->u.pair_stmt.duration) {
            buf_puts(b, " for T=");
            emit_expr(b, a, s->u.pair_stmt.duration);
        }
        buf_putc(b, ';');
        return;
    }
    case TRIAD_AST_RING: {
        buf_puts(b, pad);
        buf_puts(b, "ring(");
        for (size_t i = 0; i < s->u.ring_stmt.members_len; ++i) {
            if (i) buf_puts(b, ", ");
            buf_puts(b, s->u.ring_stmt.members[i]);
        }
        buf_putc(b, ')');
        if (s->u.ring_stmt.kappa) {
            buf_puts(b, " kappa=");
            emit_expr(b, a, s->u.ring_stmt.kappa);
        }
        if (s->u.ring_stmt.duration) {
            buf_puts(b, " for T=");
            emit_expr(b, a, s->u.ring_stmt.duration);
        }
        buf_putc(b, ';');
        return;
    }
    case TRIAD_AST_OBSERVE: {
        buf_puts(b, pad);
        buf_puts(b, "OBSERVE ");
        buf_puts(b, s->u.observe_stmt.target);
        buf_putc(b, ' ');
        for (size_t i = 0; i < s->u.observe_stmt.metrics_len; ++i) {
            if (i) buf_puts(b, ", ");
            buf_puts(b, s->u.observe_stmt.metrics[i]);
        }
        buf_putc(b, ';');
        return;
    }
    case TRIAD_AST_RUN: {
        buf_puts(b, pad);
        buf_puts(b, "run");
        if (s->u.run_stmt.duration) {
            buf_putc(b, ' ');
            emit_expr(b, a, s->u.run_stmt.duration);
        }
        buf_putc(b, ';');
        return;
    }
    case TRIAD_AST_ANNOTATION: {
        buf_puts(b, pad);
        buf_putc(b, '@');
        buf_puts(b, s->u.annotation_stmt.key);
        buf_putc(b, '(');
        buf_puts(b, s->u.annotation_stmt.args);
        buf_putc(b, ')');
        return;
    }
    case TRIAD_AST_MATCH: {
        buf_puts(b, pad);
        buf_puts(b, "match ");
        emit_expr(b, a, s->u.match_stmt.subject);
        buf_puts(b, " {");
        const char *inner = pad_str(indent + 1);
        for (size_t i = 0; i < s->u.match_stmt.cases_len; ++i) {
            const TriadMatchCase *c = &s->u.match_stmt.cases[i];
            buf_putc(b, '\n');
            buf_puts(b, inner);
            buf_puts(b, "case ");
            emit_expr(b, a, c->pattern);
            if (c->guard) {
                buf_puts(b, " if ");
                emit_expr(b, a, c->guard);
            }
            buf_puts(b, " => {");
            emit_body(b, a, c->body, c->body_len, indent + 2);
            buf_putc(b, '\n');
            buf_puts(b, inner);
            buf_putc(b, '}');
        }
        if (s->u.match_stmt.has_else) {
            buf_putc(b, '\n');
            buf_puts(b, inner);
            buf_puts(b, "else => {");
            emit_body(b, a, s->u.match_stmt.else_body,
                      s->u.match_stmt.else_body_len, indent + 2);
            buf_putc(b, '\n');
            buf_puts(b, inner);
            buf_putc(b, '}');
        }
        buf_putc(b, '\n');
        buf_puts(b, pad);
        buf_putc(b, '}');
        return;
    }
    default:
        buf_puts(b, pad);
        buf_printf(b, "/* unhandled: %s */", triad_ast_kind_name(s->kind));
        return;
    }
}

/* ──────────────────────────────────────────────────────────────────
 * Public entry point.
 * ────────────────────────────────────────────────────────────────── */

const char *triad_format_module(TriadArena         *arena,
                                const TriadAstNode *module) {
    if (!arena || !module || module->kind != TRIAD_AST_MODULE) return NULL;
    Buf b; buf_init(&b);
    for (size_t i = 0; i < module->u.module.body_len; ++i) {
        if (i) buf_putc(&b, '\n');
        emit_stmt(&b, arena, module->u.module.body[i], 0);
    }
    buf_putc(&b, '\n');
    return buf_to_arena(arena, &b);
}
