#include "triad_ir.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    void  **items;
    size_t  len;
    size_t  cap;
} PV;

static void pv_push(PV *v, void *p) {
    if (v->len == v->cap) {
        size_t nc = v->cap ? v->cap * 2 : 8;
        v->items = (void **)realloc(v->items, nc * sizeof(void *));
        v->cap = nc;
    }
    v->items[v->len++] = p;
}
static void pv_free(PV *v) { free(v->items); v->items = NULL; v->len = v->cap = 0; }

static TriadIRNode *ir_new(TriadArena *a, TriadIRKind k) {
    TriadIRNode *n = (TriadIRNode *)triad_arena_calloc(a, sizeof(TriadIRNode));
    n->kind = k;
    return n;
}

static TriadIRNode *lower_stmt(TriadArena *a, const TriadAstNode *s);
static TriadIRNode *lower_expr(TriadArena *a, const TriadAstNode *e);
static TriadIRNode *lower_match(TriadArena *a, const TriadAstNode *s);
static TriadIRNode *lower_class(TriadArena *a, const TriadAstNode *s);
static TriadIRNode *match_condition(TriadArena *a, TriadIRNode *subject,
                                    const TriadAstNode *pattern);

static TriadIRNode **lower_stmt_list(TriadArena *a, TriadAstNode *const *items,
                                     size_t n, size_t *out_len) {
    if (n == 0) { *out_len = 0; return NULL; }
    TriadIRNode **arr = (TriadIRNode **)triad_arena_alloc(a, n * sizeof(TriadIRNode *));
    for (size_t k = 0; k < n; ++k) arr[k] = lower_stmt(a, items[k]);
    *out_len = n;
    return arr;
}

static TriadIRNode **lower_expr_list(TriadArena *a, TriadAstNode *const *items,
                                     size_t n, size_t *out_len) {
    if (n == 0) { *out_len = 0; return NULL; }
    TriadIRNode **arr = (TriadIRNode **)triad_arena_alloc(a, n * sizeof(TriadIRNode *));
    for (size_t k = 0; k < n; ++k) arr[k] = lower_expr(a, items[k]);
    *out_len = n;
    return arr;
}

static TriadIRNode *lower_expr(TriadArena *a, const TriadAstNode *e) {
    if (!e) return ir_new(a, TRIAD_IR_NONE);
    switch (e->kind) {
        case TRIAD_AST_INT_LIT: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_INT);
            n->u.int_val = e->u.int_val;
            return n;
        }
        case TRIAD_AST_FLOAT_LIT: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_FLOAT);
            n->u.float_val = e->u.float_val;
            return n;
        }
        case TRIAD_AST_BOOL_LIT: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_BOOL);
            n->u.bool_val = e->u.bool_val;
            return n;
        }
        case TRIAD_AST_STRING_LIT: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_STRING);
            n->u.string_val = e->u.string_val ? e->u.string_val : "";
            return n;
        }
        case TRIAD_AST_NONE_LIT:
            return ir_new(a, TRIAD_IR_NONE);
        case TRIAD_AST_IDENT: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_IDENT);
            n->u.ident_name = e->u.ident_name;
            return n;
        }
        case TRIAD_AST_BINOP: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_BINOP);
            n->u.binop.op    = e->u.op.op;
            n->u.binop.left  = lower_expr(a, e->u.op.left);
            n->u.binop.right = lower_expr(a, e->u.op.right);
            return n;
        }
        case TRIAD_AST_UNARYOP: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_UNARYOP);
            n->u.unaryop.op      = e->u.op.op;
            n->u.unaryop.operand = lower_expr(a, e->u.op.right);
            return n;
        }
        case TRIAD_AST_CALL: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_CALL);
            n->u.call.func_or_obj = lower_expr(a, e->u.call.func_or_obj);
            n->u.call.args        = lower_expr_list(a, e->u.call.args, e->u.call.args_len, &n->u.call.args_len);
            if (e->u.call.kwargs_len) {
                TriadIRKwArg *kw = (TriadIRKwArg *)triad_arena_alloc(a, e->u.call.kwargs_len * sizeof(TriadIRKwArg));
                for (size_t k = 0; k < e->u.call.kwargs_len; ++k) {
                    kw[k].name  = e->u.call.kwargs[k].name;
                    kw[k].value = lower_expr(a, e->u.call.kwargs[k].value);
                }
                n->u.call.kwargs = kw;
                n->u.call.kwargs_len = e->u.call.kwargs_len;
            }
            return n;
        }
        case TRIAD_AST_INDEX: {
            const TriadAstNode *idx = e->u.index.index;
            if (idx && idx->kind == TRIAD_AST_SLICE) {
                TriadIRNode *n = ir_new(a, TRIAD_IR_SLICE);
                n->u.slice.obj   = lower_expr(a, e->u.index.obj);
                n->u.slice.start = idx->u.slice.start ? lower_expr(a, idx->u.slice.start) : NULL;
                n->u.slice.end   = idx->u.slice.end   ? lower_expr(a, idx->u.slice.end)   : NULL;
                n->u.slice.step  = idx->u.slice.step  ? lower_expr(a, idx->u.slice.step)  : NULL;
                return n;
            }
            TriadIRNode *n = ir_new(a, TRIAD_IR_INDEX);
            n->u.index.obj   = lower_expr(a, e->u.index.obj);
            n->u.index.index = lower_expr(a, idx);
            return n;
        }
        case TRIAD_AST_FIELD: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_FIELD);
            n->u.field.obj   = lower_expr(a, e->u.field.obj);
            n->u.field.field = e->u.field.field;
            return n;
        }
        case TRIAD_AST_LIST: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_LIST);
            n->u.list.elements = lower_expr_list(a, e->u.list.elements, e->u.list.elements_len, &n->u.list.elements_len);
            return n;
        }
        case TRIAD_AST_MAP: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_MAP);
            if (e->u.map.pairs_len) {
                TriadIRMapPair *arr = (TriadIRMapPair *)triad_arena_alloc(a, e->u.map.pairs_len * sizeof(TriadIRMapPair));
                for (size_t k = 0; k < e->u.map.pairs_len; ++k) {
                    arr[k].key   = lower_expr(a, e->u.map.pairs[k].key);
                    arr[k].value = lower_expr(a, e->u.map.pairs[k].value);
                }
                n->u.map.pairs = arr;
                n->u.map.pairs_len = e->u.map.pairs_len;
            }
            return n;
        }
        case TRIAD_AST_LAMBDA: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_FUNCTION);
            n->u.func.name = "";
            if (e->u.lambda.params_len) {
                const char **ps = (const char **)triad_arena_alloc(a, e->u.lambda.params_len * sizeof(char *));
                for (size_t k = 0; k < e->u.lambda.params_len; ++k) ps[k] = e->u.lambda.params[k].name;
                n->u.func.params = ps;
                n->u.func.params_len = e->u.lambda.params_len;
            }
            n->u.func.body = lower_stmt_list(a, e->u.lambda.body, e->u.lambda.body_len, &n->u.func.body_len);
            return n;
        }
        case TRIAD_AST_METHOD_CALL: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_METHOD_CALL);
            n->u.call.func_or_obj = lower_expr(a, e->u.call.func_or_obj);
            n->u.call.method      = e->u.call.method;
            n->u.call.args        = lower_expr_list(a, e->u.call.args, e->u.call.args_len, &n->u.call.args_len);
            if (e->u.call.kwargs_len) {
                TriadIRKwArg *kw = (TriadIRKwArg *)triad_arena_alloc(a, e->u.call.kwargs_len * sizeof(TriadIRKwArg));
                for (size_t k = 0; k < e->u.call.kwargs_len; ++k) {
                    kw[k].name  = e->u.call.kwargs[k].name;
                    kw[k].value = lower_expr(a, e->u.call.kwargs[k].value);
                }
                n->u.call.kwargs = kw;
                n->u.call.kwargs_len = e->u.call.kwargs_len;
            }
            return n;
        }
        case TRIAD_AST_ASSIGN_EXPR: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_ASSIGN_EXPR);
            n->u.assign_expr.target = lower_expr(a, e->u.assign_expr.target);
            n->u.assign_expr.value  = lower_expr(a, e->u.assign_expr.value);
            return n;
        }
        case TRIAD_AST_FSTRING: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_FSTRING);
            if (e->u.fstring.parts_len) {
                TriadIRFStringPart *arr = (TriadIRFStringPart *)triad_arena_calloc(
                    a, e->u.fstring.parts_len * sizeof(TriadIRFStringPart));
                for (size_t k = 0; k < e->u.fstring.parts_len; ++k) {
                    if (!e->u.fstring.parts[k].is_expr) {
                        arr[k].text = e->u.fstring.parts[k].text ? e->u.fstring.parts[k].text : "";
                        arr[k].expr = NULL;
                    } else {
                        arr[k].text = "";
                        arr[k].expr = lower_expr(a, e->u.fstring.parts[k].expr);
                    }
                }
                n->u.fstring.parts = arr;
                n->u.fstring.parts_len = e->u.fstring.parts_len;
            }
            return n;
        }
        case TRIAD_AST_LIST_COMP: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_LIST_COMP);
            n->u.list_comp.expr      = lower_expr(a, e->u.list_comp.expr);
            n->u.list_comp.var       = e->u.list_comp.var;
            n->u.list_comp.iter      = lower_expr(a, e->u.list_comp.iter);
            n->u.list_comp.condition = e->u.list_comp.condition ? lower_expr(a, e->u.list_comp.condition) : NULL;
            n->u.list_comp.clauses_len = 0;
            n->u.list_comp.clauses = NULL;
            return n;
        }
        case TRIAD_AST_COMPLEX_LIT: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_COMPLEX);
            n->u.complex_lit.real_val = e->u.complex_lit.real_val;
            n->u.complex_lit.imag_val = e->u.complex_lit.imag_val;
            return n;
        }
        case TRIAD_AST_BYTES_LIT: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_BYTES);
            n->u.bytes_lit.bytes_val = e->u.bytes_lit.bytes_val;
            n->u.bytes_lit.bytes_len = e->u.bytes_lit.bytes_len;
            return n;
        }
        case TRIAD_AST_TUPLE: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_TUPLE);
            n->u.tuple_lit.elements = lower_expr_list(a, e->u.list.elements, e->u.list.elements_len, &n->u.tuple_lit.elements_len);
            return n;
        }
        case TRIAD_AST_SET_LIT: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_SET);
            n->u.set_lit.elements = lower_expr_list(a, e->u.set_lit.elements, e->u.set_lit.elements_len, &n->u.set_lit.elements_len);
            return n;
        }
        case TRIAD_AST_TERNARY: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_TERNARY);
            n->u.ternary.condition = lower_expr(a, e->u.ternary.condition);
            n->u.ternary.then_val   = lower_expr(a, e->u.ternary.then_val);
            n->u.ternary.else_val   = lower_expr(a, e->u.ternary.else_val);
            return n;
        }
        case TRIAD_AST_CHAIN_CMP: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_CHAIN_CMP);
            n->u.chain_cmp.operands = lower_expr_list(a, e->u.chain_cmp.operands, e->u.chain_cmp.operands_len, &n->u.chain_cmp.operands_len);
            n->u.chain_cmp.ops = e->u.chain_cmp.ops;
            n->u.chain_cmp.ops_len = e->u.chain_cmp.ops_len;
            return n;
        }
        case TRIAD_AST_SUPER: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_SUPER);
            n->u.super_expr.args = lower_expr_list(a, e->u.super_expr.args, e->u.super_expr.args_len, &n->u.super_expr.args_len);
            return n;
        }
        case TRIAD_AST_DICT_COMP: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_DICT_COMP);
            n->u.dict_comp.key_expr   = lower_expr(a, e->u.dict_comp.key_expr);
            n->u.dict_comp.value_expr = lower_expr(a, e->u.dict_comp.value_expr);
            n->u.dict_comp.clauses_len = 0;
            n->u.dict_comp.clauses = NULL;
            return n;
        }
        case TRIAD_AST_SET_COMP: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_SET_COMP);
            n->u.set_comp.expr = lower_expr(a, e->u.set_comp.expr);
            n->u.set_comp.clauses_len = 0;
            n->u.set_comp.clauses = NULL;
            return n;
        }
        case TRIAD_AST_GEN_COMP: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_GEN_COMP);
            n->u.gen_comp.expr = lower_expr(a, e->u.gen_comp.expr);
            n->u.gen_comp.clauses_len = 0;
            n->u.gen_comp.clauses = NULL;
            return n;
        }
        case TRIAD_AST_YIELD_EXPR: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_YIELD_EXPR);
            n->u.yield_expr.value = e->u.yield_expr.value ? lower_expr(a, e->u.yield_expr.value) : NULL;
            return n;
        }
        case TRIAD_AST_AWAIT_EXPR: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_AWAIT);
            n->u.await_expr.value = lower_expr(a, e->u.await_expr.value);
            return n;
        }

        default:
            return ir_new(a, TRIAD_IR_NONE);
    }
}

static TriadIRNode *match_condition(TriadArena *a, TriadIRNode *subject,
                                    const TriadAstNode *pattern) {
    if (!pattern) {
        TriadIRNode *n = ir_new(a, TRIAD_IR_BOOL);
        n->u.bool_val = 1; return n;
    }

    if (pattern->kind == TRIAD_AST_IDENT &&
        pattern->u.ident_name && strcmp(pattern->u.ident_name, "_") == 0) {
        TriadIRNode *n = ir_new(a, TRIAD_IR_BOOL);
        n->u.bool_val = 1; return n;
    }

    switch (pattern->kind) {
        case TRIAD_AST_INT_LIT:
        case TRIAD_AST_FLOAT_LIT:
        case TRIAD_AST_BOOL_LIT:
        case TRIAD_AST_STRING_LIT:
        case TRIAD_AST_NONE_LIT:
        case TRIAD_AST_IDENT:
        case TRIAD_AST_LIST: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_BINOP);
            n->u.binop.op    = "==";
            n->u.binop.left  = subject;
            n->u.binop.right = lower_expr(a, pattern);
            return n;
        }
        default: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_BOOL);
            n->u.bool_val = 1; return n;
        }
    }
}

static TriadIRNode *lower_match(TriadArena *a, const TriadAstNode *s) {
    TriadIRNode *subject = lower_expr(a, s->u.match_stmt.subject);

    if (s->u.match_stmt.cases_len == 0) {

        TriadIRNode *n = ir_new(a, TRIAD_IR_IF);
        n->u.if_stmt.condition = subject;
        return n;
    }

    PV elifs = {0};
    TriadIRNode  *first_cond = NULL;
    TriadIRNode **first_body = NULL;
    size_t        first_body_len = 0;

    for (size_t i = 0; i < s->u.match_stmt.cases_len; ++i) {
        const TriadMatchCase *c = &s->u.match_stmt.cases[i];
        TriadIRNode  *cond = match_condition(a, subject, c->pattern);
        size_t blen = 0;
        TriadIRNode **body = lower_stmt_list(a, c->body, c->body_len, &blen);
        if (i == 0) {
            first_cond = cond;
            first_body = body;
            first_body_len = blen;
        } else {
            TriadIRElifClause *cl = (TriadIRElifClause *)triad_arena_calloc(a, sizeof(TriadIRElifClause));
            cl->cond     = cond;
            cl->body     = body;
            cl->body_len = blen;
            pv_push(&elifs, cl);
        }
    }

    int has_else = 0;
    TriadIRNode **else_body = NULL;
    size_t        else_body_len = 0;

    if (s->u.match_stmt.has_else) {
        else_body = lower_stmt_list(a, s->u.match_stmt.else_body,
                                    s->u.match_stmt.else_body_len, &else_body_len);
        has_else = 1;
    } else if (elifs.len > 0) {
        TriadIRElifClause *last = (TriadIRElifClause *)elifs.items[elifs.len - 1];
        else_body     = last->body;
        else_body_len = last->body_len;
        has_else      = 1;
        elifs.len--;
    }

    TriadIRNode *iff = ir_new(a, TRIAD_IR_IF);
    iff->u.if_stmt.condition = first_cond;
    iff->u.if_stmt.then_body = first_body;
    iff->u.if_stmt.then_body_len = first_body_len;
    if (elifs.len) {
        TriadIRElifClause *arr = (TriadIRElifClause *)triad_arena_alloc(a, elifs.len * sizeof(TriadIRElifClause));
        for (size_t k = 0; k < elifs.len; ++k) arr[k] = *(TriadIRElifClause *)elifs.items[k];
        iff->u.if_stmt.elif_clauses = arr;
        iff->u.if_stmt.elif_clauses_len = elifs.len;
    }
    pv_free(&elifs);
    iff->u.if_stmt.has_else = has_else;
    iff->u.if_stmt.else_body = else_body;
    iff->u.if_stmt.else_body_len = else_body_len;
    return iff;
}

static TriadIRNode *lower_class(TriadArena *a, const TriadAstNode *s) {
    TriadIRNode *cd = ir_new(a, TRIAD_IR_CLASS_DECL);
    cd->u.class_decl.name   = s->u.type_decl.name;
    cd->u.class_decl.parent = s->u.type_decl.parent;

    if (s->u.type_decl.fields_len) {
        const char **fs = (const char **)triad_arena_alloc(a, s->u.type_decl.fields_len * sizeof(char *));
        for (size_t k = 0; k < s->u.type_decl.fields_len; ++k) fs[k] = s->u.type_decl.fields[k].name;
        cd->u.class_decl.fields = fs;
        cd->u.class_decl.fields_len = s->u.type_decl.fields_len;
    }

    if (s->u.type_decl.methods_len) {
        TriadIRNode **ms = (TriadIRNode **)triad_arena_alloc(a, s->u.type_decl.methods_len * sizeof(TriadIRNode *));
        for (size_t k = 0; k < s->u.type_decl.methods_len; ++k) {
            const TriadAstNode *m = s->u.type_decl.methods[k];
            TriadIRNode *fn = ir_new(a, TRIAD_IR_FUNCTION);
            fn->u.func.name = m->u.fn_decl.name;
            if (m->u.fn_decl.params_len) {
                const char **ps = (const char **)triad_arena_alloc(a, m->u.fn_decl.params_len * sizeof(char *));
                for (size_t p = 0; p < m->u.fn_decl.params_len; ++p) ps[p] = m->u.fn_decl.params[p].name;
                fn->u.func.params = ps;
                fn->u.func.params_len = m->u.fn_decl.params_len;
            }
            fn->u.func.body = lower_stmt_list(a, m->u.fn_decl.body, m->u.fn_decl.body_len, &fn->u.func.body_len);
            ms[k] = fn;
        }
        cd->u.class_decl.methods = ms;
        cd->u.class_decl.methods_len = s->u.type_decl.methods_len;
    }
    return cd;
}

static TriadIRNode *lower_stmt(TriadArena *a, const TriadAstNode *s) {
    if (!s) {
        TriadIRNode *n = ir_new(a, TRIAD_IR_EXPR_STMT);
        n->u.expr_stmt.expr = ir_new(a, TRIAD_IR_NONE);
        return n;
    }
    switch (s->kind) {
        case TRIAD_AST_LET: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_LET);
            n->u.let_stmt.name     = s->u.let_stmt.name;
            n->u.let_stmt.type_ann = s->u.let_stmt.type_ann;
            n->u.let_stmt.value    = s->u.let_stmt.value ? lower_expr(a, s->u.let_stmt.value) : NULL;
            return n;
        }
        case TRIAD_AST_CONST: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_CONST);
            n->u.const_stmt.name  = s->u.const_stmt.name;
            n->u.const_stmt.value = lower_expr(a, s->u.const_stmt.value);
            return n;
        }
        case TRIAD_AST_ASSIGN: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_ASSIGN);
            n->u.assign_stmt.target = lower_expr(a, s->u.assign_stmt.target);
            n->u.assign_stmt.value  = lower_expr(a, s->u.assign_stmt.value);
            return n;
        }
        case TRIAD_AST_EXPR_STMT: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_EXPR_STMT);
            n->u.expr_stmt.expr = lower_expr(a, s->u.expr_stmt.expr);
            return n;
        }
        case TRIAD_AST_RETURN: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_RETURN);
            n->u.ret_or_throw.value = s->u.unary_value.value ? lower_expr(a, s->u.unary_value.value) : NULL;
            return n;
        }
        case TRIAD_AST_BREAK:    return ir_new(a, TRIAD_IR_BREAK);
        case TRIAD_AST_CONTINUE: return ir_new(a, TRIAD_IR_CONTINUE);
        case TRIAD_AST_IF: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_IF);
            n->u.if_stmt.condition  = lower_expr(a, s->u.if_stmt.condition);
            n->u.if_stmt.then_body  = lower_stmt_list(a, s->u.if_stmt.then_body, s->u.if_stmt.then_body_len, &n->u.if_stmt.then_body_len);
            if (s->u.if_stmt.elif_clauses_len) {
                TriadIRElifClause *arr = (TriadIRElifClause *)triad_arena_calloc(a, s->u.if_stmt.elif_clauses_len * sizeof(TriadIRElifClause));
                for (size_t k = 0; k < s->u.if_stmt.elif_clauses_len; ++k) {
                    arr[k].cond = lower_expr(a, s->u.if_stmt.elif_clauses[k].cond);
                    arr[k].body = lower_stmt_list(a, s->u.if_stmt.elif_clauses[k].body, s->u.if_stmt.elif_clauses[k].body_len, &arr[k].body_len);
                }
                n->u.if_stmt.elif_clauses = arr;
                n->u.if_stmt.elif_clauses_len = s->u.if_stmt.elif_clauses_len;
            }
            n->u.if_stmt.has_else = s->u.if_stmt.has_else;
            if (s->u.if_stmt.has_else) {
                n->u.if_stmt.else_body = lower_stmt_list(a, s->u.if_stmt.else_body, s->u.if_stmt.else_body_len, &n->u.if_stmt.else_body_len);
            }
            return n;
        }
        case TRIAD_AST_FOR: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_FOR);
            n->u.for_stmt.var  = s->u.for_stmt.var;
            n->u.for_stmt.iter = lower_expr(a, s->u.for_stmt.iter);
            n->u.for_stmt.body = lower_stmt_list(a, s->u.for_stmt.body, s->u.for_stmt.body_len, &n->u.for_stmt.body_len);
            return n;
        }
        case TRIAD_AST_WHILE: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_WHILE);
            n->u.while_stmt.condition = lower_expr(a, s->u.while_stmt.condition);
            n->u.while_stmt.body      = lower_stmt_list(a, s->u.while_stmt.body, s->u.while_stmt.body_len, &n->u.while_stmt.body_len);
            return n;
        }
        case TRIAD_AST_FN_DECL: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_FUNCTION);
            n->u.func.name = s->u.fn_decl.name;
            if (s->u.fn_decl.params_len) {
                const char **ps = (const char **)triad_arena_alloc(a, s->u.fn_decl.params_len * sizeof(char *));
                for (size_t k = 0; k < s->u.fn_decl.params_len; ++k) ps[k] = s->u.fn_decl.params[k].name;
                n->u.func.params = ps;
                n->u.func.params_len = s->u.fn_decl.params_len;
            }
            n->u.func.body = lower_stmt_list(a, s->u.fn_decl.body, s->u.fn_decl.body_len, &n->u.func.body_len);
            return n;
        }
        case TRIAD_AST_TYPE_DECL: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_TYPE_DECL);
            n->u.type_decl.name = s->u.type_decl.name;
            if (s->u.type_decl.fields_len) {
                TriadIRTypeField *arr = (TriadIRTypeField *)triad_arena_alloc(a, s->u.type_decl.fields_len * sizeof(TriadIRTypeField));
                for (size_t k = 0; k < s->u.type_decl.fields_len; ++k) {
                    arr[k].name     = s->u.type_decl.fields[k].name;
                    arr[k].type_ann = s->u.type_decl.fields[k].type_ann;
                }
                n->u.type_decl.fields = arr;
                n->u.type_decl.fields_len = s->u.type_decl.fields_len;
            }
            return n;
        }
        case TRIAD_AST_IMPORT: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_IMPORT);
            n->u.import_stmt.path     = s->u.import_stmt.path;
            n->u.import_stmt.path_len = s->u.import_stmt.path_len;
            n->u.import_stmt.alias    = s->u.import_stmt.alias;
            return n;
        }
        case TRIAD_AST_FROM_IMPORT: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_IMPORT);
            n->u.import_stmt.path      = s->u.from_import.path;
            n->u.import_stmt.path_len  = s->u.from_import.path_len;
            n->u.import_stmt.alias     = NULL;
            n->u.import_stmt.names     = s->u.from_import.names;
            n->u.import_stmt.names_len = s->u.from_import.names_len;
            return n;
        }
        case TRIAD_AST_REG: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_REG_DECL);
            n->u.reg_decl.name   = s->u.reg_stmt.name;
            n->u.reg_decl.regime = s->u.reg_stmt.regime;
            n->u.reg_decl.value  = s->u.reg_stmt.value ? lower_expr(a, s->u.reg_stmt.value) : NULL;
            return n;
        }
        case TRIAD_AST_ENTITY: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_ENTITY_DECL);
            n->u.entity_decl.name = s->u.entity_decl.name;
            n->u.entity_decl.base = s->u.entity_decl.base;
            if (s->u.entity_decl.fields_len) {
                TriadIRStrEntry *arr = (TriadIRStrEntry *)triad_arena_alloc(a, s->u.entity_decl.fields_len * sizeof(TriadIRStrEntry));
                for (size_t k = 0; k < s->u.entity_decl.fields_len; ++k) {
                    arr[k].key   = s->u.entity_decl.fields[k].key;
                    arr[k].value = lower_expr(a, s->u.entity_decl.fields[k].value);
                }
                n->u.entity_decl.fields = arr;
                n->u.entity_decl.fields_len = s->u.entity_decl.fields_len;
            }
            return n;
        }
        case TRIAD_AST_WORLD: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_WORLD_DECL);
            n->u.world_decl.name = s->u.world_decl.name;
            if (s->u.world_decl.fields_len) {
                TriadIRStrEntry *arr = (TriadIRStrEntry *)triad_arena_alloc(a, s->u.world_decl.fields_len * sizeof(TriadIRStrEntry));
                for (size_t k = 0; k < s->u.world_decl.fields_len; ++k) {
                    arr[k].key   = s->u.world_decl.fields[k].key;
                    arr[k].value = lower_expr(a, s->u.world_decl.fields[k].value);
                }
                n->u.world_decl.fields = arr;
                n->u.world_decl.fields_len = s->u.world_decl.fields_len;
            }

            if (s->u.world_decl.entities_len) {
                TriadIRNode **arr = (TriadIRNode **)triad_arena_alloc(a, s->u.world_decl.entities_len * sizeof(TriadIRNode *));
                for (size_t k = 0; k < s->u.world_decl.entities_len; ++k) {
                    const TriadAstNode *en = s->u.world_decl.entities[k];
                    TriadIRNode *ed = ir_new(a, TRIAD_IR_ENTITY_DECL);
                    ed->u.entity_decl.name = en->u.entity_decl.name;
                    ed->u.entity_decl.base = en->u.entity_decl.base;
                    if (en->u.entity_decl.fields_len) {
                        TriadIRStrEntry *fa = (TriadIRStrEntry *)triad_arena_alloc(a, en->u.entity_decl.fields_len * sizeof(TriadIRStrEntry));
                        for (size_t f = 0; f < en->u.entity_decl.fields_len; ++f) {
                            fa[f].key   = en->u.entity_decl.fields[f].key;
                            fa[f].value = lower_expr(a, en->u.entity_decl.fields[f].value);
                        }
                        ed->u.entity_decl.fields = fa;
                        ed->u.entity_decl.fields_len = en->u.entity_decl.fields_len;
                    }
                    arr[k] = ed;
                }
                n->u.world_decl.entities = arr;
                n->u.world_decl.entities_len = s->u.world_decl.entities_len;
            }
            return n;
        }
        case TRIAD_AST_SUBSTRATE: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_SUBSTRATE_DECL);
            n->u.substrate_decl.name = s->u.substrate_decl.name;
            n->u.substrate_decl.regime = s->u.substrate_decl.regime;
            n->u.substrate_decl.members = s->u.substrate_decl.members;
            n->u.substrate_decl.members_len = s->u.substrate_decl.members_len;
            n->u.substrate_decl.is_composed = s->u.substrate_decl.is_composed;
            if (s->u.substrate_decl.properties_len) {
                TriadIRStrEntry *arr = (TriadIRStrEntry *)triad_arena_alloc(a, s->u.substrate_decl.properties_len * sizeof(TriadIRStrEntry));
                for (size_t k = 0; k < s->u.substrate_decl.properties_len; ++k) {
                    arr[k].key = s->u.substrate_decl.properties[k].key;
                    arr[k].value = lower_expr(a, s->u.substrate_decl.properties[k].value);
                }
                n->u.substrate_decl.properties = arr;
                n->u.substrate_decl.properties_len = s->u.substrate_decl.properties_len;
            }
            if (s->u.substrate_decl.overrides_len) {
                TriadIRStrEntry *arr = (TriadIRStrEntry *)triad_arena_alloc(a, s->u.substrate_decl.overrides_len * sizeof(TriadIRStrEntry));
                for (size_t k = 0; k < s->u.substrate_decl.overrides_len; ++k) {
                    arr[k].key = s->u.substrate_decl.overrides[k].key;
                    arr[k].value = lower_expr(a, s->u.substrate_decl.overrides[k].value);
                }
                n->u.substrate_decl.overrides = arr;
                n->u.substrate_decl.overrides_len = s->u.substrate_decl.overrides_len;
            }
            return n;
        }
        case TRIAD_AST_OBSERVE: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_OBSERVE);
            n->u.observe.target      = s->u.observe_stmt.target;
            n->u.observe.metrics     = s->u.observe_stmt.metrics;
            n->u.observe.metrics_len = s->u.observe_stmt.metrics_len;
            return n;
        }
        case TRIAD_AST_RUN: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_RUN);
            n->u.run.duration = s->u.run_stmt.duration ? lower_expr(a, s->u.run_stmt.duration) : NULL;
            n->u.run.target = s->u.run_stmt.target;
            return n;
        }
        case TRIAD_AST_DESTRUCT_LET:
        case TRIAD_AST_MAP_DESTRUCT: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_DESTRUCT_LET);
            n->u.destruct_let.names     = s->u.destruct.names;
            n->u.destruct_let.names_len = s->u.destruct.names_len;
            n->u.destruct_let.value     = lower_expr(a, s->u.destruct.value);
            return n;
        }
        case TRIAD_AST_TRY_CATCH: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_TRY_CATCH);
            n->u.try_catch.body = lower_stmt_list(a, s->u.try_catch.body, s->u.try_catch.body_len, &n->u.try_catch.body_len);
            n->u.try_catch.catch_var = s->u.try_catch.catch_var;

            if (s->u.try_catch.catch_body_len > 0 || s->u.try_catch.catch_var) {
                n->u.try_catch.catch_body = lower_stmt_list(a, s->u.try_catch.catch_body, s->u.try_catch.catch_body_len, &n->u.try_catch.catch_body_len);
                n->u.try_catch.has_catch = 1;
            }
            if (s->u.try_catch.finally_body_len > 0) {
                n->u.try_catch.finally_body = lower_stmt_list(a, s->u.try_catch.finally_body, s->u.try_catch.finally_body_len, &n->u.try_catch.finally_body_len);
                n->u.try_catch.has_finally = 1;
            }
            return n;
        }
        case TRIAD_AST_THROW: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_THROW);
            n->u.ret_or_throw.value = s->u.unary_value.value ? lower_expr(a, s->u.unary_value.value) : NULL;
            return n;
        }
        case TRIAD_AST_WITH:
        case TRIAD_AST_ASYNC_WITH: {
            TriadIRNode *n = ir_new(a, s->kind == TRIAD_AST_ASYNC_WITH ? TRIAD_IR_ASYNC_WITH : TRIAD_IR_WITH);
            n->u.with_stmt.expr = s->u.with_stmt.expr ? lower_expr(a, s->u.with_stmt.expr) : NULL;
            n->u.with_stmt.var = s->u.with_stmt.var;
            n->u.with_stmt.body = lower_stmt_list(a, s->u.with_stmt.body, s->u.with_stmt.body_len, &n->u.with_stmt.body_len);
            return n;
        }
        case TRIAD_AST_ASSERT: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_ASSERT);
            n->u.assert_stmt.condition = lower_expr(a, s->u.assert_stmt.condition);
            n->u.assert_stmt.message = s->u.assert_stmt.message ? lower_expr(a, s->u.assert_stmt.message) : NULL;
            return n;
        }
        case TRIAD_AST_PASS:
            return ir_new(a, TRIAD_IR_PASS);
        case TRIAD_AST_DEL: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_DEL);
            n->u.del_stmt.target = lower_expr(a, s->u.del_stmt.target);
            return n;
        }
        case TRIAD_AST_ASYNC_FOR: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_ASYNC_FOR);
            n->u.for_stmt.var = s->u.for_stmt.var;
            n->u.for_stmt.iter = lower_expr(a, s->u.for_stmt.iter);
            n->u.for_stmt.body = lower_stmt_list(a, s->u.for_stmt.body, s->u.for_stmt.body_len, &n->u.for_stmt.body_len);
            return n;
        }
        case TRIAD_AST_MATCH:      return lower_match(a, s);
        case TRIAD_AST_CLASS_DECL: return lower_class(a, s);
        case TRIAD_AST_YIELD_STMT: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_YIELD);
            n->u.ret_or_throw.value = s->u.unary_value.value ? lower_expr(a, s->u.unary_value.value) : NULL;
            return n;
        }
        case TRIAD_AST_COUPLE: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_COUPLE);
            n->u.couple.src = s->u.couple_stmt.src;
            n->u.couple.dst = s->u.couple_stmt.dst;
            n->u.couple.kappa = s->u.couple_stmt.kappa ? lower_expr(a, s->u.couple_stmt.kappa) : NULL;
            n->u.couple.duration = s->u.couple_stmt.duration ? lower_expr(a, s->u.couple_stmt.duration) : NULL;
            return n;
        }
        case TRIAD_AST_PAIR: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_PAIR);
            n->u.pair.a = s->u.pair_stmt.a;
            n->u.pair.b = s->u.pair_stmt.b;
            n->u.pair.kappa = s->u.pair_stmt.kappa ? lower_expr(a, s->u.pair_stmt.kappa) : NULL;
            n->u.pair.duration = s->u.pair_stmt.duration ? lower_expr(a, s->u.pair_stmt.duration) : NULL;
            return n;
        }
        case TRIAD_AST_RING: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_RING);
            n->u.ring.members = s->u.ring_stmt.members;
            n->u.ring.members_len = s->u.ring_stmt.members_len;
            n->u.ring.kappa = s->u.ring_stmt.kappa ? lower_expr(a, s->u.ring_stmt.kappa) : NULL;
            n->u.ring.duration = s->u.ring_stmt.duration ? lower_expr(a, s->u.ring_stmt.duration) : NULL;
            return n;
        }
        case TRIAD_AST_ANNOTATION: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_ANNOTATION);
            n->u.annotation.key = s->u.annotation_stmt.key;
            n->u.annotation.args = s->u.annotation_stmt.args;
            return n;
        }
        case TRIAD_AST_SEQUENCE: {
            TriadIRNode *n = ir_new(a, TRIAD_IR_SEQUENCE);
            n->u.sequence.inputs = lower_expr(a, s->u.sequence_stmt.inputs);
            n->u.sequence.target = s->u.sequence_stmt.target;
            n->u.sequence.each_for = s->u.sequence_stmt.each_for ? lower_expr(a, s->u.sequence_stmt.each_for) : NULL;
            return n;
        }
        default:
            fprintf(stderr, "lower_stmt: unhandled AST kind %d\n", (int)s->kind);
            exit(1);
    }
}

TriadIRNode *triad_ir_lower_module(TriadArena *a, const TriadAstNode *module,
                                   TriadDiag *diag) {
    (void)diag;
    if (!module || module->kind != TRIAD_AST_MODULE) return NULL;
    TriadIRNode *mod = ir_new(a, TRIAD_IR_MODULE);
    mod->u.module.name = module->u.module.name ? module->u.module.name : "";
    mod->u.module.imports = NULL;
    mod->u.module.imports_len = 0;
    mod->u.module.body = lower_stmt_list(a, module->u.module.body,
                                         module->u.module.body_len,
                                         &mod->u.module.body_len);
    return mod;
}

const char *triad_ir_kind_name(TriadIRKind k) {
    switch (k) {
        case TRIAD_IR_INT:           return "IRInt";
        case TRIAD_IR_FLOAT:         return "IRFloat";
        case TRIAD_IR_BOOL:          return "IRBool";
        case TRIAD_IR_STRING:        return "IRString";
        case TRIAD_IR_NONE:          return "IRNone";
        case TRIAD_IR_IDENT:         return "IRIdent";
        case TRIAD_IR_BINOP:         return "IRBinOp";
        case TRIAD_IR_UNARYOP:       return "IRUnaryOp";
        case TRIAD_IR_CALL:          return "IRCall";
        case TRIAD_IR_METHOD_CALL:   return "IRMethodCall";
        case TRIAD_IR_INDEX:         return "IRIndex";
        case TRIAD_IR_SLICE:         return "IRSlice";
        case TRIAD_IR_FIELD:         return "IRField";
        case TRIAD_IR_LIST:          return "IRList";
        case TRIAD_IR_MAP:           return "IRMap";
        case TRIAD_IR_FSTRING:       return "IRFString";
        case TRIAD_IR_LIST_COMP:     return "IRListComp";
        case TRIAD_IR_ASSIGN_EXPR:   return "IRAssignExpr";
        case TRIAD_IR_LET:           return "IRLet";
        case TRIAD_IR_CONST:         return "IRConst";
        case TRIAD_IR_ASSIGN:        return "IRAssign";
        case TRIAD_IR_EXPR_STMT:     return "IRExprStmt";
        case TRIAD_IR_RETURN:        return "IRReturn";
        case TRIAD_IR_BREAK:         return "IRBreak";
        case TRIAD_IR_CONTINUE:      return "IRContinue";
        case TRIAD_IR_IF:            return "IRIf";
        case TRIAD_IR_FOR:           return "IRFor";
        case TRIAD_IR_WHILE:         return "IRWhile";
        case TRIAD_IR_FUNCTION:      return "IRFunction";
        case TRIAD_IR_TRY_CATCH:     return "IRTryCatch";
        case TRIAD_IR_THROW:         return "IRThrow";
        case TRIAD_IR_WITH:          return "IRWith";
        case TRIAD_IR_ASSERT:        return "IRAssert";
        case TRIAD_IR_PASS:          return "IRPass";
        case TRIAD_IR_DEL:           return "IRDel";
        case TRIAD_IR_ASYNC_FOR:     return "IRAsyncFor";
        case TRIAD_IR_ASYNC_WITH:    return "IRAsyncWith";
        case TRIAD_IR_TYPE_DECL:     return "IRTypeDecl";
        case TRIAD_IR_ENTITY_DECL:   return "IREntityDecl";
        case TRIAD_IR_WORLD_DECL:    return "IRWorldDecl";
        case TRIAD_IR_REG_DECL:      return "IRRegDecl";
        case TRIAD_IR_OBSERVE:       return "IRObserve";
        case TRIAD_IR_SUBSTRATE_DECL:return "IRSubstrateDecl";
        case TRIAD_IR_RUN:           return "IRRun";
        case TRIAD_IR_IMPORT:        return "IRImport";
        case TRIAD_IR_DESTRUCT_LET:  return "IRDestructLet";
        case TRIAD_IR_CLASS_DECL:    return "IRClassDecl";
        case TRIAD_IR_YIELD:         return "IRYield";
        case TRIAD_IR_COUPLE:        return "IRCouple";
        case TRIAD_IR_PAIR:          return "IRPair";
        case TRIAD_IR_RING:          return "IRRing";
        case TRIAD_IR_SEQUENCE:      return "IRSequence";
        case TRIAD_IR_ANNOTATION:    return "IRAnnotation";
        case TRIAD_IR_COMPLEX:      return "IRComplex";
        case TRIAD_IR_BYTES:        return "IRBytes";
        case TRIAD_IR_TUPLE:        return "IRTuple";
        case TRIAD_IR_SET:          return "IRSet";
        case TRIAD_IR_TERNARY:      return "IRTernary";
        case TRIAD_IR_CHAIN_CMP:    return "IRChainCmp";
        case TRIAD_IR_SUPER:        return "IRSuper";
        case TRIAD_IR_DICT_COMP:    return "IRDictComp";
        case TRIAD_IR_SET_COMP:     return "IRSetComp";
        case TRIAD_IR_GEN_COMP:     return "IRGenComp";
        case TRIAD_IR_YIELD_EXPR:   return "IRYieldExpr";
        case TRIAD_IR_AWAIT:        return "IRAwait";
        case TRIAD_IR_COMP_CLAUSE:  return "IRCompClause";
        case TRIAD_IR_MODULE:        return "IRModule";
        default:                     return "Unknown";
    }
}
