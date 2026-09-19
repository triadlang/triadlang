#include "triad_check.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char **buckets;
    size_t       cap;
    size_t       len;
} StrSet;

static size_t fnv1a(const char *s) {
    size_t h = 1469598103934665603ULL;
    for (; *s; ++s) {
        h ^= (unsigned char)*s;
        h *= 1099511628211ULL;
    }
    return h;
}

static void ss_init(StrSet *s, size_t cap) {
    if (!s) return;
    if (cap < 16) cap = 16;
    s->cap = cap;
    s->len = 0;
    s->buckets = (const char **)calloc(cap, sizeof(char *));
    if (!s->buckets) s->cap = 0;
}

static int ss_has(const StrSet *s, const char *k);

static void ss_rehash(StrSet *s, size_t new_cap) {
    if (!s || new_cap == 0) return;
    const char **old = s->buckets;
    size_t old_cap = s->cap;
    const char **nb = (const char **)calloc(new_cap, sizeof(char *));
    if (!nb) return;
    s->buckets = nb;
    s->cap = new_cap;
    s->len = 0;
    for (size_t i = 0; i < old_cap; ++i) {
        if (old[i]) {
            size_t h = fnv1a(old[i]) % new_cap;
            while (s->buckets[h]) h = (h + 1) % new_cap;
            s->buckets[h] = old[i];
            s->len++;
        }
    }
    free(old);
}

static void ss_add(StrSet *s, const char *k) {
    if (!s || !k || !s->buckets || s->cap == 0) return;
    if (ss_has(s, k)) return;
    if ((s->len + 1) * 2 > s->cap) ss_rehash(s, s->cap * 2);
    if (!s->buckets || s->cap == 0) return;
    size_t h = fnv1a(k) % s->cap;
    while (s->buckets[h]) {
        if (strcmp(s->buckets[h], k) == 0) return;
        h = (h + 1) % s->cap;
    }
    s->buckets[h] = k;
    s->len++;
}

static int ss_has(const StrSet *s, const char *k) {
    if (!s || !k || !s->buckets || s->cap == 0) return 0;
    size_t h = fnv1a(k) % s->cap;
    while (s->buckets[h]) {
        if (strcmp(s->buckets[h], k) == 0) return 1;
        h = (h + 1) % s->cap;
    }
    return 0;
}

static void ss_copy_from(StrSet *dst, const StrSet *src) {

    if (!dst || !src) return;
    ss_init(dst, src->cap);
    if (!dst->buckets) return;
    for (size_t i = 0; i < src->cap; ++i) {
        if (src->buckets && src->buckets[i]) ss_add(dst, src->buckets[i]);
    }
}

static void ss_free(StrSet *s) { if (!s) return; free(s->buckets); s->buckets = NULL; s->cap = s->len = 0; }

static const char *const BUILTINS[] = {
    "print", "input", "len", "range", "enumerate", "str", "int", "float",
    "type", "abs", "min", "max", "append", "sorted", "reversed",
    "true", "false", "none",
    "set", "dict", "list", "tuple", "zip", "map", "filter", "sum",
    "round", "any", "all", "pow", "open", "isinstance", "repr",
    "ord", "chr", "hash", "id", "iter", "next", "bool", "divmod",
    "format", "bytes", "exit", "callable", "getattr", "setattr",
    "hasattr", "sqrt", "sin", "cos", "exp", "log",
    "solver_solve", "solve", "ccall", "py_call", "py_eval", "py_exec",
    "cuda_available", "set_device", "device",
    "run_async", "sleep", "gather", "create_task",
    "async_read", "async_write", "async_fetch",
    "ml_tensor", "ml_zeros", "ml_randn", "ml_triad", "ml_forward",
    "ml_relu", "ml_sigmoid", "ml_tanh_act", "ml_softmax",
    "ml_mse_loss", "ml_cross_entropy", "ml_backward",
    "ml_item", "ml_data", "ml_seq_new", "ml_seq_set", "ml_seq_forward",
    "ml_adam", "ml_sgd", "ml_adam_step", "ml_adam_zero",
    "ml_sgd_step", "ml_sgd_zero", "ml_tensor_add", "ml_tensor_sub",
    "ml_tensor_mul", "ml_tensor_matmul", "ml_tensor_sum",
    "ml_tensor_mean", "ml_print",
    "ml_embedding", "ml_embedding_forward", "ml_layernorm",
    "ml_layernorm_forward", "ml_mha", "ml_mha_forward",
    "ml_transformer", "ml_transformer_forward", "ml_transformer_adam",
    "ml_wave", "ml_wave_forward", "ml_wave_adam",
    NULL
};

typedef struct { const char *name; int arity; } BArity;
static const BArity BUILTIN_ARITIES[] = {
    { "len", 1 }, { "abs", 1 }, { "min", -1 }, { "max", -1 },
    { "int", 1 }, { "float", 1 }, { "str", 1 }, { "type", 1 },
    { "range", -1 }, { "enumerate", 1 },
    { "sorted", 1 }, { "reversed", 1 },
    { NULL, 0 }
};

static int builtin_arity(const char *name, int *out_arity) {
    for (size_t k = 0; BUILTIN_ARITIES[k].name; ++k) {
        if (strcmp(BUILTIN_ARITIES[k].name, name) == 0) {
            *out_arity = BUILTIN_ARITIES[k].arity;
            return 1;
        }
    }
    return 0;
}

typedef struct FnEntry {
    const char     *name;
    int             arity;
    struct FnEntry *next;
} FnEntry;

typedef struct {
    FnEntry *head;
} FnMap;

static void fnmap_set(FnMap *m, const char *name, int arity) {
    for (FnEntry *e = m->head; e; e = e->next) {
        if (strcmp(e->name, name) == 0) { e->arity = arity; return; }
    }
    FnEntry *e = (FnEntry *)calloc(1, sizeof(FnEntry));
    e->name = name; e->arity = arity;
    e->next = m->head;
    m->head = e;
}

static int fnmap_get(const FnMap *m, const char *name, int *out_arity) {
    for (FnEntry *e = m->head; e; e = e->next) {
        if (strcmp(e->name, name) == 0) { *out_arity = e->arity; return 1; }
    }
    return 0;
}

static void fnmap_free(FnMap *m) {
    FnEntry *e = m->head;
    while (e) { FnEntry *n = e->next; free(e); e = n; }
    m->head = NULL;
}

typedef struct {
    const char **items;
    size_t       len;
    size_t       cap;
} ErrList;

static void err_push(TriadArena *a, ErrList *L, const char *fmt, ...) {
    char buf[512];
    va_list ap; va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap); va_end(ap);
    if (L->len == L->cap) {
        size_t nc = L->cap ? L->cap * 2 : 8;
        L->items = (const char **)realloc(L->items, nc * sizeof(char *));
        L->cap = nc;
    }
    L->items[L->len++] = triad_arena_strdup(a, buf);
}

static const char *fmt_pos(TriadArena *a, TriadPos p) {
    char buf[256];
    size_t n = 0;
    int wrote = 0;
    if (p.file && p.file[0]) {
        n += (size_t)snprintf(buf + n, sizeof(buf) - n,
                              "%sfile: %s", wrote ? ", " : "", p.file);
        wrote = 1;
    }
    if (p.line) {
        n += (size_t)snprintf(buf + n, sizeof(buf) - n,
                              "%sline: %d", wrote ? ", " : "", p.line);
        wrote = 1;
    }
    if (p.col) {
        n += (size_t)snprintf(buf + n, sizeof(buf) - n,
                              "%scol: %d", wrote ? ", " : "", p.col);
        wrote = 1;
    }
    return triad_arena_strdup(a, buf);
}

typedef struct {
    TriadArena *arena;
    ErrList     errs;
    FnMap       functions;
    StrSet      types_declared;
} Ctx;

static void check_expr(Ctx *C, const TriadAstNode *e, StrSet *scope);
static void check_stmt(Ctx *C, const TriadAstNode *s, StrSet *scope);
static void check_body(Ctx *C, TriadAstNode *const *stmts, size_t n, StrSet *scope) {
    for (size_t i = 0; i < n; ++i) check_stmt(C, stmts[i], scope);
}

static int _cmp_cstr(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static const char *suggest_name(const StrSet *scope, const char *name) {
    if (!scope || !name || !scope->buckets || scope->len == 0) return NULL;
    const char **cands = (const char **)malloc(scope->len * sizeof(char *));
    if (!cands) return NULL;
    size_t m = 0;
    for (size_t i = 0; i < scope->cap; ++i)
        if (scope->buckets[i]) cands[m++] = scope->buckets[i];
    qsort(cands, m, sizeof(char *), _cmp_cstr);
    const char *best = NULL;
    int best_dist = 4;
    size_t ln = strlen(name);
    for (size_t i = 0; i < m; ++i) {
        size_t lc = strlen(cands[i]);
        size_t diff = lc > ln ? lc - ln : ln - lc;
        if (diff > 3) continue;
        int d = 0;
        size_t k = ln < lc ? ln : lc;
        for (size_t j = 0; j < k; ++j) {
            d += (name[j] != cands[i][j]);
            if (d > 3) break;
        }
        d += (int)diff;
        if (d < best_dist && d <= 3) {
            best_dist = d;
            best = cands[i];
        }
    }
    free(cands);
    return best;
}

static void check_expr(Ctx *C, const TriadAstNode *e, StrSet *scope) {
    if (!e) return;
    switch (e->kind) {
        case TRIAD_AST_IDENT: {
            const char *nm = e->u.ident_name;
            if (!ss_has(scope, nm) && strcmp(nm, "self") != 0) {
                const char *sug = suggest_name(scope, nm);
                if (sug)
                    err_push(C->arena, &C->errs,
                             "error[E2001]: undefined variable '%s' (%s)\n"
                             "  help: did you mean '%s'?",
                             nm, fmt_pos(C->arena, e->pos), sug);
                else
                    err_push(C->arena, &C->errs,
                             "error[E2001]: undefined variable '%s' (%s)",
                             nm, fmt_pos(C->arena, e->pos));
            }
            break;
        }
        case TRIAD_AST_BINOP:
            check_expr(C, e->u.op.left, scope);
            check_expr(C, e->u.op.right, scope);
            break;
        case TRIAD_AST_UNARYOP:
            check_expr(C, e->u.op.right, scope);
            break;
        case TRIAD_AST_CALL: {
            check_expr(C, e->u.call.func_or_obj, scope);
            for (size_t i = 0; i < e->u.call.args_len; ++i)
                check_expr(C, e->u.call.args[i], scope);
            for (size_t i = 0; i < e->u.call.kwargs_len; ++i)
                check_expr(C, e->u.call.kwargs[i].value, scope);

            int has_spread = 0;
            for (size_t i = 0; i < e->u.call.args_len; ++i) {
                const TriadAstNode *a = e->u.call.args[i];
                if (a && a->kind == TRIAD_AST_UNARYOP && a->u.op.op &&
                    (strcmp(a->u.op.op, "*") == 0 || strcmp(a->u.op.op, "**") == 0))
                    has_spread = 1;
            }
            const TriadAstNode *fn = e->u.call.func_or_obj;
            if (fn && fn->kind == TRIAD_AST_IDENT && !has_spread) {
                const char *nm = fn->u.ident_name;
                int n_args = (int)e->u.call.args_len;
                int arity;
                if (builtin_arity(nm, &arity)) {
                    if (arity >= 0 && n_args != arity) {
                        err_push(C->arena, &C->errs,
                            "error[E2002]: '%s' expects %d argument(s), got %d (%s)",
                            nm, arity, n_args, fmt_pos(C->arena, e->pos));
                    }
                } else if (fnmap_get(&C->functions, nm, &arity)) {
                    if (arity >= 0 && n_args != arity) {
                        err_push(C->arena, &C->errs,
                            "error[E2002]: '%s' expects %d argument(s), got %d (%s)",
                            nm, arity, n_args, fmt_pos(C->arena, e->pos));
                    }
                }
            }
            break;
        }
        case TRIAD_AST_METHOD_CALL:
            check_expr(C, e->u.call.func_or_obj, scope);
            for (size_t i = 0; i < e->u.call.args_len; ++i)
                check_expr(C, e->u.call.args[i], scope);
            for (size_t i = 0; i < e->u.call.kwargs_len; ++i)
                check_expr(C, e->u.call.kwargs[i].value, scope);
            break;
        case TRIAD_AST_INDEX:
            check_expr(C, e->u.index.obj, scope);
            check_expr(C, e->u.index.index, scope);
            break;
        case TRIAD_AST_FIELD:
            check_expr(C, e->u.field.obj, scope);
            break;
        case TRIAD_AST_LIST:
        case TRIAD_AST_TUPLE:
            for (size_t i = 0; i < e->u.list.elements_len; ++i)
                check_expr(C, e->u.list.elements[i], scope);
            break;
        case TRIAD_AST_LIST_COMP: {
            check_expr(C, e->u.list_comp.iter, scope);
            StrSet inner; ss_copy_from(&inner, scope);
            ss_add(&inner, e->u.list_comp.var);
            check_expr(C, e->u.list_comp.expr, &inner);
            if (e->u.list_comp.condition) check_expr(C, e->u.list_comp.condition, &inner);
            ss_free(&inner);
            break;
        }
        case TRIAD_AST_MAP:
            for (size_t i = 0; i < e->u.map.pairs_len; ++i) {
                check_expr(C, e->u.map.pairs[i].key, scope);
                check_expr(C, e->u.map.pairs[i].value, scope);
            }
            break;
        case TRIAD_AST_FSTRING:
            for (size_t i = 0; i < e->u.fstring.parts_len; ++i) {
                if (e->u.fstring.parts[i].is_expr)
                    check_expr(C, e->u.fstring.parts[i].expr, scope);
            }
            break;
        case TRIAD_AST_LAMBDA: {
            StrSet inner; ss_copy_from(&inner, scope);
            for (size_t i = 0; i < e->u.lambda.params_len; ++i)
                ss_add(&inner, e->u.lambda.params[i].name);
            check_body(C, e->u.lambda.body, e->u.lambda.body_len, &inner);
            ss_free(&inner);
            break;
        }
        case TRIAD_AST_ASSIGN_EXPR:
            check_expr(C, e->u.assign_expr.target, scope);
            check_expr(C, e->u.assign_expr.value, scope);
            break;
        case TRIAD_AST_YIELD_EXPR:
            if (e->u.unary_value.value) check_expr(C, e->u.unary_value.value, scope);
            break;
        case TRIAD_AST_AWAIT_EXPR:
            check_expr(C, e->u.unary_value.value, scope);
            break;
        case TRIAD_AST_COMPLEX_LIT:
            break;
        case TRIAD_AST_BYTES_LIT:
            break;
        case TRIAD_AST_SET_LIT:
            for (size_t i = 0; i < e->u.set_lit.elements_len; i++)
                check_expr(C, e->u.set_lit.elements[i], scope);
            break;
        case TRIAD_AST_TERNARY:
            check_expr(C, e->u.ternary.condition, scope);
            check_expr(C, e->u.ternary.then_val, scope);
            check_expr(C, e->u.ternary.else_val, scope);
            break;
        case TRIAD_AST_CHAIN_CMP:
            for (size_t i = 0; i < e->u.chain_cmp.operands_len; i++)
                check_expr(C, e->u.chain_cmp.operands[i], scope);
            break;
        case TRIAD_AST_SUPER:
            for (size_t i = 0; i < e->u.super_expr.args_len; i++)
                check_expr(C, e->u.super_expr.args[i], scope);
            break;
        case TRIAD_AST_DICT_COMP:
            check_expr(C, e->u.dict_comp.key_expr, scope);
            check_expr(C, e->u.dict_comp.value_expr, scope);
            break;
        case TRIAD_AST_SET_COMP:
            check_expr(C, e->u.set_comp.expr, scope);
            break;
        case TRIAD_AST_GEN_COMP:
            check_expr(C, e->u.gen_comp.expr, scope);
            break;
        default:

            break;
    }
}

static void check_stmt(Ctx *C, const TriadAstNode *s, StrSet *scope) {
    if (!s) return;
    switch (s->kind) {
        case TRIAD_AST_LET:
            if (s->u.let_stmt.value) check_expr(C, s->u.let_stmt.value, scope);
            ss_add(scope, s->u.let_stmt.name);
            break;
        case TRIAD_AST_CONST:
            check_expr(C, s->u.const_stmt.value, scope);
            ss_add(scope, s->u.const_stmt.name);
            break;
        case TRIAD_AST_DESTRUCT_LET:
        case TRIAD_AST_MAP_DESTRUCT:
            check_expr(C, s->u.destruct.value, scope);
            for (size_t i = 0; i < s->u.destruct.names_len; ++i)
                ss_add(scope, s->u.destruct.names[i]);
            break;
        case TRIAD_AST_ASSIGN:
            check_expr(C, s->u.assign_stmt.value, scope);
            check_expr(C, s->u.assign_stmt.target, scope);
            break;
        case TRIAD_AST_EXPR_STMT:
            check_expr(C, s->u.expr_stmt.expr, scope);
            break;
        case TRIAD_AST_RETURN:
            if (s->u.unary_value.value) check_expr(C, s->u.unary_value.value, scope);
            break;
        case TRIAD_AST_PASS:
            break;
        case TRIAD_AST_ASSERT:
            check_expr(C, s->u.assert_stmt.condition, scope);
            if (s->u.assert_stmt.message) check_expr(C, s->u.assert_stmt.message, scope);
            break;
        case TRIAD_AST_DEL:
            check_expr(C, s->u.del_stmt.target, scope);
            break;
        case TRIAD_AST_IF: {
            check_expr(C, s->u.if_stmt.condition, scope);
            { StrSet inner; ss_copy_from(&inner, scope);
              check_body(C, s->u.if_stmt.then_body, s->u.if_stmt.then_body_len, &inner);
              ss_free(&inner); }
            for (size_t i = 0; i < s->u.if_stmt.elif_clauses_len; ++i) {
                check_expr(C, s->u.if_stmt.elif_clauses[i].cond, scope);
                StrSet inner; ss_copy_from(&inner, scope);
                check_body(C, s->u.if_stmt.elif_clauses[i].body,
                              s->u.if_stmt.elif_clauses[i].body_len, &inner);
                ss_free(&inner);
            }
            if (s->u.if_stmt.has_else) {
                StrSet inner; ss_copy_from(&inner, scope);
                check_body(C, s->u.if_stmt.else_body, s->u.if_stmt.else_body_len, &inner);
                ss_free(&inner);
            }
            break;
        }
        case TRIAD_AST_ASYNC_FOR:
        case TRIAD_AST_FOR: {
            check_expr(C, s->u.for_stmt.iter, scope);
            StrSet inner; ss_copy_from(&inner, scope);
            ss_add(&inner, s->u.for_stmt.var);
            check_body(C, s->u.for_stmt.body, s->u.for_stmt.body_len, &inner);
            ss_free(&inner);
            break;
        }
        case TRIAD_AST_ASYNC_WITH:
        case TRIAD_AST_WITH: {
            check_expr(C, s->u.with_stmt.expr, scope);
            StrSet inner; ss_copy_from(&inner, scope);
            if (s->u.with_stmt.var) ss_add(&inner, s->u.with_stmt.var);
            check_body(C, s->u.with_stmt.body, s->u.with_stmt.body_len, &inner);
            ss_free(&inner);
            break;
        }
        case TRIAD_AST_WHILE: {
            check_expr(C, s->u.while_stmt.condition, scope);
            StrSet inner; ss_copy_from(&inner, scope);
            check_body(C, s->u.while_stmt.body, s->u.while_stmt.body_len, &inner);
            ss_free(&inner);
            break;
        }
        case TRIAD_AST_FN_DECL: {
            int fn_arity = (int)s->u.fn_decl.params_len;
            for (size_t i = 0; i < s->u.fn_decl.params_len; ++i) {
                const TriadParam *pp = &s->u.fn_decl.params[i];
                if (pp->is_args || pp->is_kwargs || pp->default_value) { fn_arity = -1; break; }
            }
            fnmap_set(&C->functions, s->u.fn_decl.name, fn_arity);
            ss_add(scope, s->u.fn_decl.name);
            StrSet inner; ss_copy_from(&inner, scope);
            for (size_t i = 0; i < s->u.fn_decl.params_len; ++i)
                ss_add(&inner, s->u.fn_decl.params[i].name);
            check_body(C, s->u.fn_decl.body, s->u.fn_decl.body_len, &inner);
            ss_free(&inner);
            break;
        }
        case TRIAD_AST_CLASS_DECL:
        case TRIAD_AST_TYPE_DECL: {
            ss_add(scope, s->u.type_decl.name);
            ss_add(&C->types_declared, s->u.type_decl.name);
            for (size_t i = 0; i < s->u.type_decl.methods_len; ++i) {
                const TriadAstNode *m = s->u.type_decl.methods[i];
                StrSet inner; ss_copy_from(&inner, scope);
                ss_add(&inner, "self");
                for (size_t p = 0; p < m->u.fn_decl.params_len; ++p)
                    ss_add(&inner, m->u.fn_decl.params[p].name);
                check_body(C, m->u.fn_decl.body, m->u.fn_decl.body_len, &inner);
                ss_free(&inner);
            }
            break;
        }
        case TRIAD_AST_IMPORT: {
            const char *name = s->u.import_stmt.alias;
            if (!name && s->u.import_stmt.path_len > 0)
                name = s->u.import_stmt.path[0];
            if (name) ss_add(scope, name);
            break;
        }
        case TRIAD_AST_FROM_IMPORT:
            for (size_t i = 0; i < s->u.from_import.names_len; ++i) {
                const char *bound = NULL;
                if (s->u.from_import.aliases)
                    bound = s->u.from_import.aliases[i];
                if (!bound) bound = s->u.from_import.names[i];
                ss_add(scope, bound);
            }
            break;
        case TRIAD_AST_REG:
            ss_add(scope, s->u.reg_stmt.name);
            break;
        case TRIAD_AST_ENTITY:
            ss_add(scope, s->u.entity_decl.name);
            break;
        case TRIAD_AST_WORLD:
            ss_add(scope, s->u.world_decl.name);
            break;
        case TRIAD_AST_SUBSTRATE:
            if (s->u.substrate_decl.is_composed) {
                for (size_t i = 0; i < s->u.substrate_decl.members_len; ++i) {
                    const char *m = s->u.substrate_decl.members[i];
                    if (!ss_has(scope, m)) {
                        err_push(C->arena, &C->errs,
                            "error[E2012]: substrate '%s': composed_of '%s' not declared (%s)",
                            s->u.substrate_decl.name, m, fmt_pos(C->arena, s->pos));
                    }
                }
                for (size_t i = 0; i < s->u.substrate_decl.properties_len; ++i)
                    check_expr(C, s->u.substrate_decl.properties[i].value, scope);
            } else {
                for (size_t i = 0; i < s->u.substrate_decl.overrides_len; ++i)
                    check_expr(C, s->u.substrate_decl.overrides[i].value, scope);
            }
            ss_add(scope, s->u.substrate_decl.name);
            break;
        case TRIAD_AST_OBSERVE:
            if (!ss_has(scope, s->u.observe_stmt.target)) {
                err_push(C->arena, &C->errs,
                    "error[E2010]: undefined observe target '%s' (%s)",
                    s->u.observe_stmt.target, fmt_pos(C->arena, s->pos));
            }
            break;
        case TRIAD_AST_RUN:
            if (s->u.run_stmt.duration) check_expr(C, s->u.run_stmt.duration, scope);
            break;
        case TRIAD_AST_SEQUENCE:
            check_expr(C, s->u.sequence_stmt.inputs, scope);
            if (s->u.sequence_stmt.target && !ss_has(scope, s->u.sequence_stmt.target)) {
                err_push(C->arena, &C->errs,
                    "error[E2013]: undefined sequence target '%s' (%s)",
                    s->u.sequence_stmt.target, fmt_pos(C->arena, s->pos));
            }
            if (s->u.sequence_stmt.each_for)
                check_expr(C, s->u.sequence_stmt.each_for, scope);
            break;
        case TRIAD_AST_TRY_CATCH: {
            { StrSet inner; ss_copy_from(&inner, scope);
              check_body(C, s->u.try_catch.body, s->u.try_catch.body_len, &inner);
              ss_free(&inner); }
            if (s->u.try_catch.catch_body_len > 0 || s->u.try_catch.catch_var) {
                StrSet inner; ss_copy_from(&inner, scope);
                if (s->u.try_catch.catch_var) ss_add(&inner, s->u.try_catch.catch_var);
                check_body(C, s->u.try_catch.catch_body, s->u.try_catch.catch_body_len, &inner);
                ss_free(&inner);
            }
            if (s->u.try_catch.finally_body_len > 0) {
                StrSet inner; ss_copy_from(&inner, scope);
                check_body(C, s->u.try_catch.finally_body, s->u.try_catch.finally_body_len, &inner);
                ss_free(&inner);
            }
            break;
        }
        case TRIAD_AST_THROW:
            if (s->u.unary_value.value) check_expr(C, s->u.unary_value.value, scope);
            break;
        case TRIAD_AST_YIELD_STMT:
            if (s->u.unary_value.value) check_expr(C, s->u.unary_value.value, scope);
            break;
        case TRIAD_AST_COUPLE: {
            const char *names[2] = { s->u.couple_stmt.src, s->u.couple_stmt.dst };
            for (int i = 0; i < 2; ++i)
                if (!ss_has(scope, names[i]))
                    err_push(C->arena, &C->errs,
                        "error[E2011]: undefined substrate '%s' (%s)",
                        names[i], fmt_pos(C->arena, s->pos));
            break;
        }
        case TRIAD_AST_PAIR: {
            const char *names[2] = { s->u.pair_stmt.a, s->u.pair_stmt.b };
            for (int i = 0; i < 2; ++i)
                if (!ss_has(scope, names[i]))
                    err_push(C->arena, &C->errs,
                        "error[E2011]: undefined substrate '%s' (%s)",
                        names[i], fmt_pos(C->arena, s->pos));
            break;
        }
        case TRIAD_AST_RING:
            for (size_t i = 0; i < s->u.ring_stmt.members_len; ++i)
                if (!ss_has(scope, s->u.ring_stmt.members[i]))
                    err_push(C->arena, &C->errs,
                        "error[E2011]: undefined substrate '%s' (%s)",
                        s->u.ring_stmt.members[i], fmt_pos(C->arena, s->pos));
            break;
        default:

            break;
    }
}

int triad_check_module(TriadArena *arena, const TriadAstNode *module,
                       TriadCheckErrors *out) {
    if (!module || module->kind != TRIAD_AST_MODULE) return -1;

    Ctx C = {0};
    C.arena = arena;
    ss_init(&C.types_declared, 16);

    StrSet scope; ss_init(&scope, 64);
    for (size_t k = 0; BUILTINS[k]; ++k) ss_add(&scope, BUILTINS[k]);

    check_body(&C, module->u.module.body, module->u.module.body_len, &scope);

    int failed = (C.errs.len > 0) ? -1 : 0;

    if (out) {
        out->len = C.errs.len;
        if (C.errs.len) {
            const char **arr = (const char **)triad_arena_alloc(arena, C.errs.len * sizeof(char *));
            for (size_t k = 0; k < C.errs.len; ++k) arr[k] = C.errs.items[k];
            out->items = arr;
        } else out->items = NULL;
    }

    ss_free(&scope);
    ss_free(&C.types_declared);
    fnmap_free(&C.functions);
    free(C.errs.items);
    return failed;
}
