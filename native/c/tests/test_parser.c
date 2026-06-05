/* test_parser.c — Native parser self-tests. */
#include "triad_frontend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

static TriadAstNode *parse_or_die(TriadArena *a, const char *src) {
    TriadDiag diag = {0};
    TriadAstNode *m = triad_parse_source(a, src, "<test>", &diag);
    if (!m) {
        fprintf(stderr, "parse error: %s @ L%d:C%d\n",
                diag.msg ? diag.msg : "?", diag.line, diag.col);
        exit(1);
    }
    return m;
}

static void test_let_expr_stmt(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadAstNode *m = parse_or_die(&a, "let x = 1 + 2");
    CHECK(m->kind == TRIAD_AST_MODULE);
    CHECK(m->u.module.body_len == 1);
    TriadAstNode *s = m->u.module.body[0];
    CHECK(s->kind == TRIAD_AST_LET);
    CHECK(strcmp(s->u.let_stmt.name, "x") == 0);
    CHECK(s->u.let_stmt.value->kind == TRIAD_AST_BINOP);
    CHECK(strcmp(s->u.let_stmt.value->u.op.op, "+") == 0);
    triad_arena_free(&a);
}

static void test_fn_decl_and_call(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadAstNode *m = parse_or_die(&a,
        "fn add(a, b) { return a + b }\n"
        "let r = add(1, 2)");
    CHECK(m->u.module.body_len == 2);
    TriadAstNode *fn = m->u.module.body[0];
    CHECK(fn->kind == TRIAD_AST_FN_DECL);
    CHECK(fn->u.fn_decl.params_len == 2);
    CHECK(strcmp(fn->u.fn_decl.params[0].name, "a") == 0);
    CHECK(fn->u.fn_decl.body_len == 1);
    CHECK(fn->u.fn_decl.body[0]->kind == TRIAD_AST_RETURN);
    TriadAstNode *let = m->u.module.body[1];
    CHECK(let->kind == TRIAD_AST_LET);
    CHECK(let->u.let_stmt.value->kind == TRIAD_AST_CALL);
    CHECK(let->u.let_stmt.value->u.call.args_len == 2);
    triad_arena_free(&a);
}

static void test_if_elif_else(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadAstNode *m = parse_or_die(&a,
        "if x > 0 { return 1 } elif x < 0 { return -1 } else { return 0 }");
    TriadAstNode *iff = m->u.module.body[0];
    CHECK(iff->kind == TRIAD_AST_IF);
    CHECK(iff->u.if_stmt.elif_clauses_len == 1);
    CHECK(iff->u.if_stmt.has_else == 1);
    CHECK(iff->u.if_stmt.else_body_len == 1);
    triad_arena_free(&a);
}

static void test_fstring_with_expr(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadAstNode *m = parse_or_die(&a, "print(f\"x={n + 1}\")");
    TriadAstNode *call = m->u.module.body[0]->u.expr_stmt.expr;
    CHECK(call->kind == TRIAD_AST_CALL);
    TriadAstNode *fs = call->u.call.args[0];
    CHECK(fs->kind == TRIAD_AST_FSTRING);
    CHECK(fs->u.fstring.parts_len == 2);
    CHECK(fs->u.fstring.parts[1].is_expr == 1);
    CHECK(fs->u.fstring.parts[1].expr->kind == TRIAD_AST_BINOP);
    triad_arena_free(&a);
}

static void test_list_and_index(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadAstNode *m = parse_or_die(&a, "let xs = [1, 2, 3]; let y = xs[0]");
    TriadAstNode *xs = m->u.module.body[0]->u.let_stmt.value;
    CHECK(xs->kind == TRIAD_AST_LIST && xs->u.list.elements_len == 3);
    TriadAstNode *y = m->u.module.body[1]->u.let_stmt.value;
    CHECK(y->kind == TRIAD_AST_INDEX);
    triad_arena_free(&a);
}

static void test_slice(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadAstNode *m = parse_or_die(&a, "let y = xs[1:4:2]");
    TriadAstNode *ix = m->u.module.body[0]->u.let_stmt.value;
    CHECK(ix->kind == TRIAD_AST_INDEX);
    CHECK(ix->u.index.index->kind == TRIAD_AST_SLICE);
    CHECK(ix->u.index.index->u.slice.start != NULL);
    CHECK(ix->u.index.index->u.slice.end   != NULL);
    CHECK(ix->u.index.index->u.slice.step  != NULL);
    triad_arena_free(&a);
}

static void test_couple_pair_ring(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadAstNode *m = parse_or_die(&a,
        "couple a -> b kappa = 0.1\n"
        "pair (a, b) kappa = 0.2 for T = 5.0\n"
        "ring (a, b, c) kappa = 0.05 for T = 2.0");
    CHECK(m->u.module.body[0]->kind == TRIAD_AST_COUPLE);
    CHECK(strcmp(m->u.module.body[0]->u.couple_stmt.src, "a") == 0);
    CHECK(strcmp(m->u.module.body[0]->u.couple_stmt.dst, "b") == 0);
    CHECK(m->u.module.body[1]->kind == TRIAD_AST_PAIR);
    CHECK(m->u.module.body[1]->u.pair_stmt.duration != NULL);
    CHECK(m->u.module.body[2]->kind == TRIAD_AST_RING);
    CHECK(m->u.module.body[2]->u.ring_stmt.members_len == 3);
    triad_arena_free(&a);
}

static void test_reg_with_overrides(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadAstNode *m = parse_or_die(&a, "reg s: anti_collapse { L: 64.0; N: 256 }");
    TriadAstNode *r = m->u.module.body[0];
    CHECK(r->kind == TRIAD_AST_REG);
    CHECK(strcmp(r->u.reg_stmt.name, "s") == 0);
    CHECK(strcmp(r->u.reg_stmt.regime, "anti_collapse") == 0);
    CHECK(r->u.reg_stmt.has_overrides);
    CHECK(r->u.reg_stmt.overrides_len == 2);
    triad_arena_free(&a);
}

static void test_observe(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadAstNode *m = parse_or_die(&a, "observe s crystallinity, k_star over_seeds = 3");
    TriadAstNode *o = m->u.module.body[0];
    CHECK(o->kind == TRIAD_AST_OBSERVE);
    CHECK(o->u.observe_stmt.metrics_len == 2);
    CHECK(o->u.observe_stmt.over_seeds == 3);
    triad_arena_free(&a);
}

static void test_match(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadAstNode *m = parse_or_die(&a,
        "match x {\n"
        "  case 0 => { return \"zero\" }\n"
        "  case 1 => { return \"one\" }\n"
        "  else => { return \"other\" }\n"
        "}");
    TriadAstNode *mt = m->u.module.body[0];
    CHECK(mt->kind == TRIAD_AST_MATCH);
    CHECK(mt->u.match_stmt.cases_len == 2);
    CHECK(mt->u.match_stmt.has_else);
    triad_arena_free(&a);
}

static void test_class_with_methods(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadAstNode *m = parse_or_die(&a,
        "class Point { x: int; y: int; fn norm() { return self.x*self.x + self.y*self.y } }");
    TriadAstNode *c = m->u.module.body[0];
    CHECK(c->kind == TRIAD_AST_CLASS_DECL);
    CHECK(c->u.type_decl.fields_len == 2);
    CHECK(c->u.type_decl.methods_len == 1);
    triad_arena_free(&a);
}

static void test_parse_error_diag(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadDiag diag = {0};
    TriadAstNode *m = triad_parse_source(&a, "let = 1", "<t>", &diag);
    CHECK(m == NULL);
    CHECK(diag.msg != NULL);
    CHECK(diag.line == 1);
    triad_arena_free(&a);
}

int main(void) {
    test_let_expr_stmt();
    test_fn_decl_and_call();
    test_if_elif_else();
    test_fstring_with_expr();
    test_list_and_index();
    test_slice();
    test_couple_pair_ring();
    test_reg_with_overrides();
    test_observe();
    test_match();
    test_class_with_methods();
    test_parse_error_diag();
    if (failures == 0) {
        printf("test_parser: OK\n");
        return 0;
    }
    fprintf(stderr, "test_parser: %d FAILURE(S)\n", failures);
    return 1;
}
